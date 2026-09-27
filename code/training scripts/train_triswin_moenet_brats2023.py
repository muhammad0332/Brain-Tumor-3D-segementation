"""
TriSwin-MoENet training for BraTS 2023 GLI.

Architecture:
    3 x SwinUNETR experts + learned sample-wise gating + 1x1 fusion.

Targets:
    3 overlapping BraTS regions:
      channel 0: WT = labels {1, 2, 3}
      channel 1: TC = labels {1, 3}
      channel 2: ET = label  {3}

Expected JSON format:
{
  "training": [
    {
      "image": [t1c_path, t1n_path, t2f_path, t2w_path],
      "label": seg_path
    },
    ...
  ]
}

The code is intentionally configured conservatively for a ~24 GB GPU.
Increase batch size / sliding-window batch size only after checking VRAM usage.
"""

from __future__ import annotations

import inspect
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split

import monai
from monai.config import print_config
from monai.data import DataLoader, Dataset, decollate_batch, load_decathlon_datalist
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from monai.networks.nets import SwinUNETR
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    MapTransform,
    NormalizeIntensityd,
    Orientationd,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    SpatialPadd,
    Spacingd,
)
from monai.utils import set_determinism
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger


# =============================================================================
# 0. CONFIGURATION
# =============================================================================

JSON_PATH = Path(
    "/home/ali/R_and_D/BTS/2023/dataset/preprocessed_2023_dataset.json"
)
OUTPUT_DIR = Path(
    "/home/ali/R_and_D/BTS/2023/model/TriSwin_MoENet_2023"
)

SEED = 42
VAL_FRACTION = 0.10

ROI_SIZE: Tuple[int, int, int] = (96, 96, 96)
IN_CHANNELS = 4
OUT_CHANNELS = 3  # WT, TC, ET as overlapping sigmoid channels
NUM_EXPERTS = 3
FEATURE_SIZE = 48

BATCH_SIZE = 1
NUM_WORKERS = 4
NUM_SAMPLES = 1

MAX_EPOCHS = 130
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
PATIENCE = 15

SW_BATCH_SIZE = 1
SW_OVERLAP = 0.50
THRESHOLD = 0.50

PRECISION = "16-mixed"
GRADIENT_CLIP_VAL = 1.0

# Set to an integer (e.g. 20) only for debugging. None validates on the full set.
LIMIT_VAL_BATCHES = None


# =============================================================================
# 1. REPRODUCIBILITY / ENVIRONMENT
# =============================================================================

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
torch.set_float32_matmul_precision("high")


def seed_everything(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    set_determinism(seed=seed)


# =============================================================================
# 2. DATA VALIDATION AND BRATS TARGET CONVERSION
# =============================================================================


def validate_json(json_path: Path) -> None:
    if not json_path.is_file():
        raise FileNotFoundError(f"Dataset JSON not found: {json_path}")

    with json_path.open("r") as f:
        obj = json.load(f)

    items = obj.get("training", [])
    if not items:
        raise ValueError("JSON contains no non-empty 'training' list.")

    bad = []
    for i, item in enumerate(items):
        images = item.get("image", [])
        label = item.get("label")
        if len(images) != 4 or label is None:
            bad.append((i, "expected 4 image paths and 1 label path"))
            continue
        missing = [p for p in images + [label] if not Path(p).is_file()]
        if missing:
            bad.append((i, f"missing file: {missing[0]}"))

    if bad:
        lines = "\n".join(f"  entry {i}: {msg}" for i, msg in bad[:10])
        raise RuntimeError(
            f"Found {len(bad)} invalid JSON entries. First entries:\n{lines}"
        )

    print(f"Validated {len(items)} BraTS 2023 cases from: {json_path}")


class ConvertBraTS2023ToRegionsd(MapTransform):
    """
    Convert BraTS scalar segmentation to 3 overlapping binary channels.

    BraTS 2023 GLI labels used here:
      0 = background
      1 = NCR/NET
      2 = edema / surrounding non-enhancing component
      3 = enhancing tumor

    Output:
      [WT, TC, ET]
    """

    def __init__(self, keys):
        super().__init__(keys)

    def __call__(self, data):
        d = dict(data)

        for key in self.keys:
            label = d[key]

            # Ensure scalar spatial map [H, W, D]
            if label.ndim == 4 and label.shape[0] == 1:
                label = label[0]

            unique = torch.unique(label).detach().cpu().tolist()
            unexpected = set(int(v) for v in unique) - {0, 1, 2, 3}
            if unexpected:
                raise ValueError(
                    f"Unexpected BraTS labels {sorted(unexpected)}. "
                    "Expected only 0, 1, 2, 3."
                )

            wt = (label == 1) | (label == 2) | (label == 3)
            tc = (label == 1) | (label == 3)
            et = label == 3

            d[key] = torch.stack([wt, tc, et], dim=0).float()

        return d


# =============================================================================
# 3. TRANSFORMS AND DATA MODULE
# =============================================================================


def build_transforms():
    common = [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        EnsureTyped(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(
            keys=["image", "label"],
            pixdim=(1.0, 1.0, 1.0),
            mode=("bilinear", "nearest"),
        ),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
    ]

    # Crop sampling needs a scalar foreground label, so region conversion is
    # intentionally applied after sampling.
    train_transform = Compose(
        common
        + [
            SpatialPadd(keys=["image", "label"], spatial_size=ROI_SIZE),
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=ROI_SIZE,
                pos=1,
                neg=1,
                num_samples=NUM_SAMPLES,
                image_key="image",
                image_threshold=0,
            ),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
            RandScaleIntensityd(keys="image", factors=0.10, prob=0.5),
            RandShiftIntensityd(keys="image", offsets=0.10, prob=0.5),
            ConvertBraTS2023ToRegionsd(keys=["label"]),
            EnsureTyped(keys=["image", "label"], dtype=torch.float32),
        ]
    )

    val_transform = Compose(
        common
        + [
            ConvertBraTS2023ToRegionsd(keys=["label"]),
            EnsureTyped(keys=["image", "label"], dtype=torch.float32),
        ]
    )

    return train_transform, val_transform


class BraTS2023DataModule(pl.LightningDataModule):
    def __init__(self):
        super().__init__()
        self.train_files = None
        self.val_files = None
        self.train_ds = None
        self.val_ds = None

    def setup(self, stage=None):
        validate_json(JSON_PATH)
        datalist = load_decathlon_datalist(str(JSON_PATH), True, "training")

        self.train_files, self.val_files = train_test_split(
            datalist,
            test_size=VAL_FRACTION,
            random_state=SEED,
            shuffle=True,
        )

        print(f"Total cases : {len(datalist)}")
        print(f"Train cases : {len(self.train_files)}")
        print(f"Val cases   : {len(self.val_files)}")

        train_transform, val_transform = build_transforms()
        self.train_ds = Dataset(self.train_files, transform=train_transform)
        self.val_ds = Dataset(self.val_files, transform=val_transform)

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=NUM_WORKERS,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=NUM_WORKERS > 0,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=1,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=NUM_WORKERS > 0,
        )


# =============================================================================
# 4. TRISWIN-MOENET
# =============================================================================


def build_swin_unetr(
    in_channels: int,
    out_channels: int,
    feature_size: int,
    roi_size: Tuple[int, int, int],
) -> nn.Module:
    """Build SwinUNETR across older/newer MONAI constructor variants."""
    params = inspect.signature(SwinUNETR.__init__).parameters
    kwargs = dict(
        in_channels=in_channels,
        out_channels=out_channels,
        feature_size=feature_size,
        use_checkpoint=True,
    )
    if "img_size" in params:
        kwargs["img_size"] = roi_size
    return SwinUNETR(**kwargs)


class SwinExpert(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = build_swin_unetr(
            in_channels=IN_CHANNELS,
            out_channels=OUT_CHANNELS,
            feature_size=FEATURE_SIZE,
            roi_size=ROI_SIZE,
        )

    def forward(self, x):
        return self.model(x)


class GatingNetwork(nn.Module):
    """Sample-wise learned gating over the three Swin experts."""

    def __init__(self, in_channels=4, num_experts=3, hidden_dim=64):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv3d(in_channels, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.InstanceNorm3d(32, affine=True),
            nn.GELU(),
            nn.Conv3d(32, hidden_dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.InstanceNorm3d(hidden_dim, affine=True),
            nn.GELU(),
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
        )
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(hidden_dim, num_experts),
        )

    def forward(self, x):
        logits = self.gate(self.features(x))
        return torch.softmax(logits, dim=1)


class TriSwinMoENet(nn.Module):
    """
    Three SwinUNETR experts with learned adaptive fusion.

    Each expert predicts WT/TC/ET logits. The gate assigns one scalar weight per
    expert and subject/crop. Weighted logits are concatenated, then a learnable
    1x1x1 convolution produces the final 3-channel logits.
    """

    def __init__(self):
        super().__init__()
        self.experts = nn.ModuleList([SwinExpert() for _ in range(NUM_EXPERTS)])
        self.gate = GatingNetwork(
            in_channels=IN_CHANNELS,
            num_experts=NUM_EXPERTS,
            hidden_dim=64,
        )
        self.fusion = nn.Conv3d(
            OUT_CHANNELS * NUM_EXPERTS,
            OUT_CHANNELS,
            kernel_size=1,
            bias=True,
        )

    def forward(self, x):
        weights = self.gate(x)  # [B, 3]
        weighted_logits = []

        for i, expert in enumerate(self.experts):
            logits = expert(x)
            w = weights[:, i].view(-1, 1, 1, 1, 1)
            weighted_logits.append(logits * w)

        fused = torch.cat(weighted_logits, dim=1)
        return self.fusion(fused)


# =============================================================================
# 5. LIGHTNING MODULE
# =============================================================================


class TriSwinMoELightning(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.save_hyperparameters()

        self.model = TriSwinMoENet()

        # Proper loss for overlapping WT/TC/ET channels.
        self.loss_function = DiceCELoss(
            sigmoid=True,
            to_onehot_y=False,
            include_background=True,
            lambda_dice=1.0,
            lambda_ce=1.0,
        )

        self.train_dice = DiceMetric(
            include_background=True,
            reduction="mean",
            ignore_empty=False,
        )
        self.val_dice = DiceMetric(
            include_background=True,
            reduction="mean",
            ignore_empty=False,
        )
        self.val_dice_batch = DiceMetric(
            include_background=True,
            reduction="mean_batch",
            ignore_empty=False,
        )
        self.val_hd95_batch = HausdorffDistanceMetric(
            include_background=True,
            distance_metric="euclidean",
            percentile=95,
            directed=False,
            reduction="mean_batch",
            get_not_nans=False,
        )

    def forward(self, x):
        return self.model(x)

    @staticmethod
    def logits_to_binary(logits: torch.Tensor) -> torch.Tensor:
        return (torch.sigmoid(logits) >= THRESHOLD).float()

    def training_step(self, batch, batch_idx):
        images = batch["image"]
        labels = batch["label"]

        logits = self(images)
        loss = self.loss_function(logits, labels)

        preds = self.logits_to_binary(logits.detach())
        self.train_dice(y_pred=preds, y=labels)

        self.log(
            "train_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=images.shape[0],
        )
        return loss

    def on_train_epoch_end(self):
        dice = self.train_dice.aggregate()
        self.train_dice.reset()
        self.log("train_mean_dice", dice, prog_bar=True, logger=True)

    def validation_step(self, batch, batch_idx):
        images = batch["image"]
        labels = batch["label"]

        logits = sliding_window_inference(
            inputs=images,
            roi_size=ROI_SIZE,
            sw_batch_size=SW_BATCH_SIZE,
            predictor=self.forward,
            overlap=SW_OVERLAP,
            mode="gaussian",
        )

        loss = self.loss_function(logits, labels)
        preds = self.logits_to_binary(logits)

        self.val_dice(y_pred=preds, y=labels)
        self.val_dice_batch(y_pred=preds, y=labels)
        self.val_hd95_batch(y_pred=preds, y=labels)

        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=images.shape[0],
            sync_dist=True,
        )

    def on_validation_epoch_end(self):
        mean_dice = self.val_dice.aggregate()
        dice_regions = self.val_dice_batch.aggregate()
        hd95_regions = self.val_hd95_batch.aggregate()

        self.val_dice.reset()
        self.val_dice_batch.reset()
        self.val_hd95_batch.reset()

        # Region order: WT, TC, ET
        wt_dice = dice_regions[0]
        tc_dice = dice_regions[1]
        et_dice = dice_regions[2]

        wt_hd95 = hd95_regions[0]
        tc_hd95 = hd95_regions[1]
        et_hd95 = hd95_regions[2]
        mean_hd95 = torch.nanmean(hd95_regions)

        self.log("val_mean_dice", mean_dice, prog_bar=True, logger=True, sync_dist=True)
        self.log("val_wt_dice", wt_dice, logger=True, sync_dist=True)
        self.log("val_tc_dice", tc_dice, logger=True, sync_dist=True)
        self.log("val_et_dice", et_dice, logger=True, sync_dist=True)
        self.log("val_mean_hd95", mean_hd95, prog_bar=True, logger=True, sync_dist=True)
        self.log("val_wt_hd95", wt_hd95, logger=True, sync_dist=True)
        self.log("val_tc_hd95", tc_hd95, logger=True, sync_dist=True)
        self.log("val_et_hd95", et_hd95, logger=True, sync_dist=True)

        if self.trainer.is_global_zero:
            print(
                f"\nEpoch {self.current_epoch:03d} | "
                f"Mean Dice={float(mean_dice):.4f} | "
                f"WT={float(wt_dice):.4f}, TC={float(tc_dice):.4f}, ET={float(et_dice):.4f} | "
                f"Mean HD95={float(mean_hd95):.3f} mm"
            )

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=MAX_EPOCHS,
            eta_min=1e-6,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }


# =============================================================================
# 6. TRAINING
# =============================================================================


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    seed_everything(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print_config()
    print(f"PyTorch Lightning: {pl.__version__}")
    print(f"PyTorch          : {torch.__version__}")
    print(f"MONAI            : {monai.__version__}")
    print(f"CUDA available   : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU              : {torch.cuda.get_device_name(0)}")

    data_module = BraTS2023DataModule()
    model = TriSwinMoELightning()

    total_params = count_parameters(model)
    print(f"Trainable parameters: {total_params:,} ({total_params / 1e6:.2f} M)")

    checkpoint_callback = ModelCheckpoint(
        dirpath=str(OUTPUT_DIR / "checkpoints"),
        filename="triswin_moenet_brats2023-epoch{epoch:03d}-dice{val_mean_dice:.4f}",
        monitor="val_mean_dice",
        mode="max",
        save_top_k=1,
        save_last=True,
        auto_insert_metric_name=False,
    )

    early_stop_callback = EarlyStopping(
        monitor="val_mean_dice",
        mode="max",
        patience=PATIENCE,
        min_delta=1e-4,
        verbose=True,
    )

    logger = CSVLogger(
        save_dir=str(OUTPUT_DIR),
        name="logs",
    )

    trainer_kwargs = dict(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        precision=PRECISION if torch.cuda.is_available() else "32-true",
        max_epochs=MAX_EPOCHS,
        check_val_every_n_epoch=1,
        callbacks=[checkpoint_callback, early_stop_callback],
        logger=logger,
        default_root_dir=str(OUTPUT_DIR),
        num_sanity_val_steps=0,
        gradient_clip_val=GRADIENT_CLIP_VAL,
        log_every_n_steps=10,
    )

    if LIMIT_VAL_BATCHES is not None:
        trainer_kwargs["limit_val_batches"] = LIMIT_VAL_BATCHES

    trainer = pl.Trainer(**trainer_kwargs)
    trainer.fit(model=model, datamodule=data_module)

    print("\nTraining finished.")
    print(f"Best checkpoint: {checkpoint_callback.best_model_path}")
    print(f"Best val Dice  : {float(checkpoint_callback.best_model_score):.4f}" if checkpoint_callback.best_model_score is not None else "Best val Dice: N/A")


if __name__ == "__main__":
    main()
