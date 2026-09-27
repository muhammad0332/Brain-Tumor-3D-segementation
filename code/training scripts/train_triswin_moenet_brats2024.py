"""
TriSwin-MoENet training for BraTS 2024 post-treatment GLI.

Architecture:
    3 x SwinUNETR experts + learned sample-wise gating + 1x1x1 fusion.

BraTS 2024 post-treatment scalar labels assumed here:
    0 = background
    1 = NETC (non-enhancing tumor core)
    2 = SNFH (surrounding non-enhancing FLAIR hyperintensity)
    3 = ET   (enhancing tumor)
    4 = RC   (resection cavity)

Network targets are four independent/overlapping sigmoid channels:
    channel 0: WT = NETC | SNFH | ET
    channel 1: TC = NETC | ET
    channel 2: ET = ET
    channel 3: RC = RC (independent of the nested WT/TC/ET hierarchy)

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

The default configuration is conservative for a ~24 GB GPU because three full
SwinUNETR experts are resident in memory simultaneously.
"""

from __future__ import annotations

import inspect
import json
import os
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl

import monai
from monai.config import print_config
from monai.data import DataLoader, Dataset
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
    RandGaussianNoised,
    RandScaleIntensityd,
    RandShiftIntensityd,
    SpatialPadd,
    Spacingd,
)
from monai.utils import set_determinism
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import CSVLogger


# =============================================================================
# 0. CONFIGURATION
# =============================================================================

JSON_PATH = Path(
    "/home/ali/R_and_D/BTS/2024/dataset/preprocessed_2024_dataset.json"
)
OUTPUT_DIR = Path(
    "/home/ali/R_and_D/BTS/2024/model/TriSwin_MoENet_2024"
)

SEED = 2024
VAL_FRACTION = 0.10

ROI_SIZE: Tuple[int, int, int] = (96, 96, 96)
IN_CHANNELS = 4
OUT_CHANNELS = 4  # WT, TC, ET, RC
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

# Set to an integer such as 20 only for debugging. None validates full set.
LIMIT_VAL_BATCHES = None

# Set True to save the deterministic split for reproducibility.
SAVE_SPLIT_JSON = True


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
# 2. DATA LOADING / VALIDATION / GROUP-AWARE SPLIT
# =============================================================================


def case_id_from_item(item: dict) -> str:
    name = Path(item["label"]).name
    if name.endswith("-seg.nii.gz"):
        return name[: -len("-seg.nii.gz")]
    if name.endswith(".nii.gz"):
        return name[: -len(".nii.gz")]
    return Path(name).stem


def subject_id_from_case_id(case_id: str) -> str:
    """
    Group longitudinal/repeated exams when IDs end in a numeric exam token.

    Example:
        BraTS-...-00005-100 -> BraTS-...-00005
        BraTS-...-00005-101 -> BraTS-...-00005

    If your IDs do not encode repeated exams this way, each case naturally
    remains its own group or you can replace this function with your mapping.
    """
    parts = case_id.rsplit("-", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0]
    return case_id


def load_and_validate_json(json_path: Path) -> List[dict]:
    if not json_path.is_file():
        raise FileNotFoundError(f"Dataset JSON not found: {json_path}")

    with json_path.open("r") as f:
        obj = json.load(f)

    items = obj.get("training", [])
    if not items:
        raise ValueError("JSON contains no non-empty 'training' list.")

    valid = []
    problems = []

    for i, item in enumerate(items):
        images = item.get("image", [])
        label = item.get("label")

        if len(images) != 4 or label is None:
            problems.append((i, "expected 4 image paths and 1 label path"))
            continue

        missing = [p for p in images + [label] if not Path(p).is_file()]
        if missing:
            problems.append((i, f"missing file: {missing[0]}"))
            continue

        case_id = case_id_from_item(item)
        valid.append(
            {
                "image": images,
                "label": label,
                "case_id": case_id,
                "subject_id": subject_id_from_case_id(case_id),
            }
        )

    if problems:
        print(f"WARNING: skipped {len(problems)} invalid JSON entries. First 10:")
        for idx, msg in problems[:10]:
            print(f"  entry {idx}: {msg}")

    if not valid:
        raise RuntimeError("No valid cases remain after JSON/path validation.")

    valid = sorted(valid, key=lambda x: x["case_id"])
    print(f"Validated {len(valid)} BraTS 2024 cases from: {json_path}")
    return valid


def group_holdout_split(
    items: List[dict],
    val_fraction: float,
    seed: int,
) -> Tuple[List[dict], List[dict]]:
    """Deterministic subject-level split to reduce longitudinal leakage."""
    subject_ids = sorted({x["subject_id"] for x in items})
    if len(subject_ids) < 2:
        raise RuntimeError("Need at least two distinct subject groups for a split.")

    rng = np.random.default_rng(seed)
    rng.shuffle(subject_ids)

    n_val_subjects = max(1, int(round(len(subject_ids) * val_fraction)))
    n_val_subjects = min(n_val_subjects, len(subject_ids) - 1)

    val_subjects = set(subject_ids[:n_val_subjects])
    train_items = [x for x in items if x["subject_id"] not in val_subjects]
    val_items = [x for x in items if x["subject_id"] in val_subjects]

    train_subjects = {x["subject_id"] for x in train_items}
    overlap = train_subjects & val_subjects
    if overlap:
        raise RuntimeError(f"Subject leakage detected: {sorted(overlap)[:5]}")

    if not train_items or not val_items:
        raise RuntimeError("Train/validation split produced an empty partition.")

    return train_items, val_items


def save_split(train_items: List[dict], val_items: List[dict], output_path: Path) -> None:
    def clean(x: dict) -> dict:
        return {"image": x["image"], "label": x["label"], "case_id": x["case_id"], "subject_id": x["subject_id"]}

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(
            {
                "seed": SEED,
                "val_fraction": VAL_FRACTION,
                "training": [clean(x) for x in train_items],
                "validation": [clean(x) for x in val_items],
            },
            f,
            indent=2,
        )


# =============================================================================
# 3. BRATS 2024 TARGET CONVERSION
# =============================================================================


class ConvertBraTS2024ToRegionsd(MapTransform):
    """
    Convert scalar BraTS 2024 post-treatment labels to four channels.

    Scalar labels:
      0 background
      1 NETC
      2 SNFH
      3 ET
      4 RC

    Output channels:
      0 WT = NETC | SNFH | ET
      1 TC = NETC | ET
      2 ET = ET
      3 RC = RC
    """

    def __init__(self, keys):
        super().__init__(keys)

    def __call__(self, data):
        d = dict(data)

        for key in self.keys:
            label = d[key]

            # LoadImaged + EnsureChannelFirstd normally gives [1, H, W, D].
            if label.ndim == 4 and label.shape[0] == 1:
                label = label[0]

            unique = torch.unique(label).detach().cpu().tolist()
            unexpected = set(int(v) for v in unique) - {0, 1, 2, 3, 4}
            if unexpected:
                raise ValueError(
                    f"Unexpected BraTS 2024 labels {sorted(unexpected)}. "
                    "Expected only 0, 1, 2, 3, 4."
                )

            wt = (label == 1) | (label == 2) | (label == 3)
            tc = (label == 1) | (label == 3)
            et = label == 3
            rc = label == 4

            d[key] = torch.stack([wt, tc, et, rc], dim=0).float()

        return d


# =============================================================================
# 4. MONAI TRANSFORMS / DATA MODULE
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

    # Sampling is performed on the original scalar segmentation, then converted
    # to the four-channel target representation.
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
            RandGaussianNoised(keys="image", prob=0.15, mean=0.0, std=0.05),
            ConvertBraTS2024ToRegionsd(keys=["label"]),
            EnsureTyped(keys=["image", "label"], dtype=torch.float32),
        ]
    )

    val_transform = Compose(
        common
        + [
            ConvertBraTS2024ToRegionsd(keys=["label"]),
            EnsureTyped(keys=["image", "label"], dtype=torch.float32),
        ]
    )

    return train_transform, val_transform


class BraTS2024DataModule(pl.LightningDataModule):
    def __init__(self):
        super().__init__()
        self.train_files = None
        self.val_files = None
        self.train_ds = None
        self.val_ds = None

    def setup(self, stage=None):
        items = load_and_validate_json(JSON_PATH)
        self.train_files, self.val_files = group_holdout_split(
            items=items,
            val_fraction=VAL_FRACTION,
            seed=SEED,
        )

        print(f"Total cases    : {len(items)}")
        print(f"Train cases    : {len(self.train_files)}")
        print(f"Validation cases: {len(self.val_files)}")
        print(f"Train subjects : {len({x['subject_id'] for x in self.train_files})}")
        print(f"Val subjects   : {len({x['subject_id'] for x in self.val_files})}")

        if SAVE_SPLIT_JSON:
            split_path = OUTPUT_DIR / "brats2024_train_val_split.json"
            save_split(self.train_files, self.val_files, split_path)
            print(f"Saved split: {split_path}")

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
# 5. TRISWIN-MOENET
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
    """Learn one sample-wise weight per SwinUNETR expert."""

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
        return torch.softmax(self.gate(self.features(x)), dim=1)


class TriSwinMoENet(nn.Module):
    """
    Three SwinUNETR experts with learned adaptive fusion.

    Each expert predicts WT/TC/ET/RC logits. A gating network produces one
    scalar weight per expert for each input crop/volume. Weighted expert logits
    are concatenated and fused by a trainable 1x1x1 convolution.
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

    def forward(self, x, return_gates: bool = False):
        weights = self.gate(x)  # [B, 3]
        weighted_logits = []

        for i, expert in enumerate(self.experts):
            logits = expert(x)
            w = weights[:, i].view(-1, 1, 1, 1, 1)
            weighted_logits.append(logits * w)

        fused = torch.cat(weighted_logits, dim=1)
        output = self.fusion(fused)

        if return_gates:
            return output, weights
        return output


# =============================================================================
# 6. LIGHTNING MODULE
# =============================================================================


class TriSwinMoELightning(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.save_hyperparameters()

        self.model = TriSwinMoENet()

        # Four channels are region masks, not a mutually exclusive softmax map.
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
    def logits_to_regions(logits: torch.Tensor) -> torch.Tensor:
        """
        Threshold sigmoid outputs and enforce the known nested hierarchy:
          ET subset TC subset WT.
        RC remains independent.
        """
        p = (torch.sigmoid(logits) >= THRESHOLD).float()

        wt_raw = p[:, 0:1]
        tc_raw = p[:, 1:2]
        et = p[:, 2:3]
        rc = p[:, 3:4]

        tc = torch.maximum(tc_raw, et)
        wt = torch.maximum(wt_raw, tc)

        return torch.cat([wt, tc, et, rc], dim=1)

    def training_step(self, batch, batch_idx):
        images = batch["image"]
        labels = batch["label"]

        logits, gate_weights = self.model(images, return_gates=True)
        loss = self.loss_function(logits, labels)

        preds = self.logits_to_regions(logits.detach())
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

        # Useful diagnostics to verify the gate is not always choosing one expert.
        for i in range(NUM_EXPERTS):
            self.log(
                f"train_gate_{i + 1}",
                gate_weights[:, i].mean(),
                on_step=False,
                on_epoch=True,
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
        preds = self.logits_to_regions(logits)

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

        # Region order: WT, TC, ET, RC
        wt_dice, tc_dice, et_dice, rc_dice = [dice_regions[i] for i in range(4)]
        wt_hd95, tc_hd95, et_hd95, rc_hd95 = [hd95_regions[i] for i in range(4)]
        mean_hd95 = torch.nanmean(hd95_regions)

        self.log("val_mean_dice", mean_dice, prog_bar=True, logger=True, sync_dist=True)
        self.log("val_wt_dice", wt_dice, logger=True, sync_dist=True)
        self.log("val_tc_dice", tc_dice, logger=True, sync_dist=True)
        self.log("val_et_dice", et_dice, logger=True, sync_dist=True)
        self.log("val_rc_dice", rc_dice, logger=True, sync_dist=True)

        self.log("val_mean_hd95", mean_hd95, prog_bar=True, logger=True, sync_dist=True)
        self.log("val_wt_hd95", wt_hd95, logger=True, sync_dist=True)
        self.log("val_tc_hd95", tc_hd95, logger=True, sync_dist=True)
        self.log("val_et_hd95", et_hd95, logger=True, sync_dist=True)
        self.log("val_rc_hd95", rc_hd95, logger=True, sync_dist=True)

        if self.trainer.is_global_zero:
            print(
                f"\nEpoch {self.current_epoch:03d} | "
                f"Mean Dice={float(mean_dice):.4f} | "
                f"WT={float(wt_dice):.4f}, "
                f"TC={float(tc_dice):.4f}, "
                f"ET={float(et_dice):.4f}, "
                f"RC={float(rc_dice):.4f} | "
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
# 7. TRAINING
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

    data_module = BraTS2024DataModule()
    model = TriSwinMoELightning()

    total_params = count_parameters(model)
    print(f"Trainable parameters: {total_params:,} ({total_params / 1e6:.2f} M)")

    checkpoint_callback = ModelCheckpoint(
        dirpath=str(OUTPUT_DIR / "checkpoints"),
        filename="triswin_moenet_brats2024-epoch{epoch:03d}-dice{val_mean_dice:.4f}",
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

    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    logger = CSVLogger(save_dir=str(OUTPUT_DIR), name="logs")

    trainer_kwargs = dict(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        precision=PRECISION if torch.cuda.is_available() else "32-true",
        max_epochs=MAX_EPOCHS,
        check_val_every_n_epoch=1,
        callbacks=[checkpoint_callback, early_stop_callback, lr_monitor],
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
    if checkpoint_callback.best_model_score is not None:
        print(f"Best val Dice  : {float(checkpoint_callback.best_model_score):.4f}")
    else:
        print("Best val Dice  : N/A")


if __name__ == "__main__":
    main()
