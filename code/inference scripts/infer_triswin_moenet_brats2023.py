#!/usr/bin/env python3
"""
BraTS 2023 inference wrapper for TriSwin-MoENet.
This runs the BraTS 2023 branch from the shared inference implementation
and saves visualizations with WT=green, TC=blue, ET=red..
"""

from __future__ import annotations

import argparse
from pathlib import Path

from infer_triswin_moenet_all_brats import process_year


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="TriSwin-MoENet inference for BraTS 2023")
    p.add_argument("--split", default="validation", choices=["validation", "training", "all"])
    p.add_argument("--checkpoint", type=Path, default=None,
                   help="Explicit .ckpt path. If omitted, the best checkpoint is auto-discovered.")
    p.add_argument("--case-id", type=str, default=None,
                   help="Substring of case ID; processes matching cases only.")
    p.add_argument("--max-cases", type=int, default=None,
                   help="Limit number of cases, useful for a quick test.")
    p.add_argument("--output", type=Path, default=None,
                   help="Optional output directory override.")
    p.add_argument("--slice", type=int, default=None,
                   help="Fixed axial slice index. Default: automatically choose the best tumor slice.")
    p.add_argument("--no-nifti", action="store_true", help="Do not save prediction NIfTI files.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    process_year(
        year=2023,
        split=args.split,
        checkpoint_override=args.checkpoint,
        case_id_filter=args.case_id,
        max_cases=args.max_cases,
        output_override=args.output,
        save_nifti=not args.no_nifti,
        slice_index=args.slice,
    )


if __name__ == "__main__":
    main()
