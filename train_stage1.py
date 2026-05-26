#!/usr/bin/env python3
"""
Stage 1 training script.

Train only the 2D image encoder + point-conditioned MLP branch:
    SimPX -> 2D UNet feature map -> positional encoding + local image feature -> MLP -> point density

This intentionally disables the 3D refiner even if train.py has --use_3d_refiner default=True.
It reuses the existing train.py training loop, checkpoint format, dataset logic, and evaluation logic.
"""

from __future__ import annotations

from train import parse_args, train


def main() -> None:
    args = parse_args()

    # Force Stage 1 behavior.
    args.use_3d_refiner = False
    args.refiner_point_weight = 0.0
    args.refiner_volume_weight = 0.0
    args.refiner_proj_weight = 0.0
    if hasattr(args, "refiner_perc_weight"):
        args.refiner_perc_weight = 0.0

    train(args)


if __name__ == "__main__":
    main()
