#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from lp_groot_base.validator import run_full_validation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-windows", type=int, default=512)
    parser.add_argument("--reference-horizon-steps", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    report = run_full_validation(
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
        num_windows=args.num_windows,
        reference_horizon_steps=args.reference_horizon_steps,
        seed=args.seed,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

