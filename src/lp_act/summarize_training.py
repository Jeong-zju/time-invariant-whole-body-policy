"""Summarize matched standard ACT and LP-ACT V1 JSONL training logs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_records(path: Path, start_line: int = 1) -> list[dict]:
    records: list[dict] = []
    with path.open(errors="replace") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line_number < start_line:
                continue
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "step" in record and "mode" in record:
                records.append(record)
    return records


def split_records(records: list[dict]) -> tuple[list[dict], list[dict]]:
    train = [record for record in records if record.get("split") != "validation"]
    validation = [record for record in records if record.get("split") == "validation"]
    return train, validation


def summarize(records: list[dict]) -> dict:
    train, validation = split_records(records)
    if not train:
        raise ValueError("No training records were found")
    latest = max(train, key=lambda record: int(record["step"]))
    summary = {
        "mode": latest["mode"],
        "latest_step": int(latest["step"]),
        "latest_elapsed_s": float(latest["elapsed_s"]),
        "average_steps_per_s": float(latest["step"]) / max(float(latest["elapsed_s"]), 1e-9),
        "initial_train_l1": float(train[0]["l1_loss"]),
        "latest_train_l1": float(latest["l1_loss"]),
        "num_train_records": len(train),
        "num_validation_records": len(validation),
    }
    if validation:
        best = min(validation, key=lambda record: float(record["l1_loss"]))
        final = max(validation, key=lambda record: int(record["step"]))
        summary.update(
            {
                "best_validation_step": int(best["step"]),
                "best_validation_l1": float(best["l1_loss"]),
                "latest_validation_step": int(final["step"]),
                "latest_validation_l1": float(final["l1_loss"]),
            }
        )
    return summary


def write_csv(path: Path, records: list[dict]) -> None:
    keys = sorted({key for record in records for key in record})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(records)


def write_plot(path: Path, named_records: dict[str, list[dict]]) -> None:
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for label, records in named_records.items():
        train, validation = split_records(records)
        axes[0].plot(
            [record["step"] for record in train],
            [record["l1_loss"] for record in train],
            label=label,
            alpha=0.85,
        )
        axes[1].plot(
            [record["step"] for record in validation],
            [record["l1_loss"] for record in validation],
            marker="o",
            markersize=3,
            label=label,
        )
    axes[0].set(title="Training normalized L1", xlabel="optimizer step", ylabel="L1")
    axes[1].set(title="Inference validation normalized L1", xlabel="optimizer step", ylabel="L1")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend()
    figure.suptitle("Different target normalizations: compare trends, not absolute cross-policy values")
    figure.savefig(path, dpi=160)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--act-log", type=Path, required=True)
    parser.add_argument("--lp-log", type=Path, required=True)
    parser.add_argument("--act-start-line", type=int, default=1)
    parser.add_argument("--lp-start-line", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    named_records = {
        "ACT": parse_records(args.act_log, args.act_start_line),
        "LP-ACT V1": parse_records(args.lp_log, args.lp_start_line),
    }
    summaries = {name: summarize(records) for name, records in named_records.items()}
    for name, records in named_records.items():
        slug = "act" if name == "ACT" else "lpact_v1"
        train, validation = split_records(records)
        write_csv(args.output_dir / f"{slug}_train.csv", train)
        write_csv(args.output_dir / f"{slug}_validation.csv", validation)
    with (args.output_dir / "summary.json").open("w") as handle:
        json.dump(summaries, handle, indent=2, sort_keys=True)
        handle.write("\n")
    write_plot(args.output_dir / "training_curves.png", named_records)


if __name__ == "__main__":
    main()
