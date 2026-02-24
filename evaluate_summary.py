"""
Aggregate metrics across tinker runs into a single summary CSV.

Output format intentionally mirrors the "split-separated tables" style used by VML:
the CSV contains three tables (train/val/test) separated by a comment line and a
blank line, each with its own header row.

This script looks for inference metrics written by `tinker/SFT_inference.py`:
  - student_phase/metrics_{split}.json
  - student_phase/metrics_{split}_label_only.json
"""

from __future__ import annotations

import argparse
import json
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from pathlib import Path
from typing import Any, Iterable

SPLITS = ("train", "val", "test")

METRIC_COLUMNS = (
    "loss",
    "cls_loss",
    "accuracy",
    "macro_f1",
    "auroc",
    "auprc",
    "invalid_label_rate",
    "balanced_accuracy",
    "F1",
    "mcc",
    "precision",
    "recall",
    "tp",
    "fp",
    "fn",
    "tn",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Summarize tinker SFT inference metrics across runs.")
    p.add_argument(
        "--runs-dir",
        type=Path,
        default=Path("runs"),
        help="Directory containing tinker run folders.",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("summary.csv"),
        help="Output CSV path.",
    )
    p.add_argument(
        "--mode",
        choices=["reasoning", "label_only", "both"],
        default="both",
        help="Which inference metrics to include (reasoning exports, label-only metrics, or both).",
    )
    return p.parse_args()


def _parse_run_dir_name(name: str) -> dict[str, str]:
    """
    Best-effort parse:
      <method>_<YYYYMMDD>_<HHMMSS>  -> method=<method>, run_id=<YYYYMMDD>_<HHMMSS>
      otherwise -> method=<prefix>, run_id=<suffix>
    """
    parts = [p for p in name.split("_") if p]
    if len(parts) >= 3 and parts[-2].isdigit() and parts[-1].isdigit() and len(parts[-2]) == 8 and len(parts[-1]) == 6:
        return {"method": "_".join(parts[:-2]) or "unknown", "run_id": f"{parts[-2]}_{parts[-1]}"}
    if len(parts) >= 2:
        return {"method": "_".join(parts[:-1]) or "unknown", "run_id": parts[-1]}
    return {"method": "unknown", "run_id": name}


def _format_numeric(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, int):
        return str(value)
    try:
        cooked = Decimal(str(value))
    except (InvalidOperation, ValueError):
        try:
            cooked = Decimal(str(float(value)))  # type: ignore[arg-type]
        except (InvalidOperation, TypeError, ValueError):
            return str(value)
    if cooked.is_nan():
        return ""
    return str(cooked.quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP))


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _collect_metrics_for_run(run_dir: Path, *, mode: str) -> dict[str, list[dict[str, Any]]]:
    """
    Returns split -> list[rows]. There can be 2 rows per split when mode='both'
    (reasoning + label_only).
    """
    out: dict[str, list[dict[str, Any]]] = {s: [] for s in SPLITS}
    student_dir = run_dir / "student_phase"
    if not student_dir.exists():
        return out

    parsed = _parse_run_dir_name(run_dir.name)
    base_meta = {
        "run_dir": run_dir.name,
        "method": parsed["method"],
        "run_id": parsed["run_id"],
    }

    def maybe_add(split: str, path: Path, prompt_mode: str) -> None:
        if not path.exists():
            return
        blob = _read_json(path)
        metrics = blob.get("metrics") or {}
        row: dict[str, Any] = dict(base_meta)
        row["split"] = split
        row["prompt_mode"] = prompt_mode
        row["created_at"] = blob.get("created_at")
        row["checkpoint_path"] = blob.get("checkpoint_path")
        row["decision_threshold"] = blob.get("decision_threshold")
        row["n"] = blob.get("n")
        for key in METRIC_COLUMNS:
            row[key] = metrics.get(key)
        out[split].append(row)

    want_reasoning = mode in {"reasoning", "both"}
    want_label_only = mode in {"label_only", "both"}
    for split in SPLITS:
        if want_reasoning:
            maybe_add(split, student_dir / f"metrics_{split}.json", "reasoning")
        if want_label_only:
            maybe_add(split, student_dir / f"metrics_{split}_label_only.json", "label_only")
    return out


def _write_split_table(
    fh,
    *,
    split: str,
    rows: Iterable[dict[str, Any]],
) -> None:
    fh.write(f"# {split}\n")
    columns = (
        "run_dir",
        "method",
        "run_id",
        "prompt_mode",
        "created_at",
        "checkpoint_path",
        "decision_threshold",
        "n",
        *METRIC_COLUMNS,
    )
    fh.write(",".join(columns) + "\n")
    for row in rows:
        values: list[str] = []
        for col in columns:
            value = row.get(col)
            if col in {"run_dir", "method", "run_id", "prompt_mode", "created_at", "checkpoint_path"}:
                values.append("" if value is None else str(value))
            else:
                values.append(_format_numeric(value))
        fh.write(",".join(values) + "\n")
    fh.write("\n")


def main() -> None:
    args = parse_args()
    runs_dir: Path = args.runs_dir
    output: Path = args.output

    if not runs_dir.exists():
        raise FileNotFoundError(f"runs dir not found: {runs_dir}")

    rows_by_split: dict[str, list[dict[str, Any]]] = {s: [] for s in SPLITS}
    for run_dir in sorted([p for p in runs_dir.iterdir() if p.is_dir()], key=lambda p: p.name):
        collected = _collect_metrics_for_run(run_dir, mode=str(args.mode))
        for split in SPLITS:
            rows_by_split[split].extend(collected.get(split, []))

    # Keep ordering stable and readable.
    for split in SPLITS:
        rows_by_split[split].sort(key=lambda r: (str(r.get("method", "")), str(r.get("run_id", "")), str(r.get("prompt_mode", ""))))

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as fh:
        for split in SPLITS:
            _write_split_table(fh, split=split, rows=rows_by_split[split])

    print(f"Wrote summary to {output}")


if __name__ == "__main__":
    main()

