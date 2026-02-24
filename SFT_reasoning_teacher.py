#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger
from openai import OpenAI
from tqdm import tqdm

import SFT_reasoning as core

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv(dotenv_path: str | os.PathLike[str] | None = None, override: bool = False) -> bool:
        if dotenv_path is None:
            return False
        path = Path(dotenv_path)
        if not path.exists():
            return False
        loaded_any = False
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip("'\"")
            if not key:
                continue
            if override or key not in os.environ:
                os.environ[key] = value
                loaded_any = True
        return loaded_any


TEACHER_SUBDIR = "teacher_phase"
DEFAULT_DATASET_DIR = "tinker/dataset"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Reasoning SFT teacher stage (sampling + rejection)")
    p.add_argument("--config", type=str, default="tinker/configs/reasoning_sft.example.json")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--dataset-name", "--dataset_name", dest="dataset_name", type=str, default=None)
    p.add_argument("--dataset-root-dir", "--dataset_root_dir", dest="dataset_root_dir", type=str, default=None)
    p.add_argument("--rules-root-dir", "--rules_root_dir", dest="rules_root_dir", type=str, default=None)
    p.add_argument("--run-name", type=str, default=None)
    p.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default=None)
    p.add_argument("--teacher-k", type=int, default=None)
    p.add_argument("--teacher-workers", "--teacher_workers", dest="teacher_workers", type=int, default=None)
    p.add_argument("--teacher-temperature", "--teacher_temperature", dest="teacher_temperature", type=float, default=None)
    p.add_argument(
        "--selection-metric",
        choices=["macro_f1", "auroc", "auprc", "accuracy", "loss", "cls_loss"],
        default=None,
    )
    p.add_argument("--max-train-examples", type=int, default=None)
    p.add_argument("--max-val-examples", type=int, default=None)
    p.add_argument("--max-test-examples", type=int, default=None)
    return p.parse_args()


def _first_existing(paths: list[Path]) -> Path | None:
    for p in paths:
        if p.exists() and p.is_file():
            return p
    return None


def _resolve_split_paths(project_root: Path, cfg: dict[str, Any]) -> tuple[dict[str, Path], Path, str]:
    dataset_name = str(cfg.get("dataset_name", "ethos")).strip()
    dataset_root = core.resolve_path(project_root, str(cfg.get("dataset_root_dir", DEFAULT_DATASET_DIR)))
    if dataset_root is None or not dataset_root.exists():
        raise FileNotFoundError(f"Missing dataset root directory: {dataset_root}")

    dataset_dir = dataset_root / dataset_name
    aliases = {
        "train": ["train"],
        "val": ["val", "valid", "validation"],
        "test": ["test"],
    }
    split_paths: dict[str, Path] = {}
    missing_info: list[str] = []

    for split, names in aliases.items():
        candidates: list[Path] = []
        for name in names:
            # Preferred layout: dataset/<dataset_name>/<dataset_name>_<split>.csv
            candidates.append(dataset_dir / f"{dataset_name}_{name}.csv")
            # Also support generic names in dataset subfolder.
            candidates.append(dataset_dir / f"{name}.csv")
            # Also support flat layout under dataset root.
            candidates.append(dataset_root / f"{dataset_name}_{name}.csv")
            candidates.append(dataset_root / f"{name}.csv")

        resolved = _first_existing(candidates)
        if resolved is None:
            pretty = ", ".join(str(p) for p in candidates)
            missing_info.append(f"{split}: [{pretty}]")
        else:
            split_paths[split] = resolved

    if missing_info:
        raise FileNotFoundError(
            "Could not resolve dataset split files for dataset "
            f"'{dataset_name}'. Tried: {' | '.join(missing_info)}"
        )

    return split_paths, dataset_dir, dataset_name


def _resolve_rules_dir(project_root: Path, cfg: dict[str, Any], dataset_name: str) -> Path:
    rules_root = core.resolve_path(project_root, str(cfg.get("rules_root_dir", "tinker/rules")))
    if rules_root is not None:
        dataset_rules = rules_root / dataset_name
        if dataset_rules.exists() and dataset_rules.is_dir():
            return dataset_rules

    # Backward compatibility with existing rules_dir behavior.
    legacy_rules = core.resolve_path(project_root, str(cfg.get("rules_dir", "tinker/rules")))
    if legacy_rules is not None:
        legacy_dataset_rules = legacy_rules / dataset_name
        if legacy_dataset_rules.exists() and legacy_dataset_rules.is_dir():
            return legacy_dataset_rules
        if legacy_rules.exists() and legacy_rules.is_dir():
            return legacy_rules

    raise FileNotFoundError(
        "Could not resolve rules directory. Tried rules_root_dir/dataset_name and rules_dir variants."
    )


def _load_split_dataset(path: Path, cfg: dict[str, Any], split_name: str):
    attempts: list[tuple[str, str, str | None]] = [
        # Preferred for pre-split files under tinker/dataset/*.
        ("text", "label", None),
        # Fallbacks for custom datasets/configs.
        (str(cfg["text_column"]), str(cfg["label_column"]), None),
        (
            str(cfg["text_column"]),
            str(cfg["label_column"]),
            None if cfg.get("csv_sep") is None else str(cfg.get("csv_sep")),
        ),
    ]
    tried: list[str] = []
    last_exc: Exception | None = None

    for text_col, label_col, sep in attempts:
        key = f"text={text_col},label={label_col},sep={sep}"
        if key in tried:
            continue
        tried.append(key)
        try:
            df = core.load_dataset(
                data_path=path,
                text_column=text_col,
                label_column=label_col,
                csv_sep=sep,
                label_threshold=float(cfg["label_threshold"]),
            )
            logger.info(
                "Loaded {} split from {} using columns ({}, {})",
                split_name,
                path,
                text_col,
                label_col,
            )
            return df
        except Exception as exc:
            last_exc = exc

    raise RuntimeError(
        f"Failed loading split '{split_name}' from {path}. Tried: {tried}. Last error: {last_exc}"
    )


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    cfg = core.load_config(project_root, args)
    prompt_cfg = core.load_prompt_config(project_root, cfg)

    random.seed(int(cfg["seed"]))
    np.random.seed(int(cfg["seed"]))

    run_name = cfg["run_name"] or f"reasoning_sft_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    log_dir = core.resolve_path(project_root, str(cfg["log_dir"]))
    if log_dir is None:
        raise RuntimeError("log_dir resolution failed")
    run_dir = log_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    teacher_dir = run_dir / TEACHER_SUBDIR
    teacher_dir.mkdir(parents=True, exist_ok=True)
    core.setup_logger(teacher_dir)
    logger.info("Run dir: {}", run_dir)
    logger.info("Teacher artifact dir: {}", teacher_dir)

    (run_dir / "resolved_config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    env_file = core.resolve_path(project_root, str(cfg["teacher_env_file"]))
    if env_file is None or not env_file.exists():
        raise FileNotFoundError(f"Missing env file: {env_file}")
    load_dotenv(env_file, override=False)

    teacher_api_key = os.environ.get(str(cfg["teacher_api_key_env"]))
    if not teacher_api_key:
        raise RuntimeError(f"Environment variable {cfg['teacher_api_key_env']} is not set")

    split_paths, dataset_dir, dataset_name = _resolve_split_paths(project_root, cfg)
    train_path = split_paths["train"]
    val_path = split_paths["val"]
    test_path = split_paths["test"]
    logger.info(
        "Resolved dataset '{}' splits: train={}, val={}, test={}",
        dataset_name,
        train_path,
        val_path,
        test_path,
    )

    train_df = _load_split_dataset(train_path, cfg, "train")
    val_df = _load_split_dataset(val_path, cfg, "val")
    test_df = _load_split_dataset(test_path, cfg, "test")

    source_summary = {
        "train": {
            "count": int(len(train_df)),
            "label_counts": core.label_counts(train_df),
            "label_distribution": core.label_distribution(train_df),
        },
        "val": {
            "count": int(len(val_df)),
            "label_counts": core.label_counts(val_df),
            "label_distribution": core.label_distribution(val_df),
        },
        "test": {
            "count": int(len(test_df)),
            "label_counts": core.label_counts(test_df),
            "label_distribution": core.label_distribution(test_df),
        },
    }

    train_df = core.stratified_subset(
        train_df,
        int(cfg["max_train_examples"]),
        seed=int(cfg["seed"]),
        split_name="train",
    )
    val_df = core.stratified_subset(
        val_df,
        int(cfg["max_val_examples"]),
        seed=int(cfg["seed"]) + 1,
        split_name="val",
    )
    test_df = core.stratified_subset(
        test_df,
        int(cfg["max_test_examples"]),
        seed=int(cfg["seed"]) + 2,
        split_name="test",
    )

    splits_dir = run_dir / "data_splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(splits_dir / "train.csv", index=False, encoding="utf-8")
    val_df.to_csv(splits_dir / "val.csv", index=False, encoding="utf-8")
    test_df.to_csv(splits_dir / "test.csv", index=False, encoding="utf-8")

    split_summary = {
        "counts": {
            "train": int(len(train_df)),
            "val": int(len(val_df)),
            "test": int(len(test_df)),
            "total": int(len(train_df) + len(val_df) + len(test_df)),
        },
        "subset_limits": {
            "max_train_examples": int(cfg["max_train_examples"]),
            "max_val_examples": int(cfg["max_val_examples"]),
            "max_test_examples": int(cfg["max_test_examples"]),
        },
        "source_split_summary": source_summary,
        "used_split_summary": {
            "train": {
                "count": int(len(train_df)),
                "label_counts": core.label_counts(train_df),
                "label_distribution": core.label_distribution(train_df),
            },
            "val": {
                "count": int(len(val_df)),
                "label_counts": core.label_counts(val_df),
                "label_distribution": core.label_distribution(val_df),
            },
            "test": {
                "count": int(len(test_df)),
                "label_counts": core.label_counts(test_df),
                "label_distribution": core.label_distribution(test_df),
            },
        },
        "seed": int(cfg["seed"]),
        "dataset_name": dataset_name,
        "source_dir": str(dataset_dir),
        "resolved_split_paths": {k: str(v) for k, v in split_paths.items()},
    }
    (run_dir / "split_summary.json").write_text(json.dumps(split_summary, indent=2), encoding="utf-8")
    logger.info("Split sizes: train={}, val={}, test={}", len(train_df), len(val_df), len(test_df))

    rules_dir = _resolve_rules_dir(project_root, cfg, dataset_name)
    rulebook, rule_files = core.read_rulebook(rules_dir, str(cfg["rules_glob"]))
    (teacher_dir / "rulebook.txt").write_text(rulebook, encoding="utf-8")
    (teacher_dir / "rule_files.json").write_text(json.dumps(rule_files, indent=2), encoding="utf-8")
    logger.info("Loaded {} rule files from {} for dataset '{}'", len(rule_files), rules_dir, dataset_name)

    teacher_samples: list[dict[str, Any]] = []
    min_reasoning_chars = int(cfg["min_reasoning_chars"])
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    k = int(cfg["teacher_k"])
    teacher_workers = int(cfg["teacher_workers"])
    logger.info("Teacher sampling config: k={}, workers={}", k, teacher_workers)
    thread_local = threading.local()

    def get_teacher_client() -> OpenAI:
        cli = getattr(thread_local, "teacher_client", None)
        if cli is None:
            cli = OpenAI(api_key=teacher_api_key, base_url=str(cfg["teacher_base_url"]))
            thread_local.teacher_client = cli
        return cli

    def sample_one_example(idx: int, text: str, gold_label: int) -> dict[str, Any]:
        messages = core.build_teacher_messages(prompt_cfg, rulebook, text)
        accepted_for_example = False
        local_teacher_samples: list[dict[str, Any]] = []
        local_accepted: list[dict[str, Any]] = []
        local_rejected: list[dict[str, Any]] = []
        client_for_example = get_teacher_client()

        for ki in range(k):
            raw_output, parsed, error = core.request_teacher_sample(
                client_for_example,
                messages,
                model=str(cfg["teacher_model"]),
                temperature=float(cfg["teacher_temperature"]),
                max_tokens=int(cfg["teacher_max_tokens"]),
                timeout_seconds=int(cfg["teacher_request_timeout"]),
                max_retries=int(cfg["teacher_max_retries"]),
                json_mode=bool(cfg["teacher_json_mode"]),
            )

            sample = {
                "sample_id": f"train_{idx}_k{ki}",
                "example_id": f"train_{idx}",
                "split": "train",
                "k_index": ki,
                "text": text,
                "gold_label": gold_label,
                "raw_output": raw_output,
                "parse_ok": parsed is not None,
                "parse_source": None if parsed is None else parsed.source,
                "pred_label": None if parsed is None else parsed.label,
                "reasoning": None if parsed is None else parsed.reasoning,
                "error": error,
            }

            # Rejection sampling with early stop:
            # keep trying this example until first valid trace or max k attempts.
            if not sample["parse_ok"]:
                sample["reject_reason"] = sample.get("error") or "parse_failed"
                local_rejected.append(sample)
                local_teacher_samples.append(sample)
            else:
                pred_label = int(sample["pred_label"])
                reasoning = str(sample["reasoning"] or "").strip()
                if pred_label != gold_label:
                    sample["reject_reason"] = "label_mismatch"
                    local_rejected.append(sample)
                    local_teacher_samples.append(sample)
                elif len(reasoning) < min_reasoning_chars:
                    sample["reject_reason"] = "reasoning_too_short"
                    local_rejected.append(sample)
                    local_teacher_samples.append(sample)
                else:
                    local_teacher_samples.append(sample)
                    local_accepted.append(
                        {
                            "sample_id": sample["sample_id"],
                            "example_id": sample["example_id"],
                            "text": sample["text"],
                            "label": gold_label,
                            "reasoning": reasoning,
                            "teacher_pred_label": pred_label,
                            "parse_source": sample.get("parse_source"),
                        }
                    )
                    accepted_for_example = True

            sleep_seconds = float(cfg["teacher_sleep_seconds"])
            if sleep_seconds > 0:
                time.sleep(sleep_seconds)

            if accepted_for_example:
                break

        return {
            "idx": idx,
            "teacher_samples": local_teacher_samples,
            "accepted": local_accepted,
            "rejected": local_rejected,
        }

    pbar = tqdm(total=len(train_df), desc="Teacher sampling", unit="example")
    results_by_idx: dict[int, dict[str, Any]] = {}
    if teacher_workers <= 1:
        for idx, row in train_df.iterrows():
            result = sample_one_example(
                idx=int(idx),
                text=str(row["text"]),
                gold_label=int(row["label"]),
            )
            results_by_idx[int(idx)] = result
            pbar.update(1)
    else:
        with ThreadPoolExecutor(max_workers=teacher_workers) as pool:
            futures = {
                pool.submit(
                    sample_one_example,
                    int(idx),
                    str(row["text"]),
                    int(row["label"]),
                ): int(idx)
                for idx, row in train_df.iterrows()
            }
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    result = future.result()
                except Exception as exc:
                    raise RuntimeError(f"Teacher sampling failed for example {idx}: {exc}") from exc
                results_by_idx[idx] = result
                pbar.update(1)
    pbar.close()

    for idx in range(len(train_df)):
        result = results_by_idx[idx]
        teacher_samples.extend(result["teacher_samples"])
        accepted.extend(result["accepted"])
        rejected.extend(result["rejected"])

    core.write_jsonl(teacher_dir / "teacher_samples.jsonl", teacher_samples)

    accepted_example_ids = [str(x["example_id"]) for x in accepted]
    if len(set(accepted_example_ids)) != len(accepted_example_ids):
        raise RuntimeError("Internal error: more than one accepted trace found for at least one example")

    core.write_jsonl(teacher_dir / "accepted_samples.jsonl", accepted)
    core.write_jsonl(teacher_dir / "rejected_samples.jsonl", rejected)

    teacher_summary = {
        "teacher_model": str(cfg["teacher_model"]),
        "teacher_workers": teacher_workers,
        "requested_samples": int(len(teacher_samples)),
        "accepted_samples": int(len(accepted)),
        "rejected_samples": int(len(rejected)),
        "acceptance_rate": float(len(accepted) / max(1, len(teacher_samples))),
    }
    (teacher_dir / "teacher_summary.json").write_text(json.dumps(teacher_summary, indent=2), encoding="utf-8")
    logger.info(
        "Teacher samples: requested={}, accepted={}, rejected={}, acceptance_rate={:.4f}",
        teacher_summary["requested_samples"],
        teacher_summary["accepted_samples"],
        teacher_summary["rejected_samples"],
        teacher_summary["acceptance_rate"],
    )

    if not accepted:
        raise RuntimeError("No accepted teacher samples after rejection sampling")

    phase_summary = {
        "run_dir": str(run_dir),
        "teacher_dir": str(teacher_dir),
        "split_summary": split_summary,
        "teacher_summary": teacher_summary,
    }
    (teacher_dir / "teacher_phase_summary.json").write_text(json.dumps(phase_summary, indent=2), encoding="utf-8")
    logger.info(
        "Teacher phase complete | run={} | requested={} accepted={} rejected={} acceptance={:.2%}",
        run_name,
        teacher_summary["requested_samples"],
        teacher_summary["accepted_samples"],
        teacher_summary["rejected_samples"],
        float(teacher_summary["acceptance_rate"]),
    )
    logger.info("Artifacts saved at: {}", teacher_dir)


if __name__ == "__main__":
    main()
