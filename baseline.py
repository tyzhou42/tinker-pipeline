#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import tinker
from loguru import logger

import SFT_reasoning as core


BASELINE_SUBDIR = "baseline_phase"
DEFAULT_DATASET_DIR = "tinker/dataset"
DEFAULT_BASELINE_LOG_DIR = "tinker/baseline_runs"
DEFAULT_BASELINE_MODEL = "Qwen/Qwen3-8B"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run baseline inference (no finetuning) with Tinker sampling API.")
    p.add_argument("--config", type=str, default="tinker/configs/reasoning_sft.example.json")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--run-name", type=str, default=None)
    p.add_argument("--dataset-name", "--dataset_name", dest="dataset_name", type=str, default=None)
    p.add_argument("--dataset-root-dir", "--dataset_root_dir", dest="dataset_root_dir", type=str, default=None)
    p.add_argument("--rules-root-dir", "--rules_root_dir", dest="rules_root_dir", type=str, default=None)
    p.add_argument("--max-train-examples", type=int, default=None)
    p.add_argument("--max-val-examples", type=int, default=None)
    p.add_argument("--max-test-examples", type=int, default=None)
    p.add_argument("--max-eval-samples", type=int, default=None)
    p.add_argument("--eval-max-concurrency", type=int, default=None)
    p.add_argument("--max-length", type=int, default=None)
    p.add_argument("--baseline-model-name", type=str, default=DEFAULT_BASELINE_MODEL)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--log-dir", type=str, default=DEFAULT_BASELINE_LOG_DIR)
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
            candidates.append(dataset_dir / f"{dataset_name}_{name}.csv")
            candidates.append(dataset_dir / f"{name}.csv")
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


def _load_split_dataset(path: Path, cfg: dict[str, Any], split_name: str) -> pd.DataFrame:
    attempts: list[tuple[str, str, str | None]] = [
        ("text", "label", None),
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


def _create_sampling_client(service_client: Any, base_model: str) -> Any:
    create_sampling_client = getattr(service_client, "create_sampling_client", None)
    if not callable(create_sampling_client):
        raise RuntimeError("Service client does not expose create_sampling_client")

    attempts = [
        lambda: create_sampling_client(base_model=base_model),
        lambda: create_sampling_client(model_path=base_model),
        lambda: create_sampling_client(model=base_model),
        lambda: create_sampling_client(base_model),
    ]
    last_err: Exception | None = None
    for attempt in attempts:
        try:
            return attempt()
        except TypeError as exc:
            last_err = exc
            continue
    if last_err is not None:
        raise last_err
    raise RuntimeError("Unable to create sampling client")


def _get_tokenizer(service_client: Any, cfg: dict[str, Any], sampling_client: Any, base_model_name: str) -> Any:
    tok = getattr(sampling_client, "get_tokenizer", None)
    if callable(tok):
        return tok()

    create_training_client = getattr(service_client, "create_lora_training_client", None)
    if not callable(create_training_client):
        raise RuntimeError(
            "Cannot obtain tokenizer (sampling client has none; service client lacks create_lora_training_client)"
        )
    training_client = create_training_client(
        base_model=base_model_name or str(cfg["student_model_name"]),
        rank=4,
    )
    return training_client.get_tokenizer()


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_json_array(path: Path, items: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("[\n")
        for i, item in enumerate(items):
            if i:
                f.write(",\n")
            f.write(json.dumps(item, ensure_ascii=False))
        f.write("\n]\n")


def _run_split_eval(
    *,
    df: pd.DataFrame,
    tokenizer: Any,
    sampling_client: Any,
    cfg: dict[str, Any],
    prompt_cfg: core.PromptConfig,
    rulebook: str,
) -> dict[str, Any]:
    reasoning_prompt_builder = lambda text: core.build_reasoning_user_prompt(prompt_cfg, rulebook, text)
    eval_rows = core.build_eval_rows(
        frame=df,
        tokenizer=tokenizer,
        max_length=int(cfg["max_length"]),
        reasoning_prompt_builder=reasoning_prompt_builder,
        reasoning_placeholder=str(cfg["eval_reasoning_placeholder"]),
    )
    one_tokens = tokenizer.encode("1", add_special_tokens=False)
    zero_tokens = tokenizer.encode("0", add_special_tokens=False)
    return core.evaluate_binary(
        sampling_client,
        tokenizer,
        eval_rows,
        max_samples=int(cfg["max_eval_samples"]),
        one_tokens=one_tokens,
        zero_tokens=zero_tokens,
        max_concurrency=int(cfg["eval_max_concurrency"]),
        invalid_warn_rate=float(cfg["invalid_label_warn_rate"]),
    )


def _run_split_generation(
    *,
    split_name: str,
    df: pd.DataFrame,
    tokenizer: Any,
    sampling_client: Any,
    cfg: dict[str, Any],
    prompt_cfg: core.PromptConfig,
    rulebook: str,
    max_new_tokens: int,
    temperature: float,
    workers: int,
) -> list[dict[str, Any]]:
    reasoning_prompt_builder = lambda text: core.build_reasoning_user_prompt(prompt_cfg, rulebook, text)

    def submit_one(i: int, text: str, gold_label: int) -> dict[str, Any]:
        fit = core.fit_user_prompt_and_answer_to_max_length(
            tokenizer,
            user_builder=reasoning_prompt_builder,
            text=text,
            assistant_content="0",
            max_length=int(cfg["max_length"]),
        )
        if fit is None:
            return {
                "sample_id": f"{split_name}_{i}_k0",
                "example_id": f"{split_name}_{i}",
                "split": split_name,
                "k_index": 0,
                "text": text,
                "gold_label": int(gold_label),
                "raw_output": "",
                "parse_ok": False,
                "parse_source": None,
                "pred_label": None,
                "reasoning": None,
                "error": "prompt_too_long_for_max_length",
            }

        _, prompt_tokens, _ = fit
        model_input = tinker.ModelInput.from_ints(prompt_tokens)

        try:
            sp = tinker.SamplingParams(max_tokens=int(max_new_tokens), temperature=float(temperature))
            try:
                req = sampling_client.sample(prompt=model_input, num_samples=1, sampling_params=sp)
            except TypeError:
                req = sampling_client.sample(model_input, 1, sp)

            out = req.result() if hasattr(req, "result") else req
            generated_text, _ = core._extract_generated_text_and_ids(out, tokenizer)  # type: ignore[attr-defined]
            parsed, err = core.parse_teacher_output(generated_text)
            if parsed is None:
                return {
                    "sample_id": f"{split_name}_{i}_k0",
                    "example_id": f"{split_name}_{i}",
                    "split": split_name,
                    "k_index": 0,
                    "text": text,
                    "gold_label": int(gold_label),
                    "raw_output": generated_text,
                    "parse_ok": False,
                    "parse_source": None,
                    "pred_label": None,
                    "reasoning": None,
                    "error": err or "parse_failed",
                }
            return {
                "sample_id": f"{split_name}_{i}_k0",
                "example_id": f"{split_name}_{i}",
                "split": split_name,
                "k_index": 0,
                "text": text,
                "gold_label": int(gold_label),
                "raw_output": generated_text,
                "parse_ok": True,
                "parse_source": parsed.source,
                "pred_label": int(parsed.label),
                "reasoning": parsed.reasoning,
                "error": None,
            }
        except Exception as exc:
            return {
                "sample_id": f"{split_name}_{i}_k0",
                "example_id": f"{split_name}_{i}",
                "split": split_name,
                "k_index": 0,
                "text": text,
                "gold_label": int(gold_label),
                "raw_output": "",
                "parse_ok": False,
                "parse_source": None,
                "pred_label": None,
                "reasoning": None,
                "error": f"generation_error: {exc}",
            }

    rows = list(df[["text", "label"]].itertuples(index=False, name=None))
    outputs: list[dict[str, Any]] = [None] * len(rows)  # type: ignore[list-item]
    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as ex:
        futs = {}
        for i, (text, label) in enumerate(rows):
            futs[ex.submit(submit_one, i, str(text), int(label))] = i
        for fut in as_completed(futs):
            i = futs[fut]
            outputs[i] = fut.result()
    return outputs  # type: ignore[return-value]


def _build_metrics_blob(
    *,
    split_name: str,
    baseline_model_name: str,
    decision_threshold: float,
    eval_out: dict[str, Any],
) -> dict[str, Any]:
    thr = core.thresholded_classification_metrics(
        eval_out["y_true"],
        eval_out["p_one"],
        eval_out.get("invalid_flags"),
        decision_threshold,
    )
    eval_out["decision_threshold"] = float(decision_threshold)
    eval_out["y_pred"] = thr["y_pred"]
    eval_out["accuracy"] = thr["accuracy"]
    eval_out["macro_f1"] = thr["macro_f1"]

    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "split": split_name,
        "model_path": baseline_model_name,
        "decision_threshold_rule": "val_max_macro_f1",
        "decision_threshold": float(decision_threshold),
        "metrics": {
            "loss": float(eval_out["loss"]),
            "cls_loss": float(eval_out["cls_loss"]),
            "accuracy": float(eval_out["accuracy"]),
            "macro_f1": float(eval_out["macro_f1"]),
            "auroc": float(eval_out["auroc"]),
            "auprc": float(eval_out["auprc"]),
            "invalid_label_rate": float(eval_out["invalid_label_rate"]),
            "balanced_accuracy": float(thr["balanced_accuracy"]),
            "F1": float(thr["F1"]),
            "mcc": float(thr["mcc"]),
            "precision": float(thr["precision"]),
            "recall": float(thr["recall"]),
            "tp": int(thr["tp"]),
            "fp": int(thr["fp"]),
            "fn": int(thr["fn"]),
            "tn": int(thr["tn"]),
        },
    }


def _log_split_metrics(split_name: str, blob: dict[str, Any]) -> None:
    m = blob["metrics"]
    logger.info(
        "{} | loss={:.6f} cls_loss={:.6f} acc={:.6f} macro_f1={:.6f} auroc={:.6f} auprc={:.6f} invalid_rate={:.4f} thr={:.4f}",
        split_name,
        float(m["loss"]),
        float(m["cls_loss"]),
        float(m["accuracy"]),
        float(m["macro_f1"]),
        float(m["auroc"]),
        float(m["auprc"]),
        float(m["invalid_label_rate"]),
        float(blob.get("decision_threshold", 0.5)),
    )


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    cfg = core.load_config(project_root, args)
    prompt_cfg = core.load_prompt_config(project_root, cfg)

    # Baseline uses its own run root by default.
    cfg["log_dir"] = str(args.log_dir)
    baseline_model_name = str(args.baseline_model_name or DEFAULT_BASELINE_MODEL)
    cfg["student_model_name"] = baseline_model_name

    random.seed(int(cfg["seed"]))
    np.random.seed(int(cfg["seed"]))

    run_name = args.run_name or f"baseline_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    log_dir = core.resolve_path(project_root, str(cfg["log_dir"]))
    if log_dir is None:
        raise RuntimeError("log_dir resolution failed")
    run_dir = log_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    core.setup_logger(run_dir)
    logger.info("Run dir: {}", run_dir)
    logger.info("Baseline model: {}", baseline_model_name)

    (run_dir / "resolved_config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    (run_dir / "resolved_args.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

    split_paths, _, dataset_name = _resolve_split_paths(project_root, cfg)
    train_df = _load_split_dataset(split_paths["train"], cfg, "train")
    val_df = _load_split_dataset(split_paths["val"], cfg, "val")
    test_df = _load_split_dataset(split_paths["test"], cfg, "test")

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

    rules_dir = _resolve_rules_dir(project_root, cfg, dataset_name)
    rulebook, rule_files = core.read_rulebook(rules_dir, str(cfg["rules_glob"]))

    baseline_dir = run_dir / BASELINE_SUBDIR
    baseline_dir.mkdir(parents=True, exist_ok=True)
    (baseline_dir / "rulebook.txt").write_text(rulebook, encoding="utf-8")
    logger.info(
        "Loaded dataset '{}' | train={} val={} test={} | rules={} files",
        dataset_name,
        len(train_df),
        len(val_df),
        len(test_df),
        len(rule_files),
    )

    client = tinker.ServiceClient(base_url=str(cfg["tinker_base_url"])) if cfg.get("tinker_base_url") else tinker.ServiceClient()
    sampling_client = _create_sampling_client(client, baseline_model_name)
    tokenizer = _get_tokenizer(client, cfg, sampling_client, baseline_model_name)
    if not hasattr(tokenizer, "apply_chat_template"):
        raise RuntimeError("Tokenizer does not support apply_chat_template")

    split_frames = {
        "train": train_df.reset_index(drop=True),
        "val": val_df.reset_index(drop=True),
        "test": test_df.reset_index(drop=True),
    }

    val_eval = _run_split_eval(
        df=split_frames["val"],
        tokenizer=tokenizer,
        sampling_client=sampling_client,
        cfg=cfg,
        prompt_cfg=prompt_cfg,
        rulebook=rulebook,
    )
    decision_threshold = core.find_best_threshold_macro_f1(
        val_eval["y_true"],
        val_eval["p_one"],
        val_eval.get("invalid_flags"),
    )

    metrics_by_split: dict[str, dict[str, Any]] = {}
    for split_name in ("train", "val", "test"):
        eval_out = _run_split_eval(
            df=split_frames[split_name],
            tokenizer=tokenizer,
            sampling_client=sampling_client,
            cfg=cfg,
            prompt_cfg=prompt_cfg,
            rulebook=rulebook,
        )
        blob = _build_metrics_blob(
            split_name=split_name,
            baseline_model_name=baseline_model_name,
            decision_threshold=float(decision_threshold),
            eval_out=eval_out,
        )
        _log_split_metrics(split_name, blob)
        _write_json(baseline_dir / f"metrics_{split_name}.json", blob)
        metrics_by_split[split_name] = blob

    for split_name in ("train", "val", "test"):
        preds = _run_split_generation(
            split_name=split_name,
            df=split_frames[split_name],
            tokenizer=tokenizer,
            sampling_client=sampling_client,
            cfg=cfg,
            prompt_cfg=prompt_cfg,
            rulebook=rulebook,
            max_new_tokens=int(args.max_new_tokens),
            temperature=float(args.temperature),
            workers=int(args.workers),
        )
        _write_json_array(baseline_dir / f"{split_name}.json", preds)

    manifest = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "run_name": run_name,
        "dataset_name": dataset_name,
        "baseline_model_name": baseline_model_name,
        "decision_threshold_rule": "val_max_macro_f1",
        "decision_threshold": float(decision_threshold),
        "paths": {
            "baseline_dir": str(baseline_dir),
            "train_predictions": str(baseline_dir / "train.json"),
            "val_predictions": str(baseline_dir / "val.json"),
            "test_predictions": str(baseline_dir / "test.json"),
            "metrics_train": str(baseline_dir / "metrics_train.json"),
            "metrics_val": str(baseline_dir / "metrics_val.json"),
            "metrics_test": str(baseline_dir / "metrics_test.json"),
        },
    }
    _write_json(baseline_dir / "manifest.json", manifest)

    summary = {
        "run_dir": str(run_dir),
        "baseline_dir": str(baseline_dir),
        "dataset_name": dataset_name,
        "baseline_model_name": baseline_model_name,
        "decision_threshold": float(decision_threshold),
        "metrics": {
            "train": metrics_by_split["train"]["metrics"],
            "val": metrics_by_split["val"]["metrics"],
            "test": metrics_by_split["test"]["metrics"],
        },
    }
    _write_json(run_dir / "run_summary.json", summary)

    report_lines = [
        "# Baseline Inference Report",
        "",
        f"Generated at: {datetime.now().isoformat(timespec='seconds')}",
        f"Run name: {run_name}",
        f"Model: {baseline_model_name}",
        f"Decision threshold (val max macro-F1): {float(decision_threshold):.6f}",
        "",
    ]
    for split_name in ("train", "val", "test"):
        m = metrics_by_split[split_name]["metrics"]
        report_lines.extend(
            [
                f"## {split_name}",
                f"- loss: {float(m['loss']):.6f}",
                f"- cls_loss: {float(m['cls_loss']):.6f}",
                f"- accuracy: {float(m['accuracy']):.6f}",
                f"- macro_f1: {float(m['macro_f1']):.6f}",
                f"- balanced_accuracy: {float(m['balanced_accuracy']):.6f}",
                f"- F1: {float(m['F1']):.6f}",
                f"- mcc: {float(m['mcc']):.6f}",
                f"- precision: {float(m['precision']):.6f}",
                f"- recall: {float(m['recall']):.6f}",
                f"- auroc: {float(m['auroc']):.6f}",
                f"- auprc: {float(m['auprc']):.6f}",
                f"- invalid_label_rate: {float(m['invalid_label_rate']):.6f}",
                f"- tp: {int(m['tp'])}",
                f"- fp: {int(m['fp'])}",
                f"- fn: {int(m['fn'])}",
                f"- tn: {int(m['tn'])}",
                "",
            ]
        )
    (run_dir / "test_report.md").write_text("\n".join(report_lines), encoding="utf-8")

    logger.info("Baseline run complete. Artifacts saved under {}", baseline_dir)


if __name__ == "__main__":
    main()
