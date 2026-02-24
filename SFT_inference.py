#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import tinker
from loguru import logger

import SFT_reasoning as core

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None


TEACHER_SUBDIR = "teacher_phase"
STUDENT_SUBDIR = "student_phase"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run inference with a finetuned SFT checkpoint and export outputs.")
    p.add_argument("--config", type=str, default="tinker/configs/reasoning_sft.example.json")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--run-name", type=str, required=True)
    p.add_argument(
        "--teacher-run-name",
        type=str,
        default=None,
        help="Optional. If set, load teacher artifacts (rulebook + splits) from this run if missing under --run-name.",
    )
    p.add_argument(
        "--checkpoint-path",
        type=str,
        default=None,
        help="Optional override. If unset, selects best checkpoint by highest logged test macro-F1.",
    )
    p.add_argument(
        "--student-model-name",
        type=str,
        default=None,
        help="Optional override. Use this when the training wrapper overrides student_model_name via CLI flags.",
    )
    p.add_argument(
        "--lora-rank",
        type=int,
        default=None,
        help="Optional override. Use this when the training wrapper overrides lora_rank via CLI flags.",
    )
    p.add_argument(
        "--max-length",
        type=int,
        default=None,
        help="Optional override. Use this when the training wrapper overrides max_length via CLI flags.",
    )
    p.add_argument(
        "--eval-max-concurrency",
        type=int,
        default=None,
        help="Optional override for eval_max_concurrency.",
    )
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument(
        "--compare-label-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Deprecated (use --label-only/--no-label-only). Also evaluate with label_only_user_prompt_template (rules included) for comparison metrics.",
    )
    p.add_argument(
        "--reasoning",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run reasoning-prompt eval + generation exports (writes student_phase/manifest.json and per-split JSON exports).",
    )
    p.add_argument(
        "--label-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run label-only eval (metrics only; writes student_phase/manifest_label_only.json).",
    )
    p.add_argument(
        "--force",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Re-run inference even if matching artifacts already exist.",
    )
    p.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Show progress bars (default: auto based on TTY).",
    )
    return p.parse_args()


def _sha256_bytes(data: bytes) -> str:
    h = sha256()
    h.update(data)
    return h.hexdigest()


def _sha256_file(path: Path) -> str:
    h = sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _fingerprint_run_inputs(run_dir: Path, teacher_dir: Path, cfg: dict[str, Any], prompt_cfg: core.PromptConfig) -> dict[str, str]:
    splits_dir = run_dir / "data_splits"
    parts: list[str] = []
    for split in ("train", "val", "test"):
        fp = splits_dir / f"{split}.csv"
        if fp.exists():
            parts.append(f"{split}.csv:{_sha256_file(fp)}")
    rulebook_fp = teacher_dir / "rulebook.txt"
    if rulebook_fp.exists():
        parts.append(f"rulebook.txt:{_sha256_file(rulebook_fp)}")
    dataset_fingerprint = _sha256_bytes("\n".join(parts).encode("utf-8"))

    cfg_subset = {
        "student_model_name": cfg.get("student_model_name"),
        "lora_rank": cfg.get("lora_rank"),
        "max_length": cfg.get("max_length"),
        "eval_reasoning_placeholder": cfg.get("eval_reasoning_placeholder"),
        "decision_threshold_rule": "val_max_macro_f1",
        "prompt.task_instruction": prompt_cfg.task_instruction,
        "prompt.teacher_system_prompt": prompt_cfg.teacher_system_prompt,
        "prompt.reasoning_user_prompt_template": prompt_cfg.reasoning_user_prompt_template,
        "prompt.label_only_user_prompt_template": prompt_cfg.label_only_user_prompt_template,
    }
    config_fingerprint = _sha256_bytes(json.dumps(cfg_subset, sort_keys=True).encode("utf-8"))

    return {
        "dataset_fingerprint": dataset_fingerprint,
        "config_fingerprint": config_fingerprint,
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _create_sampling_client(service_client: Any, model_path: str) -> Any:
    create_sampling_client = getattr(service_client, "create_sampling_client", None)
    if not callable(create_sampling_client):
        raise RuntimeError("Service client does not expose create_sampling_client")

    attempts = [
        lambda: create_sampling_client(model_path=model_path),
        lambda: create_sampling_client(path=model_path),
        lambda: create_sampling_client(model=model_path),
        lambda: create_sampling_client(model_path),
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
    raise RuntimeError("Unable to create sampling client from checkpoint path")


def _get_tokenizer(service_client: Any, cfg: dict[str, Any], sampling_client: Any) -> Any:
    tok = getattr(sampling_client, "get_tokenizer", None)
    if callable(tok):
        return tok()

    # Fallback: create a training client to obtain a compatible tokenizer.
    create_training_client = getattr(service_client, "create_lora_training_client", None)
    if not callable(create_training_client):
        raise RuntimeError("Cannot obtain tokenizer (sampling client has none; service client lacks create_lora_training_client)")
    training_client = create_training_client(
        base_model=str(cfg["student_model_name"]),
        rank=int(cfg["lora_rank"]),
    )
    return training_client.get_tokenizer()


def _select_best_checkpoint(
    *,
    project_root: Path,
    run_dir: Path,
    run_name: str,
    checkpoint_override: str | None,
    fingerprints: dict[str, str],
) -> tuple[str, dict[str, Any] | None]:
    if checkpoint_override:
        return checkpoint_override, None

    pointer = _load_model_pointer(project_root, run_name)
    if pointer is not None:
        ckpt = str(pointer["pointer"].get("best_model_path") or pointer["pointer"].get("final_model_path") or "").strip()
        if ckpt:
            logger.info("Using model pointer checkpoint from {}: {}", pointer["pointer_path"], ckpt)
            return ckpt, {"source": "model_pointer", **pointer}

    student_dir = run_dir / STUDENT_SUBDIR
    history = student_dir / "eval_history.jsonl"
    if not history.exists():
        # No logged per-checkpoint test macro-F1; fall back to final model path.
        summary_path = run_dir / "run_summary.json"
        if not summary_path.exists():
            raise FileNotFoundError(f"Missing eval history and run summary: {history} / {summary_path}")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        final_path = str(summary.get("final_model_path") or "").strip()
        if not final_path:
            raise RuntimeError("Could not determine final_model_path from run_summary.json")
        logger.warning("No eval history found; falling back to final_model_path for inference selection.")
        return final_path, None

    best: dict[str, Any] | None = None

    for row in _read_jsonl(history):
        if row.get("dataset_fingerprint") != fingerprints["dataset_fingerprint"]:
            continue
        if row.get("config_fingerprint") != fingerprints["config_fingerprint"]:
            continue
        split = str(row.get("split", "")).strip()
        if split not in {"test", "test_final", "test_bestval"}:
            continue
        metrics = row.get("metrics") or {}
        try:
            score = float(metrics.get("macro_f1", float("-inf")))
        except Exception:
            score = float("-inf")
        if best is None:
            best = dict(row)
            best["_score"] = score
            continue
        best_score = float(best.get("_score", float("-inf")))
        if score > best_score:
            best = dict(row)
            best["_score"] = score
            continue
        if score == best_score:
            # Tie-break: most recent by step, then created_at string.
            try:
                step = int(row.get("step", -1))
            except Exception:
                step = -1
            try:
                best_step = int(best.get("step", -1))
            except Exception:
                best_step = -1
            if step > best_step:
                best = dict(row)
                best["_score"] = score
                continue
            if step == best_step:
                if str(row.get("created_at", "")) > str(best.get("created_at", "")):
                    best = dict(row)
                    best["_score"] = score

    if best is None:
        logger.warning(
            "eval_history.jsonl exists but no matching test records found for current dataset/config fingerprints. "
            "Falling back to best test record regardless of fingerprint."
        )
        best_any: dict[str, Any] | None = None
        for row in _read_jsonl(history):
            split = str(row.get("split", "")).strip()
            if split not in {"test", "test_final", "test_bestval"}:
                continue
            metrics = row.get("metrics") or {}
            try:
                score = float(metrics.get("macro_f1", float("-inf")))
            except Exception:
                score = float("-inf")
            if best_any is None or score > float(best_any.get("_score", float("-inf"))):
                best_any = dict(row)
                best_any["_score"] = score
        if best_any is not None:
            ckpt_any = str(best_any.get("checkpoint_path") or "").strip()
            if ckpt_any:
                return ckpt_any, best_any

        summary_path = run_dir / "run_summary.json"
        if summary_path.exists():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            ckpt_summary = str(summary.get("best_model_path") or summary.get("final_model_path") or "").strip()
            if ckpt_summary:
                logger.warning("Falling back to run_summary.json checkpoint selection: {}", ckpt_summary)
                return ckpt_summary, {"source": "run_summary", "summary_path": str(summary_path), "summary": summary}

        raise RuntimeError("Unable to select checkpoint for inference (no model pointer, no usable eval record, no run_summary.json).")

    ckpt = str(best.get("checkpoint_path") or "").strip()
    if not ckpt:
        raise RuntimeError("Best eval record missing checkpoint_path")
    return ckpt, best


def _write_json_array(path: Path, items: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("[\n")
        for i, item in enumerate(items):
            if i:
                f.write(",\n")
            f.write(json.dumps(item, ensure_ascii=False))
        f.write("\n]\n")


def _artifact_matches(path: Path, *, checkpoint_path: str, fingerprints: dict[str, str]) -> bool:
    if not path.exists():
        return False
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return (
        str(obj.get("checkpoint_path", "")).strip() == checkpoint_path
        and str(obj.get("dataset_fingerprint", "")).strip() == fingerprints["dataset_fingerprint"]
        and str(obj.get("config_fingerprint", "")).strip() == fingerprints["config_fingerprint"]
    )


def _load_model_pointer(project_root: Path, run_name: str) -> dict[str, Any] | None:
    """
    Load a model pointer JSON from tinker/model.

    Resolution order:
      1) tinker/model/<run_name>.json
      2) tinker/model/latest.json
    """
    model_dir = project_root / "tinker" / "model"
    for pointer_path in (model_dir / f"{run_name}.json", model_dir / "latest.json"):
        if not pointer_path.exists():
            continue
        try:
            pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("Failed to parse model pointer {}: {}", pointer_path, exc)
            continue
        pointer_run_name = str(pointer.get("run_name") or "").strip()
        if pointer_path.name != "latest.json" and pointer_run_name and pointer_run_name != run_name:
            logger.warning(
                "Model pointer {} run_name mismatch (expected {}, got {}). Ignoring.",
                pointer_path,
                run_name,
                pointer_run_name,
            )
            continue
        return {"pointer_path": str(pointer_path), "pointer": pointer}
    return None

def _ensure_teacher_artifacts(*, run_dir: Path, teacher_run_dir: Path) -> None:
    """
    Ensure run_dir has teacher artifacts and split files.
    If missing, copy from teacher_run_dir to make inference reproducible/self-contained.
    """
    dst_teacher = run_dir / TEACHER_SUBDIR
    dst_splits = run_dir / "data_splits"
    required_dst = [
        dst_teacher / "rulebook.txt",
        dst_splits / "train.csv",
        dst_splits / "val.csv",
        dst_splits / "test.csv",
    ]
    if all(p.exists() for p in required_dst):
        return

    src_teacher = teacher_run_dir / TEACHER_SUBDIR
    src_splits = teacher_run_dir / "data_splits"
    required_src = [
        src_teacher / "rulebook.txt",
        src_splits / "train.csv",
        src_splits / "val.csv",
        src_splits / "test.csv",
    ]
    missing = [str(p) for p in required_src if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Teacher run missing required artifacts for inference: {missing}")

    dst_teacher.mkdir(parents=True, exist_ok=True)
    dst_splits.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_teacher / "rulebook.txt", dst_teacher / "rulebook.txt")
    shutil.copy2(src_splits / "train.csv", dst_splits / "train.csv")
    shutil.copy2(src_splits / "val.csv", dst_splits / "val.csv")
    shutil.copy2(src_splits / "test.csv", dst_splits / "test.csv")

    (dst_teacher / "teacher_source.json").write_text(
        json.dumps(
            {
                "copied_at": datetime.now().isoformat(timespec="seconds"),
                "teacher_run_dir": str(teacher_run_dir),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def _run_split_eval(
    *,
    split_name: str,
    df: pd.DataFrame,
    tokenizer: Any,
    sampling_client: Any,
    cfg: dict[str, Any],
    prompt_cfg: core.PromptConfig,
    rulebook: str,
    show_progress: bool,
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
    eval_out = core.evaluate_binary(
        sampling_client,
        tokenizer,
        eval_rows,
        max_samples=0,
        one_tokens=one_tokens,
        zero_tokens=zero_tokens,
        max_concurrency=int(cfg["eval_max_concurrency"]),
        invalid_warn_rate=float(cfg["invalid_label_warn_rate"]),
        show_progress=show_progress,
        progress_desc=f"Eval {split_name}",
    )
    return eval_out


def _build_label_only_eval_rows(
    *,
    split_name: str,
    df: pd.DataFrame,
    tokenizer: Any,
    max_length: int,
    prompt_cfg: core.PromptConfig,
    rulebook: str,
) -> list[dict[str, Any]]:
    """
    Build eval rows for label-only prompting.

    These rows are compatible with core.evaluate_binary(): they include {"text","label","prompt_tokens"}.
    prompt_tokens must end in an open assistant turn so we can probe logprobs of "0"/"1".
    """
    label_prompt_builder = lambda text: core.build_label_only_user_prompt(prompt_cfg, rulebook, text)
    rows: list[dict[str, Any]] = []
    dropped = 0

    for _, row in df.iterrows():
        text = str(row["text"])
        label = int(row["label"])
        # Fit prompt to max_length by reserving a tiny dummy answer token ("0").
        fit = core.fit_user_prompt_and_answer_to_max_length(
            tokenizer,
            user_builder=label_prompt_builder,
            text=text,
            assistant_content="0",
            max_length=max_length,
        )
        if fit is None:
            dropped += 1
            continue
        _, prompt_tokens, _ = fit
        rows.append({"text": text, "label": label, "prompt_tokens": prompt_tokens})

    if dropped > 0:
        logger.warning("Label-only eval dropped {} rows for split '{}' due to max_length/template mismatch", dropped, split_name)
    if not rows:
        raise RuntimeError(f"No usable label-only eval rows after tokenization for split '{split_name}'")
    return rows


def _run_label_only_split_eval(
    *,
    split_name: str,
    df: pd.DataFrame,
    tokenizer: Any,
    sampling_client: Any,
    cfg: dict[str, Any],
    prompt_cfg: core.PromptConfig,
    rulebook: str,
    show_progress: bool,
) -> dict[str, Any]:
    eval_rows = _build_label_only_eval_rows(
        split_name=split_name,
        df=df,
        tokenizer=tokenizer,
        max_length=int(cfg["max_length"]),
        prompt_cfg=prompt_cfg,
        rulebook=rulebook,
    )
    one_tokens = tokenizer.encode("1", add_special_tokens=False)
    zero_tokens = tokenizer.encode("0", add_special_tokens=False)
    return core.evaluate_binary(
        sampling_client,
        tokenizer,
        eval_rows,
        max_samples=0,
        one_tokens=one_tokens,
        zero_tokens=zero_tokens,
        max_concurrency=int(cfg["eval_max_concurrency"]),
        invalid_warn_rate=float(cfg["invalid_label_warn_rate"]),
        show_progress=show_progress,
        progress_desc=f"Eval label-only {split_name}",
    )


def _run_split_generation(
    *,
    split_name: str,
    df: pd.DataFrame,
    tokenizer: Any,
    sampling_client: Any,
    prompt_cfg: core.PromptConfig,
    rulebook: str,
    max_new_tokens: int,
    temperature: float,
    workers: int,
    show_progress: bool,
) -> list[dict[str, Any]]:
    if not hasattr(tokenizer, "apply_chat_template"):
        raise RuntimeError("Tokenizer does not support apply_chat_template")

    reasoning_prompt_builder = lambda text: core.build_reasoning_user_prompt(prompt_cfg, rulebook, text)

    types_mod = getattr(tinker, "types", None)
    sampling_params_cls = None if types_mod is None else getattr(types_mod, "SamplingParams", None)

    def submit_one(i: int, text: str, gold_label: int) -> dict[str, Any]:
        user_prompt = reasoning_prompt_builder(text)
        prompt_tokens = core.chat_input_ids(
            tokenizer,
            [{"role": "user", "content": user_prompt}],
            add_generation_prompt=True,
        )
        model_input = tinker.ModelInput.from_ints(prompt_tokens)

        try:
            req = None
            if sampling_params_cls is not None:
                sp = sampling_params_cls(max_tokens=int(max_new_tokens), temperature=float(temperature))
                try:
                    req = sampling_client.sample(prompt=model_input, num_samples=1, sampling_params=sp)
                except TypeError:
                    # Positional signature: sample(model_input, num_samples, sampling_params)
                    req = sampling_client.sample(model_input, 1, sp)
            else:
                # Best-effort fallback for older/newer variants.
                try:
                    req = sampling_client.sample(model_input, 1, {"max_tokens": int(max_new_tokens), "temperature": float(temperature)})
                except TypeError:
                    req = sampling_client.sample(model_input, 1)

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
    pbar = None
    if tqdm is not None:
        pbar = tqdm(total=len(rows), desc=f"Generate {split_name}", unit="ex", disable=not show_progress)
    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as ex:
        futs = {}
        for i, (text, label) in enumerate(rows):
            futs[ex.submit(submit_one, i, str(text), int(label))] = i
        for fut in as_completed(futs):
            i = futs[fut]
            outputs[i] = fut.result()
            if pbar is not None:
                pbar.update(1)
    if pbar is not None:
        pbar.close()
    return outputs  # type: ignore[return-value]


def _print_metrics(split: str, metrics: dict[str, Any]) -> None:
    logger.info(
        "{} | loss={:.6f} cls_loss={:.6f} acc={:.6f} macro_f1={:.6f} auroc={:.6f} auprc={:.6f} invalid_rate={:.4f} thr={:.4f}",
        split,
        float(metrics["loss"]),
        float(metrics["cls_loss"]),
        float(metrics["accuracy"]),
        float(metrics["macro_f1"]),
        float(metrics["auroc"]),
        float(metrics["auprc"]),
        float(metrics["invalid_label_rate"]),
        float(metrics.get("decision_threshold", 0.5)),
    )


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    cfg = core.load_config(project_root, args)
    prompt_cfg = core.load_prompt_config(project_root, cfg)

    env_file = core.resolve_path(project_root, str(cfg.get("teacher_env_file", "")))
    if env_file is not None and env_file.exists():
        core.load_dotenv(env_file, override=False)

    show_progress = bool(args.progress) if args.progress is not None else bool(sys.stderr.isatty())
    do_reasoning = bool(getattr(args, "reasoning", True))
    do_label_only = bool(getattr(args, "label_only", False)) or bool(getattr(args, "compare_label_only", False))
    if not do_reasoning and not do_label_only:
        raise ValueError("Nothing to do: set --reasoning and/or --label-only.")

    random.seed(int(cfg["seed"]))
    np.random.seed(int(cfg["seed"]))

    run_name = str(args.run_name)
    log_dir = core.resolve_path(project_root, str(cfg["log_dir"]))
    if log_dir is None:
        raise RuntimeError("log_dir resolution failed")
    run_dir = log_dir / run_name
    if not run_dir.exists():
        pointer = _load_model_pointer(project_root, run_name)
        pointer_run_name = "" if pointer is None else str(pointer["pointer"].get("run_name") or "").strip()
        if pointer_run_name and (log_dir / pointer_run_name).exists():
            logger.warning(
                "Run directory '{}' not found; using run from model pointer {}: {}",
                run_name,
                pointer["pointer_path"],
                pointer_run_name,
            )
            run_name = pointer_run_name
            run_dir = log_dir / run_name
        else:
            raise FileNotFoundError(f"Missing run directory: {run_dir}")

    student_dir = run_dir / STUDENT_SUBDIR
    student_dir.mkdir(parents=True, exist_ok=True)

    teacher_run_name = str(args.teacher_run_name).strip() if args.teacher_run_name else run_name
    teacher_run_dir = log_dir / teacher_run_name
    if not teacher_run_dir.exists():
        raise FileNotFoundError(f"Missing teacher run directory: {teacher_run_dir}")
    _ensure_teacher_artifacts(run_dir=run_dir, teacher_run_dir=teacher_run_dir)

    teacher_dir = run_dir / TEACHER_SUBDIR
    if not teacher_dir.exists():
        raise FileNotFoundError(f"Missing teacher artifact dir: {teacher_dir}")

    # Use run-local rulebook and splits so inference is tied to the exact run config/data.
    rulebook_path = teacher_dir / "rulebook.txt"
    if not rulebook_path.exists():
        raise FileNotFoundError(f"Missing rulebook: {rulebook_path}")
    rulebook = rulebook_path.read_text(encoding="utf-8")

    splits_dir = run_dir / "data_splits"
    split_paths = {s: splits_dir / f"{s}.csv" for s in ("train", "val", "test")}
    for s, p in split_paths.items():
        if not p.exists():
            raise FileNotFoundError(f"Missing split file: {p}")

    splits: dict[str, pd.DataFrame] = {}
    for s, p in split_paths.items():
        df = pd.read_csv(p, encoding="utf-8")
        if "text" not in df.columns or "label" not in df.columns:
            raise KeyError(f"Split {s} must contain columns text,label: {p}")
        splits[s] = df[["text", "label"]].copy()

    fingerprints = _fingerprint_run_inputs(run_dir, teacher_dir, cfg, prompt_cfg)

    checkpoint_path, best_record = _select_best_checkpoint(
        project_root=project_root,
        run_dir=run_dir,
        run_name=run_name,
        checkpoint_override=args.checkpoint_path,
        fingerprints=fingerprints,
    )
    logger.info("Selected checkpoint for inference: {}", checkpoint_path)
    if best_record is not None:
        logger.info(
            "Selection basis: test macro_f1={:.6f} step={} created_at={}",
            float((best_record.get("metrics") or {}).get("macro_f1", float("nan"))),
            best_record.get("step"),
            best_record.get("created_at"),
        )

    api_key = os.environ.get("TINKER_API_KEY")
    if not api_key:
        hint = f"Set TINKER_API_KEY in your environment or add it to {env_file}." if env_file is not None else "Set TINKER_API_KEY in your environment."
        raise RuntimeError(f"TINKER_API_KEY is not set. {hint}")

    client_kwargs: dict[str, Any] = {"api_key": api_key}
    if cfg.get("tinker_base_url"):
        client_kwargs["base_url"] = str(cfg["tinker_base_url"])
    client = tinker.ServiceClient(**client_kwargs)
    sampling_client = _create_sampling_client(client, checkpoint_path)
    tokenizer = _get_tokenizer(client, cfg, sampling_client)

    if do_reasoning:
        manifest: dict[str, Any] = {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "run_name": run_name,
            "checkpoint_path": checkpoint_path,
            "dataset_fingerprint": fingerprints["dataset_fingerprint"],
            "config_fingerprint": fingerprints["config_fingerprint"],
            "outputs": {},
        }

        # Compute decision threshold once on val (maximize macro-F1), then apply to all splits.
        val_eval_for_threshold = _run_split_eval(
            split_name="val",
            df=splits["val"],
            tokenizer=tokenizer,
            sampling_client=sampling_client,
            cfg=cfg,
            prompt_cfg=prompt_cfg,
            rulebook=rulebook,
            show_progress=show_progress,
        )
        decision_threshold = core.find_best_threshold_macro_f1(
            val_eval_for_threshold["y_true"],
            val_eval_for_threshold["p_one"],
            val_eval_for_threshold.get("invalid_flags"),
        )
        manifest["decision_threshold"] = float(decision_threshold)

        for split_name in ("train", "val", "test"):
            out_path = student_dir / f"{split_name}.json"
            metrics_path = student_dir / f"metrics_{split_name}.json"

            reuse = (not args.force) and _artifact_matches(
                metrics_path,
                checkpoint_path=checkpoint_path,
                fingerprints=fingerprints,
            ) and out_path.exists()

            if reuse:
                metrics_obj = json.loads(metrics_path.read_text(encoding="utf-8"))
                metrics = metrics_obj.get("metrics") or {}
                _print_metrics(split_name, metrics)
                manifest["outputs"][split_name] = {
                    "reused": True,
                    "predictions_path": str(out_path),
                    "metrics_path": str(metrics_path),
                }
                continue

            df = splits[split_name]

            eval_out = _run_split_eval(
                split_name=split_name,
                df=df,
                tokenizer=tokenizer,
                sampling_client=sampling_client,
                cfg=cfg,
                prompt_cfg=prompt_cfg,
                rulebook=rulebook,
                show_progress=show_progress,
            )
            thr_metrics = core.thresholded_classification_metrics(
                eval_out["y_true"],
                eval_out["p_one"],
                eval_out.get("invalid_flags"),
                decision_threshold,
            )
            eval_out["decision_threshold"] = float(decision_threshold)
            eval_out["y_pred"] = thr_metrics["y_pred"]
            eval_out["accuracy"] = thr_metrics["accuracy"]
            eval_out["macro_f1"] = thr_metrics["macro_f1"]

            metrics = {
                "decision_threshold": float(decision_threshold),
                "loss": float(eval_out["loss"]),
                "cls_loss": float(eval_out["cls_loss"]),
                "accuracy": float(eval_out["accuracy"]),
                "macro_f1": float(eval_out["macro_f1"]),
                "auroc": float(eval_out["auroc"]),
                "auprc": float(eval_out["auprc"]),
                "invalid_label_rate": float(eval_out["invalid_label_rate"]),
                "balanced_accuracy": float(thr_metrics["balanced_accuracy"]),
                "F1": float(thr_metrics["F1"]),
                "mcc": float(thr_metrics["mcc"]),
                "precision": float(thr_metrics["precision"]),
                "recall": float(thr_metrics["recall"]),
                "tp": int(thr_metrics["tp"]),
                "fp": int(thr_metrics["fp"]),
                "fn": int(thr_metrics["fn"]),
                "tn": int(thr_metrics["tn"]),
            }
            _print_metrics(split_name, metrics)

            preds = _run_split_generation(
                split_name=split_name,
                df=df,
                tokenizer=tokenizer,
                sampling_client=sampling_client,
                prompt_cfg=prompt_cfg,
                rulebook=rulebook,
                max_new_tokens=int(args.max_new_tokens),
                temperature=float(args.temperature),
                workers=int(args.workers),
                show_progress=show_progress,
            )
            _write_json_array(out_path, preds)

            metrics_blob = {
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "checkpoint_path": checkpoint_path,
                "split": split_name,
                "dataset_fingerprint": fingerprints["dataset_fingerprint"],
                "config_fingerprint": fingerprints["config_fingerprint"],
                "decision_threshold": float(decision_threshold),
                "n": int(len(df)),
                "predictions_path": str(out_path),
                "metrics": metrics,
            }
            metrics_path.write_text(json.dumps(metrics_blob, indent=2, ensure_ascii=False), encoding="utf-8")

            manifest["outputs"][split_name] = {
                "reused": False,
                "predictions_path": str(out_path),
                "metrics_path": str(metrics_path),
            }

        (student_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info("Inference exports saved under: {}", student_dir)

    # Optional comparison: label-only prompt (rules included). Only metrics are computed and saved.
    if do_label_only:
        logger.info("Running label-only comparison eval (no generation export).")
        label_only_manifest: dict[str, Any] = {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "checkpoint_path": checkpoint_path,
            "dataset_fingerprint": fingerprints["dataset_fingerprint"],
            "config_fingerprint": fingerprints["config_fingerprint"],
            "decision_threshold_rule": "val_max_macro_f1",
            "outputs": {},
        }

        val_eval_lo = _run_label_only_split_eval(
            split_name="val",
            df=splits["val"],
            tokenizer=tokenizer,
            sampling_client=sampling_client,
            cfg=cfg,
            prompt_cfg=prompt_cfg,
            rulebook=rulebook,
            show_progress=show_progress,
        )
        decision_threshold_lo = core.find_best_threshold_macro_f1(
            val_eval_lo["y_true"],
            val_eval_lo["p_one"],
            val_eval_lo.get("invalid_flags"),
        )
        label_only_manifest["decision_threshold"] = float(decision_threshold_lo)

        for split_name in ("train", "val", "test"):
            metrics_path = student_dir / f"metrics_{split_name}_label_only.json"
            reuse = (not args.force) and _artifact_matches(
                metrics_path,
                checkpoint_path=checkpoint_path,
                fingerprints=fingerprints,
            )
            if reuse:
                metrics_obj = json.loads(metrics_path.read_text(encoding="utf-8"))
                metrics = metrics_obj.get("metrics") or {}
                _print_metrics(f"{split_name}/label_only", metrics)
                label_only_manifest["outputs"][split_name] = {"reused": True, "metrics_path": str(metrics_path)}
                continue

            eval_out = _run_label_only_split_eval(
                split_name=split_name,
                df=splits[split_name],
                tokenizer=tokenizer,
                sampling_client=sampling_client,
                cfg=cfg,
                prompt_cfg=prompt_cfg,
                rulebook=rulebook,
                show_progress=show_progress,
            )
            thr_metrics = core.thresholded_classification_metrics(
                eval_out["y_true"],
                eval_out["p_one"],
                eval_out.get("invalid_flags"),
                decision_threshold_lo,
            )
            eval_out["decision_threshold"] = float(decision_threshold_lo)
            eval_out["y_pred"] = thr_metrics["y_pred"]
            eval_out["accuracy"] = thr_metrics["accuracy"]
            eval_out["macro_f1"] = thr_metrics["macro_f1"]

            metrics = {
                "decision_threshold": float(decision_threshold_lo),
                "loss": float(eval_out["loss"]),
                "cls_loss": float(eval_out["cls_loss"]),
                "accuracy": float(eval_out["accuracy"]),
                "macro_f1": float(eval_out["macro_f1"]),
                "auroc": float(eval_out["auroc"]),
                "auprc": float(eval_out["auprc"]),
                "invalid_label_rate": float(eval_out["invalid_label_rate"]),
                "balanced_accuracy": float(thr_metrics["balanced_accuracy"]),
                "F1": float(thr_metrics["F1"]),
                "mcc": float(thr_metrics["mcc"]),
                "precision": float(thr_metrics["precision"]),
                "recall": float(thr_metrics["recall"]),
                "tp": int(thr_metrics["tp"]),
                "fp": int(thr_metrics["fp"]),
                "fn": int(thr_metrics["fn"]),
                "tn": int(thr_metrics["tn"]),
            }
            _print_metrics(f"{split_name}/label_only", metrics)

            metrics_blob = {
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "checkpoint_path": checkpoint_path,
                "split": split_name,
                "dataset_fingerprint": fingerprints["dataset_fingerprint"],
                "config_fingerprint": fingerprints["config_fingerprint"],
                "decision_threshold": float(decision_threshold_lo),
                "n": int(len(splits[split_name])),
                "metrics": metrics,
            }
            metrics_path.write_text(json.dumps(metrics_blob, indent=2, ensure_ascii=False), encoding="utf-8")
            label_only_manifest["outputs"][split_name] = {"reused": False, "metrics_path": str(metrics_path)}

        (student_dir / "manifest_label_only.json").write_text(
            json.dumps(label_only_manifest, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.info("Label-only comparison metrics saved under: {}", student_dir)


if __name__ == "__main__":
    main()
