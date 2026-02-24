#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import tarfile
from datetime import datetime
from hashlib import sha256
from pathlib import Path
from typing import Any
from urllib.request import urlopen

import numpy as np
import pandas as pd
import tinker
import wandb
from loguru import logger

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
STUDENT_SUBDIR = "student_phase"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Reasoning SFT student stage (train/eval from teacher artifacts)")
    p.add_argument("--config", type=str, default="tinker/configs/reasoning_sft.example.json")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--run-name", type=str, required=True)
    p.add_argument(
        "--teacher-run-name",
        type=str,
        default=None,
        help="Optional. Load teacher artifacts (accepted samples + rulebook + splits) from this run, but save student outputs under --run-name.",
    )
    p.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default=None)
    p.add_argument("--teacher-k", type=int, default=None)
    p.add_argument(
        "--selection-metric",
        choices=["macro_f1", "auroc", "auprc", "accuracy", "loss", "cls_loss"],
        default=None,
    )
    p.add_argument("--student-model-name", type=str, default=None)
    p.add_argument("--lora-rank", type=int, default=None)
    p.add_argument("--num-epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--learning-rate", type=float, default=None)
    p.add_argument("--max-length", type=int, default=None)
    p.add_argument("--eval-interval", type=int, default=None)
    p.add_argument("--max-eval-samples", type=int, default=None)
    p.add_argument("--eval-max-concurrency", type=int, default=None)
    p.add_argument(
        "--use-emitted-label-metrics",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Compute discrete metrics (accuracy/macro-F1/etc.) from the model-emitted 0/1 label. "
             "Still uses P(1) to compute AUROC/AUPRC. When enabled, skips val threshold search.",
    )
    p.add_argument(
        "--save-resume-checkpoint",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save a resumable trainer state checkpoint with save_state() at end of training.",
    )
    p.add_argument(
        "--download-adapter",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Download LoRA adapter archive (weights+config) from saved Tinker checkpoint(s).",
    )
    p.add_argument(
        "--adapter-output-subdir",
        type=str,
        default="model",
        help="Subfolder under run_dir used for downloaded adapter artifacts.",
    )
    p.add_argument(
        "--adapter-download-timeout-seconds",
        type=int,
        default=600,
        help="HTTP timeout for adapter archive download.",
    )
    return p.parse_args()

def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content, encoding="utf-8")
    tmp.replace(path)


def _save_model_pointer_files(project_root: Path, run_name: str, payload: dict[str, Any]) -> dict[str, str]:
    """
    Save stable references to the most recent trained model paths under tinker/model.

    This makes it easy to locate Tinker checkpoint paths later without hunting through run logs.
    """
    model_dir = project_root / "tinker" / "model"
    model_dir.mkdir(parents=True, exist_ok=True)

    run_fp = model_dir / f"{run_name}.json"
    latest_fp = model_dir / "latest.json"
    latest_txt = model_dir / "latest.txt"

    text = json.dumps(payload, indent=2, ensure_ascii=False)
    _atomic_write_text(run_fp, text)
    _atomic_write_text(latest_fp, text)
    _atomic_write_text(
        latest_txt,
        "\n".join(
            [
                f"run_name={payload.get('run_name')}",
                f"final_model_path={payload.get('final_model_path')}",
                f"best_model_path={payload.get('best_model_path')}",
                f"resume_state_path={payload.get('resume_state_path')}",
            ]
        )
        + "\n",
    )

    return {
        "model_dir": str(model_dir),
        "run_pointer": str(run_fp),
        "latest_pointer": str(latest_fp),
        "latest_text": str(latest_txt),
    }


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _unwrap_future(value: Any) -> Any:
    if hasattr(value, "result"):
        return value.result()
    return value


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
    # Use run-local artifacts (splits + rulebook) so the fingerprint is stable across machines.
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


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")

def _copy_teacher_artifacts(*, teacher_run_dir: Path, run_dir: Path) -> tuple[Path, Path]:
    """
    Copy the minimum teacher artifacts and splits into run_dir so the student run is self-contained.
    Returns (teacher_dir, splits_dir) under run_dir.
    """
    src_teacher = teacher_run_dir / TEACHER_SUBDIR
    src_splits = teacher_run_dir / "data_splits"
    required = [
        src_teacher / "accepted_samples.jsonl",
        src_teacher / "teacher_summary.json",
        src_teacher / "rulebook.txt",
        src_splits / "train.csv",
        src_splits / "val.csv",
        src_splits / "test.csv",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Teacher run missing required artifacts: {missing}")

    dst_teacher = run_dir / TEACHER_SUBDIR
    dst_teacher.mkdir(parents=True, exist_ok=True)
    dst_splits = run_dir / "data_splits"
    dst_splits.mkdir(parents=True, exist_ok=True)

    shutil.copy2(src_teacher / "accepted_samples.jsonl", dst_teacher / "accepted_samples.jsonl")
    shutil.copy2(src_teacher / "teacher_summary.json", dst_teacher / "teacher_summary.json")
    shutil.copy2(src_teacher / "rulebook.txt", dst_teacher / "rulebook.txt")
    shutil.copy2(src_splits / "train.csv", dst_splits / "train.csv")
    shutil.copy2(src_splits / "val.csv", dst_splits / "val.csv")
    shutil.copy2(src_splits / "test.csv", dst_splits / "test.csv")

    _write_json(
        dst_teacher / "teacher_source.json",
        {
            "copied_at": datetime.now().isoformat(timespec="seconds"),
            "teacher_run_dir": str(teacher_run_dir),
        },
    )
    return dst_teacher, dst_splits


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _save_eval_artifacts(
    *,
    student_dir: Path,
    checkpoint_path: str,
    checkpoint_name: str,
    step: int,
    split_name: str,
    eval_out: dict[str, Any],
    fingerprints: dict[str, str],
) -> None:
    # Flatten into a stable JSON blob for later selection/reuse.
    derived = core.binary_metrics(eval_out["y_true"], eval_out["y_pred"], eval_out["p_one"])
    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "discrete_metrics_source": str(eval_out.get("discrete_metrics_source", "threshold")),
        "checkpoint_path": checkpoint_path,
        "checkpoint_name": checkpoint_name,
        "step": int(step),
        "split": split_name,
        "dataset_fingerprint": fingerprints["dataset_fingerprint"],
        "config_fingerprint": fingerprints["config_fingerprint"],
        "metrics": {
            "loss": float(eval_out["loss"]),
            "cls_loss": float(eval_out["cls_loss"]),
            "accuracy": float(eval_out["accuracy"]),
            "macro_f1": float(eval_out["macro_f1"]),
            "auroc": float(eval_out["auroc"]),
            "auprc": float(eval_out["auprc"]),
            "invalid_label_rate": float(eval_out["invalid_label_rate"]),
            "decision_threshold": float(eval_out.get("decision_threshold", 0.5)),
            # confusion-derived
            "balanced_accuracy": float(derived["balanced_accuracy"]),
            "F1": float(derived["F1"]),
            "mcc": float(derived["mcc"]),
            "precision": float(derived["precision"]),
            "recall": float(derived["recall"]),
            "tp": int(derived["tp"]),
            "fp": int(derived["fp"]),
            "fn": int(derived["fn"]),
            "tn": int(derived["tn"]),
        },
    }
    _write_json(student_dir / f"metrics_{split_name}_step_{step}.json", payload)
    _append_jsonl(student_dir / "eval_history.jsonl", payload)


def _extract_text_field(obj: Any, keys: list[str]) -> str | None:
    if obj is None:
        return None

    if isinstance(obj, str):
        s = obj.strip()
        return s or None

    if isinstance(obj, dict):
        for key in keys:
            value = obj.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return None

    for key in keys:
        if hasattr(obj, key):
            value = getattr(obj, key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return None


def _extract_sampling_model_path(sampling_client: Any) -> str | None:
    return _extract_text_field(
        sampling_client,
        ["model_path", "tinker_path", "checkpoint_path", "path", "id"],
    )


def _create_sampling_client_from_path(service_client: Any, model_path: str) -> Any:
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


def _save_sampler_checkpoint(
    training_client: Any,
    service_client: Any,
    name: str,
) -> tuple[Any, str]:
    save_for_sampler = getattr(training_client, "save_weights_for_sampler", None)
    if callable(save_for_sampler):
        response = _unwrap_future(save_for_sampler(name=name))
        model_path = _extract_text_field(response, ["path", "model_path", "tinker_path", "id"])
        if not model_path:
            raise RuntimeError("save_weights_for_sampler did not return a checkpoint path")
        sampling_client = _create_sampling_client_from_path(service_client, model_path)
        return sampling_client, model_path

    # Backward-compatible fallback for older runtimes.
    logger.warning("save_weights_for_sampler not available; falling back to save_weights_and_get_sampling_client")
    sampling_client = training_client.save_weights_and_get_sampling_client(name=name)
    model_path = _extract_sampling_model_path(sampling_client) or name
    return sampling_client, model_path


def _save_resume_state(training_client: Any, name: str) -> str | None:
    save_state = getattr(training_client, "save_state", None)
    if not callable(save_state):
        logger.warning("Training client does not expose save_state(); resume checkpoint not saved")
        return None

    try:
        response = _unwrap_future(save_state(name=name))
    except Exception as exc:
        logger.warning("save_state failed: {}", exc)
        return None

    state_path = _extract_text_field(
        response,
        ["state_path", "path", "tinker_path", "checkpoint_path", "id"],
    )
    if state_path is None:
        logger.warning("save_state returned an unknown response shape; cannot extract state path")
    return state_path


def _get_checkpoint_archive_url(client: Any, checkpoint_path: str) -> str | None:
    create_rest_client = getattr(client, "create_rest_client", None)
    if not callable(create_rest_client):
        logger.warning("Service client does not expose create_rest_client(); cannot download adapter")
        return None

    rest_client = create_rest_client()
    getter = getattr(rest_client, "get_checkpoint_archive_url_from_tinker_path", None)
    if not callable(getter):
        logger.warning("REST client missing get_checkpoint_archive_url_from_tinker_path; cannot download adapter")
        return None

    response: Any = None
    call_attempts = [
        lambda: getter(checkpoint_path),
        lambda: getter(tinker_path=checkpoint_path),
        lambda: getter(path=checkpoint_path),
    ]
    for attempt in call_attempts:
        try:
            response = _unwrap_future(attempt())
            break
        except TypeError:
            continue
        except Exception as exc:
            logger.warning("Checkpoint archive URL request failed for {}: {}", checkpoint_path, exc)
            return None

    if response is None:
        logger.warning("Could not call get_checkpoint_archive_url_from_tinker_path with supported signature")
        return None

    url = _extract_text_field(response, ["url", "download_url", "signed_url", "path"])
    if not url:
        logger.warning("Checkpoint archive URL response did not contain a URL")
        return None
    return url


def _safe_extract_tar(archive_path: Path, dest_dir: Path) -> list[str]:
    extracted_files: list[str] = []
    with tarfile.open(archive_path, mode="r:*") as tf:
        dest_resolved = dest_dir.resolve()
        for member in tf.getmembers():
            member_path = (dest_dir / member.name).resolve()
            if not str(member_path).startswith(str(dest_resolved)):
                raise ValueError(f"Unsafe tar path detected: {member.name}")
        tf.extractall(dest_dir)

    for fp in sorted(dest_dir.rglob("*")):
        if fp.is_file():
            extracted_files.append(str(fp.relative_to(dest_dir)))
    return extracted_files


def _download_checkpoint_archive(url: str, out_path: Path, timeout_seconds: int) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(url, timeout=timeout_seconds) as resp, out_path.open("wb") as f:
        shutil.copyfileobj(resp, f)


def _download_adapter_bundle(
    client: Any,
    checkpoint_path: str | None,
    output_dir: Path,
    timeout_seconds: int,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "checkpoint_path": checkpoint_path,
        "output_dir": str(output_dir),
        "downloaded": False,
        "extracted": False,
        "archive_path": None,
        "files": [],
        "error": None,
    }

    if not checkpoint_path:
        result["error"] = "missing_checkpoint_path"
        return result

    if not checkpoint_path.startswith("tinker://"):
        result["error"] = "checkpoint_path_not_tinker_uri"
        return result

    signed_url = _get_checkpoint_archive_url(client, checkpoint_path)
    if not signed_url:
        result["error"] = "failed_to_get_archive_url"
        return result

    archive_path = output_dir / "adapter_archive.tar"
    result["archive_path"] = str(archive_path)
    try:
        _download_checkpoint_archive(signed_url, archive_path, timeout_seconds=timeout_seconds)
        result["downloaded"] = True
    except Exception as exc:
        result["error"] = f"download_failed: {exc}"
        return result

    try:
        extracted_files = _safe_extract_tar(archive_path, output_dir)
        result["files"] = extracted_files
        result["extracted"] = True
    except Exception as exc:
        result["error"] = f"extract_failed: {exc}"

    return result


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    cfg = core.load_config(project_root, args)
    prompt_cfg = core.load_prompt_config(project_root, cfg)
    use_emitted_label_metrics = bool(getattr(args, "use_emitted_label_metrics", False))
    discrete_pred_source = "emitted" if use_emitted_label_metrics else "prob"
    discrete_metrics_source = "emitted_label" if use_emitted_label_metrics else "threshold_val_max_macro_f1"

    if cfg.get("run_name") is None:
        raise RuntimeError("run_name is required for student stage")

    random.seed(int(cfg["seed"]))
    np.random.seed(int(cfg["seed"]))

    run_name = str(cfg["run_name"])
    log_dir = core.resolve_path(project_root, str(cfg["log_dir"]))
    if log_dir is None:
        raise RuntimeError("log_dir resolution failed")
    run_dir = log_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    core.setup_logger(run_dir)
    logger.info("Run dir: {}", run_dir)

    teacher_run_name = str(args.teacher_run_name).strip() if args.teacher_run_name else run_name
    teacher_run_dir = log_dir / teacher_run_name
    if not teacher_run_dir.exists():
        raise FileNotFoundError(f"Missing teacher run directory: {teacher_run_dir}")

    if teacher_run_name != run_name:
        logger.info(
            "Sourcing teacher artifacts from run '{}' and saving student outputs under '{}'",
            teacher_run_name,
            run_name,
        )
        teacher_dir, splits_dir = _copy_teacher_artifacts(teacher_run_dir=teacher_run_dir, run_dir=run_dir)
    else:
        teacher_dir = run_dir / TEACHER_SUBDIR
        splits_dir = run_dir / "data_splits"
        if not teacher_dir.exists():
            raise FileNotFoundError(f"Missing teacher artifact dir: {teacher_dir}")
        for fp in [splits_dir / "train.csv", splits_dir / "val.csv", splits_dir / "test.csv"]:
            if not fp.exists():
                raise FileNotFoundError(f"Missing split file: {fp}")
    logger.info("Loading teacher artifacts from {}", teacher_dir)

    student_dir = run_dir / STUDENT_SUBDIR
    student_dir.mkdir(parents=True, exist_ok=True)

    env_file = core.resolve_path(project_root, str(cfg["teacher_env_file"]))
    if env_file is not None and env_file.exists():
        load_dotenv(env_file, override=False)

    accepted_path = teacher_dir / "accepted_samples.jsonl"
    if not accepted_path.exists():
        raise FileNotFoundError(f"Missing accepted samples: {accepted_path}")
    accepted = read_jsonl(accepted_path)
    if not accepted:
        raise RuntimeError("No accepted samples in teacher artifacts")

    teacher_summary_path = teacher_dir / "teacher_summary.json"
    if not teacher_summary_path.exists():
        raise FileNotFoundError(f"Missing teacher summary: {teacher_summary_path}")
    teacher_summary = json.loads(teacher_summary_path.read_text(encoding="utf-8"))

    train_split = splits_dir / "train.csv"
    val_split = splits_dir / "val.csv"
    test_split = splits_dir / "test.csv"
    for fp in [train_split, val_split, test_split]:
        if not fp.exists():
            raise FileNotFoundError(f"Missing split file: {fp}")

    train_df = pd.read_csv(train_split, encoding="utf-8")
    val_df = pd.read_csv(val_split, encoding="utf-8")
    test_df = pd.read_csv(test_split, encoding="utf-8")

    if "text" not in val_df.columns or "label" not in val_df.columns:
        raise KeyError(f"val split must contain columns text,label: {val_split}")
    if "text" not in test_df.columns or "label" not in test_df.columns:
        raise KeyError(f"test split must contain columns text,label: {test_split}")

    rulebook_path = teacher_dir / "rulebook.txt"
    if not rulebook_path.exists():
        raise FileNotFoundError(f"Missing rulebook: {rulebook_path}")
    rulebook = rulebook_path.read_text(encoding="utf-8")

    client = tinker.ServiceClient(base_url=str(cfg["tinker_base_url"])) if cfg.get("tinker_base_url") else tinker.ServiceClient()
    training_client = client.create_lora_training_client(
        base_model=str(cfg["student_model_name"]),
        rank=int(cfg["lora_rank"]),
    )

    tokenizer = training_client.get_tokenizer()
    if not hasattr(tokenizer, "apply_chat_template"):
        raise RuntimeError("Tokenizer does not support apply_chat_template")

    one_tokens = tokenizer.encode("1", add_special_tokens=False)
    zero_tokens = tokenizer.encode("0", add_special_tokens=False)
    if len(one_tokens) == 0 or len(zero_tokens) == 0:
        raise RuntimeError("Tokenizer returned empty tokens for labels '1' or '0'")

    reasoning_prompt_builder = lambda text: core.build_reasoning_user_prompt(prompt_cfg, rulebook, text)

    train_rows = core.build_train_examples(
        accepted_rows=accepted,
        tokenizer=tokenizer,
        max_length=int(cfg["max_length"]),
        reasoning_prompt_builder=reasoning_prompt_builder,
    )
    val_rows = core.build_eval_rows(
        frame=val_df,
        tokenizer=tokenizer,
        max_length=int(cfg["max_length"]),
        reasoning_prompt_builder=reasoning_prompt_builder,
        reasoning_placeholder=str(cfg["eval_reasoning_placeholder"]),
    )
    test_rows = core.build_eval_rows(
        frame=test_df,
        tokenizer=tokenizer,
        max_length=int(cfg["max_length"]),
        reasoning_prompt_builder=reasoning_prompt_builder,
        reasoning_placeholder=str(cfg["eval_reasoning_placeholder"]),
    )

    logger.info("Usable tokenized rows: train={}, val={}, test={}", len(train_rows), len(val_rows), len(test_rows))

    fingerprints = _fingerprint_run_inputs(run_dir, teacher_dir, cfg, prompt_cfg)

    if bool(cfg["save_sft_jsonl"]):
        sft_rows = [{"messages": ex["messages"], "label": ex["label"]} for ex in train_rows]
        core.write_jsonl(run_dir / "train_sft.jsonl", sft_rows)

    wandb_run = wandb.init(
        project=str(cfg["wandb_project"]),
        entity=cfg.get("wandb_entity"),
        name=run_name,
        config=cfg,
        mode=str(cfg["wandb_mode"]),
        dir=str(run_dir),
    )
    wandb.log(
        {
            "teacher/requested_samples": teacher_summary["requested_samples"],
            "teacher/accepted_samples": teacher_summary["accepted_samples"],
            "teacher/acceptance_rate": teacher_summary["acceptance_rate"],
        },
        step=0,
    )

    rng = random.Random(int(cfg["seed"]))
    batches_per_epoch = math.ceil(len(train_rows) / int(cfg["batch_size"]))
    total_steps = int(cfg["num_epochs"]) * batches_per_epoch

    selection_metric = str(cfg["selection_metric"])
    selection_direction = core.SELECTION_DIRECTIONS[selection_metric]
    best_val_selection_score = float("inf") if selection_direction == "min" else float("-inf")
    best_val_metric_value = float("nan")
    best_val_macro_f1 = -1.0
    best_client = None
    best_model_path = ""
    best_step = -1
    best_decision_threshold = 0.5

    step = 0
    for epoch in range(int(cfg["num_epochs"])):
        order = list(range(len(train_rows)))
        rng.shuffle(order)

        for i in range(0, len(order), int(cfg["batch_size"])):
            step_client = None
            do_eval = (step % int(cfg["eval_interval"]) == 0) or (step == total_steps - 1)

            if do_eval:
                step_client, step_model_path = _save_sampler_checkpoint(
                    training_client=training_client,
                    service_client=client,
                    name=f"{run_name}_step_{step}_eval",
                )
                val_eval = core.evaluate_binary(
                    step_client,
                    tokenizer,
                    val_rows,
                    max_samples=int(cfg["max_eval_samples"]),
                    one_tokens=one_tokens,
                    zero_tokens=zero_tokens,
                    max_concurrency=int(cfg["eval_max_concurrency"]),
                    invalid_warn_rate=float(cfg["invalid_label_warn_rate"]),
                    discrete_pred_source=discrete_pred_source,
                )
                test_eval = core.evaluate_binary(
                    step_client,
                    tokenizer,
                    test_rows,
                    max_samples=int(cfg["max_eval_samples"]),
                    one_tokens=one_tokens,
                    zero_tokens=zero_tokens,
                    max_concurrency=int(cfg["eval_max_concurrency"]),
                    invalid_warn_rate=float(cfg["invalid_label_warn_rate"]),
                    discrete_pred_source=discrete_pred_source,
                )

                decision_threshold: float | None = None
                if not use_emitted_label_metrics:
                    decision_threshold = core.find_best_threshold_macro_f1(
                        val_eval["y_true"],
                        val_eval["p_one"],
                        val_eval.get("invalid_flags"),
                    )
                    val_thr = core.thresholded_classification_metrics(
                        val_eval["y_true"],
                        val_eval["p_one"],
                        val_eval.get("invalid_flags"),
                        decision_threshold,
                    )
                    test_thr = core.thresholded_classification_metrics(
                        test_eval["y_true"],
                        test_eval["p_one"],
                        test_eval.get("invalid_flags"),
                        decision_threshold,
                    )

                    # Override hard-prediction dependent metrics with val-calibrated threshold.
                    val_eval["decision_threshold"] = decision_threshold
                    val_eval["y_pred"] = val_thr["y_pred"]
                    val_eval["accuracy"] = val_thr["accuracy"]
                    val_eval["macro_f1"] = val_thr["macro_f1"]
                    test_eval["decision_threshold"] = decision_threshold
                    test_eval["y_pred"] = test_thr["y_pred"]
                    test_eval["accuracy"] = test_thr["accuracy"]
                    test_eval["macro_f1"] = test_thr["macro_f1"]

                val_eval["discrete_metrics_source"] = discrete_metrics_source
                test_eval["discrete_metrics_source"] = discrete_metrics_source

                wandb_payload = {
                    "val/loss": val_eval["loss"],
                    "val/cls_loss": val_eval["cls_loss"],
                    "val/accuracy": val_eval["accuracy"],
                    "val/macro_f1": val_eval["macro_f1"],
                    "val/auroc": val_eval["auroc"],
                    "val/auprc": val_eval["auprc"],
                    "val/invalid_label_rate": val_eval["invalid_label_rate"],
                    "test/loss": test_eval["loss"],
                    "test/cls_loss": test_eval["cls_loss"],
                    "test/accuracy": test_eval["accuracy"],
                    "test/macro_f1": test_eval["macro_f1"],
                    "test/auroc": test_eval["auroc"],
                    "test/auprc": test_eval["auprc"],
                    "test/invalid_label_rate": test_eval["invalid_label_rate"],
                }
                if decision_threshold is not None:
                    wandb_payload["val/decision_threshold"] = decision_threshold
                    wandb_payload["test/decision_threshold"] = decision_threshold
                wandb.log(wandb_payload, step=step)

                best_val_macro_f1 = max(best_val_macro_f1, float(val_eval["macro_f1"]))
                selection_metric_value = float(val_eval.get(selection_metric, float("nan")))
                selection_candidate = core._selection_score_for_compare(selection_metric, selection_metric_value)

                logger.info(
                    "Step {} | val: loss={:.6f}, cls_loss={:.6f}, acc={:.6f}, f1={:.6f}, auroc={:.6f}, auprc={:.6f}, invalid_rate={:.4f}{} sel({})={:.6f} | test: loss={:.6f}, cls_loss={:.6f}, acc={:.6f}, f1={:.6f}, auroc={:.6f}, auprc={:.6f}, invalid_rate={:.4f}{}",
                    step,
                    float(val_eval["loss"]),
                    float(val_eval["cls_loss"]),
                    float(val_eval["accuracy"]),
                    float(val_eval["macro_f1"]),
                    float(val_eval["auroc"]),
                    float(val_eval["auprc"]),
                    float(val_eval["invalid_label_rate"]),
                    "" if decision_threshold is None else f", thr={float(decision_threshold):.4f}",
                    selection_metric,
                    selection_metric_value,
                    float(test_eval["loss"]),
                    float(test_eval["cls_loss"]),
                    float(test_eval["accuracy"]),
                    float(test_eval["macro_f1"]),
                    float(test_eval["auroc"]),
                    float(test_eval["auprc"]),
                    float(test_eval["invalid_label_rate"]),
                    "" if decision_threshold is None else f", thr={float(decision_threshold):.4f}",
                )

                _save_eval_artifacts(
                    student_dir=student_dir,
                    checkpoint_path=step_model_path,
                    checkpoint_name=f"{run_name}_step_{step}_eval",
                    step=step,
                    split_name="val",
                    eval_out=val_eval,
                    fingerprints=fingerprints,
                )
                _save_eval_artifacts(
                    student_dir=student_dir,
                    checkpoint_path=step_model_path,
                    checkpoint_name=f"{run_name}_step_{step}_eval",
                    step=step,
                    split_name="test",
                    eval_out=test_eval,
                    fingerprints=fingerprints,
                )

                if core._is_better_selection(selection_metric, selection_candidate, best_val_selection_score):
                    best_val_selection_score = selection_candidate
                    best_val_metric_value = selection_metric_value
                    best_client = step_client
                    best_model_path = step_model_path
                    best_step = step
                    if decision_threshold is not None:
                        best_decision_threshold = float(decision_threshold)

            batch = [train_rows[j] for j in order[i : i + int(cfg["batch_size"])]]
            batch_data = [x["datum"] for x in batch]

            if str(cfg["lr_schedule"]) == "constant":
                lr = float(cfg["learning_rate"])
            else:
                progress = step / max(total_steps - 1, 1)
                lr = float(cfg["learning_rate"]) * (1.0 - (1.0 - float(cfg["min_lr_ratio"])) * progress)

            adam = tinker.AdamParams(
                learning_rate=lr,
                beta1=float(cfg["adam_beta1"]),
                beta2=float(cfg["adam_beta2"]),
                eps=float(cfg["adam_eps"]),
            )

            fwd = training_client.forward_backward(batch_data, loss_fn="cross_entropy")
            opt = training_client.optim_step(adam)
            fwd_result = fwd.result()
            opt.result()

            trainer_loss = None
            if "loss" in fwd_result.metrics:
                trainer_loss = float(fwd_result.metrics["loss"])
            else:
                total_w_logprob = 0.0
                total_w = 0.0
                for out, ex in zip(fwd_result.loss_fn_outputs, batch, strict=True):
                    lp = np.asarray(out["logprobs"].to_numpy(), dtype=np.float64).reshape(-1)
                    w = np.asarray(ex["datum"].loss_fn_inputs["weights"].to_numpy(), dtype=np.float64).reshape(-1)
                    if lp.shape[0] != w.shape[0]:
                        raise ValueError(f"logprobs/weights length mismatch: {lp.shape[0]} vs {w.shape[0]}")
                    total_w_logprob += float(np.dot(lp, w))
                    total_w += float(w.sum())
                trainer_loss = float(-total_w_logprob / max(total_w, 1e-9))

            if step_client is None:
                step_client, _ = _save_sampler_checkpoint(
                    training_client=training_client,
                    service_client=client,
                    name=f"{run_name}_step_{step}_train",
                )

            batch_eval_rows = [{"text": ex["text"], "label": ex["label"], "prompt_tokens": ex["eval_prompt_tokens"]} for ex in batch]
            train_eval = core.evaluate_binary(
                step_client,
                tokenizer,
                batch_eval_rows,
                max_samples=0,
                one_tokens=one_tokens,
                zero_tokens=zero_tokens,
                max_concurrency=int(cfg["eval_max_concurrency"]),
                invalid_warn_rate=float(cfg["invalid_label_warn_rate"]),
                discrete_pred_source=discrete_pred_source,
            )

            wandb.log(
                {
                    "train/trainer_loss": trainer_loss,
                    "train/eval_loss": train_eval["loss"],
                    "train/eval_cls_loss": train_eval["cls_loss"],
                    "train/accuracy": train_eval["accuracy"],
                    "train/auroc": train_eval["auroc"],
                    "train/auprc": train_eval["auprc"],
                    "train/invalid_label_rate": train_eval["invalid_label_rate"],
                    "train/lr": lr,
                    "epoch": epoch,
                },
                step=step,
            )

            logger.info(
                "Step {} | epoch {} | train: trainer_loss={:.6f}, eval_loss={:.6f}, eval_cls_loss={:.6f}, acc={:.6f}, auroc={:.6f}, auprc={:.6f}, invalid_rate={:.4f}, lr={:.6e}",
                step,
                epoch,
                float(trainer_loss),
                float(train_eval["loss"]),
                float(train_eval["cls_loss"]),
                float(train_eval["accuracy"]),
                float(train_eval["auroc"]),
                float(train_eval["auprc"]),
                float(train_eval["invalid_label_rate"]),
                float(lr),
            )

            step += 1

    final_client, final_model_path = _save_sampler_checkpoint(
        training_client=training_client,
        service_client=client,
        name=f"{run_name}_final",
    )

    final_decision_threshold = 0.5
    final_val_eval = core.evaluate_binary(
        final_client,
        tokenizer,
        val_rows,
        max_samples=0,
        one_tokens=one_tokens,
        zero_tokens=zero_tokens,
        max_concurrency=int(cfg["eval_max_concurrency"]),
        invalid_warn_rate=float(cfg["invalid_label_warn_rate"]),
        discrete_pred_source=discrete_pred_source,
    )
    if not use_emitted_label_metrics:
        final_decision_threshold = core.find_best_threshold_macro_f1(
            final_val_eval["y_true"],
            final_val_eval["p_one"],
            final_val_eval.get("invalid_flags"),
        )
        final_val_thr = core.thresholded_classification_metrics(
            final_val_eval["y_true"],
            final_val_eval["p_one"],
            final_val_eval.get("invalid_flags"),
            final_decision_threshold,
        )
        final_val_eval["decision_threshold"] = float(final_decision_threshold)
        final_val_eval["y_pred"] = final_val_thr["y_pred"]
        final_val_eval["accuracy"] = final_val_thr["accuracy"]
        final_val_eval["macro_f1"] = final_val_thr["macro_f1"]
    final_val_eval["discrete_metrics_source"] = discrete_metrics_source
    _save_eval_artifacts(
        student_dir=student_dir,
        checkpoint_path=final_model_path,
        checkpoint_name=f"{run_name}_final",
        step=step,
        split_name="val_final",
        eval_out=final_val_eval,
        fingerprints=fingerprints,
    )

    final_eval = core.evaluate_binary(
        final_client,
        tokenizer,
        test_rows,
        max_samples=0,
        one_tokens=one_tokens,
        zero_tokens=zero_tokens,
        max_concurrency=int(cfg["eval_max_concurrency"]),
        invalid_warn_rate=float(cfg["invalid_label_warn_rate"]),
        discrete_pred_source=discrete_pred_source,
    )
    if not use_emitted_label_metrics:
        final_test_thr = core.thresholded_classification_metrics(
            final_eval["y_true"],
            final_eval["p_one"],
            final_eval.get("invalid_flags"),
            final_decision_threshold,
        )
        final_eval["decision_threshold"] = float(final_decision_threshold)
        final_eval["y_pred"] = final_test_thr["y_pred"]
        final_eval["accuracy"] = final_test_thr["accuracy"]
        final_eval["macro_f1"] = final_test_thr["macro_f1"]
    final_eval["discrete_metrics_source"] = discrete_metrics_source
    final_metrics = core.binary_metrics(final_eval["y_true"], final_eval["y_pred"], final_eval["p_one"])
    _save_eval_artifacts(
        student_dir=student_dir,
        checkpoint_path=final_model_path,
        checkpoint_name=f"{run_name}_final",
        step=step,
        split_name="test_final",
        eval_out=final_eval,
        fingerprints=fingerprints,
    )

    if best_client is None:
        best_client = final_client
        best_model_path = final_model_path
        best_step = step - 1
        if not use_emitted_label_metrics:
            best_decision_threshold = float(final_decision_threshold)

    resume_state_path = None
    if args.save_resume_checkpoint:
        resume_state_path = _save_resume_state(training_client, name=f"{run_name}_resume_state")

    adapter_downloads: dict[str, Any] = {}
    if args.download_adapter:
        model_root = run_dir / args.adapter_output_subdir
        model_root.mkdir(parents=True, exist_ok=True)
        final_download_checkpoint = final_model_path if final_model_path.startswith("tinker://") else resume_state_path
        best_download_checkpoint = best_model_path if best_model_path.startswith("tinker://") else None

        adapter_downloads["final"] = _download_adapter_bundle(
            client=client,
            checkpoint_path=final_download_checkpoint,
            output_dir=model_root / "final",
            timeout_seconds=int(args.adapter_download_timeout_seconds),
        )
        if best_download_checkpoint is None:
            adapter_downloads["best_val"] = {
                "checkpoint_path": best_model_path,
                "output_dir": str(model_root / "best_val"),
                "downloaded": False,
                "extracted": False,
                "archive_path": None,
                "files": [],
                "error": "missing_tinker_checkpoint_path_for_best",
            }
        elif final_download_checkpoint == best_download_checkpoint:
            adapter_downloads["best_val"] = {
                "checkpoint_path": best_model_path,
                "output_dir": str(model_root / "best_val"),
                "downloaded": False,
                "extracted": False,
                "archive_path": None,
                "files": [],
                "error": "skipped_same_as_final",
            }
        else:
            adapter_downloads["best_val"] = _download_adapter_bundle(
                client=client,
                checkpoint_path=best_download_checkpoint,
                output_dir=model_root / "best_val",
                timeout_seconds=int(args.adapter_download_timeout_seconds),
            )

        for tag, info in adapter_downloads.items():
            if info.get("error"):
                logger.warning("Adapter download [{}] warning: {}", tag, info["error"])
            else:
                logger.info(
                    "Adapter download [{}] complete: {} files at {}",
                    tag,
                    len(info.get("files", [])),
                    info.get("output_dir"),
                )

    best_eval = core.evaluate_binary(
        best_client,
        tokenizer,
        test_rows,
        max_samples=0,
        one_tokens=one_tokens,
        zero_tokens=zero_tokens,
        max_concurrency=int(cfg["eval_max_concurrency"]),
        invalid_warn_rate=float(cfg["invalid_label_warn_rate"]),
        discrete_pred_source=discrete_pred_source,
    )
    if not use_emitted_label_metrics:
        best_eval_thr = core.thresholded_classification_metrics(
            best_eval["y_true"],
            best_eval["p_one"],
            best_eval.get("invalid_flags"),
            best_decision_threshold,
        )
        best_eval["decision_threshold"] = float(best_decision_threshold)
        best_eval["y_pred"] = best_eval_thr["y_pred"]
        best_eval["accuracy"] = best_eval_thr["accuracy"]
        best_eval["macro_f1"] = best_eval_thr["macro_f1"]
    best_eval["discrete_metrics_source"] = discrete_metrics_source
    best_metrics = core.binary_metrics(best_eval["y_true"], best_eval["y_pred"], best_eval["p_one"])
    _save_eval_artifacts(
        student_dir=student_dir,
        checkpoint_path=best_model_path,
        checkpoint_name=f"{run_name}_bestval",
        step=best_step,
        split_name="test_bestval",
        eval_out=best_eval,
        fingerprints=fingerprints,
    )

    wandb_payload = {
        "final_test/accuracy": final_metrics["accuracy"],
        "final_test/balanced_accuracy": final_metrics["balanced_accuracy"],
        "final_test/f1": final_metrics["F1"],
        "final_test/auroc": final_metrics["auroc"],
        "final_test/auprc": final_metrics["auprc"],
        "final_test/invalid_label_rate": final_eval["invalid_label_rate"],
        "bestval_test/accuracy": best_metrics["accuracy"],
        "bestval_test/balanced_accuracy": best_metrics["balanced_accuracy"],
        "bestval_test/f1": best_metrics["F1"],
        "bestval_test/auroc": best_metrics["auroc"],
        "bestval_test/auprc": best_metrics["auprc"],
        "bestval_test/invalid_label_rate": best_eval["invalid_label_rate"],
        "selection/metric_name": selection_metric,
        "selection/best_val_metric": best_val_metric_value,
        "best_step": best_step,
        "discrete_metrics_source": discrete_metrics_source,
    }
    if not use_emitted_label_metrics:
        wandb_payload["best_val/decision_threshold"] = float(best_decision_threshold)
        wandb_payload["final/decision_threshold"] = float(final_decision_threshold)
    wandb.log(wandb_payload, step=step)

    report_path = run_dir / "test_report.md"
    blocks = [("final model", final_model_path, final_metrics, final_eval), ("best-val model", best_model_path, best_metrics, best_eval)]
    lines = ["# Reasoning SFT Test Report", "", f"Generated at: {datetime.now().isoformat(timespec='seconds')}", ""]
    lines += [f"- resume_state_path: {resume_state_path}", ""]
    for tag, model_path, m, eval_info in blocks:
        lines += [
            f"## {tag}",
            "",
            f"- final model or best-val model: {tag}",
            f"- model_path: {model_path}",
            f"- accuracy: {m['accuracy']:.6f}",
            f"- balanced_accuracy: {m['balanced_accuracy']:.6f}",
            f"- F1: {m['F1']:.6f}",
            f"- mcc: {m['mcc']:.6f}",
            f"- precision: {m['precision']:.6f}",
            f"- recall: {m['recall']:.6f}",
            f"- auroc: {m['auroc']:.6f}",
            f"- auprc: {m['auprc']:.6f}",
            f"- invalid_label_rate: {float(eval_info['invalid_label_rate']):.6f}",
            f"- tp: {m['tp']}",
            f"- fp: {m['fp']}",
            f"- fn: {m['fn']}",
            f"- tn: {m['tn']}",
            "",
        ]
    report_path.write_text("\n".join(lines), encoding="utf-8")

    summary = {
        "run_dir": str(run_dir),
        "teacher_dir": str(teacher_dir),
        "report_path": str(report_path),
        "final_model_path": final_model_path,
        "best_model_path": best_model_path,
        "resume_state_path": resume_state_path,
        "adapter_downloads": adapter_downloads,
        "best_step": best_step,
        "best_val_macro_f1": best_val_macro_f1,
        "selection_metric": selection_metric,
        "best_val_selection_score": best_val_selection_score,
        "best_val_selection_metric_value": best_val_metric_value,
        "teacher_summary": teacher_summary,
    }
    (run_dir / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    model_pointer_info: dict[str, str] | None = None
    if args.save_resume_checkpoint:
        # Only write stable pointers when the user explicitly enabled checkpoint saving.
        payload = {
            "saved_at": datetime.now().isoformat(timespec="seconds"),
            "run_name": run_name,
            "final_model_path": final_model_path,
            "best_model_path": best_model_path,
            "resume_state_path": resume_state_path,
        }
        model_pointer_info = _save_model_pointer_files(project_root, run_name, payload)

    wandb_run.finish()
    logger.info(
        "Student phase complete | run={} | best_step={} | final_path={} | best_path={}",
        run_name,
        best_step,
        final_model_path,
        best_model_path,
    )
    logger.info("Metrics/report saved at: {}", report_path)
    if model_pointer_info is not None:
        logger.info(
            "Saved model pointer files under {} (latest: {})",
            model_pointer_info["model_dir"],
            model_pointer_info["latest_pointer"],
        )


if __name__ == "__main__":
    main()
