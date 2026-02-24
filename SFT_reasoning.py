#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import math
import os
import random
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterator

import numpy as np
import pandas as pd
import tinker
import wandb
import yaml
from loguru import logger
from openai import OpenAI
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from tqdm import tqdm

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

DEFAULT_PROMPT_FILE = "tinker/prompts/sft_reasoning.yaml"
DEFAULT_PROMPTS: dict[str, str] = {
    "task_instruction": "Decide whether the following text is hate speech.",
    "teacher_system_prompt": (
        "You are a careful binary classifier. "
        "Always return strict JSON with keys 'reasoning' and 'label'."
    ),
    "reasoning_user_prompt_template": (
        "{task_instruction}\n\n"
        "Use the rulebook below to reason before selecting the label.\n\n"
        "Rulebook:\n{rulebook}\n\n"
        "Text:\n{text}\n\n"
        "Return a JSON object with keys:\n"
        '- "reasoning": concise rule-grounded explanation\n'
        '- "label": integer 0 or 1\n'
    ),
    "label_only_user_prompt_template": (
        "{task_instruction}\n\n"
        "Use the rulebook below, then return only the final label.\n\n"
        "Rulebook:\n{rulebook}\n\n"
        "Text:\n{text}\n\n"
        "Respond with only one token: 0 or 1."
    ),
}


DEFAULT_CONFIG: dict[str, Any] = {
    "dataset_root_dir": "tinker/dataset",
    "dataset_name": "ethos",
    "data_path": "datasets/ethos/Ethos_Dataset_Binary.csv",
    "text_column": "comment",
    "label_column": "isHate",
    "label_threshold": 0.5,
    "csv_sep": ";",
    "seed": 42,
    "prompt_file": DEFAULT_PROMPT_FILE,
    "task_instruction": DEFAULT_PROMPTS["task_instruction"],
    "rules_root_dir": "tinker/rules",
    "rules_dir": "tinker/rules",
    "rules_glob": "*.txt",
    "teacher_env_file": "extraction/.env",
    "teacher_api_key_env": "DEEPSEEK_API_KEY",
    "teacher_base_url": "https://api.deepseek.com",
    "teacher_model": "deepseek/deepseek-chat",
    "teacher_temperature": 0.7,
    "teacher_max_tokens": 300,
    "teacher_k": 3,
    "teacher_workers": 1,
    "teacher_max_retries": 3,
    "teacher_request_timeout": 90,
    "teacher_json_mode": True,
    "teacher_sleep_seconds": 0.0,
    "student_model_name": "Qwen3-8B",
    "tinker_base_url": None,
    "lora_rank": 32,
    "ttl_seconds": 7 * 24 * 3600,
    "num_epochs": 3,
    "batch_size": 16,
    "learning_rate": 2e-4,
    "adam_beta1": 0.9,
    "adam_beta2": 0.95,
    "adam_eps": 1e-8,
    "lr_schedule": "linear",
    "min_lr_ratio": 0.1,
    "max_length": 2048,
    "eval_interval": 10,
    "max_eval_samples": 0,
    "eval_max_concurrency": 64,
    "eval_reasoning_placeholder": "Reasoning intentionally omitted for label scoring.",
    "invalid_label_warn_rate": 0.10,
    "selection_metric": "macro_f1",
    "min_reasoning_chars": 20,
    "max_train_examples": 0,
    "max_val_examples": 0,
    "max_test_examples": 0,
    "log_dir": "tinker/runs",
    "run_name": None,
    "save_sft_jsonl": True,
    "wandb_project": "reasoning-sft-tinker",
    "wandb_entity": None,
    "wandb_mode": "online",
}


@dataclass
class ParsedTeacherOutput:
    reasoning: str
    label: int
    source: str


@dataclass
class PromptConfig:
    task_instruction: str
    teacher_system_prompt: str
    reasoning_user_prompt_template: str
    label_only_user_prompt_template: str


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Reasoning SFT with DeepSeek teacher + Tinker student")
    p.add_argument("--config", type=str, default="tinker/configs/reasoning_sft.example.json")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--dataset-name", "--dataset_name", dest="dataset_name", type=str, default=None)
    p.add_argument("--dataset-root-dir", "--dataset_root_dir", dest="dataset_root_dir", type=str, default=None)
    p.add_argument("--rules-root-dir", "--rules_root_dir", dest="rules_root_dir", type=str, default=None)
    p.add_argument("--run-name", type=str, default=None)
    p.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default=None)
    p.add_argument("--teacher-k", type=int, default=None)
    p.add_argument("--teacher-workers", "--teacher_workers", dest="teacher_workers", type=int, default=None)
    p.add_argument(
        "--selection-metric",
        choices=["macro_f1", "auroc", "auprc", "accuracy", "loss", "cls_loss"],
        default=None,
    )
    p.add_argument("--max-train-examples", type=int, default=None)
    p.add_argument("--max-val-examples", type=int, default=None)
    p.add_argument("--max-test-examples", type=int, default=None)
    return p.parse_args()


def resolve_path(project_root: Path, path_like: str | None) -> Path | None:
    if path_like is None:
        return None
    p = Path(path_like)
    return p if p.is_absolute() else (project_root / p)


def load_config(project_root: Path, args: argparse.Namespace) -> dict[str, Any]:
    cfg = dict(DEFAULT_CONFIG)
    cfg_path = resolve_path(project_root, args.config)
    if cfg_path is not None and cfg_path.exists():
        loaded = json.loads(cfg_path.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            raise ValueError(f"Config must be a JSON object: {cfg_path}")
        cfg.update(loaded)

    run_name_arg = getattr(args, "run_name", None)
    seed_arg = getattr(args, "seed", None)
    dataset_name_arg = getattr(args, "dataset_name", None)
    dataset_root_dir_arg = getattr(args, "dataset_root_dir", None)
    rules_root_dir_arg = getattr(args, "rules_root_dir", None)
    wandb_mode_arg = getattr(args, "wandb_mode", None)
    teacher_k_arg = getattr(args, "teacher_k", None)
    teacher_workers_arg = getattr(args, "teacher_workers", None)
    teacher_temperature_arg = getattr(args, "teacher_temperature", None)
    selection_metric_arg = getattr(args, "selection_metric", None)
    max_train_examples_arg = getattr(args, "max_train_examples", None)
    max_val_examples_arg = getattr(args, "max_val_examples", None)
    max_test_examples_arg = getattr(args, "max_test_examples", None)
    student_model_name_arg = getattr(args, "student_model_name", None)
    lora_rank_arg = getattr(args, "lora_rank", None)
    num_epochs_arg = getattr(args, "num_epochs", None)
    batch_size_arg = getattr(args, "batch_size", None)
    learning_rate_arg = getattr(args, "learning_rate", None)
    max_length_arg = getattr(args, "max_length", None)
    eval_interval_arg = getattr(args, "eval_interval", None)
    max_eval_samples_arg = getattr(args, "max_eval_samples", None)
    eval_max_concurrency_arg = getattr(args, "eval_max_concurrency", None)

    if run_name_arg is not None:
        cfg["run_name"] = run_name_arg
    if seed_arg is not None:
        cfg["seed"] = int(seed_arg)
    if dataset_name_arg is not None:
        cfg["dataset_name"] = str(dataset_name_arg)
    if dataset_root_dir_arg is not None:
        cfg["dataset_root_dir"] = str(dataset_root_dir_arg)
    if rules_root_dir_arg is not None:
        cfg["rules_root_dir"] = str(rules_root_dir_arg)
    if wandb_mode_arg is not None:
        cfg["wandb_mode"] = wandb_mode_arg
    if teacher_k_arg is not None:
        cfg["teacher_k"] = teacher_k_arg
    if teacher_workers_arg is not None:
        cfg["teacher_workers"] = int(teacher_workers_arg)
    if teacher_temperature_arg is not None:
        cfg["teacher_temperature"] = float(teacher_temperature_arg)
    if selection_metric_arg is not None:
        cfg["selection_metric"] = selection_metric_arg
    if max_train_examples_arg is not None:
        cfg["max_train_examples"] = int(max_train_examples_arg)
    if max_val_examples_arg is not None:
        cfg["max_val_examples"] = int(max_val_examples_arg)
    if max_test_examples_arg is not None:
        cfg["max_test_examples"] = int(max_test_examples_arg)
    if student_model_name_arg is not None:
        cfg["student_model_name"] = str(student_model_name_arg)
    if lora_rank_arg is not None:
        cfg["lora_rank"] = int(lora_rank_arg)
    if num_epochs_arg is not None:
        cfg["num_epochs"] = int(num_epochs_arg)
    if batch_size_arg is not None:
        cfg["batch_size"] = int(batch_size_arg)
    if learning_rate_arg is not None:
        cfg["learning_rate"] = float(learning_rate_arg)
    if max_length_arg is not None:
        cfg["max_length"] = int(max_length_arg)
    if eval_interval_arg is not None:
        cfg["eval_interval"] = int(eval_interval_arg)
    if max_eval_samples_arg is not None:
        cfg["max_eval_samples"] = int(max_eval_samples_arg)
    if eval_max_concurrency_arg is not None:
        cfg["eval_max_concurrency"] = int(eval_max_concurrency_arg)

    if int(cfg["teacher_k"]) <= 0:
        raise ValueError("teacher_k must be > 0")
    if int(cfg["teacher_workers"]) <= 0:
        raise ValueError("teacher_workers must be > 0")
    if int(cfg["batch_size"]) <= 0:
        raise ValueError("batch_size must be > 0")
    if int(cfg["num_epochs"]) <= 0:
        raise ValueError("num_epochs must be > 0")
    if int(cfg["max_length"]) <= 64:
        raise ValueError("max_length must be > 64")
    if int(cfg["eval_interval"]) <= 0:
        raise ValueError("eval_interval must be > 0")
    if int(cfg["eval_max_concurrency"]) <= 0:
        raise ValueError("eval_max_concurrency must be > 0")
    if float(cfg["invalid_label_warn_rate"]) < 0.0 or float(cfg["invalid_label_warn_rate"]) > 1.0:
        raise ValueError("invalid_label_warn_rate must be in [0, 1]")
    if str(cfg["selection_metric"]) not in {"macro_f1", "auroc", "auprc", "accuracy", "loss", "cls_loss"}:
        raise ValueError("selection_metric must be one of: macro_f1, auroc, auprc, accuracy, loss, cls_loss")
    if not str(cfg.get("dataset_name", "")).strip():
        raise ValueError("dataset_name must be a non-empty string")
    if int(cfg["max_train_examples"]) < 0:
        raise ValueError("max_train_examples must be >= 0")
    if int(cfg["max_val_examples"]) < 0:
        raise ValueError("max_val_examples must be >= 0")
    if int(cfg["max_test_examples"]) < 0:
        raise ValueError("max_test_examples must be >= 0")

    return cfg


def load_prompt_config(project_root: Path, cfg: dict[str, Any]) -> PromptConfig:
    prompt_path = resolve_path(project_root, str(cfg.get("prompt_file", DEFAULT_PROMPT_FILE)))
    if prompt_path is None or not prompt_path.exists():
        raise FileNotFoundError(f"Missing prompt file: {prompt_path}")

    loaded = yaml.safe_load(prompt_path.read_text(encoding="utf-8"))
    if loaded is None:
        loaded = {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Prompt file must contain a YAML object: {prompt_path}")

    merged = dict(DEFAULT_PROMPTS)
    for key in DEFAULT_PROMPTS:
        if key in loaded and loaded[key] is not None:
            merged[key] = str(loaded[key])

    # Keep backward compatibility for configs that already set task_instruction.
    if cfg.get("task_instruction") is not None:
        merged["task_instruction"] = str(cfg["task_instruction"])

    return PromptConfig(
        task_instruction=merged["task_instruction"],
        teacher_system_prompt=merged["teacher_system_prompt"],
        reasoning_user_prompt_template=merged["reasoning_user_prompt_template"],
        label_only_user_prompt_template=merged["label_only_user_prompt_template"],
    )


def setup_logger(run_dir: Path) -> None:
    logger.remove()
    logger.add(sys.stderr, level="INFO", format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}")
    logger.add(run_dir / "train.log", level="INFO")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_binary_label(value: Any, threshold: float) -> int | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if isinstance(value, (bool, np.bool_)):
        return int(value)
    if isinstance(value, (int, np.integer)):
        v = int(value)
        return v if v in {0, 1} else None
    if isinstance(value, (float, np.floating)):
        v = float(value)
        if v in {0.0, 1.0}:
            return int(v)
        return 1 if v >= threshold else 0

    s = str(value).strip().lower()
    if not s:
        return None
    if s in {"0", "no", "false", "negative"}:
        return 0
    if s in {"1", "yes", "true", "positive"}:
        return 1
    try:
        numeric = float(s)
    except ValueError:
        return None
    if numeric in {0.0, 1.0}:
        return int(numeric)
    return 1 if numeric >= threshold else 0


def load_dataset(
    data_path: Path,
    text_column: str,
    label_column: str,
    csv_sep: str | None,
    label_threshold: float,
) -> pd.DataFrame:
    suffix = data_path.suffix.lower()
    if suffix == ".jsonl":
        rows: list[dict[str, Any]] = []
        with data_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
        frame = pd.DataFrame(rows)
    else:
        sep = csv_sep or ","
        try:
            frame = pd.read_csv(data_path, sep=sep, encoding="utf-8", encoding_errors="replace", on_bad_lines="skip")
        except Exception:
            frame = pd.read_csv(data_path, sep=";", encoding="utf-8", encoding_errors="replace", on_bad_lines="skip")

    if text_column not in frame.columns:
        raise KeyError(f"Missing text column '{text_column}' in {data_path}. Columns={list(frame.columns)}")
    if label_column not in frame.columns:
        raise KeyError(f"Missing label column '{label_column}' in {data_path}. Columns={list(frame.columns)}")

    out = frame[[text_column, label_column]].copy()
    out = out.dropna(subset=[text_column, label_column])

    texts: list[str] = []
    labels: list[int] = []
    for _, row in out.iterrows():
        text = str(row[text_column]).strip()
        if not text:
            continue
        label = parse_binary_label(row[label_column], threshold=label_threshold)
        if label is None:
            continue
        texts.append(text)
        labels.append(label)

    if not texts:
        raise RuntimeError("No usable rows after parsing text/label")

    return pd.DataFrame({"text": texts, "label": labels})


def label_distribution(frame: pd.DataFrame) -> dict[str, float]:
    if "label" not in frame.columns or frame.empty:
        return {}
    vc = frame["label"].value_counts(normalize=True).sort_index()
    out: dict[str, float] = {}
    for k, v in vc.items():
        try:
            key = str(int(k))
        except Exception:
            key = str(k)
        out[key] = float(v)
    return out


def label_counts(frame: pd.DataFrame) -> dict[str, int]:
    if "label" not in frame.columns or frame.empty:
        return {}
    vc = frame["label"].value_counts().sort_index()
    out: dict[str, int] = {}
    for k, v in vc.items():
        try:
            key = str(int(k))
        except Exception:
            key = str(k)
        out[key] = int(v)
    return out


def stratified_subset(
    frame: pd.DataFrame,
    max_examples: int,
    *,
    seed: int,
    split_name: str,
) -> pd.DataFrame:
    n_total = len(frame)
    if max_examples <= 0 or n_total <= max_examples:
        return frame.reset_index(drop=True)

    if "label" not in frame.columns:
        logger.warning(
            "Split '{}' missing label column; using random subset n={} from {} rows",
            split_name,
            max_examples,
            n_total,
        )
        return frame.sample(n=max_examples, random_state=seed, replace=False).reset_index(drop=True)

    class_counts = frame["label"].value_counts()
    n_classes = int(class_counts.shape[0])
    if n_classes < 2:
        logger.warning(
            "Split '{}' has <2 classes; using random subset n={} from {} rows",
            split_name,
            max_examples,
            n_total,
        )
        return frame.sample(n=max_examples, random_state=seed, replace=False).reset_index(drop=True)
    if max_examples < n_classes:
        logger.warning(
            "Requested {} examples for split '{}' but there are {} classes; using random subset",
            max_examples,
            split_name,
            n_classes,
        )
        return frame.sample(n=max_examples, random_state=seed, replace=False).reset_index(drop=True)

    try:
        sampled, _ = train_test_split(
            frame,
            train_size=max_examples,
            stratify=frame["label"],
            random_state=seed,
            shuffle=True,
        )
        return sampled.reset_index(drop=True)
    except ValueError as exc:
        logger.warning(
            "Stratified subset failed for split '{}' ({}); using random subset",
            split_name,
            exc,
        )
        return frame.sample(n=max_examples, random_state=seed, replace=False).reset_index(drop=True)


def read_rulebook(rules_dir: Path, pattern: str) -> tuple[str, list[str]]:
    files = sorted(f for f in rules_dir.glob(pattern) if f.is_file())
    if not files:
        raise FileNotFoundError(f"No rule files found in {rules_dir} matching {pattern}")

    chunks: list[str] = []
    names: list[str] = []
    for fp in files:
        content = fp.read_text(encoding="utf-8").strip()
        if not content:
            continue
        names.append(fp.name)
        chunks.append(f"[RULE FILE: {fp.name}]\n{content}")

    if not chunks:
        raise RuntimeError(f"All rule files are empty in {rules_dir}")

    return "\n\n".join(chunks), names

def strip_code_fence(text: str) -> str:
    s = text.strip()
    if not s.startswith("```"):
        return s
    s = re.sub(r"^```(?:json)?\\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\\s*```$", "", s)
    return s.strip()


def iter_json_object_candidates(text: str) -> Iterator[str]:
    depth = 0
    start = -1
    in_string = False
    escape = False
    for idx, ch in enumerate(text):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
            continue

        if ch == "{":
            if depth == 0:
                start = idx
            depth += 1
            continue

        if ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start >= 0:
                    yield text[start : idx + 1]
            continue


def normalize_label_value(value: Any) -> int | None:
    if isinstance(value, (bool, np.bool_)):
        return int(value)
    if isinstance(value, (int, np.integer)):
        v = int(value)
        return v if v in {0, 1} else None
    if isinstance(value, (float, np.floating)):
        v = float(value)
        if v in {0.0, 1.0}:
            return int(v)
        return None

    s = str(value).strip().lower()
    if s in {"0", "no", "false", "negative"}:
        return 0
    if s in {"1", "yes", "true", "positive"}:
        return 1
    return None


def parse_reasoning_label_from_obj(obj: Any) -> ParsedTeacherOutput | None:
    if isinstance(obj, list) and obj:
        obj = obj[0]
    if not isinstance(obj, dict):
        return None

    norm: dict[str, Any] = {str(k).strip().lower(): v for k, v in obj.items()}
    reasoning_keys = ["reasoning", "rationale", "analysis", "explanation", "thought"]
    label_keys = ["label", "pred_label", "prediction", "predicted_label", "class"]

    reasoning_val: Any = None
    for key in reasoning_keys:
        if key in norm:
            reasoning_val = norm[key]
            break

    label_val: Any = None
    for key in label_keys:
        if key in norm:
            label_val = norm[key]
            break

    if label_val is None and "output" in norm and isinstance(norm["output"], dict):
        nested = parse_reasoning_label_from_obj(norm["output"])
        if nested is not None:
            return ParsedTeacherOutput(reasoning=nested.reasoning, label=nested.label, source="json_nested")

    label = normalize_label_value(label_val)
    if label is None:
        return None

    reasoning = "" if reasoning_val is None else str(reasoning_val).strip()
    if not reasoning:
        return None

    return ParsedTeacherOutput(reasoning=reasoning, label=label, source="json")


def parse_teacher_output(raw: str) -> tuple[ParsedTeacherOutput | None, str | None]:
    raw = raw.strip()
    if not raw:
        return None, "empty_response"

    candidates: list[str] = []
    seen: set[str] = set()

    def push(candidate: str) -> None:
        c = candidate.strip()
        if c and c not in seen:
            seen.add(c)
            candidates.append(c)

    push(raw)
    stripped = strip_code_fence(raw)
    push(stripped)
    for c in iter_json_object_candidates(raw):
        push(c)
    for c in iter_json_object_candidates(stripped):
        push(c)

    for candidate in candidates:
        parsed_obj: Any | None = None
        try:
            parsed_obj = json.loads(candidate)
        except Exception:
            try:
                parsed_obj = ast.literal_eval(candidate)
            except Exception:
                parsed_obj = None

        if parsed_obj is None:
            continue

        parsed = parse_reasoning_label_from_obj(parsed_obj)
        if parsed is not None:
            return parsed, None

    match = re.search(
        r'(?is)"?\b(?:label|prediction|predicted[_\s]?label)\b"?\s*[:=]\s*(0|1|yes|no|true|false)\b',
        raw,
    )
    if match:
        label = normalize_label_value(match.group(1))
        if label is not None:
            reasoning = re.sub(re.escape(match.group(0)), "", raw, count=1).strip(" \n:-")
            if not reasoning:
                reasoning = raw
            return ParsedTeacherOutput(reasoning=reasoning, label=label, source="regex"), None

    return None, "parse_failed"


def build_reasoning_user_prompt(prompt_cfg: PromptConfig, rulebook: str, text: str) -> str:
    return prompt_cfg.reasoning_user_prompt_template.format(
        task_instruction=prompt_cfg.task_instruction,
        rulebook=rulebook,
        text=text,
    )


def build_label_only_user_prompt(prompt_cfg: PromptConfig, rulebook: str, text: str) -> str:
    return prompt_cfg.label_only_user_prompt_template.format(
        task_instruction=prompt_cfg.task_instruction,
        rulebook=rulebook,
        text=text,
    )


def build_teacher_messages(prompt_cfg: PromptConfig, rulebook: str, text: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": prompt_cfg.teacher_system_prompt,
        },
        {
            "role": "user",
            "content": build_reasoning_user_prompt(prompt_cfg, rulebook, text),
        },
    ]


def request_teacher_sample(
    client: OpenAI,
    messages: list[dict[str, str]],
    *,
    model: str,
    temperature: float,
    max_tokens: int,
    timeout_seconds: int,
    max_retries: int,
    json_mode: bool,
) -> tuple[str, ParsedTeacherOutput | None, str | None]:
    last_error: str | None = None
    use_json_mode = json_mode

    for attempt in range(max_retries):
        try:
            kwargs: dict[str, Any] = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "timeout": timeout_seconds,
            }
            if use_json_mode:
                kwargs["response_format"] = {"type": "json_object"}

            response = client.chat.completions.create(**kwargs)
            if not response.choices:
                last_error = "no_choices"
            else:
                content = response.choices[0].message.content or ""
                parsed, parse_error = parse_teacher_output(content)
                if parsed is not None:
                    return content, parsed, None
                last_error = parse_error or "parse_failed"

        except Exception as exc:
            message = str(exc)
            if use_json_mode and "response_format" in message.lower():
                use_json_mode = False
                last_error = "response_format_unsupported"
                continue
            last_error = message

        if attempt + 1 < max_retries:
            time.sleep(min(4.0, 1.0 + attempt))

    return "", None, last_error

def _coerce_input_ids(out: Any) -> list[int]:
    ids: Any = out

    if isinstance(out, dict) and "input_ids" in out:
        ids = out["input_ids"]
    elif hasattr(out, "input_ids"):
        ids = getattr(out, "input_ids")

    if hasattr(ids, "tolist"):
        ids = ids.tolist()

    if isinstance(ids, tuple):
        ids = list(ids)

    # Some tokenizers may return batched ids: [[...]] for a single conversation.
    if isinstance(ids, list) and ids and isinstance(ids[0], (list, tuple)):
        ids = list(ids[0])

    if not isinstance(ids, list):
        ids = list(ids)

    coerced: list[int] = []
    for tok in ids:
        if isinstance(tok, (int, np.integer)):
            coerced.append(int(tok))
            continue
        if isinstance(tok, str):
            s = tok.strip()
            if re.fullmatch(r"-?\d+", s):
                coerced.append(int(s))
                continue
        raise ValueError(f"Unsupported token type from chat template: {type(tok)} value={tok!r}")
    return coerced


def chat_input_ids(
    tokenizer: Any,
    messages: list[dict[str, str]],
    *,
    add_generation_prompt: bool = False,
) -> list[int]:
    out = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=add_generation_prompt,
    )
    return _coerce_input_ids(out)


def find_sublist_indices(full: list[int], sub: list[int]) -> list[int]:
    n, m = len(full), len(sub)
    if m == 0 or n < m:
        return []
    matches: list[int] = []
    for i in range(n - m + 1):
        if full[i : i + m] == sub:
            matches.append(i)
    return matches


def fit_messages_to_max_length(
    tokenizer: Any,
    user_builder: Callable[[str], str],
    text: str,
    assistant_content: str,
    max_length: int,
) -> tuple[list[dict[str, str]], list[int], list[int]] | None:
    text_tokens = tokenizer.encode(text, add_special_tokens=False)
    if not text_tokens:
        text_tokens = tokenizer.encode(" ", add_special_tokens=False)

    answer_tokens = tokenizer.encode(assistant_content, add_special_tokens=False)
    if not answer_tokens:
        return None

    best_messages: list[dict[str, str]] | None = None
    best_ids: list[int] | None = None

    lo, hi = 1, len(text_tokens)
    while lo <= hi:
        mid = (lo + hi) // 2
        candidate_text = tokenizer.decode(text_tokens[:mid])
        messages = [
            {"role": "user", "content": user_builder(candidate_text)},
            {"role": "assistant", "content": assistant_content},
        ]
        input_ids = chat_input_ids(tokenizer, messages, add_generation_prompt=False)

        if len(input_ids) <= max_length:
            best_messages = messages
            best_ids = input_ids
            lo = mid + 1
        else:
            hi = mid - 1

    if best_messages is None or best_ids is None:
        return None

    return best_messages, best_ids, answer_tokens


def fit_user_prompt_and_answer_to_max_length(
    tokenizer: Any,
    user_builder: Callable[[str], str],
    text: str,
    assistant_content: str,
    max_length: int,
) -> tuple[list[dict[str, str]], list[int], list[int]] | None:
    text_tokens = tokenizer.encode(text, add_special_tokens=False)
    if not text_tokens:
        text_tokens = tokenizer.encode(" ", add_special_tokens=False)

    answer_tokens = tokenizer.encode(assistant_content, add_special_tokens=False)
    if not answer_tokens:
        return None

    best_messages: list[dict[str, str]] | None = None
    best_prompt_tokens: list[int] | None = None

    lo, hi = 1, len(text_tokens)
    while lo <= hi:
        mid = (lo + hi) // 2
        candidate_text = tokenizer.decode(text_tokens[:mid])
        messages = [
            {"role": "user", "content": user_builder(candidate_text)},
            {"role": "assistant", "content": assistant_content},
        ]
        prompt_tokens = chat_input_ids(
            tokenizer,
            [{"role": "user", "content": user_builder(candidate_text)}],
            add_generation_prompt=True,
        )

        if len(prompt_tokens) + len(answer_tokens) <= max_length:
            best_messages = messages
            best_prompt_tokens = prompt_tokens
            lo = mid + 1
        else:
            hi = mid - 1

    if best_messages is None or best_prompt_tokens is None:
        return None
    return best_messages, best_prompt_tokens, answer_tokens


def build_train_examples(
    accepted_rows: list[dict[str, Any]],
    tokenizer: Any,
    max_length: int,
    reasoning_prompt_builder: Callable[[str], str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    dropped = 0
    drop_reasons: dict[str, int] = {}

    for sample in accepted_rows:
        text = str(sample["text"])
        label = int(sample["label"])
        assistant_content = json.dumps(
            {"reasoning": str(sample["reasoning"]), "label": label},
            ensure_ascii=False,
        )

        reasoning_fit = fit_user_prompt_and_answer_to_max_length(
            tokenizer,
            user_builder=reasoning_prompt_builder,
            text=text,
            assistant_content=assistant_content,
            max_length=max_length,
        )
        if reasoning_fit is None:
            dropped += 1
            drop_reasons["fit_none"] = drop_reasons.get("fit_none", 0) + 1
            continue

        messages, prompt_tokens, answer_tokens = reasoning_fit
        full_tokens = prompt_tokens + answer_tokens
        if len(full_tokens) < 2:
            dropped += 1
            drop_reasons["full_too_short"] = drop_reasons.get("full_too_short", 0) + 1
            continue

        label_key_match = re.search(r'"label"\s*:\s*', assistant_content)
        if label_key_match is None:
            dropped += 1
            drop_reasons["missing_label_key"] = drop_reasons.get("missing_label_key", 0) + 1
            continue
        label_probe_assistant_prefix = assistant_content[: label_key_match.end()]
        probe_tokens = tokenizer.encode(label_probe_assistant_prefix, add_special_tokens=False)
        if not probe_tokens:
            dropped += 1
            drop_reasons["empty_probe_tokens"] = drop_reasons.get("empty_probe_tokens", 0) + 1
            continue
        if len(prompt_tokens) + len(probe_tokens) > max_length:
            dropped += 1
            drop_reasons["label_probe_too_long"] = drop_reasons.get("label_probe_too_long", 0) + 1
            continue
        # Important: use the open assistant generation prefix from prompt_tokens.
        # Appending probe tokens keeps eval in "assistant is speaking" mode.
        eval_prompt_tokens = prompt_tokens + probe_tokens

        weights = [0.0] * len(prompt_tokens) + [1.0] * len(answer_tokens)
        datum = tinker.Datum(
            model_input=tinker.ModelInput.from_ints(full_tokens[:-1]),
            loss_fn_inputs={
                "target_tokens": full_tokens[1:],
                "weights": weights[1:],
            },
        )

        rows.append(
            {
                "sample_id": sample["sample_id"],
                "text": text,
                "label": label,
                "messages": messages,
                "datum": datum,
                "eval_prompt_tokens": eval_prompt_tokens,
            }
        )

    if dropped > 0:
        logger.warning("Dropped {} accepted rows due to max_length/template mismatch", dropped)
        logger.warning("Drop breakdown: {}", json.dumps(drop_reasons, ensure_ascii=False))
    if not rows:
        raise RuntimeError("No usable training rows after tokenization")
    return rows


def build_eval_rows(
    frame: pd.DataFrame,
    tokenizer: Any,
    max_length: int,
    reasoning_prompt_builder: Callable[[str], str],
    reasoning_placeholder: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    dropped = 0

    for _, row in frame.iterrows():
        text = str(row["text"])
        label = int(row["label"])
        label_probe_assistant_prefix = (
            '{"reasoning": ' + json.dumps(reasoning_placeholder, ensure_ascii=False) + ', "label": '
        )

        fit = fit_user_prompt_and_answer_to_max_length(
            tokenizer,
            user_builder=reasoning_prompt_builder,
            text=text,
            assistant_content=label_probe_assistant_prefix,
            max_length=max_length,
        )
        if fit is None:
            dropped += 1
            continue

        _, prompt_tokens, probe_tokens = fit
        rows.append({"text": text, "label": label, "prompt_tokens": prompt_tokens + probe_tokens})

    if dropped > 0:
        logger.warning("Dropped {} eval rows due to max_length/template mismatch", dropped)
    if not rows:
        raise RuntimeError("No usable eval rows after tokenization")
    return rows


def submit_completion_logprob(
    sampling_client: tinker.SamplingClient,
    prompt_tokens: list[int],
    answer_tokens: list[int],
) -> tuple[Any, int, int, int]:
    seq = prompt_tokens + answer_tokens
    fut = sampling_client.compute_logprobs(tinker.ModelInput.from_ints(seq))
    start = len(prompt_tokens)
    end = start + len(answer_tokens)
    return fut, start, end, len(answer_tokens)


def resolve_completion_logprob(fut: Any, start: int, end: int, answer_len: int) -> float:
    logprobs = fut.result()
    answer_logprobs = logprobs[start:end]
    if len(answer_logprobs) != answer_len:
        raise ValueError(
            f"compute_logprobs length mismatch: got {len(answer_logprobs)} answer logprobs, expected {answer_len}"
        )

    total = 0.0
    for lp in answer_logprobs:
        if lp is None:
            total += -1e9
        else:
            total += float(lp)
    return total


def normalize_binary_logprobs(lp_one: float, lp_zero: float) -> tuple[float, float]:
    denom = float(np.logaddexp(lp_one, lp_zero))
    p_one = float(np.exp(lp_one - denom))
    p_zero = float(np.exp(lp_zero - denom))
    return p_one, p_zero


def predict_with_threshold(
    p_one: list[float],
    invalid_flags: list[bool] | None,
    threshold: float,
) -> list[int]:
    preds: list[int] = []
    for i, p in enumerate(p_one):
        if invalid_flags is not None and i < len(invalid_flags) and invalid_flags[i]:
            preds.append(0)  # preserve fallback behavior for invalid emitted labels
        else:
            preds.append(1 if float(p) >= float(threshold) else 0)
    return preds


def find_best_threshold_macro_f1(
    y_true: list[int],
    p_one: list[float],
    invalid_flags: list[bool] | None = None,
) -> float:
    """
    Choose a decision threshold on P(1) that maximizes macro-F1 on the given labels.

    Notes:
    - Invalid rows (if provided) are forced to predict 0 regardless of threshold.
    - Candidate thresholds are derived from unique P(1) values plus endpoints.
    """
    if not y_true or not p_one or len(y_true) != len(p_one):
        return 0.5

    # Use only finite scores for threshold candidates.
    scores: list[float] = []
    for i, p in enumerate(p_one):
        if invalid_flags is not None and i < len(invalid_flags) and invalid_flags[i]:
            continue
        try:
            pf = float(p)
        except Exception:
            continue
        if math.isfinite(pf):
            scores.append(pf)

    if not scores:
        return 0.5

    uniq = sorted(set(scores))
    candidates: list[float] = [0.0] + uniq + [1.0]

    best_thr = 0.5
    best_f1 = float("-inf")
    best_acc = float("-inf")

    for thr in candidates:
        y_pred = predict_with_threshold(p_one, invalid_flags, thr)
        f1 = float(f1_score(y_true, y_pred, average="macro", labels=[0, 1], zero_division=0))
        if f1 > best_f1:
            best_f1 = f1
            best_thr = float(thr)
            best_acc = float(accuracy_score(y_true, y_pred))
            continue
        if f1 == best_f1:
            acc = float(accuracy_score(y_true, y_pred))
            # tie-break: higher accuracy, then threshold closer to 0.5
            if acc > best_acc:
                best_acc = acc
                best_thr = float(thr)
            elif acc == best_acc:
                if abs(float(thr) - 0.5) < abs(best_thr - 0.5):
                    best_thr = float(thr)

    return float(best_thr)


def thresholded_classification_metrics(
    y_true: list[int],
    p_one: list[float],
    invalid_flags: list[bool] | None,
    threshold: float,
) -> dict[str, Any]:
    y_pred = predict_with_threshold(p_one, invalid_flags, threshold)
    out = {
        "decision_threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)) if y_true else 0.0,
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=[0, 1], zero_division=0)) if y_true else 0.0,
        "y_pred": y_pred,
    }
    out.update(binary_metrics(y_true, y_pred, p_one))
    return out


def _unwrap_future(value: Any) -> Any:
    if hasattr(value, "result"):
        return value.result()
    return value


def _coerce_token_list(value: Any) -> list[int] | None:
    if value is None:
        return None
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, (list, tuple)):
        return None

    tokens: list[int] = []
    for tok in value:
        if isinstance(tok, (int, np.integer)):
            tokens.append(int(tok))
            continue
        if isinstance(tok, str):
            s = tok.strip()
            if re.fullmatch(r"-?\d+", s):
                tokens.append(int(s))
                continue
        return None
    return tokens if tokens else None


def _extract_tokens_from_sequence_obj(seq: Any) -> list[int] | None:
    if isinstance(seq, dict):
        for key in ("tokens", "token_ids", "ids", "output_ids", "generated_ids"):
            parsed = _coerce_token_list(seq.get(key))
            if parsed is not None:
                return parsed
        return None

    for key in ("tokens", "token_ids", "ids", "output_ids", "generated_ids"):
        parsed = _coerce_token_list(getattr(seq, key, None))
        if parsed is not None:
            return parsed
    return None


def _extract_tokens_from_sample_response(generated: Any) -> list[int] | None:
    if isinstance(generated, dict):
        container = generated.get("sequences")
        if not isinstance(container, (list, tuple)):
            container = generated.get("samples")
        if isinstance(container, (list, tuple)) and container:
            return _extract_tokens_from_sequence_obj(container[0])
        return _extract_tokens_from_sequence_obj(generated)

    container = getattr(generated, "sequences", None)
    if not isinstance(container, (list, tuple)):
        container = getattr(generated, "samples", None)
    if isinstance(container, (list, tuple)) and container:
        return _extract_tokens_from_sequence_obj(container[0])

    return _extract_tokens_from_sequence_obj(generated)


def _extract_generated_text_and_ids(generated: Any, tokenizer: Any) -> tuple[str, list[int] | None]:
    text: str | None = None
    token_ids: list[int] | None = None

    sample_tokens = _extract_tokens_from_sample_response(generated)
    if sample_tokens:
        token_ids = sample_tokens

    if isinstance(generated, str):
        text = generated
    else:
        parsed_direct = _coerce_token_list(generated)
        if parsed_direct is not None:
            token_ids = parsed_direct
    if token_ids is None and isinstance(generated, dict):
        for key in ("token_ids", "tokens", "ids", "output_ids", "generated_ids"):
            parsed = _coerce_token_list(generated.get(key))
            if parsed is not None:
                token_ids = parsed
                break
        for key in ("text", "output_text", "completion", "content", "generated_text"):
            value = generated.get(key)
            if isinstance(value, str):
                text = value
                break
    elif token_ids is None and not isinstance(generated, str):
        for key in ("token_ids", "tokens", "ids", "output_ids", "generated_ids"):
            parsed = _coerce_token_list(getattr(generated, key, None))
            if parsed is not None:
                token_ids = parsed
                break
        for key in ("text", "output_text", "completion", "content", "generated_text"):
            value = getattr(generated, key, None)
            if isinstance(value, str):
                text = value
                break

    if text is None and token_ids:
        try:
            text = tokenizer.decode(token_ids)
        except Exception:
            text = None

    return (text or ""), token_ids


def _submit_generate_label_output(
    sampling_client: Any,
    prompt_tokens: list[int],
    max_new_tokens: int,
) -> tuple[Any | None, str | None]:
    model_input = tinker.ModelInput.from_ints(prompt_tokens)
    method_names = ["sample", "generate", "complete", "completion"]
    last_error: str | None = None

    sample_method = getattr(sampling_client, "sample", None)
    if sample_method is not None:
        try:
            types_mod = getattr(tinker, "types", None)
            sampling_params_cls = None if types_mod is None else getattr(types_mod, "SamplingParams", None)
            if sampling_params_cls is not None:
                sampling_params = sampling_params_cls(
                    max_tokens=max_new_tokens,
                    temperature=0.0,
                )
                try:
                    return (
                        sample_method(
                            prompt=model_input,
                            num_samples=1,
                            sampling_params=sampling_params,
                        ),
                        None,
                    )
                except TypeError:
                    # Also support positional signature shown in docs examples.
                    return sample_method(model_input, 1, sampling_params), None
        except Exception as exc:
            last_error = f"sample: {exc}"

    call_specs = [
        lambda m: m(model_input, max_new_tokens=max_new_tokens, temperature=0.0),
        lambda m: m(model_input, max_tokens=max_new_tokens, temperature=0.0),
        lambda m: m(input=model_input, max_new_tokens=max_new_tokens, temperature=0.0),
        lambda m: m(input=model_input, max_tokens=max_new_tokens, temperature=0.0),
        lambda m: m(prompt=model_input, max_new_tokens=max_new_tokens, temperature=0.0),
        lambda m: m(prompt=model_input, max_tokens=max_new_tokens, temperature=0.0),
        lambda m: m(model_input, max_new_tokens=max_new_tokens),
        lambda m: m(model_input, max_tokens=max_new_tokens),
        lambda m: m(input=model_input, max_new_tokens=max_new_tokens),
        lambda m: m(input=model_input, max_tokens=max_new_tokens),
    ]

    # Fallback path for older/newer client variations.
    for method_name in method_names:
        method = getattr(sampling_client, method_name, None)
        if method is None:
            continue
        for call_fn in call_specs:
            try:
                return call_fn(method), None
            except TypeError as exc:
                last_error = f"{method_name}: {exc}"
            except Exception as exc:
                last_error = f"{method_name}: {exc}"
                break

    if last_error is None:
        last_error = "no generation method available on sampling_client"
    return None, last_error


def _parse_generated_label(
    generated_text: str,
    generated_ids: list[int] | None,
    one_tokens: list[int],
    zero_tokens: list[int],
) -> int | None:
    if generated_ids:
        if len(generated_ids) >= len(one_tokens) and generated_ids[: len(one_tokens)] == one_tokens:
            return 1
        if len(generated_ids) >= len(zero_tokens) and generated_ids[: len(zero_tokens)] == zero_tokens:
            return 0

    parsed, _ = parse_teacher_output(generated_text)
    if parsed is not None:
        return int(parsed.label)

    m = re.match(r"^\s*([01])(?:\D|$)", generated_text)
    if m:
        return int(m.group(1))
    return None


def submit_emitted_label_request(
    sampling_client: Any,
    prompt_tokens: list[int],
    one_tokens: list[int],
    zero_tokens: list[int],
) -> tuple[Any | None, str | None]:
    max_new_tokens = max(1, len(one_tokens), len(zero_tokens), 16)
    return _submit_generate_label_output(
        sampling_client=sampling_client,
        prompt_tokens=prompt_tokens,
        max_new_tokens=max_new_tokens,
    )


def resolve_emitted_label_request(
    generation_request: Any,
    tokenizer: Any,
    one_tokens: list[int],
    zero_tokens: list[int],
) -> tuple[int | None, str]:
    try:
        generated = _unwrap_future(generation_request)
    except Exception as exc:
        return None, f"generation_error: {exc}"

    generated_text, generated_ids = _extract_generated_text_and_ids(generated, tokenizer)
    label = _parse_generated_label(generated_text, generated_ids, one_tokens, zero_tokens)
    if label is None:
        return None, generated_text.strip()
    return label, generated_text.strip()


def evaluate_binary(
    sampling_client: Any,
    tokenizer: Any,
    rows: list[dict[str, Any]],
    max_samples: int,
    one_tokens: list[int],
    zero_tokens: list[int],
    max_concurrency: int,
    invalid_warn_rate: float,
    show_progress: bool = False,
    progress_desc: str = "Eval",
    discrete_pred_source: str = "prob",
) -> dict[str, Any]:
    if max_concurrency <= 0:
        raise ValueError("max_concurrency must be > 0")
    if discrete_pred_source not in {"prob", "emitted"}:
        raise ValueError("discrete_pred_source must be 'prob' or 'emitted'")

    rows_eval = rows if max_samples <= 0 else rows[:max_samples]

    y_true: list[int] = []
    y_pred: list[int] = []
    p_one_all: list[float] = []
    p_zero_all: list[float] = []
    cls_losses: list[float] = []
    total_nll = 0.0
    total_tokens = 0.0
    invalid_label_count = 0
    invalid_examples: list[str] = []
    invalid_flags: list[bool] = []

    in_flight: list[
        tuple[dict[str, Any], tuple[Any, int, int, int], tuple[Any, int, int, int], Any | None, str | None]
    ] = []
    pending_request_units = 0
    pbar = tqdm(total=len(rows_eval), desc=progress_desc, unit="ex", disable=not show_progress)

    def consume_record(
        record: tuple[dict[str, Any], tuple[Any, int, int, int], tuple[Any, int, int, int], Any | None, str | None]
    ) -> None:
        ex, one_req, zero_req, generation_request, generation_submit_error = record
        if generation_request is not None and generation_submit_error is None:
            emitted_label, emitted_text = resolve_emitted_label_request(
                generation_request=generation_request,
                tokenizer=tokenizer,
                one_tokens=one_tokens,
                zero_tokens=zero_tokens,
            )
        else:
            emitted_label = None
            emitted_text = f"generation_error: {generation_submit_error or 'submission_failed'}"

        lp_one = resolve_completion_logprob(*one_req)
        lp_zero = resolve_completion_logprob(*zero_req)
        p_one, p_zero = normalize_binary_logprobs(lp_one, lp_zero)

        is_invalid = emitted_label not in {0, 1}
        invalid_flags.append(bool(is_invalid))
        if is_invalid:
            nonlocal invalid_label_count
            invalid_label_count += 1
            p_one, p_zero = 0.5, 0.5
            if len(invalid_examples) < 5:
                invalid_examples.append(
                    f"sample_text={str(ex.get('text', ''))[:80]!r} emitted={emitted_text[:120]!r}"
                )
            pred = 0
        else:
            if discrete_pred_source == "emitted":
                pred = int(emitted_label)
            else:
                pred = 1 if p_one >= p_zero else 0

        true = int(ex["label"])
        y_true.append(true)
        y_pred.append(pred)
        p_one_all.append(p_one)
        p_zero_all.append(p_zero)

        p_true = p_one if true == 1 else p_zero
        nll = float(-np.log(max(p_true, 1e-12)))
        cls_losses.append(nll)

        nonlocal total_nll, total_tokens
        total_nll += nll
        total_tokens += 1.0
        pbar.update(1)

    try:
        for ex in rows_eval:
            generation_request, generation_submit_error = submit_emitted_label_request(
                sampling_client=sampling_client,
                prompt_tokens=ex["prompt_tokens"],
                one_tokens=one_tokens,
                zero_tokens=zero_tokens,
            )
            one_req = submit_completion_logprob(sampling_client, ex["prompt_tokens"], one_tokens)
            zero_req = submit_completion_logprob(sampling_client, ex["prompt_tokens"], zero_tokens)
            in_flight.append((ex, one_req, zero_req, generation_request, generation_submit_error))
            pending_request_units += 2 + (1 if generation_request is not None else 0)

            while pending_request_units >= max_concurrency and in_flight:
                record = in_flight.pop(0)
                pending_request_units -= 2 + (1 if record[3] is not None else 0)
                consume_record(record)

        while in_flight:
            record = in_flight.pop(0)
            pending_request_units -= 2 + (1 if record[3] is not None else 0)
            consume_record(record)
    finally:
        pbar.close()

    auroc = float("nan")
    auprc = float("nan")
    if y_true and len(set(y_true)) > 1:
        try:
            auroc = float(roc_auc_score(y_true, p_one_all))
        except ValueError:
            auroc = float("nan")
        try:
            auprc = float(average_precision_score(y_true, p_one_all))
        except ValueError:
            auprc = float("nan")

    invalid_label_rate = float(invalid_label_count / max(1, len(y_true)))

    if invalid_label_count > 0:
        logger.warning(
            "Detected {} invalid emitted labels during eval; defaulted to pred=0 and P(1)=0.5 for those rows. Examples: {}",
            invalid_label_count,
            " | ".join(invalid_examples),
        )
    if invalid_label_rate >= float(invalid_warn_rate):
        logger.warning(
            "High invalid emitted-label rate {:.2%} (threshold {:.2%}). Eval metrics may be unreliable.",
            invalid_label_rate,
            float(invalid_warn_rate),
        )

    return {
        "loss": float(total_nll / max(total_tokens, 1.0)),
        "cls_loss": float(np.mean(cls_losses)) if cls_losses else 0.0,
        "accuracy": float(accuracy_score(y_true, y_pred)) if y_true else 0.0,
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=[0, 1], zero_division=0)) if y_true else 0.0,
        "auroc": auroc,
        "auprc": auprc,
        "invalid_label_count": int(invalid_label_count),
        "invalid_label_rate": invalid_label_rate,
        "invalid_flags": invalid_flags,
        "y_true": y_true,
        "y_pred": y_pred,
        "p_one": p_one_all,
        "p_zero": p_zero_all,
    }


def binary_metrics(y_true: list[int], y_pred: list[int], p_one: list[float] | None = None) -> dict[str, Any]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "F1": float(f1_score(y_true, y_pred, average="binary", zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average="binary", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="binary", zero_division=0)),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }
    if p_one is not None and len(p_one) == len(y_true) and len(set(y_true)) > 1:
        try:
            out["auroc"] = float(roc_auc_score(y_true, p_one))
        except ValueError:
            out["auroc"] = float("nan")
        try:
            out["auprc"] = float(average_precision_score(y_true, p_one))
        except ValueError:
            out["auprc"] = float("nan")
    else:
        out["auroc"] = float("nan")
        out["auprc"] = float("nan")
    return out


SELECTION_DIRECTIONS: dict[str, str] = {
    "loss": "min",
    "cls_loss": "min",
    "accuracy": "max",
    "macro_f1": "max",
    "auroc": "max",
    "auprc": "max",
}


def _selection_score_for_compare(metric_name: str, metric_value: Any) -> float:
    direction = SELECTION_DIRECTIONS[metric_name]
    try:
        value = float(metric_value)
    except Exception:
        return float("inf") if direction == "min" else float("-inf")
    if math.isnan(value):
        return float("inf") if direction == "min" else float("-inf")
    return value


def _is_better_selection(metric_name: str, candidate: float, current_best: float) -> bool:
    direction = SELECTION_DIRECTIONS[metric_name]
    if direction == "min":
        return candidate < current_best
    return candidate > current_best

def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    cfg = load_config(project_root, args)
    prompt_cfg = load_prompt_config(project_root, cfg)

    random.seed(int(cfg["seed"]))
    np.random.seed(int(cfg["seed"]))

    run_name = cfg["run_name"] or f"reasoning_sft_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    log_dir = resolve_path(project_root, str(cfg["log_dir"]))
    if log_dir is None:
        raise RuntimeError("log_dir resolution failed")
    run_dir = log_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    setup_logger(run_dir)
    logger.info("Run dir: {}", run_dir)

    (run_dir / "resolved_config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    env_file = resolve_path(project_root, str(cfg["teacher_env_file"]))
    if env_file is None or not env_file.exists():
        raise FileNotFoundError(f"Missing env file: {env_file}")
    load_dotenv(env_file, override=False)

    teacher_api_key = os.environ.get(str(cfg["teacher_api_key_env"]))
    if not teacher_api_key:
        raise RuntimeError(f"Environment variable {cfg['teacher_api_key_env']} is not set")

    data_path = resolve_path(project_root, str(cfg["data_path"]))
    if data_path is None or not data_path.exists():
        raise FileNotFoundError(f"Missing data path: {data_path}")

    df = load_dataset(
        data_path=data_path,
        text_column=str(cfg["text_column"]),
        label_column=str(cfg["label_column"]),
        csv_sep=None if cfg.get("csv_sep") is None else str(cfg.get("csv_sep")),
        label_threshold=float(cfg["label_threshold"]),
    )

    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        stratify=df["label"],
        random_state=int(cfg["seed"]),
        shuffle=True,
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        stratify=temp_df["label"],
        random_state=int(cfg["seed"]),
        shuffle=True,
    )

    source_summary = {
        "train": {
            "count": int(len(train_df)),
            "label_counts": label_counts(train_df),
            "label_distribution": label_distribution(train_df),
        },
        "val": {
            "count": int(len(val_df)),
            "label_counts": label_counts(val_df),
            "label_distribution": label_distribution(val_df),
        },
        "test": {
            "count": int(len(test_df)),
            "label_counts": label_counts(test_df),
            "label_distribution": label_distribution(test_df),
        },
    }

    train_df = stratified_subset(
        train_df,
        int(cfg["max_train_examples"]),
        seed=int(cfg["seed"]),
        split_name="train",
    )
    val_df = stratified_subset(
        val_df,
        int(cfg["max_val_examples"]),
        seed=int(cfg["seed"]) + 1,
        split_name="val",
    )
    test_df = stratified_subset(
        test_df,
        int(cfg["max_test_examples"]),
        seed=int(cfg["seed"]) + 2,
        split_name="test",
    )

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

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
                "label_counts": label_counts(train_df),
                "label_distribution": label_distribution(train_df),
            },
            "val": {
                "count": int(len(val_df)),
                "label_counts": label_counts(val_df),
                "label_distribution": label_distribution(val_df),
            },
            "test": {
                "count": int(len(test_df)),
                "label_counts": label_counts(test_df),
                "label_distribution": label_distribution(test_df),
            },
        },
        "seed": int(cfg["seed"]),
    }
    (run_dir / "split_summary.json").write_text(json.dumps(split_summary, indent=2), encoding="utf-8")
    logger.info("Split sizes: train={}, val={}, test={}", len(train_df), len(val_df), len(test_df))

    rules_dir = resolve_path(project_root, str(cfg["rules_dir"]))
    if rules_dir is None or not rules_dir.exists():
        raise FileNotFoundError(f"Missing rules directory: {rules_dir}")
    rulebook, rule_files = read_rulebook(rules_dir, str(cfg["rules_glob"]))
    (run_dir / "rulebook.txt").write_text(rulebook, encoding="utf-8")
    (run_dir / "rule_files.json").write_text(json.dumps(rule_files, indent=2), encoding="utf-8")
    logger.info("Loaded {} rule files from {}", len(rule_files), rules_dir)

    teacher_client = OpenAI(api_key=teacher_api_key, base_url=str(cfg["teacher_base_url"]))

    teacher_samples: list[dict[str, Any]] = []
    k = int(cfg["teacher_k"])
    pbar = tqdm(total=len(train_df) * k, desc="Teacher sampling", unit="sample")

    for idx, row in train_df.iterrows():
        text = str(row["text"])
        gold_label = int(row["label"])
        messages = build_teacher_messages(prompt_cfg, rulebook, text)

        for ki in range(k):
            raw_output, parsed, error = request_teacher_sample(
                teacher_client,
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
            teacher_samples.append(sample)
            pbar.update(1)

            sleep_seconds = float(cfg["teacher_sleep_seconds"])
            if sleep_seconds > 0:
                time.sleep(sleep_seconds)

    pbar.close()

    write_jsonl(run_dir / "teacher_samples.jsonl", teacher_samples)

    min_reasoning_chars = int(cfg["min_reasoning_chars"])
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []

    for sample in teacher_samples:
        if not sample["parse_ok"]:
            sample["reject_reason"] = sample.get("error") or "parse_failed"
            rejected.append(sample)
            continue

        pred_label = int(sample["pred_label"])
        gold_label = int(sample["gold_label"])
        reasoning = str(sample["reasoning"] or "").strip()

        if pred_label != gold_label:
            sample["reject_reason"] = "label_mismatch"
            rejected.append(sample)
            continue

        if len(reasoning) < min_reasoning_chars:
            sample["reject_reason"] = "reasoning_too_short"
            rejected.append(sample)
            continue

        accepted.append(
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

    write_jsonl(run_dir / "accepted_samples.jsonl", accepted)
    write_jsonl(run_dir / "rejected_samples.jsonl", rejected)

    teacher_summary = {
        "teacher_model": str(cfg["teacher_model"]),
        "requested_samples": int(len(teacher_samples)),
        "accepted_samples": int(len(accepted)),
        "rejected_samples": int(len(rejected)),
        "acceptance_rate": float(len(accepted) / max(1, len(teacher_samples))),
    }
    (run_dir / "teacher_summary.json").write_text(json.dumps(teacher_summary, indent=2), encoding="utf-8")
    logger.info(
        "Teacher samples: requested={}, accepted={}, rejected={}, acceptance_rate={:.4f}",
        teacher_summary["requested_samples"],
        teacher_summary["accepted_samples"],
        teacher_summary["rejected_samples"],
        teacher_summary["acceptance_rate"],
    )

    if not accepted:
        raise RuntimeError("No accepted teacher samples after rejection sampling")

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

    reasoning_prompt_builder = lambda text: build_reasoning_user_prompt(prompt_cfg, rulebook, text)

    train_rows = build_train_examples(
        accepted_rows=accepted,
        tokenizer=tokenizer,
        max_length=int(cfg["max_length"]),
        reasoning_prompt_builder=reasoning_prompt_builder,
    )
    val_rows = build_eval_rows(
        frame=val_df,
        tokenizer=tokenizer,
        max_length=int(cfg["max_length"]),
        reasoning_prompt_builder=reasoning_prompt_builder,
        reasoning_placeholder=str(cfg["eval_reasoning_placeholder"]),
    )
    test_rows = build_eval_rows(
        frame=test_df,
        tokenizer=tokenizer,
        max_length=int(cfg["max_length"]),
        reasoning_prompt_builder=reasoning_prompt_builder,
        reasoning_placeholder=str(cfg["eval_reasoning_placeholder"]),
    )

    logger.info("Usable tokenized rows: train={}, val={}, test={}", len(train_rows), len(val_rows), len(test_rows))

    if bool(cfg["save_sft_jsonl"]):
        sft_rows = [{"messages": ex["messages"], "label": ex["label"]} for ex in train_rows]
        write_jsonl(run_dir / "train_sft.jsonl", sft_rows)

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
    selection_direction = SELECTION_DIRECTIONS[selection_metric]
    best_val_selection_score = float("inf") if selection_direction == "min" else float("-inf")
    best_val_metric_value = float("nan")
    best_val_macro_f1 = -1.0
    best_client = None
    best_model_path = ""
    best_step = -1

    step = 0
    for epoch in range(int(cfg["num_epochs"])):
        order = list(range(len(train_rows)))
        rng.shuffle(order)

        for i in range(0, len(order), int(cfg["batch_size"])):
            step_client = None
            do_eval = (step % int(cfg["eval_interval"]) == 0) or (step == total_steps - 1)

            if do_eval:
                step_client = training_client.save_weights_and_get_sampling_client(name=f"{run_name}_step_{step}_eval")
                val_eval = evaluate_binary(
                    step_client,
                    tokenizer,
                    val_rows,
                    max_samples=int(cfg["max_eval_samples"]),
                    one_tokens=one_tokens,
                    zero_tokens=zero_tokens,
                    max_concurrency=int(cfg["eval_max_concurrency"]),
                    invalid_warn_rate=float(cfg["invalid_label_warn_rate"]),
                )
                test_eval = evaluate_binary(
                    step_client,
                    tokenizer,
                    test_rows,
                    max_samples=int(cfg["max_eval_samples"]),
                    one_tokens=one_tokens,
                    zero_tokens=zero_tokens,
                    max_concurrency=int(cfg["eval_max_concurrency"]),
                    invalid_warn_rate=float(cfg["invalid_label_warn_rate"]),
                )

                wandb.log(
                    {
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
                    },
                    step=step,
                )

                best_val_macro_f1 = max(best_val_macro_f1, float(val_eval["macro_f1"]))
                selection_metric_value = float(val_eval.get(selection_metric, float("nan")))
                selection_candidate = _selection_score_for_compare(selection_metric, selection_metric_value)

                logger.info(
                    "Step {} | val: loss={:.6f}, cls_loss={:.6f}, acc={:.6f}, f1={:.6f}, auroc={:.6f}, auprc={:.6f}, invalid_rate={:.4f}, sel({})={:.6f} | test: loss={:.6f}, cls_loss={:.6f}, acc={:.6f}, f1={:.6f}, auroc={:.6f}, auprc={:.6f}, invalid_rate={:.4f}",
                    step,
                    float(val_eval["loss"]),
                    float(val_eval["cls_loss"]),
                    float(val_eval["accuracy"]),
                    float(val_eval["macro_f1"]),
                    float(val_eval["auroc"]),
                    float(val_eval["auprc"]),
                    float(val_eval["invalid_label_rate"]),
                    selection_metric,
                    selection_metric_value,
                    float(test_eval["loss"]),
                    float(test_eval["cls_loss"]),
                    float(test_eval["accuracy"]),
                    float(test_eval["macro_f1"]),
                    float(test_eval["auroc"]),
                    float(test_eval["auprc"]),
                    float(test_eval["invalid_label_rate"]),
                )

                if _is_better_selection(selection_metric, selection_candidate, best_val_selection_score):
                    best_val_selection_score = selection_candidate
                    best_val_metric_value = selection_metric_value
                    best_client = step_client
                    best_model_path = f"model at step {step}"
                    best_step = step

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
                step_client = training_client.save_weights_and_get_sampling_client(name=f"{run_name}_step_{step}_train")

            batch_eval_rows = [{"text": ex["text"], "label": ex["label"], "prompt_tokens": ex["eval_prompt_tokens"]} for ex in batch]
            train_eval = evaluate_binary(
                step_client,
                tokenizer,
                batch_eval_rows,
                max_samples=0,
                one_tokens=one_tokens,
                zero_tokens=zero_tokens,
                max_concurrency=int(cfg["eval_max_concurrency"]),
                invalid_warn_rate=float(cfg["invalid_label_warn_rate"]),
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

    final_client = training_client.save_weights_and_get_sampling_client(name=f"{run_name}_final")
    final_model_path = "final model"

    final_eval = evaluate_binary(
        final_client,
        tokenizer,
        test_rows,
        max_samples=0,
        one_tokens=one_tokens,
        zero_tokens=zero_tokens,
        max_concurrency=int(cfg["eval_max_concurrency"]),
        invalid_warn_rate=float(cfg["invalid_label_warn_rate"]),
    )
    final_metrics = binary_metrics(final_eval["y_true"], final_eval["y_pred"], final_eval["p_one"])

    if best_client is None:
        best_client = final_client
        best_model_path = final_model_path
        best_step = step - 1

    best_eval = evaluate_binary(
        best_client,
        tokenizer,
        test_rows,
        max_samples=0,
        one_tokens=one_tokens,
        zero_tokens=zero_tokens,
        max_concurrency=int(cfg["eval_max_concurrency"]),
        invalid_warn_rate=float(cfg["invalid_label_warn_rate"]),
    )
    best_metrics = binary_metrics(best_eval["y_true"], best_eval["y_pred"], best_eval["p_one"])

    wandb.log(
        {
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
        },
        step=step,
    )

    report_path = run_dir / "test_report.md"
    blocks = [
        ("final model", final_model_path, final_metrics, final_eval),
        ("best-val model", best_model_path, best_metrics, best_eval),
    ]
    lines = ["# Reasoning SFT Test Report", "", f"Generated at: {datetime.now().isoformat(timespec='seconds')}", ""]
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
        "report_path": str(report_path),
        "final_model_path": final_model_path,
        "best_model_path": best_model_path,
        "best_step": best_step,
        "best_val_macro_f1": best_val_macro_f1,
        "selection_metric": selection_metric,
        "best_val_selection_score": best_val_selection_score,
        "best_val_selection_metric_value": best_val_metric_value,
        "teacher_summary": teacher_summary,
    }
    (run_dir / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    wandb_run.finish()
    logger.info("Run summary: {}", json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
