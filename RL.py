#!/usr/bin/env python3
"""GRPO-style RL training for hate speech binary classification using the Tinker API.

This script continues training from an SFT checkpoint (loaded from model/latest.json)
using Group Relative Policy Optimization (GRPO) with classification correctness as reward.

Pipeline overview:
  1. Load SFT checkpoint from model/latest.json (or --init-model-path / --init-state-path).
  2. Build a simple RL prompt (no rulebook) so the model must reason from knowledge.
  3. For each training step:
       a. Save current weights → get a SamplingClient.
       b. For each prompt in the batch, sample `group_size` rollouts.
       c. Compute correctness reward (+1 / -1) per rollout.
          Optional: add thinking bonus for longer reasoning.
       d. GRPO: compute per-group advantages = (r - group_mean) / (group_std + eps).
       e. Optional KL penalty: adjust advantages to stay close to reference model.
       f. Build Tinker Datums and call forward_backward + optim_step.
  4. Periodic eval (every --eval-interval steps):
       - Val set: logprob P("1")/P("0") → val-calibrated threshold → macro-F1.
       - Hard-sample accuracy (90 train examples teacher rejected).
       - Easy-sample accuracy (90 random non-hard train examples; for comparison).
       - Log all metrics to WandB.
  5. Early stopping: stop after `patience` evals without val macro-F1 improvement.
     Save best model checkpoint separately.
  6. Final test evaluation (on best or final model):
       - Greedy generation → reasoning + emitted label.
       - Logprob probing → P(1)/P(0) for AUROC/AUPRC.
       - Save (text, reasoning, predicted_label, true_label) to test_results.json.
  7. Save RL model pointer to model/rl_latest.json.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import tinker
import wandb
from loguru import logger
from sklearn.metrics import f1_score as sklearn_f1_score

import SFT_reasoning as core

# ── Optional dotenv loading ───────────────────────────────────────────────────
try:
    from dotenv import load_dotenv
except ImportError:

    def load_dotenv(dotenv_path: str | os.PathLike[str] | None = None, override: bool = False) -> bool:  # type: ignore[misc]
        """Minimal fallback if python-dotenv is not installed."""
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
            key, value = key.strip(), value.strip().strip("'\"")
            if not key:
                continue
            if override or key not in os.environ:
                os.environ[key] = value
                loaded_any = True
        return loaded_any


# ── Constants ─────────────────────────────────────────────────────────────────
TEACHER_SUBDIR = "teacher_phase"
STUDENT_SUBDIR = "student_phase"
RL_SUBDIR = "rl_phase"

# Default API keys (overridden by environment variables or CLI args).
_DEFAULT_TINKER_API_KEY = "tml-Rf8AxE8WUvLuefoXbxOjBRRextBanIz0ZWbTv6Fz0jfBv3GA57uKItoWkLYBIDjNAAAAA"
_DEFAULT_WANDB_API_KEY = "tml-SDViQX9HaHNejzj2RquoYGneIhb6LgIXBOAgnMhwmr8MBojMqnJUmZela7q22nN2CAAAA"

# Thinking reward: reasoning longer than this many characters gets the full bonus.
# Below this, the bonus scales linearly with length. Set to encourage non-trivial reasoning.
_THINKING_REWARD_THRESHOLD_CHARS = 500

# Number of easy (non-hard) train samples to monitor separately during training.
_EASY_MONITOR_COUNT = 90


# ── Label parsing ─────────────────────────────────────────────────────────────

def _extract_emitted_label(text: str) -> tuple[int | None, str]:
    """Parse a 0/1 label from a generated completion. Returns (label, debug_source).

    Tries JSON parsing first (to match the SFT JSON output format), then falls back
    to several regex patterns for robustness. Returns (None, "unparsed") if all fail.
    """
    raw = (text or "").strip()
    if not raw:
        return None, "empty"

    # --- JSON parsing (primary path, matches SFT output format) ---
    try:
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1 and end > start:
            obj = json.loads(raw[start : end + 1])
            if isinstance(obj, dict) and "label" in obj:
                v = obj["label"]
                if isinstance(v, (bool, int)):
                    iv = int(v)
                    if iv in {0, 1}:
                        return iv, "json.label"
                if isinstance(v, str):
                    m = re.fullmatch(r"\s*([01])\s*", v)
                    if m:
                        return int(m.group(1)), "json.label_str"
    except Exception:
        pass

    # --- Regex fallbacks for malformed JSON or unexpected text outputs ---
    m = re.search(r'(?is)"label"\s*:\s*([01])\b', raw)
    if m:
        return int(m.group(1)), "regex.json_label"
    m = re.search(r"(?im)^\s*label\s*[:=]\s*([01])\b", raw)
    if m:
        return int(m.group(1)), "regex.label_line"
    m = re.match(r"^\s*([01])(?:\D|$)", raw)
    if m:
        return int(m.group(1)), "regex.leading_digit"

    return None, "unparsed"


def _extract_reasoning_from_json(text: str) -> str:
    """Extract the 'reasoning' field from a JSON-formatted model output.

    Falls back to the full text if JSON parsing fails. Used for test result collection.
    """
    raw = (text or "").strip()
    try:
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1:
            obj = json.loads(raw[start : end + 1])
            if isinstance(obj, dict) and "reasoning" in obj:
                return str(obj["reasoning"])
    except Exception:
        pass
    return raw  # fallback: return full generated text


def _extract_reasoning_char_count(text: str) -> int:
    """Count characters in the 'reasoning' field of a JSON output.

    Used for the optional thinking reward: longer reasoning gets a higher bonus.
    Falls back to total text length if JSON parsing fails.
    """
    raw = (text or "").strip()
    try:
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1:
            obj = json.loads(raw[start : end + 1])
            if isinstance(obj, dict) and "reasoning" in obj:
                return len(str(obj["reasoning"]))
    except Exception:
        pass
    return len(raw)


# ── Logprob fallback ──────────────────────────────────────────────────────────

def _compute_generated_token_logprobs(
    sampling_client: tinker.SamplingClient,
    *,
    prompt_tokens: list[int],
    generated_tokens: list[int],
) -> list[float] | None:
    """Fallback when SamplingClient.sample() doesn't return per-token logprobs.

    Calls compute_logprobs on the full sequence (prompt + generation) and slices
    out the logprobs for the generated portion. Returns None on any failure.
    """
    if not generated_tokens:
        return None
    seq = list(prompt_tokens) + list(generated_tokens)
    try:
        fut = sampling_client.compute_logprobs(tinker.ModelInput.from_ints(seq))
        lp_full = fut.result()
    except Exception:
        return None
    prompt_len = len(prompt_tokens)
    lp_slice = lp_full[prompt_len : prompt_len + len(generated_tokens)]
    if len(lp_slice) != len(generated_tokens):
        return None
    out: list[float] = []
    for lp in lp_slice:
        if lp is None:
            return None
        try:
            out.append(float(lp))
        except Exception:
            return None
    return out


# ── Logprob-only evaluation ───────────────────────────────────────────────────

def _eval_logprob_only(
    sampling_client: tinker.SamplingClient,
    tokenizer: Any,  # unused at runtime (rows are pre-tokenized); kept for API compatibility
    eval_rows: list[dict[str, Any]],
    *,
    max_samples: int,
    one_tokens: list[int],
    zero_tokens: list[int],
    max_concurrency: int,
) -> dict[str, Any]:
    """Evaluate via logprob probing: compute P("1"|prompt) and P("0"|prompt).

    Faster than generation-based eval because it only needs a single forward pass per
    example (no sampling). The eval_rows must contain pre-tokenized 'prompt_tokens'
    that include a label-probe prefix (e.g., up to '{"reasoning": "...", "label": ')
    so we're measuring p(label token | context).

    Uses a sliding window of concurrent requests for throughput control.

    Returns a dict with y_true, p_one, p_zero, accuracy, cls_loss, auroc, auprc, etc.
    """
    rows_eval = eval_rows if max_samples <= 0 else eval_rows[:max_samples]
    if not rows_eval:
        return {
            "loss": 0.0, "cls_loss": 0.0, "invalid_label_count": 0,
            "invalid_label_rate": 0.0, "invalid_flags": [],
            "y_true": [], "y_pred": [], "p_one": [], "p_zero": [],
        }

    y_true: list[int] = []
    p_one_all: list[float] = []
    p_zero_all: list[float] = []
    cls_losses: list[float] = []

    # Each entry: (example_dict, one_logprob_request, zero_logprob_request)
    in_flight: list[tuple[dict[str, Any], tuple[Any, int, int, int], tuple[Any, int, int, int]]] = []
    pending = 0

    def _consume(record: tuple) -> None:
        """Resolve one in-flight request and accumulate y_true / P(1) / P(0) / loss."""
        ex, one_req, zero_req = record
        lp1 = core.resolve_completion_logprob(*one_req)
        lp0 = core.resolve_completion_logprob(*zero_req)
        # Softmax normalization: P(1) = exp(lp1) / (exp(lp1) + exp(lp0))
        denom = float(np.logaddexp(lp1, lp0))
        p1 = float(math.exp(lp1 - denom))
        p0 = float(math.exp(lp0 - denom))
        y = int(ex["label"])
        y_true.append(y)
        p_one_all.append(p1)
        p_zero_all.append(p0)
        # Binary cross-entropy: used as a training-proxy loss metric for monitoring
        cls_losses.append(-(y * math.log(max(p1, 1e-12)) + (1 - y) * math.log(max(p0, 1e-12))))

    for ex in rows_eval:
        prompt_tokens = list(ex["prompt_tokens"])
        # Submit async logprob requests for label tokens "1" and "0"
        one_req = core.submit_completion_logprob(sampling_client, prompt_tokens, one_tokens)
        zero_req = core.submit_completion_logprob(sampling_client, prompt_tokens, zero_tokens)
        in_flight.append((ex, one_req, zero_req))
        pending += 1
        # Enforce concurrency limit by draining the oldest request when we hit the cap
        if pending >= max_concurrency:
            _consume(in_flight.pop(0))
            pending -= 1

    # Drain all remaining in-flight requests
    while in_flight:
        _consume(in_flight.pop(0))

    # Compute hard predictions using a default 0.5 threshold.
    # Callers should override with a val-calibrated threshold for real metrics.
    y_pred = [1 if p >= 0.5 else 0 for p in p_one_all]
    base = core.binary_metrics(y_true, y_pred, p_one=p_one_all) if y_true else {}
    out = {
        "loss": float(np.mean(cls_losses)) if cls_losses else 0.0,
        "cls_loss": float(np.mean(cls_losses)) if cls_losses else 0.0,
        "invalid_label_count": 0, "invalid_label_rate": 0.0,
        "invalid_flags": [False for _ in y_true],
        "y_true": y_true, "y_pred": y_pred, "p_one": p_one_all, "p_zero": p_zero_all,
    }
    for k in ["accuracy", "balanced_accuracy", "F1", "mcc", "precision", "recall",
              "tp", "fp", "fn", "tn", "auroc", "auprc"]:
        out[k] = base.get(k, out.get(k))
    return out


# ── Disk I/O ──────────────────────────────────────────────────────────────────

def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    """Append a JSON row to a .jsonl file (creating parent directories as needed)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


# ── Dataset utilities ─────────────────────────────────────────────────────────

def _read_latest_pointer(project_root: Path) -> dict[str, Any] | None:
    """Read model/latest.json, which records paths from the most recent SFT run."""
    fp = project_root / "model" / "latest.json"
    if not fp.exists():
        return None
    try:
        return json.loads(fp.read_text(encoding="utf-8"))
    except Exception:
        return None


def _resolve_split_paths(project_root: Path, cfg: dict[str, Any]) -> tuple[dict[str, Path], Path, str]:
    """Resolve absolute paths to train/val/test CSV files for the configured dataset."""
    dataset_root_dir = core.resolve_path(project_root, str(cfg["dataset_root_dir"]))
    if dataset_root_dir is None:
        raise RuntimeError("dataset_root_dir resolution failed")
    dataset_name = str(cfg["dataset_name"])
    dataset_dir = dataset_root_dir / dataset_name
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Missing dataset dir: {dataset_dir}")

    def _find_split(split: str) -> Path:
        candidates = [
            dataset_dir / f"{dataset_name}_{split}.csv",
            dataset_dir / f"{split}.csv",
            dataset_dir / f"{dataset_name}-{split}.csv",
        ]
        for fp in candidates:
            if fp.exists():
                return fp
        raise FileNotFoundError(f"No split file for '{split}' under {dataset_dir}. Tried: {candidates}")

    return (
        {"train": _find_split("train"), "val": _find_split("val"), "test": _find_split("test")},
        dataset_dir,
        dataset_name,
    )


def _load_split_dataset(path: Path, cfg: dict[str, Any], split_name: str) -> pd.DataFrame:
    """Load a dataset split from CSV, trying multiple known column-name schemas.

    Ethos uses columns ('text', 'label') with comma separator.
    Falls back to config-specified columns on failure.
    """
    preferred = [
        ("text", "label", None),
        ("comment", "isHate", ";"),
        (str(cfg.get("text_column", "text")), str(cfg.get("label_column", "label")), cfg.get("csv_sep", None)),
    ]
    tried: list[str] = []
    last_exc: Exception | None = None
    for text_col, label_col, sep in preferred:
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
            logger.info("Loaded {} split from {} using columns ({}, {})", split_name, path, text_col, label_col)
            return df
        except Exception as exc:
            last_exc = exc
    raise RuntimeError(f"Failed loading '{split_name}' from {path}. Tried: {tried}. Last error: {last_exc}")


def _parse_stop_arg(raw: str | None) -> list[str] | None:
    """Parse a comma-separated stop-string argument into a list. Returns None if empty."""
    if raw is None:
        return None
    s = str(raw).strip()
    if not s:
        return None
    parts = [p.strip() for p in s.split(",")]
    parts = [p for p in parts if p]
    return parts or None


# ── Prompt construction ───────────────────────────────────────────────────────

def _build_rl_user_prompt(task_instruction: str, text: str) -> str:
    """Build the RL training prompt: task instruction + text, NO rulebook.

    Crucially, we omit the rulebook that was present during SFT. The model must now
    learn to classify using its own internalized knowledge and chain-of-thought reasoning,
    rather than being guided by explicit rules. The JSON output format is kept identical
    to SFT to avoid unnecessary distributional shift.
    """
    return (
        f"{task_instruction}\n\n"
        f"Text: {text}\n\n"
        'Return a JSON object with keys:\n'
        '- "reasoning": your step-by-step reasoning\n'
        '- "label": integer 0 or 1'
    )


# ── Tokenization helpers ──────────────────────────────────────────────────────

def _tokenize_prompt_rows(
    df: pd.DataFrame,
    tokenizer: Any,
    prompt_builder: Callable[[str], str],
    max_length: int,
    extra_cols: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Pre-tokenize a DataFrame into generation-ready prompt rows.

    Uses binary search (via core.fit_user_prompt_and_answer_to_max_length) to truncate
    long texts to fit within max_length while preserving as much text as possible.
    The returned prompt_tokens end with the 'add_generation_prompt' suffix so the model
    can immediately start generating its response.

    Args:
        df: DataFrame with 'text' and 'label' columns.
        tokenizer: HuggingFace tokenizer for the student model.
        prompt_builder: Callable (text: str) -> user_prompt_str.
        max_length: Maximum total sequence length (prompt + dummy answer).
        extra_cols: Additional column names from df to carry through (e.g., 'hard').

    Returns:
        List of dicts with keys: 'text', 'label', 'prompt_tokens', + any extra_cols.
    """
    rows: list[dict[str, Any]] = []
    # Dummy assistant answer used only for length computation, not for content.
    dummy_answer = '{"reasoning": "x", "label": 0}'

    for _, row in df.iterrows():
        text = str(row["text"])
        label = int(row["label"])

        fit = core.fit_user_prompt_and_answer_to_max_length(
            tokenizer,
            user_builder=prompt_builder,
            text=text,
            assistant_content=dummy_answer,
            max_length=max_length,
        )
        if fit is None:
            # Text is too long even after maximum truncation; skip this example.
            continue

        _, prompt_tokens, _ = fit  # only need prompt_tokens (with add_generation_prompt=True)
        entry: dict[str, Any] = {"text": text, "label": label, "prompt_tokens": list(prompt_tokens)}
        for col in (extra_cols or []):
            if col in df.columns and col in row.index:
                entry[col] = row[col]
        rows.append(entry)

    return rows


# ── Eval helpers ──────────────────────────────────────────────────────────────

def _apply_threshold_accuracy(p_one: list[float], y_true: list[int], threshold: float) -> float:
    """Compute classification accuracy after applying a decision threshold to P(label=1)."""
    if not y_true:
        return 0.0
    preds = [1 if p >= threshold else 0 for p in p_one]
    return float(np.mean([int(p == t) for p, t in zip(preds, y_true)]))


def _run_periodic_eval(
    *,
    training_client: tinker.TrainingClient,
    step: int,
    val_eval_rows: list[dict[str, Any]],
    hard_eval_rows: list[dict[str, Any]],
    easy_eval_rows: list[dict[str, Any]],
    one_tokens: list[int],
    zero_tokens: list[int],
    max_concurrency: int,
    rl_dir: Path,
    wandb_run: Any,
) -> dict[str, Any]:
    """Run val + hard-sample + easy-sample evaluation at a given training step.

    Saves the current model weights (creating a named Tinker checkpoint), runs
    logprob-based P(1)/P(0) evaluation, calibrates the decision threshold on val
    to maximize macro-F1, then uses that threshold for hard/easy accuracy.

    All metrics are logged to WandB and val metrics are also saved to disk.

    Returns a dict with:
        'val_macro_f1'       float  -- used for early stopping comparison
        'decision_threshold' float  -- val-calibrated, applied to hard/easy subsets
        'hard_accuracy'      float | None
        'easy_accuracy'      float | None
        'eval_sampling_client' tinker.SamplingClient  -- reuse for best-model save
    """
    # Save current weights and create a named sampling client for this eval step.
    # We name it so checkpoints are traceable in the Tinker dashboard.
    eval_sampling_client = training_client.save_weights_and_get_sampling_client(
        name=f"rl_step_{step}_eval"
    )

    # ── Validation evaluation ─────────────────────────────────────────────────
    val_eval = _eval_logprob_only(
        eval_sampling_client, tokenizer=None, eval_rows=val_eval_rows,
        max_samples=0, one_tokens=one_tokens, zero_tokens=zero_tokens,
        max_concurrency=max_concurrency,
    )
    # Calibrate the decision threshold on val to maximize macro-F1.
    # This threshold is then applied consistently to hard/easy subsets.
    decision_threshold = core.find_best_threshold_macro_f1(
        val_eval["y_true"], val_eval["p_one"], val_eval.get("invalid_flags"),
    )
    val_thr = core.thresholded_classification_metrics(
        val_eval["y_true"], val_eval["p_one"], val_eval.get("invalid_flags"), decision_threshold,
    )
    val_macro_f1 = float(val_thr.get("macro_f1", 0.0))

    # Persist val metrics to disk for post-hoc analysis across runs.
    val_metrics_path = rl_dir / f"metrics_val_step_{step}.json"
    val_metrics_path.write_text(
        json.dumps(
            {
                "step": step,
                "decision_threshold_rule": "val_max_macro_f1",
                "decision_threshold": float(decision_threshold),
                **{k: (float(v) if isinstance(v, (int, float)) else v) for k, v in val_thr.items()
                   if k != "y_pred"},
                "auroc": float(val_eval.get("auroc", float("nan"))),
                "auprc": float(val_eval.get("auprc", float("nan"))),
                "loss": float(val_eval.get("loss", 0.0)),
                "cls_loss": float(val_eval.get("cls_loss", 0.0)),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    logger.info(
        "VAL@step {} | thr={:.4f} acc={:.4f} macro_f1={:.4f} auroc={:.4f} auprc={:.4f}",
        step, float(decision_threshold),
        float(val_thr.get("accuracy", 0.0)), val_macro_f1,
        float(val_eval.get("auroc", float("nan"))),
        float(val_eval.get("auprc", float("nan"))),
    )

    # ── Hard-sample evaluation ────────────────────────────────────────────────
    # Hard samples are train examples where the teacher model repeatedly failed to
    # produce the correct label (rejected by rejection sampling). They represent the
    # most difficult or label-noisy examples. Monitoring them separately reveals
    # whether RL is improving performance on hard cases vs. just easy ones.
    hard_accuracy: float | None = None
    if hard_eval_rows:
        hard_eval = _eval_logprob_only(
            eval_sampling_client, tokenizer=None, eval_rows=hard_eval_rows,
            max_samples=0, one_tokens=one_tokens, zero_tokens=zero_tokens,
            max_concurrency=max_concurrency,
        )
        hard_accuracy = _apply_threshold_accuracy(
            hard_eval["p_one"], hard_eval["y_true"], decision_threshold
        )
        logger.info("HARD@step {} | acc={:.4f} (n={})", step, hard_accuracy, len(hard_eval_rows))

    # ── Easy-sample evaluation ────────────────────────────────────────────────
    # Easy samples are a fixed random subset of non-hard train examples.
    # Comparing hard vs. easy accuracy gives insight into whether RL improves
    # performance on challenging cases or only on examples it already handles well.
    easy_accuracy: float | None = None
    if easy_eval_rows:
        easy_eval = _eval_logprob_only(
            eval_sampling_client, tokenizer=None, eval_rows=easy_eval_rows,
            max_samples=0, one_tokens=one_tokens, zero_tokens=zero_tokens,
            max_concurrency=max_concurrency,
        )
        easy_accuracy = _apply_threshold_accuracy(
            easy_eval["p_one"], easy_eval["y_true"], decision_threshold
        )
        logger.info("EASY@step {} | acc={:.4f} (n={})", step, easy_accuracy, len(easy_eval_rows))

    # ── WandB logging ─────────────────────────────────────────────────────────
    if wandb_run is not None:
        wandb_metrics: dict[str, Any] = {
            "val/loss": float(val_eval.get("loss", 0.0)),
            "val/accuracy": float(val_thr.get("accuracy", 0.0)),
            "val/macro_f1": val_macro_f1,
            "val/auroc": float(val_eval.get("auroc", float("nan"))),
            "val/auprc": float(val_eval.get("auprc", float("nan"))),
            "val/decision_threshold": float(decision_threshold),
        }
        if hard_accuracy is not None:
            wandb_metrics["train/hard_accuracy"] = hard_accuracy
        if easy_accuracy is not None:
            wandb_metrics["train/easy_accuracy"] = easy_accuracy
        wandb_run.log(wandb_metrics, step=step)

    return {
        "val_macro_f1": val_macro_f1,
        "decision_threshold": float(decision_threshold),
        "hard_accuracy": hard_accuracy,
        "easy_accuracy": easy_accuracy,
        "eval_sampling_client": eval_sampling_client,
    }


def _run_test_generation_eval(
    *,
    sampling_client: tinker.SamplingClient,
    tokenizer: Any,
    test_gen_rows: list[dict[str, Any]],
    test_logprob_rows: list[dict[str, Any]],
    one_tokens: list[int],
    zero_tokens: list[int],
    max_new_tokens: int,
    max_concurrency: int,
    rl_dir: Path,
    wandb_run: Any,
    step: int,
) -> dict[str, Any]:
    """Run final test set evaluation using greedy generation + logprob probing.

    Two-phase approach:
      Phase 1 — Greedy generation (temperature=0): collect generated text, extract
        reasoning and emitted label. These are used for acc / macro-F1 / precision /
        recall / balanced_acc / tp / fp / fn / tn.
      Phase 2 — Logprob probing: compute P("1") / P("0") for AUROC / AUPRC.
        Per spec: auroc/auprc use normalized P(1)/P(0); other metrics use actual labels.

    Saves:
      - test_results.json: list of {text, reasoning, predicted_label, true_label}
      - test_metrics.json: all classification metrics

    Returns dict with all test metrics.
    """
    logger.info("Running test generation evaluation on {} examples...", len(test_gen_rows))

    # ── Phase 1: Greedy generation ────────────────────────────────────────────
    # Temperature=0 (greedy) for deterministic, reproducible test results.
    gen_params = tinker.SamplingParams(
        max_tokens=max_new_tokens,
        temperature=0.0,
        top_k=-1,
        top_p=1.0,
    )

    test_results: list[dict[str, Any]] = []
    emitted_y_true: list[int] = []
    emitted_y_pred: list[int] = []

    for ex in test_gen_rows:
        prompt_input = tinker.ModelInput.from_ints(list(ex["prompt_tokens"]))
        try:
            resp = sampling_client.sample(
                prompt_input, num_samples=1, sampling_params=gen_params
            ).result()
        except Exception as exc:
            logger.warning("Test generation failed for text={!r}: {}", ex["text"][:60], exc)
            # On failure, record with an empty response and fallback prediction of 0.
            test_results.append({
                "text": ex["text"],
                "reasoning": "",
                "predicted_label": None,
                "true_label": int(ex["label"]),
            })
            emitted_y_true.append(int(ex["label"]))
            emitted_y_pred.append(0)
            continue

        sequences = list(resp.sequences or [])
        if not sequences or not sequences[0].tokens:
            gen_text = ""
        else:
            gen_text = tokenizer.decode(list(sequences[0].tokens), skip_special_tokens=False)

        # Extract label and reasoning from generated text
        pred, _ = _extract_emitted_label(gen_text)
        reasoning = _extract_reasoning_from_json(gen_text)

        test_results.append({
            "text": ex["text"],
            "reasoning": reasoning,
            "predicted_label": pred,
            "true_label": int(ex["label"]),
        })
        emitted_y_true.append(int(ex["label"]))
        # Use 0 as fallback when label cannot be parsed (treats invalid as "not hate speech")
        emitted_y_pred.append(int(pred) if pred in {0, 1} else 0)

    # ── Phase 2: Logprob probing for AUROC/AUPRC ──────────────────────────────
    # Generation gives hard predictions; logprobs give calibrated probabilities.
    # We use logprob P(1)/P(0) for area-under-curve metrics as specified.
    logprob_eval = _eval_logprob_only(
        sampling_client, tokenizer=None, eval_rows=test_logprob_rows,
        max_samples=0, one_tokens=one_tokens, zero_tokens=zero_tokens,
        max_concurrency=max_concurrency,
    )
    p_one = logprob_eval.get("p_one", [])

    # ── Compute test metrics ──────────────────────────────────────────────────
    # Non-AUC metrics use emitted labels; AUC metrics use logprob P(1).
    emitted_base = core.binary_metrics(emitted_y_true, emitted_y_pred, p_one=p_one if p_one else None)
    macro_f1 = float(sklearn_f1_score(emitted_y_true, emitted_y_pred, average="macro", zero_division=0))

    test_metrics: dict[str, Any] = {
        "step": step,
        "n_test": len(test_gen_rows),
        "n_invalid_label": sum(1 for r in test_results if r["predicted_label"] is None),
        "accuracy": emitted_base.get("accuracy", 0.0),
        "balanced_accuracy": emitted_base.get("balanced_accuracy", 0.0),
        "macro_f1": macro_f1,
        "precision": emitted_base.get("precision", 0.0),
        "recall": emitted_base.get("recall", 0.0),
        "tp": emitted_base.get("tp", 0),
        "fp": emitted_base.get("fp", 0),
        "fn": emitted_base.get("fn", 0),
        "tn": emitted_base.get("tn", 0),
        # AUC metrics from logprob P(1) (per spec: use normalized P(1)/P(0))
        "auroc": emitted_base.get("auroc", float("nan")),
        "auprc": emitted_base.get("auprc", float("nan")),
    }

    # Save detailed per-example results and aggregate metrics to disk
    (rl_dir / "test_results.json").write_text(
        json.dumps(test_results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (rl_dir / "test_metrics.json").write_text(
        json.dumps(
            {k: (float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else v)
             for k, v in test_metrics.items()},
            indent=2,
        ),
        encoding="utf-8",
    )

    logger.info(
        "TEST | acc={:.4f} balanced_acc={:.4f} macro_f1={:.4f} "
        "auroc={:.4f} auprc={:.4f} precision={:.4f} recall={:.4f} "
        "tp={} fp={} fn={} tn={}",
        float(test_metrics.get("accuracy", 0.0)),
        float(test_metrics.get("balanced_accuracy", 0.0)),
        float(test_metrics.get("macro_f1", 0.0)),
        float(test_metrics.get("auroc", float("nan"))),
        float(test_metrics.get("auprc", float("nan"))),
        float(test_metrics.get("precision", 0.0)),
        float(test_metrics.get("recall", 0.0)),
        int(test_metrics.get("tp", 0)),
        int(test_metrics.get("fp", 0)),
        int(test_metrics.get("fn", 0)),
        int(test_metrics.get("tn", 0)),
    )

    # Log scalar test metrics to WandB for easy comparison across runs
    if wandb_run is not None:
        wandb_run.log(
            {f"test/{k}": (float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else v)
             for k, v in test_metrics.items() if isinstance(v, (int, float))},
            step=step,
        )

    return test_metrics


def _save_rl_model_pointer(
    project_root: Path,
    run_name: str,
    best_step: int,
    best_val_macro_f1: float,
    best_model_path: str | None,
    final_model_path: str,
    final_state_path: str,
) -> None:
    """Write RL training result paths to model/rl_latest.json.

    Stored separately from model/latest.json (which records SFT results) to avoid
    overwriting the SFT checkpoint. The RL best/final paths can be used to
    initialize future training runs via --init-model-path.
    """
    pointer = {
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "run_name": run_name,
        "best_step": best_step,
        "best_val_macro_f1": float(best_val_macro_f1),
        "best_model_path": best_model_path,
        "final_model_path": final_model_path,
        "final_state_path": final_state_path,
    }
    # Write to model/rl_latest.json (same level as SFT's model/latest.json).
    out_path = project_root / "model" / "rl_latest.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(pointer, indent=2), encoding="utf-8")
    logger.info("Saved RL model pointer to {}", out_path)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "GRPO-style RL with Tinker for hate speech classification.\n\n"
            "Loads an SFT checkpoint from model/latest.json (or --init-model-path),\n"
            "then trains with classification correctness reward (group-centered GRPO).\n"
            "No rulebook in RL prompts — the model must reason from internalized knowledge."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--config", type=str, default="configs/reasoning_sft.example.json",
                   help="Path to JSON config file (relative to project root or absolute).")
    p.add_argument("--seed", type=int, default=None, help="Random seed (overrides config).")
    p.add_argument("--run-name", type=str, default=None,
                   help="WandB run name and output directory name (auto-generated if omitted).")
    p.add_argument("--dataset-name", "--dataset_name", dest="dataset_name", type=str, default=None)
    p.add_argument("--dataset-root-dir", "--dataset_root_dir", dest="dataset_root_dir", type=str, default=None)

    # ── Model initialization ────────────────────────────────────────────────
    p.add_argument(
        "--init-model-path", type=str, default=None,
        help="Tinker sampler_weights path to start RL from (e.g. tinker://.../sampler_weights/...)."
             " If omitted, reads from model/latest.json.",
    )
    p.add_argument(
        "--init-state-path", type=str, default=None,
        help="Optional full trainer state path for reliable weight initialization"
             " (weights + optimizer). Preferred over --init-model-path.",
    )
    p.add_argument(
        "--init-checkpoint", choices=["best", "final"], default="final",
        help="When loading from model/latest.json, use 'best' or 'final' checkpoint (default: final).",
    )
    p.add_argument(
        "--kl-ref-model-path", type=str, default=None,
        help="Reference model path for KL penalty (defaults to init model path).",
    )

    # ── RL training hyperparameters ─────────────────────────────────────────
    p.add_argument("--lora-rank", type=int, default=None,
                   help="LoRA rank (overrides config). Use the same rank as SFT.")
    p.add_argument("--num-epochs", type=int, default=1,
                   help="Number of passes over the training set.")
    p.add_argument("--batch-size", type=int, default=8,
                   help="Number of prompts per training step.")
    p.add_argument(
        "--loss-fn", type=str, default="ppo", choices=["importance_sampling", "ppo", "cispo"],
        help="RL loss function for forward_backward(). 'ppo' recommended.",
    )
    p.add_argument("--ppo-clip-coef", type=float, default=0.2,
                   help="PPO clip coefficient (ε): clips ratio to [1-ε, 1+ε]. Default: 0.2.")
    p.add_argument("--num-substeps", type=int, default=1,
                   help="Optimizer updates per rollout batch (reuses same rollouts). Default: 1.")
    p.add_argument(
        "--normalize-advantages", action=argparse.BooleanOptionalAction, default=True,
        help="If True (default), GRPO normalization: (r-mean)/(std+1e-8) per group.",
    )
    p.add_argument("--group-size", type=int, default=8,
                   help="Number of rollouts per prompt (GRPO group size). Default: 8.")
    p.add_argument("--max-train-examples", type=int, default=None, help="0 = use all.")
    p.add_argument("--max-val-examples", type=int, default=None, help="0 = use all.")
    p.add_argument("--learning-rate", type=float, default=1e-5)
    p.add_argument("--grad-clip-norm", type=float, default=1.0,
                   help="Gradient clipping norm. 0 disables clipping.")
    p.add_argument("--weight-decay", type=float, default=0.01)

    # ── Sampling (rollout generation) ───────────────────────────────────────
    p.add_argument("--rollout-max-tokens", type=int, default=512,
                   help="Max tokens to generate per rollout. Should accommodate full JSON output.")
    p.add_argument("--rollout-temperature", type=float, default=1.0,
                   help="Sampling temperature for rollouts. 1.0 = no scaling.")
    p.add_argument("--rollout-top-p", type=float, default=1.0)
    p.add_argument("--rollout-top-k", type=int, default=-1)
    p.add_argument("--rollout-stop", type=str, default=None,
                   help="Comma-separated stop strings for rollouts (leave empty for none).")
    p.add_argument("--sample-max-concurrency", type=int, default=64,
                   help="Max concurrent sampling requests per training step.")

    # ── KL penalty ─────────────────────────────────────────────────────────
    p.add_argument("--kl-beta", type=float, default=0.0,
                   help="KL penalty coefficient β. 0 disables KL penalty (default).")

    # ── Thinking reward (optional) ──────────────────────────────────────────
    p.add_argument(
        "--thinking-reward-coef", type=float, default=0.0,
        help=(
            "Optional additive reward bonus for longer reasoning. "
            f"Bonus = coef * min(1.0, reasoning_chars / {_THINKING_REWARD_THRESHOLD_CHARS}). "
            "0 disables (default). Encourages the model to generate more thoughtful responses."
        ),
    )

    # ── Eval and checkpointing ──────────────────────────────────────────────
    p.add_argument("--eval-interval", type=int, default=50,
                   help="Run val/hard/easy eval every N training steps.")
    p.add_argument("--eval-max-concurrency", type=int, default=64,
                   help="Max concurrent logprob requests during evaluation.")
    p.add_argument("--early-stopping-patience", type=int, default=5,
                   help="Stop training after N consecutive evals without val macro-F1 improvement.")
    p.add_argument("--save-interval", type=int, default=200,
                   help="Save state + sampler checkpoint every N steps.")
    p.add_argument("--ttl-seconds", type=int, default=7 * 24 * 3600,
                   help="Time-to-live for Tinker checkpoints (default: 7 days).")
    p.add_argument("--log-dir", type=str, default=None,
                   help="Override log_dir from config.")
    p.add_argument(
        "--log-rollout-samples", type=int, default=4,
        help=(
            "Number of rollout samples to write to rollouts.jsonl per training step. "
            "Useful for inspecting actual model outputs and verifying training is healthy. "
            "0 disables rollout logging."
        ),
    )

    # ── API keys ────────────────────────────────────────────────────────────
    p.add_argument("--tinker-api-key", type=str, default=None,
                   help="Tinker API key (overrides TINKER_API_KEY env var and built-in default).")
    p.add_argument("--wandb-api-key", type=str, default=None,
                   help="WandB API key (overrides WANDB_API_KEY env var and built-in default).")
    p.add_argument("--wandb-project", type=str, default=None,
                   help="WandB project name (overrides config).")

    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:  # noqa: C901 (complexity OK for a training script)
    args = parse_args()
    project_root = Path(__file__).resolve().parent

    # ── API key setup ─────────────────────────────────────────────────────────
    # Priority: CLI arg > env var > built-in default.
    tinker_api_key = args.tinker_api_key or os.environ.get("TINKER_API_KEY") or _DEFAULT_TINKER_API_KEY
    wandb_api_key = args.wandb_api_key or os.environ.get("WANDB_API_KEY") or _DEFAULT_WANDB_API_KEY
    # Expose the key to Tinker's internal auth mechanism (reads from env).
    os.environ["TINKER_API_KEY"] = tinker_api_key
    # Log in to WandB (relogin=True allows switching keys between runs).
    wandb.login(key=wandb_api_key, relogin=True)

    # ── Config and logging setup ──────────────────────────────────────────────
    cfg = core.load_config(project_root, args)

    # Apply CLI overrides that supersede config values
    if args.seed is not None:
        cfg["seed"] = int(args.seed)
    if args.dataset_name is not None:
        cfg["dataset_name"] = str(args.dataset_name)
    if args.dataset_root_dir is not None:
        cfg["dataset_root_dir"] = str(args.dataset_root_dir)
    if args.lora_rank is not None:
        cfg["lora_rank"] = int(args.lora_rank)
    if args.max_train_examples is not None:
        cfg["max_train_examples"] = int(args.max_train_examples)
    if args.max_val_examples is not None:
        cfg["max_val_examples"] = int(args.max_val_examples)
    if args.log_dir is not None:
        cfg["log_dir"] = str(args.log_dir)

    seed = int(cfg["seed"])
    random.seed(seed)
    np.random.seed(seed)

    run_name = args.run_name or f"grpo_rl_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    log_dir = core.resolve_path(project_root, str(cfg.get("log_dir", "runs")))
    if log_dir is None:
        raise RuntimeError("log_dir resolution failed")
    run_dir = log_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    rl_dir = run_dir / RL_SUBDIR
    rl_dir.mkdir(parents=True, exist_ok=True)
    core.setup_logger(rl_dir)

    # Persist configs for reproducibility
    (run_dir / "resolved_config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    (rl_dir / "resolved_args.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    logger.info("Run dir: {}", run_dir)
    logger.info("RL dir: {}", rl_dir)

    # ── SFT checkpoint loading ────────────────────────────────────────────────
    init_model_path = args.init_model_path
    init_state_path: str | None = args.init_state_path or None

    if not init_model_path:
        # Read from the model pointer written by the SFT training script
        latest = _read_latest_pointer(project_root)
        if latest is None:
            raise RuntimeError(
                "Missing --init-model-path and could not read model/latest.json. "
                "Pass --init-model-path tinker://.../sampler_weights/..."
            )
        model_key = "best_model_path" if args.init_checkpoint == "best" else "final_model_path"
        init_model_path = str(latest.get(model_key) or "")
        if not init_model_path:
            raise RuntimeError(f"model/latest.json missing key '{model_key}'")
        # For the final checkpoint, we also have a full resumable trainer state.
        # This is more reliable than reinitializing from sampler_weights alone.
        if args.init_checkpoint == "final":
            init_state_path = str(latest.get("resume_state_path") or "").strip() or None
        elif args.init_checkpoint == "best" and init_state_path is None:
            logger.warning(
                "Init checkpoint is 'best' but no state path available. "
                "For reliable init, use --init-checkpoint final or --init-state-path."
            )

    # The KL reference model defaults to the same init model path.
    # This means we penalize divergence from the SFT starting point.
    kl_ref_model_path = args.kl_ref_model_path or init_model_path

    # ── Dataset loading ───────────────────────────────────────────────────────
    split_paths, dataset_dir, dataset_name = _resolve_split_paths(project_root, cfg)
    train_df = _load_split_dataset(split_paths["train"], cfg, "train")
    val_df = _load_split_dataset(split_paths["val"], cfg, "val")
    test_df = _load_split_dataset(split_paths["test"], cfg, "test")

    train_df = core.stratified_subset(train_df, int(cfg["max_train_examples"]), seed=seed, split_name="train")
    val_df = core.stratified_subset(val_df, int(cfg["max_val_examples"]), seed=seed + 1, split_name="val")

    logger.info(
        "Dataset '{}' | train={} val={} test={}",
        dataset_name, len(train_df), len(val_df), len(test_df),
    )

    # ── Hard/easy sample identification ──────────────────────────────────────
    # The train_with_hard_tag.csv has a 'hard' column (1=hard, 0=easy).
    # Hard samples are those where the teacher model failed to produce the correct
    # label (rejected by rejection sampling), i.e., challenging or noisy examples.
    hard_train_df: pd.DataFrame | None = None
    easy_monitor_df: pd.DataFrame | None = None

    hard_tag_path = dataset_dir / "train_with_hard_tag.csv"
    if hard_tag_path.exists():
        hard_tag_df = pd.read_csv(hard_tag_path)
        # Align column names: fall back to index merge if 'text' column exists
        if "hard" in hard_tag_df.columns and "text" in hard_tag_df.columns:
            hard_mask = hard_tag_df["hard"].astype(int) == 1
            hard_train_df = hard_tag_df[hard_mask][["text", "label"]].reset_index(drop=True)
            easy_all_df = hard_tag_df[~hard_mask][["text", "label"]].reset_index(drop=True)
            # Fix label column dtype
            hard_train_df["label"] = hard_train_df["label"].astype(int)
            easy_all_df["label"] = easy_all_df["label"].astype(int)
            # Fix up label values in case they're continuous (apply same threshold)
            threshold = float(cfg.get("label_threshold", 0.5))
            hard_train_df["label"] = (hard_train_df["label"].astype(float) >= threshold).astype(int)
            easy_all_df["label"] = (easy_all_df["label"].astype(float) >= threshold).astype(int)
            # Fixed random subset of easy samples for consistent monitoring across runs
            n_easy = min(_EASY_MONITOR_COUNT, len(easy_all_df))
            easy_monitor_df = easy_all_df.sample(n=n_easy, random_state=seed).reset_index(drop=True)
            logger.info(
                "Hard/easy split: {} hard, {} easy ({} selected for monitoring)",
                len(hard_train_df), len(easy_all_df), n_easy,
            )
        else:
            logger.warning("train_with_hard_tag.csv found but missing 'hard' or 'text' column — skipping.")
    else:
        logger.warning("train_with_hard_tag.csv not found at {}; skipping hard/easy monitoring.", hard_tag_path)

    # ── Tinker client setup ───────────────────────────────────────────────────
    service_client = (
        tinker.ServiceClient(base_url=str(cfg["tinker_base_url"]))
        if cfg.get("tinker_base_url")
        else tinker.ServiceClient()
    )

    # Initialize the RL TrainingClient from the SFT checkpoint.
    # Using init_state_path (full weights + optimizer) is the most reliable path.
    # Without it, we re-create a LoRA client and hope the base_model supports
    # direct initialization from sampler_weights (model-dependent behavior).
    if init_state_path is not None:
        logger.info("Initializing RL training client from state: {}", init_state_path)
        training_client = service_client.create_lora_training_client(
            base_model=str(cfg["student_model_name"]),
            rank=int(cfg["lora_rank"]),
            seed=seed,
        )
        training_client.load_state(init_state_path).result()
    else:
        logger.info("Initializing RL training client from model path: {}", init_model_path)
        training_client = service_client.create_lora_training_client(
            base_model=str(init_model_path),
            rank=int(cfg["lora_rank"]),
            seed=seed,
        )

    tokenizer = training_client.get_tokenizer()

    # Optional KL reference model.
    # If kl_beta > 0, we penalize the policy for drifting away from the reference.
    # This prevents reward hacking (e.g., the model collapsing to always output "0").
    base_sampling_client: tinker.SamplingClient | None = None
    if float(args.kl_beta) > 0.0:
        ref = str(kl_ref_model_path)
        logger.info("Creating KL reference client from: {}", ref)
        if ref.startswith("tinker://"):
            base_sampling_client = service_client.create_sampling_client(model_path=ref)
        else:
            base_sampling_client = service_client.create_sampling_client(base_model=ref)

    # ── Prompt and tokenization setup ────────────────────────────────────────
    task_instruction = str(cfg.get("task_instruction", "Decide whether the following text is hate speech."))
    reasoning_placeholder = str(cfg.get("eval_reasoning_placeholder", "Reasoning intentionally omitted for label scoring."))
    max_length = int(cfg["max_length"])

    # The RL prompt builder: simple instruction + text, NO rulebook.
    # This is the key difference from SFT — the model must reason without explicit rules.
    rl_prompt_builder: Callable[[str], str] = lambda text: _build_rl_user_prompt(task_instruction, text)

    # Encode label probe tokens (single tokens for "1" and "0").
    # These are used for logprob-based P(label=1) / P(label=0) evaluation.
    one_tokens = tokenizer.encode("1", add_special_tokens=False)
    zero_tokens = tokenizer.encode("0", add_special_tokens=False)
    if not one_tokens or not zero_tokens:
        raise RuntimeError("Failed to tokenize '1'/'0' for label probing.")
    logger.info("Label probe tokens: one={}, zero={}", one_tokens, zero_tokens)

    # Pre-tokenize evaluation rows (val, hard, easy) with the RL prompt + label-probe prefix.
    # The label-probe prefix lets us measure P("1"|context) and P("0"|context).
    # Using the SAME prompt as training ensures consistent distribution.
    logger.info("Pre-tokenizing val eval rows ({} examples)...", len(val_df))
    val_eval_rows = core.build_eval_rows(
        val_df,
        tokenizer=tokenizer,
        max_length=max_length,
        reasoning_prompt_builder=rl_prompt_builder,
        reasoning_placeholder=reasoning_placeholder,
    )
    logger.info("Val eval rows: {}", len(val_eval_rows))

    # Hard sample eval rows.
    # Wrapped in try-except: if all rows fail tokenization (extremely unlikely for short tweets),
    # we log a warning and disable hard-set monitoring rather than crashing the entire run.
    hard_eval_rows: list[dict[str, Any]] = []
    if hard_train_df is not None and len(hard_train_df) > 0:
        logger.info("Pre-tokenizing hard eval rows ({} examples)...", len(hard_train_df))
        try:
            hard_eval_rows = core.build_eval_rows(
                hard_train_df,
                tokenizer=tokenizer,
                max_length=max_length,
                reasoning_prompt_builder=rl_prompt_builder,
                reasoning_placeholder=reasoning_placeholder,
            )
            logger.info("Hard eval rows: {}", len(hard_eval_rows))
        except RuntimeError as exc:
            logger.warning("Hard eval rows disabled ({}). Hard-set monitoring will be skipped.", exc)

    # Easy sample eval rows.
    # Same defensive wrapper as hard_eval_rows.
    easy_eval_rows: list[dict[str, Any]] = []
    if easy_monitor_df is not None and len(easy_monitor_df) > 0:
        logger.info("Pre-tokenizing easy eval rows ({} examples)...", len(easy_monitor_df))
        try:
            easy_eval_rows = core.build_eval_rows(
                easy_monitor_df,
                tokenizer=tokenizer,
                max_length=max_length,
                reasoning_prompt_builder=rl_prompt_builder,
                reasoning_placeholder=reasoning_placeholder,
            )
            logger.info("Easy eval rows: {}", len(easy_eval_rows))
        except RuntimeError as exc:
            logger.warning("Easy eval rows disabled ({}). Easy-set monitoring will be skipped.", exc)

    # Pre-tokenize test set for final evaluation.
    # test_gen_rows: plain prompt_tokens for greedy generation (to get reasoning + label).
    # test_logprob_rows: probe-prefixed prompt_tokens for P(1)/P(0) logprob eval.
    logger.info("Pre-tokenizing test rows ({} examples)...", len(test_df))
    test_gen_rows = _tokenize_prompt_rows(test_df, tokenizer, rl_prompt_builder, max_length)
    try:
        test_logprob_rows = core.build_eval_rows(
            test_df,
            tokenizer=tokenizer,
            max_length=max_length,
            reasoning_prompt_builder=rl_prompt_builder,
            reasoning_placeholder=reasoning_placeholder,
        )
    except RuntimeError as exc:
        # If all test rows fail (shouldn't happen in practice), fall back to an empty list.
        # _run_test_generation_eval() handles mismatched / empty logprob rows gracefully.
        logger.warning("Test logprob rows empty ({}). AUROC/AUPRC will not be computed.", exc)
        test_logprob_rows = []
    logger.info("Test gen rows: {}, test logprob rows: {}", len(test_gen_rows), len(test_logprob_rows))

    # Pre-tokenize train prompts for RL rollout sampling.
    logger.info("Pre-tokenizing train prompts ({} examples)...", len(train_df))
    train_prompts: list[dict[str, Any]] = _tokenize_prompt_rows(
        train_df, tokenizer, rl_prompt_builder, max_length
    )
    if not train_prompts:
        raise RuntimeError("No training prompts available after tokenization.")
    logger.info("Train prompts: {}", len(train_prompts))

    # ── WandB initialization ──────────────────────────────────────────────────
    # Wrapped in try-except: if WandB login fails (e.g. invalid key or network issue),
    # training continues without remote logging. All metrics are still saved locally
    # to rl_history.jsonl and per-step JSON files.
    wandb_project = args.wandb_project or str(cfg.get("wandb_project", "reasoning-rl-tinker"))
    wandb_run = None
    try:
        wandb_run = wandb.init(
            project=wandb_project,
            entity=cfg.get("wandb_entity") or None,
            name=run_name,
            config={**vars(args), **{k: v for k, v in cfg.items() if isinstance(v, (str, int, float, bool))}},
            mode=str(cfg.get("wandb_mode", "online")),
            dir=str(run_dir),
        )
        logger.info("WandB run: {}", wandb_run.url if wandb_run else "disabled")
    except Exception as exc:
        logger.warning("WandB init failed ({}). Training will continue without remote logging.", exc)
        wandb_run = None

    # ── Rollout sampling parameters ───────────────────────────────────────────
    stop_list = _parse_stop_arg(args.rollout_stop)
    sampling_params = tinker.SamplingParams(
        max_tokens=int(args.rollout_max_tokens),
        seed=seed,
        stop=stop_list,
        temperature=float(args.rollout_temperature),
        top_k=int(args.rollout_top_k),
        top_p=float(args.rollout_top_p),
    )

    # ── Optimizer configuration ───────────────────────────────────────────────
    # tinker.AdamParams only accepts learning_rate, beta1, beta2, eps.
    # weight_decay and grad_clip_norm are not supported by the Tinker API.
    adam_params = tinker.AdamParams(
        learning_rate=float(args.learning_rate),
    )

    # ── Loss function config ──────────────────────────────────────────────────
    loss_fn_config: dict[str, float] | None = None
    if str(args.loss_fn) in {"ppo", "cispo"}:
        # PPO and CISPO require explicit clipping thresholds for the importance ratio.
        eps = float(args.ppo_clip_coef)
        loss_fn_config = {
            "clip_low_threshold": 1.0 - eps,
            "clip_high_threshold": 1.0 + eps,
        }

    # ── Training schedule ─────────────────────────────────────────────────────
    steps_per_epoch = int(math.ceil(len(train_prompts) / max(int(args.batch_size), 1)))
    total_steps = steps_per_epoch * int(args.num_epochs)
    logger.info(
        "RL schedule: epochs={} batch_size={} steps_per_epoch={} total_steps={} group_size={}",
        int(args.num_epochs), int(args.batch_size), steps_per_epoch, total_steps, int(args.group_size),
    )

    # ── Early stopping state ──────────────────────────────────────────────────
    best_val_macro_f1 = float("-inf")
    best_model_sampler_path: str | None = None
    best_step = -1
    patience_counter = 0
    stop_training = False  # flag to break out of double loop

    # ── Training history ──────────────────────────────────────────────────────
    history_path = rl_dir / "rl_history.jsonl"
    if history_path.exists():
        history_path.unlink()  # start fresh for this run

    global_step = 0

    # ═══════════════════════════════════════════════════════════════════════════
    # Training loop
    # ═══════════════════════════════════════════════════════════════════════════
    for epoch in range(int(args.num_epochs)):
        if stop_training:
            break

        # Shuffle training prompts at the start of each epoch with a per-epoch seed
        # so that each epoch sees examples in a different order.
        rng = random.Random(seed + epoch)
        rng.shuffle(train_prompts)

        for b in range(0, len(train_prompts), int(args.batch_size)):
            if stop_training:
                break

            batch = train_prompts[b : b + int(args.batch_size)]
            global_step += 1

            # ── Rollout sampling ──────────────────────────────────────────────
            # Save current policy weights and create a sampling client.
            # This snapshot is what we sample from (on-policy rollouts).
            sampling_client = training_client.save_weights_and_get_sampling_client()

            # Submit async sample requests for all prompts in the batch.
            # Each prompt gets `group_size` independent rollouts for GRPO centering.
            sample_futs: list[tuple[dict[str, Any], Any]] = []
            in_flight = 0
            for ex in batch:
                prompt_input = tinker.ModelInput.from_ints(list(ex["prompt_tokens"]))
                try:
                    fut = sampling_client.sample(
                        prompt=prompt_input,
                        num_samples=int(args.group_size),
                        sampling_params=sampling_params,
                    )
                except Exception as exc:
                    logger.warning("sample() failed for idx={} error={}", ex.get("idx", "?"), exc)
                    continue
                sample_futs.append((ex, fut))
                in_flight += 1

                # Enforce concurrency limit with soft backpressure:
                # wait for the oldest in-flight request before submitting more.
                if in_flight >= int(args.sample_max_concurrency):
                    oldest_idx = len(sample_futs) - in_flight
                    try:
                        _ = sample_futs[oldest_idx][1].result()
                    except Exception:
                        pass
                    in_flight -= 1

            # ── Collect rollouts and compute rewards ──────────────────────────
            all_datums: list[tinker.Datum] = []
            datum_meta: list[dict[str, int]] = []

            # Aggregate metrics across the batch
            batch_rewards: list[float] = []       # all rollout rewards (flat)
            batch_correct: list[int] = []         # correctness flags (flat)
            batch_invalid: list[int] = []         # parse-failure flags (flat)
            batch_thinking_lengths: list[int] = []  # generated token counts (flat)

            # Per-group statistics: one entry per prompt in the batch.
            # reward_mean = mean of per-group means, reward_std = mean of per-group stds.
            group_reward_means: list[float] = []
            group_reward_stds: list[float] = []

            # KL penalty bookkeeping (used only if kl_beta > 0)
            kl_logprob_diffs: list[float] = []

            # Rollout log records collected this step (written to rollouts.jsonl below).
            rollout_records: list[dict[str, Any]] = []

            for ex, fut in sample_futs:
                try:
                    resp: tinker.SampleResponse = fut.result()
                except Exception as exc:
                    logger.warning("Sampling result failed for idx={} error={}", ex.get("idx", "?"), exc)
                    continue
                sequences = list(resp.sequences or [])
                if not sequences:
                    continue

                # Process each rollout in this prompt's group
                rewards: list[float] = []         # rewards for this group
                per_seq: list[dict[str, Any]] = []  # metadata per rollout

                for seq in sequences:
                    tokens = list(seq.tokens or [])
                    if not tokens:
                        continue

                    # Obtain per-token logprobs (needed for importance sampling loss).
                    # The sampler usually returns logprobs directly; fall back to
                    # compute_logprobs if they're missing or malformed.
                    raw_lps = list(seq.logprobs or [])
                    lps: list[float] | None = None
                    if len(raw_lps) == len(tokens) and not any(x is None for x in raw_lps):
                        try:
                            lps = [float(x) for x in raw_lps]  # type: ignore[arg-type]
                        except Exception:
                            lps = None
                    if lps is None:
                        lps = _compute_generated_token_logprobs(
                            sampling_client,
                            prompt_tokens=list(ex["prompt_tokens"]),
                            generated_tokens=tokens,
                        )
                    if lps is None or len(lps) != len(tokens):
                        continue  # skip if logprobs are unavailable

                    # Decode generated tokens to text for label extraction
                    text = tokenizer.decode(tokens, skip_special_tokens=False) if tokens else ""

                    # Extract the emitted binary label from the generated text
                    pred, pred_src = _extract_emitted_label(text)
                    valid = pred in {0, 1}
                    correct = int(valid and int(pred) == int(ex["label"]))

                    # --- Reward computation ---
                    # Base reward: +1 if correct, -1 if incorrect or unparseable.
                    # We treat parse failures as wrong answers (reward = -1).
                    reward = 1.0 if correct else -1.0

                    # Optional thinking bonus: reward longer/more detailed reasoning.
                    # This encourages the model to generate thoughtful responses rather
                    # than just outputting a label with minimal reasoning.
                    if float(args.thinking_reward_coef) > 0.0:
                        reasoning_chars = _extract_reasoning_char_count(text)
                        # Scale linearly from 0 to full bonus at the threshold.
                        thinking_bonus = float(args.thinking_reward_coef) * min(
                            1.0, reasoning_chars / _THINKING_REWARD_THRESHOLD_CHARS
                        )
                        reward += thinking_bonus

                    rewards.append(reward)
                    per_seq.append({
                        "tokens": tokens,
                        "logprobs": lps,
                        "text": text,
                        "pred": pred,
                        "pred_src": pred_src,
                        "valid": bool(valid),
                        "correct": int(correct),
                        "reward": float(reward),
                    })

                    # Collect for rollout log (written at end of step).
                    # Truncate input text to keep the log file readable.
                    rollout_records.append({
                        "step": int(global_step),
                        "epoch": int(epoch),
                        "input_text": str(ex["text"])[:300],
                        "true_label": int(ex["label"]),
                        "generated": text,
                        "n_tokens": len(tokens),
                        "pred_label": pred,
                        "pred_src": pred_src,
                        "correct": int(correct),
                        "reward": float(reward),
                    })

                if not rewards:
                    continue  # skip prompts where all rollouts failed

                # --- GRPO advantage computation ---
                # Center rewards within the group (subtract group mean).
                # Normalize by group std if enabled (reduces gradient variance).
                mean_reward = float(np.mean(rewards))
                std_reward = float(np.std(rewards))

                if args.normalize_advantages:
                    # (r - mean) / (std + eps): normalized GRPO advantages
                    denom = std_reward + 1e-8
                    advantages = [float((r - mean_reward) / denom) for r in rewards]
                else:
                    # (r - mean): centered but not normalized
                    advantages = [float(r - mean_reward) for r in rewards]

                # Track per-group statistics for WandB logging.
                # reward_mean metric = average of group means across batch.
                # reward_std metric = average of group stds across batch.
                group_reward_means.append(mean_reward)
                group_reward_stds.append(std_reward)

                # --- Build Tinker Datums for the importance_sampling / PPO loss ---
                prompt_tokens = list(ex["prompt_tokens"])
                prompt_input = tinker.ModelInput.from_ints(prompt_tokens)
                # ob_len = number of observation (prompt) tokens minus the last one.
                # This is the standard "right-shift" offset used in the Tinker cookbook.
                ob_len = prompt_input.length - 1

                for seq_i, seq_info in enumerate(per_seq):
                    sampled_tokens = list(seq_info["tokens"])
                    sampled_lps = list(seq_info["logprobs"])
                    if len(sampled_tokens) < 1 or len(sampled_lps) != len(sampled_tokens):
                        continue

                    # The model_input is prompt + generated tokens (excluding last token,
                    # which becomes the final target). This follows the "right-shifted"
                    # causal LM convention: input[:-1] predicts target[1:].
                    model_input = prompt_input.append(
                        tinker.EncodedTextChunk(tokens=sampled_tokens[:-1])
                    )

                    # Pad target tokens and logprobs with zeros for the prompt portion.
                    # Only the generated tokens contribute to the RL loss (masked by advantage=0
                    # for the prompt, non-zero advantage for generated tokens).
                    target_tokens = [0] * ob_len + sampled_tokens
                    padded_logprobs = [0.0] * ob_len + sampled_lps
                    adv = float(advantages[seq_i])
                    padded_advantages = [0.0] * ob_len + [adv] * len(sampled_tokens)

                    datum = tinker.Datum(
                        model_input=model_input,
                        loss_fn_inputs={
                            "target_tokens": tinker.TensorData.from_numpy(
                                np.asarray(target_tokens, dtype=np.int32)
                            ),
                            "logprobs": tinker.TensorData.from_numpy(
                                np.asarray(padded_logprobs, dtype=np.float32)
                            ),
                            "advantages": tinker.TensorData.from_numpy(
                                np.asarray(padded_advantages, dtype=np.float32)
                            ),
                        },
                    )
                    all_datums.append(datum)
                    datum_meta.append({"ob_len": int(ob_len)})
                    batch_rewards.append(float(seq_info["reward"]))
                    batch_correct.append(int(seq_info["correct"]))
                    batch_invalid.append(0 if seq_info["valid"] else 1)
                    # Track thinking length = total generated tokens per rollout
                    batch_thinking_lengths.append(len(seq_info["tokens"]))

            if not all_datums:
                logger.warning(
                    "Step {}: no usable datums (all sampling or parsing failed)", global_step
                )
                continue

            # ── Optional KL penalty (reward shaping) ──────────────────────────
            # Compute logprob differences between the current policy and the reference.
            # Adjust advantages to penalize divergence from the reference model.
            # This prevents the policy from drifting too far from the SFT checkpoint,
            # which can cause reward hacking or catastrophic forgetting.
            if base_sampling_client is not None and float(args.kl_beta) > 0.0:
                # Submit logprob requests for the full sequence (prompt + completion)
                # under the reference model.
                base_futs: list[Any] = []
                for datum in all_datums:
                    # Reconstruct full sequence by appending the final target token.
                    final_tok = int(datum.loss_fn_inputs["target_tokens"].data[-1])
                    full_seq = datum.model_input.append_int(final_tok)
                    base_futs.append(base_sampling_client.compute_logprobs(full_seq))

                base_logprobs_list: list[list[float | None]] = [f.result() for f in base_futs]

                # Compute centered logprob differences on action (generated) positions only.
                # diff = log π_current(a|s) - log π_ref(a|s) (positive = drifted away from ref)
                # avg_diff = batch-averaged drift
                diffs_all: list[float] = []
                for datum, base_lp, meta in zip(all_datums, base_logprobs_list, datum_meta):
                    sampled_lp = datum.loss_fn_inputs["logprobs"].data
                    # base_lp is indexed from position 1 (shifted by the Tinker API convention)
                    base_aligned = base_lp[1 : 1 + len(sampled_lp)]
                    if len(base_aligned) != len(sampled_lp):
                        continue
                    ob_len = int(meta.get("ob_len", 0))
                    for j, (s, b) in enumerate(zip(sampled_lp, base_aligned)):
                        if j < ob_len:
                            continue  # skip prompt positions
                        if b is None or not math.isfinite(float(b)) or not math.isfinite(float(s)):
                            continue
                        diffs_all.append(float(s) - float(b))

                # Mean divergence across all action tokens in this batch.
                avg_diff = float(np.mean(diffs_all)) if diffs_all else 0.0

                # Update advantages by replacing TensorData (never mutate .data in place —
                # TensorData may return an immutable sequence from .data).
                # Formula: advantage += β * (avg_diff - diff_i)
                # Penalizes tokens that diverge more than average from the reference,
                # and rewards tokens that diverge less (centered KL penalty).
                for datum, base_lp, meta in zip(all_datums, base_logprobs_list, datum_meta):
                    sampled_lp = list(datum.loss_fn_inputs["logprobs"].data)
                    base_aligned = base_lp[1 : 1 + len(sampled_lp)]
                    if len(base_aligned) != len(sampled_lp):
                        continue
                    ob_len = int(meta.get("ob_len", 0))
                    # Build a per-position KL adjustment array (same length as advantages).
                    adv_np = datum.loss_fn_inputs["advantages"].to_numpy().copy()
                    for i, (s, b) in enumerate(zip(sampled_lp, base_aligned)):
                        if i < ob_len:
                            continue  # skip prompt positions
                        if b is None or not math.isfinite(float(b)) or not math.isfinite(float(s)):
                            continue
                        diff = float(s) - float(b)
                        adv_np[i] += float(args.kl_beta) * (avg_diff - diff)
                        kl_logprob_diffs.append(diff)
                    # Replace TensorData with the updated numpy array.
                    datum.loss_fn_inputs["advantages"] = tinker.TensorData.from_numpy(adv_np)

            # ── Gradient update ───────────────────────────────────────────────
            # Run forward_backward and optim_step (possibly multiple times if num_substeps > 1,
            # which reuses the same rollouts for multiple gradient updates — a form of PPO).
            if int(args.num_substeps) <= 0:
                raise ValueError("--num-substeps must be >= 1")

            fb_out = None
            optim_result = None
            for _substep in range(int(args.num_substeps)):
                fb_out = training_client.forward_backward(
                    all_datums, loss_fn=str(args.loss_fn), loss_fn_config=loss_fn_config
                ).result()
                # Save optim_result — it carries the actual loss scalar in .metrics["loss:sum"].
                # ForwardBackwardOutput does NOT have a .loss attribute; loss lives in optim_result.
                optim_result = training_client.optim_step(adam_params=adam_params).result()

            # ── Step-level metrics computation ────────────────────────────────
            # reward_mean = mean of per-group means (not mean of all rollouts).
            # This matches GRPO semantics where the group is the unit of centering.
            reward_mean = float(np.mean(group_reward_means)) if group_reward_means else 0.0
            # reward_std = mean of per-group stds (captures within-group reward variance).
            reward_std = float(np.mean(group_reward_stds)) if group_reward_stds else 0.0
            acc = float(np.mean(batch_correct)) if batch_correct else 0.0
            invalid_rate = float(np.mean(batch_invalid)) if batch_invalid else 0.0
            # thinking_length = mean number of generated tokens across all rollouts
            thinking_length = float(np.mean(batch_thinking_lengths)) if batch_thinking_lengths else 0.0
            # kl_mean = mean log-ratio between current and reference policy (only if KL enabled)
            kl_mean = float(np.mean(kl_logprob_diffs)) if kl_logprob_diffs else 0.0
            # loss_val: scalar training loss reported by the Tinker loss function.
            # "loss:sum" lives in fb_out.metrics (ForwardBackwardOutput.metrics: Dict[str, float]),
            # NOT in optim_result.metrics (which contains optimizer diagnostics like gradient norms).
            # For cross_entropy and importance_sampling this key is documented; PPO uses the same
            # convention internally even though it is not explicitly listed in the docs.
            loss_val = 0.0
            if fb_out is not None and hasattr(fb_out, "metrics") and fb_out.metrics:
                loss_val = float(fb_out.metrics.get("loss:sum", 0.0))

            # Persist step metrics to jsonl for post-hoc analysis
            step_row = {
                "step": int(global_step),
                "epoch": int(epoch),
                "loss": loss_val,
                "reward_mean": reward_mean,
                "reward_std": reward_std,
                "batch_acc": acc,
                "invalid_rate": invalid_rate,
                "thinking_length": thinking_length,
                "kl_logprob_diff_mean": kl_mean,
                "kl_beta": float(args.kl_beta),
                "loss_fn": str(args.loss_fn),
                "num_substeps": int(args.num_substeps),
                "n_datums": int(len(all_datums)),
            }
            _append_jsonl(history_path, step_row)

            # ── Rollout log ───────────────────────────────────────────────────
            # Write a sample of rollouts to rollouts.jsonl so training behaviour
            # can be inspected by reading the file (e.g. tail -f rollouts.jsonl).
            # Only the first --log-rollout-samples entries are written to keep
            # the file size manageable.
            n_log = int(args.log_rollout_samples)
            if n_log > 0 and rollout_records:
                for rec in rollout_records[:n_log]:
                    _append_jsonl(rl_dir / "rollouts.jsonl", rec)

            # ── WandB step logging ────────────────────────────────────────────
            if wandb_run is not None:
                wandb_metrics: dict[str, Any] = {
                    "train/loss": loss_val,
                    "train/accuracy": acc,
                    "train/reward_mean": reward_mean,
                    "train/reward_std": reward_std,
                    "train/thinking_length": thinking_length,
                    "train/invalid_rate": invalid_rate,
                    "epoch": epoch,
                }
                if float(args.kl_beta) > 0.0:
                    wandb_metrics["train/kl"] = kl_mean
                wandb_run.log(wandb_metrics, step=global_step)

            logger.info(
                "Step {}/{} | epoch {} | loss={:.4f} reward={:.3f} "
                "acc={:.3f} invalid={:.3f} thinking={:.0f}tok kl={:.4f}",
                global_step, total_steps, epoch,
                loss_val, reward_mean, acc, invalid_rate, thinking_length, kl_mean,
            )

            # ── Periodic evaluation ───────────────────────────────────────────
            if int(args.eval_interval) > 0 and global_step % int(args.eval_interval) == 0:
                eval_result = _run_periodic_eval(
                    training_client=training_client,
                    step=global_step,
                    val_eval_rows=val_eval_rows,
                    hard_eval_rows=hard_eval_rows,
                    easy_eval_rows=easy_eval_rows,
                    one_tokens=one_tokens,
                    zero_tokens=zero_tokens,
                    max_concurrency=int(args.eval_max_concurrency),
                    rl_dir=rl_dir,
                    wandb_run=wandb_run,
                )
                current_macro_f1 = eval_result["val_macro_f1"]

                # ── Early stopping + best model tracking ──────────────────────
                if current_macro_f1 > best_val_macro_f1:
                    # New best! Save this as the best model checkpoint.
                    best_val_macro_f1 = current_macro_f1
                    best_step = global_step
                    patience_counter = 0

                    # We can reuse the eval sampling client's underlying weights
                    # by saving them as a named "best" checkpoint.
                    best_sampler_res = training_client.save_weights_for_sampler(
                        name=f"rl_best_step_{global_step}",
                    ).result()
                    best_model_sampler_path = str(best_sampler_res.path)
                    logger.info(
                        "New best! val macro_f1={:.4f} @ step {} | path={}",
                        best_val_macro_f1, global_step, best_model_sampler_path,
                    )
                    # Save best model info to disk
                    (rl_dir / "best_model_info.json").write_text(
                        json.dumps({
                            "step": best_step,
                            "val_macro_f1": float(best_val_macro_f1),
                            "sampler_path": best_model_sampler_path,
                        }, indent=2),
                        encoding="utf-8",
                    )
                else:
                    # No improvement this eval.
                    patience_counter += 1
                    logger.info(
                        "No improvement (patience {}/{}). Best: macro_f1={:.4f} @ step {}",
                        patience_counter, int(args.early_stopping_patience),
                        best_val_macro_f1, best_step,
                    )
                    # Trigger early stopping if patience is exhausted.
                    if 0 < int(args.early_stopping_patience) <= patience_counter:
                        logger.info(
                            "Early stopping triggered at step {} (patience={} exceeded).",
                            global_step, int(args.early_stopping_patience),
                        )
                        stop_training = True
                        break  # break inner for-loop; outer loop checks stop_training

            # ── Periodic checkpointing ────────────────────────────────────────
            if int(args.save_interval) > 0 and global_step % int(args.save_interval) == 0:
                ckpt_name = f"rl_step_{global_step}"
                state_path = training_client.save_state(name=ckpt_name).result().path
                sampler_path = training_client.save_weights_for_sampler(name=ckpt_name).result().path
                _append_jsonl(
                    rl_dir / "checkpoints.jsonl",
                    {"step": int(global_step), "state_path": str(state_path), "sampler_path": str(sampler_path)},
                )
                logger.info("Checkpoint {} | state={} | sampler={}", ckpt_name, state_path, sampler_path)

    # ═══════════════════════════════════════════════════════════════════════════
    # Post-training: save final checkpoint and run test evaluation
    # ═══════════════════════════════════════════════════════════════════════════

    # ── Final checkpoint ──────────────────────────────────────────────────────
    logger.info("Saving final RL checkpoint...")
    final_state_path = training_client.save_state(name="rl_final").result().path
    final_sampler_path = training_client.save_weights_for_sampler(name="rl_final").result().path

    logger.info("Final state: {}", final_state_path)
    logger.info("Final sampler: {}", final_sampler_path)

    # ── Test evaluation ───────────────────────────────────────────────────────
    # Use the best model if available (tracked during training), else the final model.
    # Best model = checkpoint with highest val macro-F1.
    if best_model_sampler_path is not None:
        logger.info("Using best model for test eval (step={}, macro_f1={:.4f})", best_step, best_val_macro_f1)
        if best_model_sampler_path.startswith("tinker://"):
            test_sampling_client = service_client.create_sampling_client(
                model_path=best_model_sampler_path
            )
        else:
            test_sampling_client = service_client.create_sampling_client(
                base_model=best_model_sampler_path
            )
    else:
        logger.info("No best model saved; using final model for test eval.")
        test_sampling_client = training_client.save_weights_and_get_sampling_client(
            name="rl_final_eval"
        )

    _run_test_generation_eval(
        sampling_client=test_sampling_client,
        tokenizer=tokenizer,
        test_gen_rows=test_gen_rows,
        test_logprob_rows=test_logprob_rows,
        one_tokens=one_tokens,
        zero_tokens=zero_tokens,
        max_new_tokens=int(args.rollout_max_tokens),
        max_concurrency=int(args.eval_max_concurrency),
        rl_dir=rl_dir,
        wandb_run=wandb_run,
        step=global_step,
    )

    # ── Model pointer ─────────────────────────────────────────────────────────
    _save_rl_model_pointer(
        project_root=project_root,
        run_name=run_name,
        best_step=best_step,
        best_val_macro_f1=best_val_macro_f1,
        best_model_path=best_model_sampler_path,
        final_model_path=str(final_sampler_path),
        final_state_path=str(final_state_path),
    )

    # ── Wrap up ───────────────────────────────────────────────────────────────
    (rl_dir / "final_paths.json").write_text(
        json.dumps({
            "final_state_path": str(final_state_path),
            "final_sampler_path": str(final_sampler_path),
            "best_step": best_step,
            "best_val_macro_f1": float(best_val_macro_f1),
            "best_model_sampler_path": best_model_sampler_path,
        }, indent=2),
        encoding="utf-8",
    )

    if wandb_run is not None:
        wandb_run.finish()

    logger.info(
        "RL training complete. Best: macro_f1={:.4f} @ step {}. "
        "Final: {} | Best: {}",
        best_val_macro_f1, best_step, final_sampler_path, best_model_sampler_path,
    )


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user.")
        sys.exit(130)
