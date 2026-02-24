#!/usr/bin/env python3
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
from typing import Any

import numpy as np
import pandas as pd
import tinker
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
RL_SUBDIR = "rl_phase"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "GRPO-style RL with Tinker.\n\n"
            "Reward: +1 if emitted label matches gold, else -1.\n"
            "Optional KL penalty: reward shaping that penalizes deviation from a reference model.\n"
            "Val monitoring: logprob-based P(1)/P(0) eval with a decision threshold calibrated on val to max macro-F1."
        )
    )
    p.add_argument("--config", type=str, default="tinker/configs/reasoning_sft.example.json")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--run-name", type=str, default=None)
    p.add_argument("--dataset-name", "--dataset_name", dest="dataset_name", type=str, default=None)
    p.add_argument("--dataset-root-dir", "--dataset_root_dir", dest="dataset_root_dir", type=str, default=None)
    p.add_argument("--rules-root-dir", "--rules_root_dir", dest="rules_root_dir", type=str, default=None)

    # Model init / reference model (for KL penalty)
    p.add_argument(
        "--init-model-path",
        type=str,
        default=None,
        help="Model path to start RL from (e.g. tinker://.../sampler_weights/...). If omitted, tries tinker/model/latest.json.",
    )
    p.add_argument(
        "--init-state-path",
        type=str,
        default=None,
        help="Optional. If set, initializes the RL TrainingClient weights via TrainingClient.load_state(path).",
    )
    p.add_argument(
        "--init-checkpoint",
        choices=["best", "final"],
        default="final",
        help="If --init-model-path is omitted, choose which pointer from tinker/model/latest.json to use.",
    )
    p.add_argument(
        "--kl-ref-model-path",
        type=str,
        default=None,
        help="Reference model path for KL penalty (defaults to init model).",
    )

    # RL training knobs
    p.add_argument("--lora-rank", type=int, default=None)
    p.add_argument("--num-epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument(
        "--loss-fn",
        type=str,
        default="ppo",
        choices=["importance_sampling", "ppo", "cispo"],
        help="RL loss function to use for forward_backward().",
    )
    p.add_argument(
        "--ppo-clip-coef",
        type=float,
        default=0.2,
        help="PPO clip coefficient (passed via loss_fn_config when --loss-fn=ppo).",
    )
    p.add_argument(
        "--num-substeps",
        type=int,
        default=1,
        help="Number of optimizer updates per sampled rollout batch (reuses the same rollouts).",
    )
    p.add_argument(
        "--normalize-advantages",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If enabled, normalize advantages within each group by (r-mean)/(std+1e-8).",
    )
    p.add_argument("--group-size", type=int, default=4, help="Number of rollouts per prompt (GRPO group size).")
    p.add_argument("--max-train-examples", type=int, default=None, help="0 means use all.")
    p.add_argument("--max-val-examples", type=int, default=None, help="0 means use all.")
    p.add_argument("--learning-rate", type=float, default=1e-5)
    p.add_argument("--grad-clip-norm", type=float, default=0.0)
    p.add_argument("--weight-decay", type=float, default=0.0)

    # Sampling knobs (policy rollouts)
    p.add_argument("--rollout-max-tokens", type=int, default=256)
    p.add_argument("--rollout-temperature", type=float, default=1.0)
    p.add_argument("--rollout-top-p", type=float, default=1.0)
    p.add_argument("--rollout-top-k", type=int, default=-1)
    p.add_argument(
        "--rollout-stop",
        type=str,
        default=None,
        help="Optional stop string for rollouts (comma-separated list). Leave empty to use model defaults.",
    )
    p.add_argument(
        "--sample-max-concurrency",
        type=int,
        default=64,
        help="Max in-flight sampling requests per step (prompts).",
    )

    # KL penalty knobs
    p.add_argument(
        "--kl-beta",
        type=float,
        default=0.0,
        help="KL penalty coefficient. 0 disables KL penalty.",
    )

    # Eval/checkpoint knobs
    p.add_argument("--eval-interval", type=int, default=50)
    p.add_argument("--eval-max-concurrency", type=int, default=64)
    p.add_argument("--save-interval", type=int, default=200)
    p.add_argument("--ttl-seconds", type=int, default=7 * 24 * 3600)
    p.add_argument("--log-dir", type=str, default=None)
    return p.parse_args()


def _read_latest_pointer(project_root: Path) -> dict[str, Any] | None:
    fp = project_root / "tinker" / "model" / "latest.json"
    if not fp.exists():
        return None
    try:
        return json.loads(fp.read_text(encoding="utf-8"))
    except Exception:
        return None


def _resolve_split_paths(project_root: Path, cfg: dict[str, Any]) -> tuple[dict[str, Path], Path, str]:
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
        raise FileNotFoundError(f"Missing split file for '{split}' under {dataset_dir}. Tried: {candidates}")

    return (
        {"train": _find_split("train"), "val": _find_split("val"), "test": _find_split("test")},
        dataset_dir,
        dataset_name,
    )


def _load_split_dataset(path: Path, cfg: dict[str, Any], split_name: str) -> pd.DataFrame:
    # Try common schemas first, but allow overrides via config.
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
    raise RuntimeError(f"Failed loading split '{split_name}' from {path}. Tried: {tried}. Last error: {last_exc}")


def _parse_stop_arg(raw: str | None) -> list[str] | None:
    if raw is None:
        return None
    s = str(raw).strip()
    if not s:
        return None
    parts = [p.strip() for p in s.split(",")]
    parts = [p for p in parts if p]
    return parts or None


def _extract_emitted_label(text: str) -> tuple[int | None, str]:
    """
    Parse 0/1 from a generated completion. Returns (label, debug_source).
    """
    raw = (text or "").strip()
    if not raw:
        return None, "empty"

    # Try JSON first.
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

    # Regex fallback.
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


def _compute_generated_token_logprobs(
    sampling_client: tinker.SamplingClient,
    *,
    prompt_tokens: list[int],
    generated_tokens: list[int],
) -> list[float] | None:
    """
    Fallback when SamplingClient.sample() does not return per-token logprobs.

    Returns logprobs aligned to generated_tokens (same length), or None on failure.
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


def _eval_logprob_only(
    sampling_client: tinker.SamplingClient,
    tokenizer: Any,
    eval_rows: list[dict[str, Any]],
    *,
    max_samples: int,
    one_tokens: list[int],
    zero_tokens: list[int],
    max_concurrency: int,
) -> dict[str, Any]:
    rows_eval = eval_rows if max_samples <= 0 else eval_rows[:max_samples]
    if not rows_eval:
        return {
            "loss": 0.0,
            "cls_loss": 0.0,
            "invalid_label_count": 0,
            "invalid_label_rate": 0.0,
            "invalid_flags": [],
            "y_true": [],
            "y_pred": [],
            "p_one": [],
            "p_zero": [],
        }

    y_true: list[int] = []
    p_one_all: list[float] = []
    p_zero_all: list[float] = []
    cls_losses: list[float] = []

    in_flight: list[tuple[dict[str, Any], tuple[Any, int, int, int], tuple[Any, int, int, int]]] = []
    pending = 0

    def _consume(record: tuple[dict[str, Any], tuple[Any, int, int, int], tuple[Any, int, int, int]]) -> None:
        ex, one_req, zero_req = record
        lp1 = core.resolve_completion_logprob(*one_req)
        lp0 = core.resolve_completion_logprob(*zero_req)
        denom = float(np.logaddexp(lp1, lp0))
        p1 = float(math.exp(lp1 - denom))
        p0 = float(math.exp(lp0 - denom))
        y = int(ex["label"])
        y_true.append(y)
        p_one_all.append(p1)
        p_zero_all.append(p0)
        cls_losses.append(-(y * math.log(max(p1, 1e-12)) + (1 - y) * math.log(max(p0, 1e-12))))

    for ex in rows_eval:
        prompt_tokens = list(ex["prompt_tokens"])
        one_req = core.submit_completion_logprob(sampling_client, prompt_tokens, one_tokens)
        zero_req = core.submit_completion_logprob(sampling_client, prompt_tokens, zero_tokens)
        in_flight.append((ex, one_req, zero_req))
        pending += 1

        if pending >= max_concurrency:
            _consume(in_flight.pop(0))
            pending -= 1

    while in_flight:
        _consume(in_flight.pop(0))

    # Placeholder hard preds at 0.5; caller will override with val-calibrated threshold.
    y_pred = [1 if p >= 0.5 else 0 for p in p_one_all]
    base = core.binary_metrics(y_true, y_pred, p_one=p_one_all) if y_true else {}
    out = {
        "loss": float(np.mean(cls_losses)) if cls_losses else 0.0,
        "cls_loss": float(np.mean(cls_losses)) if cls_losses else 0.0,
        "invalid_label_count": 0,
        "invalid_label_rate": 0.0,
        "invalid_flags": [False for _ in y_true],
        "y_true": y_true,
        "y_pred": y_pred,
        "p_one": p_one_all,
        "p_zero": p_zero_all,
    }
    out.update({k: base.get(k, out.get(k)) for k in ["accuracy", "balanced_accuracy", "F1", "mcc", "precision", "recall", "tp", "fp", "fn", "tn", "auroc", "auprc"]})
    return out


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    cfg = core.load_config(project_root, args)
    prompt_cfg = core.load_prompt_config(project_root, cfg)

    if args.seed is not None:
        cfg["seed"] = int(args.seed)
    if args.dataset_name is not None:
        cfg["dataset_name"] = str(args.dataset_name)
    if args.dataset_root_dir is not None:
        cfg["dataset_root_dir"] = str(args.dataset_root_dir)
    if args.rules_root_dir is not None:
        cfg["rules_root_dir"] = str(args.rules_root_dir)
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
    log_dir = core.resolve_path(project_root, str(cfg.get("log_dir", "tinker/runs")))
    if log_dir is None:
        raise RuntimeError("log_dir resolution failed")
    run_dir = log_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    rl_dir = run_dir / RL_SUBDIR
    rl_dir.mkdir(parents=True, exist_ok=True)
    core.setup_logger(rl_dir)

    (run_dir / "resolved_config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    (rl_dir / "resolved_args.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    logger.info("Run dir: {}", run_dir)
    logger.info("RL dir: {}", rl_dir)

    init_model_path = args.init_model_path
    init_state_path: str | None = None
    if args.init_state_path:
        init_state_path = str(args.init_state_path).strip() or None
    if not init_model_path:
        latest = _read_latest_pointer(project_root)
        if latest is None:
            raise RuntimeError(
                "Missing --init-model-path and could not read tinker/model/latest.json. "
                "Pass --init-model-path tinker://.../sampler_weights/...."
            )
        # Model pointers are written by SFT_reasoning_student.py
        model_key = "best_model_path" if args.init_checkpoint == "best" else "final_model_path"
        init_model_path = str(latest.get(model_key) or "")
        if not init_model_path:
            raise RuntimeError(f"tinker/model/latest.json missing {model_key}")
        # For the final checkpoint we also have a resumable state (weights + optimizer).
        if args.init_checkpoint == "final":
            init_state_path = str(latest.get("resume_state_path") or "").strip() or None
        elif args.init_checkpoint == "best" and init_state_path is None:
            if isinstance(init_model_path, str) and init_model_path.startswith("tinker://") and "/sampler_weights/" in init_model_path:
                logger.warning(
                    "Init checkpoint is 'best' but no init_state_path is available. "
                    "Will try base_model='{}' (may not be supported). "
                    "For a reliable init, use --init-checkpoint final or provide --init-state-path.",
                    init_model_path,
                )

    kl_ref_model_path = args.kl_ref_model_path or init_model_path

    split_paths, dataset_dir, dataset_name = _resolve_split_paths(project_root, cfg)
    train_df = _load_split_dataset(split_paths["train"], cfg, "train")
    val_df = _load_split_dataset(split_paths["val"], cfg, "val")
    test_df = _load_split_dataset(split_paths["test"], cfg, "test")

    train_df = core.stratified_subset(train_df, int(cfg["max_train_examples"]), seed=seed, split_name="train")
    val_df = core.stratified_subset(val_df, int(cfg["max_val_examples"]), seed=seed + 1, split_name="val")

    logger.info(
        "Dataset '{}' | train={} val={} test={} (test is loaded but not used for reward or threshold).",
        dataset_name,
        len(train_df),
        len(val_df),
        len(test_df),
    )

    rules_root = core.resolve_path(project_root, str(cfg["rules_root_dir"]))
    if rules_root is None:
        raise RuntimeError("rules_root_dir resolution failed")
    rules_dir = rules_root / dataset_name
    rulebook, rule_files = core.read_rulebook(rules_dir, str(cfg["rules_glob"]))
    (rl_dir / "rulebook.txt").write_text(rulebook, encoding="utf-8")
    logger.info("Loaded {} rule files from {}", len(rule_files), rules_dir)

    # Service + training client
    service_client = (
        tinker.ServiceClient(base_url=str(cfg["tinker_base_url"]))
        if cfg.get("tinker_base_url")
        else tinker.ServiceClient()
    )
    if init_state_path is not None:
        # Most reliable: load a full saved trainer state from SFT (weights + optimizer state).
        # This avoids depending on whether Tinker supports using sampler_weights as base_model.
        training_client = service_client.create_lora_training_client(
            base_model=str(cfg["student_model_name"]),
            rank=int(cfg["lora_rank"]),
            seed=seed,
        )
        training_client.load_state(init_state_path).result()
        logger.info("Initialized RL training weights from resume_state_path={}", init_state_path)
    else:
        # Best-effort fallback: try to start from a sampler_weights path directly.
        training_client = service_client.create_lora_training_client(
            base_model=str(init_model_path),
            rank=int(cfg["lora_rank"]),
            seed=seed,
        )
        logger.info("Initialized RL training weights from init_model_path={}", init_model_path)
    tokenizer = training_client.get_tokenizer()

    base_sampling_client: tinker.SamplingClient | None = None
    if float(args.kl_beta) > 0.0:
        # Support both tinker:// checkpoint URIs and base model names.
        ref = str(kl_ref_model_path)
        if ref.startswith("tinker://"):
            base_sampling_client = service_client.create_sampling_client(model_path=ref)
        else:
            base_sampling_client = service_client.create_sampling_client(base_model=ref)

    reasoning_prompt_builder = lambda t: core.build_reasoning_user_prompt(prompt_cfg, rulebook, t)

    stop_list = _parse_stop_arg(args.rollout_stop)
    sampling_params = tinker.SamplingParams(
        max_tokens=int(args.rollout_max_tokens),
        seed=seed,
        stop=stop_list,
        temperature=float(args.rollout_temperature),
        top_k=int(args.rollout_top_k),
        top_p=float(args.rollout_top_p),
    )

    one_tokens = tokenizer.encode("1", add_special_tokens=False)
    zero_tokens = tokenizer.encode("0", add_special_tokens=False)
    if not one_tokens or not zero_tokens:
        raise RuntimeError("Failed to tokenize '1'/'0' for label probing")

    # Pre-tokenize eval prompts once (val thresholding/eval uses logprob-only probing).
    val_eval_rows = core.build_eval_rows(
        val_df,
        tokenizer=tokenizer,
        max_length=int(cfg["max_length"]),
        reasoning_prompt_builder=reasoning_prompt_builder,
        reasoning_placeholder=str(cfg["eval_reasoning_placeholder"]),
    )

    # Pre-tokenize RL prompts (open generation prefix).
    train_prompts: list[dict[str, Any]] = []
    for i, row in train_df.reset_index(drop=True).iterrows():
        text = str(row["text"])
        label = int(row["label"])
        # Fit the *prompt* to max_length by reserving a tiny dummy answer ("0").
        # This avoids sampling failures when raw texts are long.
        fit = core.fit_user_prompt_and_answer_to_max_length(
            tokenizer,
            user_builder=reasoning_prompt_builder,
            text=text,
            assistant_content="0",
            max_length=int(cfg["max_length"]),
        )
        if fit is None:
            continue
        _, prompt_tokens, _ = fit
        train_prompts.append({"idx": int(i), "text": text, "label": label, "prompt_tokens": prompt_tokens})

    if not train_prompts:
        raise RuntimeError("No training prompts available after preprocessing")

    steps_per_epoch = int(math.ceil(len(train_prompts) / max(int(args.batch_size), 1)))
    total_steps = steps_per_epoch * int(args.num_epochs)
    logger.info(
        "RL schedule: epochs={} batch_size={} steps_per_epoch={} total_steps={} group_size={}",
        int(args.num_epochs),
        int(args.batch_size),
        steps_per_epoch,
        total_steps,
        int(args.group_size),
    )

    history_path = rl_dir / "rl_history.jsonl"
    if history_path.exists():
        history_path.unlink()

    adam_params = tinker.AdamParams(
        learning_rate=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
        grad_clip_norm=float(args.grad_clip_norm),
    )

    global_step = 0
    for epoch in range(int(args.num_epochs)):
        rng = random.Random(seed + epoch)
        rng.shuffle(train_prompts)

        for b in range(0, len(train_prompts), int(args.batch_size)):
            batch = train_prompts[b : b + int(args.batch_size)]
            global_step += 1

            # Create a sampling client for current policy weights.
            sampling_client = training_client.save_weights_and_get_sampling_client()

            # Submit sample requests for each prompt (limited in-flight).
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
                    logger.warning("sample() failed for idx={} error={}", ex.get("idx"), exc)
                    continue
                sample_futs.append((ex, fut))
                in_flight += 1

                if in_flight >= int(args.sample_max_concurrency):
                    # Wait for the oldest in-flight request to finish (but keep it in the list).
                    # This enforces backpressure without dropping any prompts.
                    oldest_idx = len(sample_futs) - in_flight
                    try:
                        _ = sample_futs[oldest_idx][1].result()
                    except Exception:
                        # We'll handle the exception when consuming this future below.
                        pass
                    in_flight -= 1

            # Collect samples.
            all_datums: list[tinker.Datum] = []
            datum_meta: list[dict[str, int]] = []
            batch_rewards: list[float] = []
            batch_correct: list[int] = []
            batch_invalid: list[int] = []

            # KL penalty bookkeeping
            kl_logprob_diffs: list[float] = []

            for ex, fut in sample_futs:
                try:
                    resp: tinker.SampleResponse = fut.result()
                except Exception as exc:
                    logger.warning("Sampling request failed for idx={} error={}", ex.get("idx"), exc)
                    continue
                sequences = list(resp.sequences or [])
                if not sequences:
                    continue

                # Compute rewards per rollout (group) for this prompt.
                rewards: list[float] = []
                per_seq: list[dict[str, Any]] = []
                for seq in sequences:
                    tokens = list(seq.tokens or [])
                    if not tokens:
                        continue
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
                        continue
                    text = tokenizer.decode(tokens, skip_special_tokens=False) if tokens else ""
                    pred, src = _extract_emitted_label(text)
                    valid = pred in {0, 1}
                    correct = int(valid and int(pred) == int(ex["label"]))
                    reward = 1.0 if correct else -1.0
                    rewards.append(reward)
                    per_seq.append(
                        {
                            "tokens": tokens,
                            "logprobs": lps,
                            "text": text,
                            "pred": pred,
                            "pred_src": src,
                            "valid": bool(valid),
                            "correct": int(correct),
                            "reward": float(reward),
                        }
                    )

                # GRPO-style group centering.
                mean_reward = float(np.mean(rewards)) if rewards else 0.0
                std_reward = float(np.std(rewards)) if rewards else 0.0
                if args.normalize_advantages:
                    denom = std_reward + 1e-8
                    advantages = [float((r - mean_reward) / denom) for r in rewards]
                else:
                    advantages = [float(r - mean_reward) for r in rewards]

                # Build datums for importance_sampling loss.
                prompt_tokens = list(ex["prompt_tokens"])
                prompt_input = tinker.ModelInput.from_ints(prompt_tokens)
                ob_len = prompt_input.length - 1  # aligns with the cookbook pattern

                for seq_i, seq_info in enumerate(per_seq):
                    sampled_tokens = list(seq_info["tokens"])
                    sampled_lps = list(seq_info["logprobs"])
                    if len(sampled_tokens) < 1 or len(sampled_lps) != len(sampled_tokens):
                        continue

                    # Exclude last token from model input; keep it as the final target.
                    model_input = prompt_input.append(tinker.EncodedTextChunk(tokens=sampled_tokens[:-1]))

                    target_tokens = [0] * ob_len + sampled_tokens
                    padded_logprobs = [0.0] * ob_len + sampled_lps
                    adv = float(advantages[seq_i])
                    padded_advantages = [0.0] * ob_len + [adv] * len(sampled_tokens)

                    datum = tinker.Datum(
                        model_input=model_input,
                        loss_fn_inputs={
                            "target_tokens": tinker.TensorData.from_numpy(np.asarray(target_tokens, dtype=np.int32)),
                            "logprobs": tinker.TensorData.from_numpy(np.asarray(padded_logprobs, dtype=np.float32)),
                            "advantages": tinker.TensorData.from_numpy(np.asarray(padded_advantages, dtype=np.float32)),
                        },
                    )
                    all_datums.append(datum)
                    datum_meta.append({"ob_len": int(ob_len)})
                    batch_rewards.append(float(seq_info["reward"]))
                    batch_correct.append(int(seq_info["correct"]))
                    batch_invalid.append(0 if seq_info["valid"] else 1)

            if not all_datums:
                logger.warning("Step {}: no usable datums (sampling failed or parsing failed)", global_step)
                continue

            # Optional KL penalty (reward shaping) against a reference model.
            if base_sampling_client is not None and float(args.kl_beta) > 0.0:
                # Compute base logprobs for each full sequence (prompt + completion + final target token).
                base_futs: list[Any] = []
                for datum in all_datums:
                    tt = datum.loss_fn_inputs["target_tokens"].data
                    final_tok = int(tt[-1])
                    full_seq = datum.model_input.append_int(final_tok)
                    base_futs.append(base_sampling_client.compute_logprobs(full_seq))

                base_logprobs_list: list[list[float | None]] = [f.result() for f in base_futs]

                # Compute centered logprob diffs on masked action positions (GRPO-style centering).
                diffs_all: list[float] = []
                for datum, base_lp, meta in zip(all_datums, base_logprobs_list, datum_meta):
                    sampled_lp = datum.loss_fn_inputs["logprobs"].data
                    # Align base logprobs to model_input positions (skip first token like cookbook).
                    base_aligned = base_lp[1 : 1 + len(sampled_lp)]
                    if len(base_aligned) != len(sampled_lp):
                        continue
                    ob_len = int(meta.get("ob_len", 0))
                    for j, (s, b) in enumerate(zip(sampled_lp, base_aligned)):
                        if j < ob_len:
                            continue
                        if b is None or not math.isfinite(float(b)) or not math.isfinite(float(s)):
                            continue
                        diffs_all.append(float(s) - float(b))

                avg_diff = float(np.mean(diffs_all)) if diffs_all else 0.0

                # Update advantages in-place.
                for datum, base_lp, meta in zip(all_datums, base_logprobs_list, datum_meta):
                    sampled_lp = datum.loss_fn_inputs["logprobs"].data
                    adv = datum.loss_fn_inputs["advantages"].data
                    base_aligned = base_lp[1 : 1 + len(sampled_lp)]
                    if len(base_aligned) != len(sampled_lp):
                        continue
                    ob_len = int(meta.get("ob_len", 0))
                    for i, (s, b) in enumerate(zip(sampled_lp, base_aligned)):
                        if i < ob_len:
                            continue
                        if b is None or not math.isfinite(float(b)) or not math.isfinite(float(s)):
                            continue
                        diff = float(s) - float(b)
                        # Reward shaping term: +beta * (avg_diff - diff), centered across batch.
                        adv[i] = float(adv[i]) + float(args.kl_beta) * (avg_diff - diff)
                        kl_logprob_diffs.append(diff)

            if int(args.num_substeps) <= 0:
                raise ValueError("num_substeps must be >= 1")

            loss_fn_config: dict[str, float] | None = None
            if str(args.loss_fn) in {"ppo", "cispo"}:
                # Tinker expects explicit thresholds for ratio clipping.
                # See: https://tinker-docs.thinkingmachines.ai/losses
                eps = float(args.ppo_clip_coef)
                loss_fn_config = {
                    "clip_low_threshold": 1.0 - eps,
                    "clip_high_threshold": 1.0 + eps,
                }

            fb_out = None
            for _substep in range(int(args.num_substeps)):
                fb_out = training_client.forward_backward(
                    all_datums,
                    loss_fn=str(args.loss_fn),
                    loss_fn_config=loss_fn_config,
                ).result()
                _ = training_client.optim_step(adam_params=adam_params).result()

            # Step logging
            mean_reward = float(np.mean(batch_rewards)) if batch_rewards else 0.0
            acc = float(np.mean(batch_correct)) if batch_correct else 0.0
            invalid_rate = float(np.mean(batch_invalid)) if batch_invalid else 0.0
            kl_mean = float(np.mean(kl_logprob_diffs)) if kl_logprob_diffs else 0.0

            row = {
                "step": int(global_step),
                "epoch": int(epoch),
                "loss": float(getattr(fb_out, "loss", 0.0) or 0.0) if fb_out is not None else 0.0,
                "mean_reward": mean_reward,
                "batch_acc": acc,
                "invalid_rate": invalid_rate,
                "kl_logprob_diff_mean": kl_mean,
                "kl_beta": float(args.kl_beta),
                "loss_fn": str(args.loss_fn),
                "num_substeps": int(args.num_substeps),
                "n_datums": int(len(all_datums)),
            }
            _append_jsonl(history_path, row)

            if global_step == 1 or global_step % 10 == 0:
                logger.info(
                    "Step {} | epoch {} | loss={:.6f} reward={:.3f} acc={:.3f} invalid={:.3f} kl_diff_mean={:.4f}",
                    global_step,
                    epoch,
                    row["loss"],
                    mean_reward,
                    acc,
                    invalid_rate,
                    kl_mean,
                )

            # Periodic val monitoring (logprob-only eval + val-calibrated threshold).
            if int(args.eval_interval) > 0 and global_step % int(args.eval_interval) == 0:
                eval_sampling_client = training_client.save_weights_and_get_sampling_client(
                    name=f"rl_step_{global_step}_eval"
                )
                val_eval = _eval_logprob_only(
                    eval_sampling_client,
                    tokenizer=tokenizer,
                    eval_rows=val_eval_rows,
                    max_samples=0,
                    one_tokens=one_tokens,
                    zero_tokens=zero_tokens,
                    max_concurrency=int(args.eval_max_concurrency),
                )
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
                (rl_dir / f"metrics_val_step_{global_step}.json").write_text(
                    json.dumps(
                        {
                            "decision_threshold_rule": "val_max_macro_f1",
                            "decision_threshold": float(decision_threshold),
                            **{k: float(v) if isinstance(v, (int, float)) else v for k, v in val_thr.items()},
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
                    global_step,
                    float(decision_threshold),
                    float(val_thr.get("accuracy", 0.0)),
                    float(val_thr.get("macro_f1", 0.0)),
                    float(val_eval.get("auroc", float("nan"))),
                    float(val_eval.get("auprc", float("nan"))),
                )

            # Periodic checkpointing for sampler/resume.
            if int(args.save_interval) > 0 and global_step % int(args.save_interval) == 0:
                ckpt_name = f"rl_step_{global_step}"
                state_path = training_client.save_state(name=ckpt_name, ttl_seconds=int(args.ttl_seconds)).result().path
                sampler_path = (
                    training_client.save_weights_for_sampler(
                        name=ckpt_name,
                        ttl_seconds=int(args.ttl_seconds),
                    )
                    .result()
                    .path
                )
                (rl_dir / "checkpoints.jsonl").open("a", encoding="utf-8").write(
                    json.dumps(
                        {
                            "step": int(global_step),
                            "state_path": str(state_path),
                            "sampler_path": str(sampler_path),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                logger.info("Saved checkpoint {} | state={} | sampler={}", ckpt_name, state_path, sampler_path)

    # Final checkpoint
    final_name = "rl_final"
    final_state_path = training_client.save_state(name=final_name, ttl_seconds=int(args.ttl_seconds)).result().path
    final_sampler_path = (
        training_client.save_weights_for_sampler(name=final_name, ttl_seconds=int(args.ttl_seconds)).result().path
    )
    (rl_dir / "final_paths.json").write_text(
        json.dumps({"final_state_path": final_state_path, "final_sampler_path": final_sampler_path}, indent=2),
        encoding="utf-8",
    )
    logger.info("RL complete | final_state_path={} | final_sampler_path={}", final_state_path, final_sampler_path)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.warning("Interrupted.")
        sys.exit(130)
