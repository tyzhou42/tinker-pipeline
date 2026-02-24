#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from datetime import datetime
from pathlib import Path

from tqdm import tqdm
import numpy as np
import pandas as pd
import tinker
import wandb
import yaml
from loguru import logger
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

DEFAULT_PROMPTS = {
    "instruction": "Decide whether the following text is hate speech.",
    "user_prompt_template": "{instruction}\nText: {text}",
}
DEFAULT_PROMPTS_PATH = Path(__file__).resolve().parent / "prompts" / "sft.yaml"
DEFAULT_DATA_PATH = "/u/zhe3/fin/hate_speech_detection_tinker/data/Ethos_Dataset_Binary.csv"
DEFAULT_PRICES_JSON = "/u/zhe3/fin/hate_speech_detection_tinker/prices.json"


def binary_metrics(y_true: list[int], y_pred: list[int]) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
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


def load_prompt_config(path: Path) -> dict[str, str]:
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")

    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if loaded is None:
        loaded = {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Prompt file must contain a YAML object: {path}")

    cfg = dict(DEFAULT_PROMPTS)
    for key in DEFAULT_PROMPTS:
        if key in loaded and loaded[key] is not None:
            cfg[key] = str(loaded[key])

    return cfg


def make_messages(comment: str, label: int, prompt_cfg: dict[str, str]) -> list[dict[str, str]]:
    user_content = prompt_cfg["user_prompt_template"].format(
        instruction=prompt_cfg["instruction"],
        text=comment,
    )
    return [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": "Yes" if label == 1 else "No"},
    ]


def build_examples(frame: pd.DataFrame, tokenizer, max_length: int, prompt_cfg: dict[str, str]) -> list[dict]:
    rows: list[dict] = []
    dropped = 0
    for _, row in frame.iterrows():
        comment = str(row["comment"])
        label = int(row["label"])

        encoded_comment = tokenizer.encode(comment, add_special_tokens=False)
        encoded_comment = encoded_comment[: max(1, max_length - 10)]  # leave space for answer tokens
        comment = tokenizer.decode(encoded_comment)

        messages = make_messages(comment, label, prompt_cfg)
        encoded_messages = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
        )["input_ids"]
        encoded_answer = tokenizer.encode(messages[1]["content"], add_special_tokens=False)

        def find_sublist_indices(A, B):
            n, m = len(A), len(B)
            res = []
            for i in range(n - m + 1):
                if A[i:i + m] == B:
                    res.append(i)
            return res

        answer_matches = find_sublist_indices(encoded_messages, encoded_answer)
        answer_idx = answer_matches[-1]
        prompt_tokens = encoded_messages[:answer_idx]
        answer_tokens = encoded_answer
        full_tokens = prompt_tokens + answer_tokens

        if len(full_tokens) < 2:
            dropped += 1
            continue

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
                "comment": comment,
                "label": label,
                "messages": messages,
                "prompt_tokens": prompt_tokens,
                "datum": datum,
            }
        )

    if dropped > 0:
        logger.warning("Dropped {} rows due to max_length limits", dropped)
    if not rows:
        raise RuntimeError("No usable rows after tokenization")
    return rows


def submit_completion_logprob(
        sampling_client: tinker.SamplingClient,
        prompt_tokens: list[int],
        answer_tokens: list[int],
):
    seq = prompt_tokens + answer_tokens
    fut = sampling_client.compute_logprobs(tinker.ModelInput.from_ints(seq))
    start = len(prompt_tokens)
    end = start + len(answer_tokens)
    return fut, start, end, len(answer_tokens)


def resolve_completion_logprob(fut, start: int, end: int, answer_len: int) -> float:
    logprobs = fut.result()
    answer_logprobs = logprobs[start:end]
    if len(answer_logprobs) != answer_len:
        raise ValueError(
            f"compute_logprobs length mismatch: got {len(answer_logprobs)} answer logprobs, expected {answer_len}"
        )

    total = 0.0
    for lp in answer_logprobs:
        if lp is None:
            logger.error("Received None logprob for tokens {} to {}, treating as -1e9", start, end)
            total += -1e9
        else:
            total += float(lp)
    return total


def evaluate(
        sampling_client: tinker.SamplingClient,
        rows: list[dict],
        max_samples: int,
        yes_tokens: list[int],
        no_tokens: list[int],
        max_concurrency: int,
) -> dict:
    if max_concurrency <= 0:
        raise ValueError("max_concurrency must be > 0")

    rows_eval = rows if max_samples <= 0 else rows[:max_samples]
    y_true: list[int] = []
    y_pred: list[int] = []
    cls_losses: list[float] = []
    total_nll = 0.0
    total_tokens = 0.0

    in_flight: list[tuple[dict, tuple, tuple]] = []

    def consume_record(record: tuple[dict, tuple, tuple]) -> None:
        ex, yes_req, no_req = record
        lp_yes = resolve_completion_logprob(*yes_req)
        lp_no = resolve_completion_logprob(*no_req)
        pred = 1 if lp_yes >= lp_no else 0
        true = int(ex["label"])

        y_true.append(true)
        y_pred.append(pred)
        true_lp = lp_yes if true == 1 else lp_no
        cls_losses.append(float(np.logaddexp(lp_yes, lp_no) - true_lp))
        answer_len = len(yes_tokens) if true == 1 else len(no_tokens)
        nonlocal total_nll, total_tokens
        total_nll += float(-true_lp)
        total_tokens += float(answer_len)

    for ex in rows_eval:
        yes_req = submit_completion_logprob(sampling_client, ex["prompt_tokens"], yes_tokens)
        no_req = submit_completion_logprob(sampling_client, ex["prompt_tokens"], no_tokens)
        in_flight.append((ex, yes_req, no_req))

        # Each record holds 2 outstanding requests (Yes/No).
        while len(in_flight) * 2 >= max_concurrency:
            consume_record(in_flight.pop(0))

    while in_flight:
        consume_record(in_flight.pop(0))

    return {
        "loss": float(total_nll / total_tokens),
        "cls_loss": float(np.mean(cls_losses)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=[0, 1], zero_division=0)),
        "y_true": y_true,
        "y_pred": y_pred,
    }


def estimate_cost(
        args: argparse.Namespace,
        train_rows: list[dict],
        val_rows: list[dict],
        test_rows: list[dict],
        yes_tokens: list[int],
        no_tokens: list[int],
        total_steps: int,
        price: dict,
) -> dict:
    def eval_prefill_tokens(rows: list[dict]) -> int:
        total = 0
        for ex in rows:
            p = ex["prompt_tokens"]
            total += len(p[: args.max_length - len(yes_tokens)]) + len(yes_tokens)
            total += len(p[: args.max_length - len(no_tokens)]) + len(no_tokens)
        return total

    train_tokens_per_epoch = sum(len(ex["datum"].model_input.to_ints()) for ex in train_rows)
    train_tokens_total = train_tokens_per_epoch * args.num_epochs

    train_eval_prefill = eval_prefill_tokens(train_rows) * args.num_epochs
    val_for_eval = val_rows if args.max_eval_samples <= 0 else val_rows[: args.max_eval_samples]
    test_for_eval = test_rows if args.max_eval_samples <= 0 else test_rows[: args.max_eval_samples]
    eval_steps = [s for s in range(total_steps) if (s % args.eval_interval == 0) or (s == total_steps - 1)]
    periodic_prefill = (eval_prefill_tokens(val_for_eval) + eval_prefill_tokens(test_for_eval)) * len(eval_steps)
    final_prefill = eval_prefill_tokens(test_rows) * 2

    prefill_tokens = train_eval_prefill + periodic_prefill + final_prefill
    sample_tokens = 0
    prefill_cost = prefill_tokens / 1_000_000 * float(price["prefill"])
    sample_cost = sample_tokens / 1_000_000 * float(price["sample"])
    train_cost = train_tokens_total / 1_000_000 * float(price["train"])

    return {
        "prefill_tokens": int(prefill_tokens),
        "sample_tokens": int(sample_tokens),
        "train_tokens": int(train_tokens_total),
        "prefill_cost_usd": float(prefill_cost),
        "sample_cost_usd": float(sample_cost),
        "train_cost_usd": float(train_cost),
        "total_cost_usd": float(prefill_cost + sample_cost + train_cost),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ethos hate-speech SFT with Tinker LoRA")
    p.add_argument("--data-path", type=str, default=DEFAULT_DATA_PATH)
    p.add_argument("--prices-json", type=str, default=DEFAULT_PRICES_JSON)
    p.add_argument("--prompts-path", type=str, default=str(DEFAULT_PROMPTS_PATH))
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--model-name", type=str, default="meta-llama/Llama-3.2-1B-Instruct")
    p.add_argument("--base-url", type=str, default=None)
    p.add_argument("--lora-rank", type=int, default=32)
    p.add_argument("--ttl-seconds", type=int, default=7 * 24 * 3600)

    p.add_argument("--num-epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--learning-rate", type=float, default=2e-4)
    p.add_argument("--adam-beta1", type=float, default=0.9)
    p.add_argument("--adam-beta2", type=float, default=0.95)
    p.add_argument("--adam-eps", type=float, default=1e-8)
    p.add_argument("--lr-schedule", choices=["constant", "linear"], default="linear")
    p.add_argument("--min-lr-ratio", type=float, default=0.1)

    p.add_argument("--max-length", type=int, default=1024)
    p.add_argument("--eval-interval", type=int, default=10)
    p.add_argument("--max-eval-samples", type=int, default=0)
    p.add_argument("--eval-max-concurrency", type=int, default=64)

    p.add_argument("--log-dir", type=str, default="/u/zhe3/fin/hate_speech_detection_tinker/runs")
    p.add_argument("--run-name", type=str, default=None)
    p.add_argument("--save-sft-jsonl", action="store_true")

    p.add_argument("--wandb-project", type=str, default="hate-speech-tinker")
    p.add_argument("--wandb-entity", type=str, default=None)
    p.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default="online")
    return p.parse_args()


def setup_logger(run_dir: Path) -> None:
    logger.remove()
    logger.add(sys.stderr, level="INFO", format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}")
    logger.add(run_dir / "train.log", level="INFO")


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be > 0")
    if args.num_epochs <= 0:
        raise ValueError("--num-epochs must be > 0")
    if args.max_length <= 1:
        raise ValueError("--max-length must be > 1")
    if args.eval_interval <= 0:
        raise ValueError("--eval-interval must be > 0")
    if args.max_eval_samples < 0:
        raise ValueError("--max-eval-samples must be >= 0")
    if args.eval_max_concurrency <= 0:
        raise ValueError("--eval-max-concurrency must be > 0")

    random.seed(args.seed)
    np.random.seed(args.seed)
    prompt_cfg = load_prompt_config(Path(args.prompts_path))

    run_name = args.run_name or f"ethos_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = Path(args.log_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    setup_logger(run_dir)
    logger.info("Run dir: {}", run_dir)

    df = pd.read_csv(args.data_path, sep=";", encoding="utf-8", encoding_errors="replace", on_bad_lines="skip")
    # df = df.sample(n=100, random_state=42)

    df = df[["comment", "isHate"]].dropna().copy()
    df["isHate"] = pd.to_numeric(df["isHate"], errors="coerce")
    df = df.dropna(subset=["isHate"])
    df["label"] = (df["isHate"] >= 0.5).astype(int)

    train_df, temp_df = train_test_split(
        df, test_size=0.30, stratify=df["label"], random_state=args.seed, shuffle=True
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, stratify=temp_df["label"], random_state=args.seed, shuffle=True
    )
    # save splits for reproducibility
    splits_dir = run_dir / "data_splits"
    splits_dir.mkdir(exist_ok=True)
    train_df.to_csv(splits_dir / "train.csv", index=False, encoding="utf-8")
    val_df.to_csv(splits_dir / "val.csv", index=False, encoding="utf-8")
    test_df.to_csv(splits_dir / "test.csv", index=False, encoding="utf-8")

    logger.info("Split sizes: train={}, val={}, test={}", len(train_df), len(val_df), len(test_df))

    client = tinker.ServiceClient(base_url=args.base_url) if args.base_url else tinker.ServiceClient()
    training_client = client.create_lora_training_client(
        base_model=args.model_name,
        rank=args.lora_rank,
    )
    tokenizer = training_client.get_tokenizer()
    if not hasattr(tokenizer, "apply_chat_template"):
        raise RuntimeError("Tokenizer does not support apply_chat_template")
    yes_tokens = tokenizer.encode("Yes", add_special_tokens=False)
    no_tokens = tokenizer.encode("No", add_special_tokens=False)
    if len(yes_tokens) == 0 or len(no_tokens) == 0:
        raise RuntimeError("Tokenizer returned empty tokens for 'Yes' or 'No'")
    logger.info(f"Yes tokens: {yes_tokens}, No tokens: {no_tokens}")

    train_rows = build_examples(train_df.reset_index(drop=True), tokenizer, args.max_length, prompt_cfg)
    val_rows = build_examples(val_df.reset_index(drop=True), tokenizer, args.max_length, prompt_cfg)
    test_rows = build_examples(test_df.reset_index(drop=True), tokenizer, args.max_length, prompt_cfg)
    logger.info("Usable rows: train={}, val={}, test={}", len(train_rows), len(val_rows), len(test_rows))

    if args.save_sft_jsonl:
        for name, rows in [("train", train_rows), ("val", val_rows), ("test", test_rows)]:
            out = run_dir / f"{name}_sft.jsonl"
            with out.open("w", encoding="utf-8") as f:
                for ex in rows:
                    f.write(json.dumps({"messages": ex["messages"], "label": ex["label"]}, ensure_ascii=False) + "\n")

    split_summary = {
        "seed": args.seed,
        "counts": {
            "train": len(train_rows),
            "val": len(val_rows),
            "test": len(test_rows),
            "total": len(train_rows) + len(val_rows) + len(test_rows),
        },
    }
    (run_dir / "split_summary.json").write_text(json.dumps(split_summary, indent=2), encoding="utf-8")

    batches_per_epoch = math.ceil(len(train_rows) / args.batch_size)
    total_steps = args.num_epochs * batches_per_epoch

    prices = json.loads(Path(args.prices_json).read_text(encoding="utf-8"))
    short = args.model_name.split("/")[-1]
    price = None
    for key in [args.model_name, short, short.replace("-Instruct-2507", ""), short.replace("-Instruct", "")]:
        if key in prices:
            price = prices[key]
            break
    if price is None:
        raise ValueError(f"Price not found for {args.model_name}. Add it to {args.prices_json}")

    cost = estimate_cost(
        args=args,
        train_rows=train_rows,
        val_rows=val_rows,
        test_rows=test_rows,
        yes_tokens=yes_tokens,
        no_tokens=no_tokens,
        total_steps=total_steps,
        price=price,
    )
    (run_dir / "cost_estimate.json").write_text(json.dumps(cost, indent=2), encoding="utf-8")

    logger.info(
        "Price for {}: prefill=${}/M, sample=${}/M, train=${}/M",
        args.model_name,
        price["prefill"],
        price["sample"],
        price["train"],
    )
    logger.info(
        "Estimated tokens -> prefill={}, sample={}, train={}",
        cost["prefill_tokens"],
        cost["sample_tokens"],
        cost["train_tokens"],
    )
    logger.info(
        "Estimated cost (USD) -> prefill={:.6f}, sample={:.6f}, train={:.6f}, total={:.6f}",
        cost["prefill_cost_usd"],
        cost["sample_cost_usd"],
        cost["train_cost_usd"],
        cost["total_cost_usd"],
    )

    wandb_run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=run_name,
        config=vars(args),
        mode=args.wandb_mode,
        dir=str(run_dir),
    )

    rng = random.Random(args.seed)
    best_val_macro_f1 = -1.0
    best_client = None
    best_model_path = ""
    best_step = -1

    step = 0
    for epoch in range(args.num_epochs):
        order = list(range(len(train_rows)))
        rng.shuffle(order)

        for i in range(0, len(order), args.batch_size):

            do_eval = (step % args.eval_interval == 0) or (step == total_steps - 1)
            if do_eval:
                step_client = training_client.save_weights_and_get_sampling_client(name=f"{run_name}_step_{step}_eval")
                val_eval = evaluate(
                    step_client,
                    val_rows,
                    max_samples=args.max_eval_samples,
                    yes_tokens=yes_tokens,
                    no_tokens=no_tokens,
                    max_concurrency=args.eval_max_concurrency,
                )
                test_eval = evaluate(
                    step_client,
                    test_rows,
                    max_samples=args.max_eval_samples,
                    yes_tokens=yes_tokens,
                    no_tokens=no_tokens,
                    max_concurrency=args.eval_max_concurrency,
                )
                info_dict = {
                    "val/loss": val_eval["loss"],
                    "val/cls_loss": val_eval["cls_loss"],
                    "val/accuracy": val_eval["accuracy"],
                    "val/macro_f1": val_eval["macro_f1"],
                    "test/loss": test_eval["loss"],
                    "test/cls_loss": test_eval["cls_loss"],
                    "test/accuracy": test_eval["accuracy"],
                    "test/macro_f1": test_eval["macro_f1"],
                }
                wandb.log(info_dict, step=step)
                logger.info(
                    "Step {} | val: loss={:.6f}, cls_loss={:.6f}, acc={:.6f}, f1={:.6f} | test: loss={:.6f}, cls_loss={:.6f}, acc={:.6f}, f1={:.6f}",
                    step,
                    float(val_eval["loss"]),
                    float(val_eval["cls_loss"]),
                    float(val_eval["accuracy"]),
                    float(val_eval["macro_f1"]),
                    float(test_eval["loss"]),
                    float(test_eval["cls_loss"]),
                    float(test_eval["accuracy"]),
                    float(test_eval["macro_f1"]),
                )

                if float(val_eval["macro_f1"]) > best_val_macro_f1:
                    best_val_macro_f1 = float(val_eval["macro_f1"])
                    best_client = step_client
                    best_model_path = 'model at step {}'.format(step)
                    best_step = step

            batch = [train_rows[j] for j in order[i: i + args.batch_size]]
            batch_data = [x["datum"] for x in batch]

            if args.lr_schedule == "constant":
                lr = args.learning_rate
            else:
                progress = step / max(total_steps - 1, 1)
                lr = args.learning_rate * (1.0 - (1.0 - args.min_lr_ratio) * progress)

            adam = tinker.AdamParams(
                learning_rate=lr,
                beta1=args.adam_beta1,
                beta2=args.adam_beta2,
                eps=args.adam_eps,
            )

            fwd = training_client.forward_backward(batch_data, loss_fn="cross_entropy")
            opt = training_client.optim_step(adam)
            fwd_result = fwd.result()
            opt.result()

            total_w_logprob = 0.0
            total_w = 0.0
            for out, ex in zip(fwd_result.loss_fn_outputs, batch, strict=True):
                lp = np.asarray(out["logprobs"].to_numpy(), dtype=np.float64).reshape(-1)
                w = np.asarray(ex["datum"].loss_fn_inputs["weights"].to_numpy(), dtype=np.float64).reshape(-1)
                if lp.shape[0] != w.shape[0]:
                    raise ValueError(f"logprobs/weights length mismatch: {lp.shape[0]} vs {w.shape[0]}")
                total_w_logprob += float(np.dot(lp, w))
                total_w += float(w.sum())
            trainer_loss = -total_w_logprob / total_w
            trainer_returned_loss = trainer_loss
            if "loss" in fwd_result.metrics:
                trainer_returned_loss = float(fwd_result.metrics["loss"])

            step_client = training_client.save_weights_and_get_sampling_client(name=f"{run_name}_step_{step}")
            train_eval = evaluate(
                step_client,
                batch,
                max_samples=0,
                yes_tokens=yes_tokens,
                no_tokens=no_tokens,
                max_concurrency=args.eval_max_concurrency,
            )
            info_dict = {
                "train/trainer_loss": trainer_returned_loss,
                "train/eval_loss": train_eval["loss"],
                "train/eval_cls_loss": train_eval["cls_loss"],
                "train/accuracy": train_eval["accuracy"],
                "train/lr": lr,
                "epoch": epoch,
            }
            wandb.log(info_dict, step=step)
            logger.info(
                "Step {} | epoch {} | train: trainer_loss={:.6f}, eval_loss={:.6f}, eval_cls_loss={:.6f}, acc={:.6f}, lr={:.6e}",
                step,
                epoch,
                float(info_dict["train/trainer_loss"]),
                float(info_dict["train/eval_loss"]),
                float(info_dict["train/eval_cls_loss"]),
                float(info_dict["train/accuracy"]),
                float(info_dict["train/lr"]),
                )

            step += 1

    final_client = training_client.save_weights_and_get_sampling_client(name=f"{run_name}_final")
    final_model_path = 'final model'

    final_eval = evaluate(
        final_client,
        test_rows,
        max_samples=0,
        yes_tokens=yes_tokens,
        no_tokens=no_tokens,
        max_concurrency=args.eval_max_concurrency,
    )
    final_metrics = binary_metrics(final_eval["y_true"], final_eval["y_pred"])

    if best_client is None:
        best_client = final_client
        best_model_path = final_model_path
        best_step = step - 1

    best_eval = evaluate(
        best_client,
        test_rows,
        max_samples=0,
        yes_tokens=yes_tokens,
        no_tokens=no_tokens,
        max_concurrency=args.eval_max_concurrency,
    )
    best_metrics = binary_metrics(best_eval["y_true"], best_eval["y_pred"])

    wandb.log(
        {
            "final_test/accuracy": final_metrics["accuracy"],
            "final_test/balanced_accuracy": final_metrics["balanced_accuracy"],
            "final_test/f1": final_metrics["F1"],
            "bestval_test/accuracy": best_metrics["accuracy"],
            "bestval_test/balanced_accuracy": best_metrics["balanced_accuracy"],
            "bestval_test/f1": best_metrics["F1"],
            "best_step": best_step,
        },
        step=step,
    )

    report_path = run_dir / "test_report.md"
    blocks = [("final model", final_model_path, final_metrics), ("best-val model", best_model_path, best_metrics)]
    lines = ["# Ethos Hate Speech Test Report", "", f"Generated at: {datetime.now().isoformat(timespec='seconds')}", ""]
    for tag, model_path, m in blocks:
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
    }
    (run_dir / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    wandb_run.finish()
    logger.info("Run summary: {}", json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
