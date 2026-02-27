#!/usr/bin/env bash
# =============================================================================
# run_rl.sh — GRPO RL training for hate speech classification (Ethos dataset)
#
# Usage:
#   cd /home/zhongmouhe/p2/tinker-pipeline
#   bash run_rl.sh
#
# Override any parameter by setting the variable before running, e.g.:
#   GROUP_SIZE=16 NUM_EPOCHS=2 bash run_rl.sh
#
# The script loads the SFT checkpoint from model/latest.json automatically.
# Results are saved under runs/<run_name>/rl_phase/.
# WandB metrics are logged to project "reasoning-rl-tinker".
# =============================================================================
set -euo pipefail

export WANDB_API_KEY="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"

# ── Run identification ────────────────────────────────────────────────────────
# Auto-generate a timestamped name, or pass RUN_NAME=<name> to fix it.
RUN_NAME="${RUN_NAME:-grpo_rl_$(date +%Y%m%d_%H%M%S)}"

# ── RL training hyperparameters ───────────────────────────────────────────────
# group_size: number of rollouts per prompt for GRPO centering.
#   Higher = better gradient estimates, more GPU compute per step.
GROUP_SIZE="${GROUP_SIZE:-8}"

# batch_size: number of prompts per training step.
#   With 700 train examples: ~88 steps/epoch.
BATCH_SIZE="${BATCH_SIZE:-8}"

# num_epochs: passes over the training set.
#   With early stopping (patience=5), training often stops before this.
NUM_EPOCHS="${NUM_EPOCHS:-3}"

# learning_rate: LoRA LR for RL. Typically 5-10x lower than SFT LR.
LEARNING_RATE="${LEARNING_RATE:-1e-5}"

# loss function: ppo (recommended), importance_sampling, or cispo.
LOSS_FN="${LOSS_FN:-ppo}"

# ppo_clip_coef: PPO ε clipping ratio [1-ε, 1+ε]. 0.2 is standard.
PPO_CLIP="${PPO_CLIP:-0.2}"

# ── KL penalty (optional) ─────────────────────────────────────────────────────
# Set > 0 to penalize drift from the SFT checkpoint. 0 disables.
# Recommended range: 0.01 – 0.1. Start with 0 to see how much RL helps.
KL_BETA="${KL_BETA:-0.0}"

# ── Thinking reward (optional) ────────────────────────────────────────────────
# Additive bonus = coef * min(1.0, reasoning_chars / 500).
# 0 disables. Try 0.1 to gently encourage longer reasoning.
THINKING_REWARD_COEF="${THINKING_REWARD_COEF:-0.0}"

# ── Rollout sampling ──────────────────────────────────────────────────────────
# max tokens to generate per rollout. Must accommodate full JSON output.
ROLLOUT_MAX_TOKENS="${ROLLOUT_MAX_TOKENS:-512}"
# temperature for rollout diversity. 1.0 = no scaling.
ROLLOUT_TEMPERATURE="${ROLLOUT_TEMPERATURE:-1.0}"

# ── Evaluation and checkpointing ─────────────────────────────────────────────
# Run val/hard/easy eval every N training steps.
EVAL_INTERVAL="${EVAL_INTERVAL:-20}"
# Early stopping: stop after N consecutive evals without val macro-F1 improvement.
EARLY_STOPPING_PATIENCE="${EARLY_STOPPING_PATIENCE:-5}"
# Save a full checkpoint every N steps (0 to disable periodic saves).
SAVE_INTERVAL="${SAVE_INTERVAL:-100}"

# ── Optimizer ─────────────────────────────────────────────────────────────────
GRAD_CLIP_NORM="${GRAD_CLIP_NORM:-1.0}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
NUM_SUBSTEPS="${NUM_SUBSTEPS:-1}"

# ── SFT checkpoint selection ──────────────────────────────────────────────────
# "final" = use final_model_path + resume_state_path from model/latest.json (recommended).
# "best"  = use best_model_path (highest val macro-F1 during SFT).
INIT_CHECKPOINT="${INIT_CHECKPOINT:-final}"

# ── Directories ───────────────────────────────────────────────────────────────
# Must be run from the tinker-pipeline project root.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "============================================================"
echo "  GRPO RL Training"
echo "  Run name  : $RUN_NAME"
echo "  Epochs    : $NUM_EPOCHS"
echo "  Batch size: $BATCH_SIZE"
echo "  Group size: $GROUP_SIZE"
echo "  LR        : $LEARNING_RATE"
echo "  Loss fn   : $LOSS_FN"
echo "  KL beta   : $KL_BETA"
echo "  Eval every: $EVAL_INTERVAL steps"
echo "  Patience  : $EARLY_STOPPING_PATIENCE evals"
echo "  Init from : model/latest.json ($INIT_CHECKPOINT checkpoint)"
echo "============================================================"

python3 RL.py \
    --run-name                   "$RUN_NAME"                \
    --init-checkpoint            "$INIT_CHECKPOINT"          \
    \
    --num-epochs                 "$NUM_EPOCHS"               \
    --batch-size                 "$BATCH_SIZE"               \
    --group-size                 "$GROUP_SIZE"               \
    --learning-rate              "$LEARNING_RATE"            \
    --loss-fn                    "$LOSS_FN"                  \
    --ppo-clip-coef              "$PPO_CLIP"                 \
    --num-substeps               "$NUM_SUBSTEPS"             \
    --normalize-advantages                                   \
    \
    --kl-beta                    "$KL_BETA"                  \
    --thinking-reward-coef       "$THINKING_REWARD_COEF"     \
    \
    --rollout-max-tokens         "$ROLLOUT_MAX_TOKENS"       \
    --rollout-temperature        "$ROLLOUT_TEMPERATURE"      \
    \
    --eval-interval              "$EVAL_INTERVAL"            \
    --early-stopping-patience    "$EARLY_STOPPING_PATIENCE"  \
    --save-interval              "$SAVE_INTERVAL"            \
    \
    --grad-clip-norm             "$GRAD_CLIP_NORM"           \
    --weight-decay               "$WEIGHT_DECAY"             \
    \
    --eval-max-concurrency       64                          \
    --sample-max-concurrency     64                          \
    --rollout-top-k              -1                          \
    --rollout-top-p              1.0

echo "============================================================"
echo "  Training complete. Results in: runs/$RUN_NAME/rl_phase/"
echo "  Model pointer: model/rl_latest.json"
echo "============================================================"
