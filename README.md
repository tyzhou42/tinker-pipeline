# Tinker (Reasoning SFT + Inference)

This folder contains an end-to-end **teacher → student (LoRA) → inference/export** pipeline for binary text classification with optional rule-grounded reasoning.

## Code map (what to run)

### Orchestration
- `tinker/run_reasoning_two_stage.sh` — primary entrypoint. Runs:
  - Stage 1 (teacher sampling + rejection) → `SFT_reasoning_teacher.py`
  - Stage 2 (student LoRA train/eval) → `SFT_reasoning_student.py`
  - Stage 3 (inference/export) → `SFT_inference.py`
  - Each stage can be enabled/disabled independently via flags at the top of the script.

### Stages
- `tinker/SFT_reasoning_teacher.py` — generates `{reasoning,label}` with a teacher LLM, applies rejection rules, and writes:
  - `runs/<run>/teacher_phase/accepted_samples.jsonl`
  - `runs/<run>/teacher_phase/rulebook.txt`
  - `runs/<run>/data_splits/{train,val,test}.csv`
- `tinker/SFT_reasoning_student.py` — trains a Tinker LoRA student on accepted samples and periodically evaluates.
  - Writes per-checkpoint metrics to `runs/<run>/student_phase/eval_history.jsonl` and `metrics_*.json`.
  - Writes a run summary to `runs/<run>/run_summary.json`.
  - Writes model pointer files to `tinker/model/` (when enabled) so downstream inference can reliably find the latest/best checkpoint.
- `tinker/SFT_inference.py` — runs eval + exports predictions for a selected checkpoint.
  - Reasoning mode: eval + generation exports to `runs/<run>/student_phase/{train,val,test}.json` plus `metrics_{split}.json`.
  - Label-only mode: eval-only metrics to `runs/<run>/student_phase/metrics_{split}_label_only.json`.
  - Toggles:
    - `--reasoning/--no-reasoning`
    - `--label-only/--no-label-only`

### Shared core (not legacy)
- `tinker/SFT_reasoning.py` — shared “core” module imported as `core` by teacher/student/inference.
  - Owns: config/prompt loading, rulebook building, token-fitting utilities, evaluation + thresholding helpers, parsing utilities, and Tinker sampling wrappers.
  - It also contains a `main()` for a monolithic run, but the recommended path is the stage scripts + `run_reasoning_two_stage.sh`.

## Config, prompts, rules, dataset
- `tinker/configs/` — JSON configs (e.g. `reasoning_sft.example.json`).
- `tinker/prompts/` — prompt templates (YAML).
- `tinker/rules/` — rulebook `.txt` files (concatenated into `rulebook.txt` per run).
- `tinker/dataset/` — dataset splits (commonly `dataset/<name>/{train,val,test}.csv` or `<name>_{split}.csv`).

## Outputs and artifacts
- `tinker/runs/<run_name>/` — all artifacts for a run:
  - `teacher_phase/` and `student_phase/`
  - `data_splits/`
  - `train.log`, `test_report.md`, `run_summary.json`
- `tinker/model/` — lightweight pointers to the latest/best model checkpoint paths:
  - `latest.json`
  - `<run_name>.json`

## Cross-run summarization
- `tinker/evaluate_summary.py` — aggregates inference metrics across `tinker/runs/*` into a single CSV with 3 split tables:
  - train table
  - val table
  - test table
  - Example: `python3 tinker/evaluate_summary.py --runs-dir tinker/runs --output tinker/summary.csv --mode both`

## Environment variables
Common keys (depending on which stages you run):
- `DEEPSEEK_API_KEY` — teacher sampling (if using DeepSeek teacher).
- `TINKER_API_KEY` — required for student training and inference (Tinker client).
- `HF_TOKEN` / `HUGGINGFACE_HUB_TOKEN` — optional; avoids unauthenticated Hugging Face download warnings.

## Legacy / experimental scripts
- `tinker/SFT_classification.py` — older standalone SFT classification script (not used by the two-stage runner).
- `tinker/baseline.py` and `tinker/baseline_runs/` — local baselines/experiments.
- `tinker/RL.py` — reinforcement-learning style experiments (separate from the reasoning SFT pipeline).

