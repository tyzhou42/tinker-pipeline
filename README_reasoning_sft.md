# Reasoning SFT (Teacher + Student)

## What it does
- Loads binary text dataset.
- Loads rulebook by concatenating `tinker/rules/*.txt`.
- Uses `deepseek-chat` teacher to generate JSON `{reasoning, label}`.
- Applies rejection sampling (`pred_label == gold_label`, minimum reasoning length).
- Fine-tunes a Tinker LoRA student on accepted reasoning traces.
- Logs training/eval metrics to Weights & Biases.

## Required inputs
1. Put rules under `tinker/rules/` as `.txt` files.
2. Ensure `extraction/.env` contains `DEEPSEEK_API_KEY` and `TINKER_API_KEY`.
3. Adjust config at `tinker/configs/reasoning_sft.example.json`.

## Run
```bash
cd /home/tzhou3/Financial-LLM-for-Analyst-Anomaly-Detection
python3 tinker/SFT_reasoning.py --config tinker/configs/reasoning_sft.example.json
```

## Key outputs
- `tinker/runs/<run_name>/teacher_samples.jsonl`
- `tinker/runs/<run_name>/accepted_samples.jsonl`
- `tinker/runs/<run_name>/rejected_samples.jsonl`
- `tinker/runs/<run_name>/train_sft.jsonl` (if enabled)
- `tinker/runs/<run_name>/train.log`
- `tinker/runs/<run_name>/test_report.md`
- `tinker/runs/<run_name>/run_summary.json`
