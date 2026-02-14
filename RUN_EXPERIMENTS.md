# Experiment Runner (No Colab-specific steps)

This repository now includes `src/exp03_runner.py` to:

1. Rebuild a GPTQ-compressed EXAONE model from `LGAI-EXAONE/EXAONE-4.0-1.2B`
2. Measure proxy score terms for leaderboard formula:
   - `PerfNorm` (proxy from CE loss ratio vs base model)
   - `SpeedNorm` (proxy from decode sec/token ratio vs base model)
   - `ScoreProxy = max(0.5*PerfNorm + 0.5*SpeedNorm, 0)`
3. Avoid re-running known duplicate experiments (Exp_03 family from history)
4. Save only the best model once, then create `submit.zip` in required format

## Run

```bash
python -m src.exp03_runner --out-dir artifacts/exp03_runner --final-model-dir artifacts/final_model --submission-zip submit.zip
```

Optional:

```bash
python -m src.exp03_runner --include-exp03-baseline
```

## Outputs

- `artifacts/exp03_runner/results.csv`
- `artifacts/exp03_runner/results.json`
- `artifacts/exp03_runner/summary.json`
- `artifacts/final_model/*`
- `submit.zip` (contains top-level `model/` folder only)

## Notes

- No Colab-only code is used (`!pip`, `google.colab`, `files.download` are not used).
- This runner is designed for CUDA GPU environments compatible with EXAONE + GPTQ.
- Public leaderboard score and this local `ScoreProxy` are not numerically identical.
