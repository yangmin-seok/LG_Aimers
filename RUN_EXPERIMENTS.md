# Experiment Runner (No Colab-specific steps)

This repository now includes `src/exp03_runner.py` to:

1. Rebuild a GPTQ-compressed EXAONE model from `LGAI-EXAONE/EXAONE-4.0-1.2B`
2. Measure score terms with competition formula:
   - `PerfNorm = Perf_model / Perf_base_model`
   - `SpeedNorm = 1 - (sec_per_tok_model / sec_per_tok_base)`
   - `Score = max(0.5*PerfNorm + 0.5*SpeedNorm, 0)`
3. Keep CE loss as diagnostic output only (not used in score)
4. Always rerun `Exp_03` once as `exp03_anchor` (unless explicitly skipped)
   - If `--exp03-model-dir` exists, it loads that model and skips re-quantization.
5. Calibrate predicted score to known `Exp_03=0.605`:
   - `ScorePred_AnchoredToExp03 = 0.605 + (Score - Score_exp03_anchor)`
6. Avoid re-running known duplicate experiments (except the `exp03_anchor` run)
7. Save only the best model once, then create `submit.zip` in required format

## Run

```bash
python -m src.exp03_runner \
  --out-dir artifacts/exp03_runner \
  --final-model-dir artifacts/final_model \
  --submission-zip submit.zip \
  --perf-source external \
  --perf-cmd-template "python eval_perf.py --model_dir {model_dir}"
```

Optional:

```bash
python -m src.exp03_runner --skip-exp03-anchor
```

Use local prebuilt Exp_03 model:

```bash
python -m src.exp03_runner --exp03-model-dir ./model_QPTQ_layer5_24
```

## Outputs

- `artifacts/exp03_runner/results.csv` (includes `Perf`, `PerfNorm`, `SpeedNorm`, `Score`, `ce_loss`)
- `artifacts/exp03_runner/results.json`
- `artifacts/exp03_runner/summary.json`
- `artifacts/final_model/*`
- `submit.zip` (contains top-level `model/` folder only)

## Notes

- No Colab-only code is used (`!pip`, `google.colab`, `files.download` are not used).
- This runner is designed for CUDA GPU environments compatible with EXAONE + GPTQ.
- If `--perf-source external` is used with the same evaluator pipeline, score definition matches the competition formula.
- `ce_loss` is printed for diagnostics and is intentionally excluded from score calculation.
