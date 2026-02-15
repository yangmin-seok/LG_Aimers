# Experiment Runner (`src/exp03_runner.py`)

`src/exp03_runner.py`는 EXAONE 모델에 GPTQ를 적용하고, 후보 실험들을 비교해 최종 제출 산출물(`submit.zip`)을 생성합니다.

## What it does

1. `LGAI-EXAONE/EXAONE-4.0-1.2B` 기반 GPTQ 양자화 실험 실행
2. 대회 점수식 형태로 지표 계산
   - `PerfNorm = Perf_model / Perf_base_model`
   - `SpeedNorm = 1 - (sec_per_tok_model / sec_per_tok_base)`
   - `Score = max(0.5*PerfNorm + 0.5*SpeedNorm, 0)`
3. `exp03_anchor` 기준 보정 점수 계산
   - `ScorePred_AnchoredToExp03 = 0.605 + (Score - Score_exp03_anchor)`
4. 중복 시그니처 실험 스킵(단, `exp03_anchor`는 예외)
5. 최고 점수 모델 저장 후 `submit.zip` 생성

---

## Colab quick start

아래 버전으로 먼저 설치하세요.

```python
!pip install -U torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1
```

필요 패키지 설치(프로젝트 기준):

```python
!pip install -r requirements.txt
```

이후 `src/exp03_runner.py` 상단의 **대문자 상수**를 수정한 뒤 실행하면 됩니다.

```python
!python -m src.exp03_runner
```

---

## Configuration (edit constants in file)

`argparse` 대신, 파일 상단 상수로 설정합니다.

- `OUT_DIR`
- `FINAL_MODEL_DIR`
- `SUBMISSION_ZIP`
- `EVAL_SAMPLES`
- `DTYPE` (`"bfloat16"` or `"float16"`)
- `SKIP_EXP03_ANCHOR`
- `EXP03_PUBLIC_SCORE`
- `EXP03_MODEL_DIR`
- `PERF_SOURCE` (`"external"`, `"ce_proxy"`, or `"token_acc"`)
- `PERF_CMD_TEMPLATE` (예: `python eval_perf.py --model_dir {model_dir}`)
- `BASE_MODEL_DIR`
- `BASE_CE_LOSS`, `BASE_SPEED_SEC_PER_TOKEN`, `BASE_PERF` (베이스 측정값 재사용 시)

예시:

```python
OUT_DIR = "content/drive/MyDrive/model"
DTYPE = "bfloat16"
PERF_SOURCE = "external"
PERF_CMD_TEMPLATE = "python eval_perf.py --model_dir {model_dir}"
```

베이스 모델은 한 번만 측정하고 재사용할 수 있습니다:

```python
BASE_CE_LOSS = 1.768302
BASE_SPEED_SEC_PER_TOKEN = 0.044679
BASE_PERF = 0.565432
```

위 3개를 모두 채우면 베이스 모델 로드/평가를 건너뜁니다.

외부 평가 스크립트 없이 정확도까지 같이 보려면 아래처럼 토큰 정확도를 Perf로 쓸 수 있습니다:

```python
PERF_SOURCE = "token_acc"
```

`token_acc`는 현재 `EVAL_SAMPLES` 구간에서 next-token accuracy를 계산해 Perf로 사용합니다.

---

## Outputs

- `${OUT_DIR}/results.csv`
- `${OUT_DIR}/results.json`
- `${OUT_DIR}/summary.json`
- `${FINAL_MODEL_DIR}/*`
- `${SUBMISSION_ZIP}` (top-level `model/` 폴더를 포함한 zip)

---

## Notes

- `PERF_SOURCE="external"`인데 `PERF_CMD_TEMPLATE`가 비어 있으면 자동으로 `ce_proxy`로 fallback 됩니다.
- 외부 평가기 없이 정확도 측정이 필요하면 `PERF_SOURCE="token_acc"`를 사용하세요.
- `ce_loss`는 진단용으로 출력되며 공식 점수 계산에는 직접 사용되지 않습니다.
- CUDA GPU 환경(모델 다운로드/양자화 가능)이 필요합니다.
