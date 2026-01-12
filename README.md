# LG_Aimers
LG Aimers 8기

# 가상환경 설정
```text
python -m venv lg_env
source lg_env/Scripts/activate
pip install -r requirements.txt
```

# .env
HF_TOKEN=your_huggingface_token_here

# 폴더 구조
```text
project_root/
├── .env                # API 토큰 등 환경 변수
├── requirements.txt    # 의존성 패키지 목록
├── LG_env/             # 가상환경 폴더
├── analysis_results/   # 분석 결과 이미지 저장 폴더
└── src/
    ├── __init__.py
    ├── config.py       # 모델 ID 및 경로 설정
    └── wanda_visualization.py  # 메인 분석 및 시각화 스크립트
```
