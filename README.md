# LG_Aimers 8기
```text
LG Aimers 8기
```
# 가상환경 설정
```text
python -m venv lg_env
source lg_env/Scripts/activate
pip install -r requirements.txt
```

# .env
```text
HF_TOKEN=your_huggingface_token_here
```

# 폴더 구조
```text
project_root/
├── .env                # API 토큰 등 환경 변수
├── requirements.txt    # 의존성 패키지 목록
└── src/
    ├── __init__.py
    ├── config.py       # 모델 ID 및 경로 설정
    ├── wanda_visualization.py  # Wanda 시각화 파일
    ├── activation_visualization.py  # Activation 분포 시각화 파일
    ├── wanda_visualizations/   # wanda 시각화 폴더
    └── activation_distribution/ # Activation 시각화 폴더 
```
