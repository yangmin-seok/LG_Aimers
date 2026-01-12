# LG_Aimers
LG Aimers 8기

# 가상환경 설정
source lg_env/Scripts/activate

# .env
HF_TOKEN=your_huggingface_token_here

# 폴더 구조
project_root/
├── .env                 # API 토큰 등 환경 변수
├── requirements.txt     # 의존성 패키지 목록
├── lg_env/              # 가상환경 폴더
└── src/
    ├── __init__.py
    ├── config.py        # 모델 ID 및 경로 설정
    └── wanda_visualization.py  # 메인 분석 및 시각화 스크립트
    └── wanda_visualizations # wanda 시각화 이미지 폴더
    