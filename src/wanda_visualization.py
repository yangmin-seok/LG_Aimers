import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from dotenv import load_dotenv
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader

# config.py에서 설정 임포트
from src.config import MODEL_ID, DATASET_ID, OUTPUT_DIR

# .env 파일 로드 
load_dotenv(os.path.join(os.path.dirname(__file__), '../.env'))
HF_TOKEN = os.getenv("HF_TOKEN")

def load_model_and_tokenizer(model_id, token):
    """
    분석에 필요한 사전 학습된 LLM 모델과 토크나이저를 로드합니다.
    
    Wanda는 가중치 업데이트가 필요 없는 원샷(One-shot) 방식이므로 
    모델을 추론 모드(eval)로 사용합니다
    """
    print(f"🚀 Loading model: {model_id}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        token=token,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, token=token)

    # 패딩 토큰이 없는 경우 EOS 토큰을 패딩 토큰으로 설정
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

def get_calibration_loader(dataset_id, tokenizer, num_samples=128, max_length=512):
    """
    활성화 값 통계(Activation Norm)를 계산하기 위한 소량의 보정 데이터셋을 준비합니다.
    """
    print(f"📦 Loading dataset: {dataset_id}...")
    ds = load_dataset(dataset_id, split=f"train[:{num_samples}]")

    def preprocess(examples):
        # Text 형태로 변환
        return [tokenizer.apply_chat_template(convo, add_generation_prompt=True, tokenize=False) 
                for convo in examples["conversations"]]

    texts = preprocess(ds)
    
    def collate_fn(batch_texts):
        return tokenizer(batch_texts, padding=True, truncation=True, 
                         max_length=max_length, return_tensors="pt")

    return DataLoader(texts, batch_size=2, shuffle=False, collate_fn=collate_fn)

def calculate_wanda_metric(model, dataloader):
    """
    각 레이어의 가중치 행렬(W)과 입력 활성화 값의 l2 노름(||X||_2)을 계산하여 
    Wanda 중요도 지표를 생성합니다.
    
    이 지표는 가중치 크기뿐만 아니라 입력 피처의 스케일을 반영하여 모델 예측에 핵심적인 이상치 피처를 보호합니다.
    """
    model.eval()
    device = model.device
    layer_stats = []
    
    # 모델 아키텍처에 따른 레이어 접근 
    layers = model.model.layers
    
    for i, layer in enumerate(layers):
        print(f"🔍 Analyzing Layer {i}...")
        target_modules = {name: mod for name, mod in layer.named_modules() if isinstance(mod, nn.Linear)}
        activations = {}

        def get_hook(name):
            def hook(m, inp, out):
                # 논문의 수식 S_ij = |W_ij| * ||X_j||_2를 위한 l2 norm 계산 
                # x shape: (Batch, Seq, Hidden)
                x = inp[0].detach().float() 
                if name not in activations:
                    activations[name] = torch.sum(x**2, dim=(0, 1))
                else:
                    activations[name] += torch.sum(x**2, dim=(0, 1))
            return hook

        hooks = [mod.register_forward_hook(get_hook(name)) for name, mod in target_modules.items()]
        
        # 보정 데이터를 통해 Activation 통계 수집
        batch = next(iter(dataloader)).to(device)
        with torch.no_grad():
            model(**batch)
            
        for h in hooks: h.remove() # 메모리 정리

        layer_results = {}
        for name, mod in target_modules.items():
            W = mod.weight.detach().float().abs().cpu()
            # 제곱합의 제곱근으로 최종 l2 노름 완성 
            x_l2_norm = torch.sqrt(activations[name]).cpu()
            
            # Wanda 점수 계산: (C_out, C_in) 가중치 행렬에 (C_in,) 노름 벡터를 브로드캐스팅 곱셈
            wanda_matrix = W * x_l2_norm.unsqueeze(0)
            layer_results[name] = wanda_matrix

        layer_stats.append(layer_results)
        torch.cuda.empty_cache()
        
    return layer_stats

def visualize_wanda_distribution(layer_idx, layer_data, output_dir):
    """
    Wanda 지표의 레이어별 분포를 시각화하여 파일로 저장합니다.
    
    좌측 Heatmap은 가중치의 국소적 강도를 보여주며, 
    우측 라인 그래프는 전체 입력 차원 중 성능에 결정적인 
    이상치 피처(Outlier peaks)를 식별하게 해줍니다[cite: 113, 339].
    """
    num_modules = len(layer_data)
    fig, axes = plt.subplots(num_modules, 2, figsize=(15, 5 * num_modules))
    
    for idx, (name, wanda) in enumerate(layer_data.items()):
        # 1. Heatmap: 국부적인 결합 강도 확인 (150x150 샘플링)
        sns.heatmap(wanda[:150, :150].numpy(), ax=axes[idx, 0], cmap="viridis", cbar=True)
        axes[idx, 0].set_title(f"{name} Wanda Heatmap\nMean: {wanda.mean():.6f}")
        
        # 2. 1D Importance Chart: 전체 차원에 대한 이상치(Outlier) 피크 확인
        feature_importance = wanda.mean(dim=0) 
        axes[idx, 1].plot(feature_importance.numpy(), color='tab:red', alpha=0.7)
        axes[idx, 1].set_title(f"{name} Feature Importance (All Input Dims)")
        axes[idx, 1].set_xlabel("Input Dimension Index")
        axes[idx, 1].set_ylabel("Mean Wanda Score")
        axes[idx, 1].grid(True, alpha=0.3)

    plt.suptitle(f"Layer {layer_idx} Wanda Metric Analysis\n|W| * ||X||2", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # OUTPUT_DIR은 config.py에 정의된 상위 폴더 경로를 사용
    full_output_path = os.path.join(os.path.dirname(__file__), output_dir)
    os.makedirs(full_output_path, exist_ok=True)
    
    save_name = os.path.join(full_output_path, f"layer_{layer_idx:02d}_analysis.png")
    plt.savefig(save_name)
    plt.close()

if __name__ == "__main__":
    # 1. 모델 및 토크나이저 초기화
    model, tokenizer = load_model_and_tokenizer(MODEL_ID, HF_TOKEN)
    
    # 2. 보정 데이터셋 로드
    calib_loader = get_calibration_loader(DATASET_ID, tokenizer)
    
    # 3. Wanda 메트릭 계산 (l2 노름 기반)
    all_layer_stats = calculate_wanda_metric(model, calib_loader)
    
    # 4. 결과 시각화 및 저장
    print(f"🎨 Generating visualization files...")
    for i, layer_data in enumerate(all_layer_stats):
        visualize_wanda_distribution(i, layer_data, OUTPUT_DIR)
        
    print(f"\n✅ Analysis complete! Check the results in: {os.path.abspath(os.path.join(os.path.dirname(__file__), OUTPUT_DIR))}")