import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from tqdm import tqdm
import math
from scipy.stats import kurtosis
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from dotenv import load_dotenv
from src.config import MODEL_ID, DATASET_ID 

# .env 파일 로드 
load_dotenv(os.path.join(os.path.dirname(__file__), '../.env'))
HF_TOKEN = os.getenv("HF_TOKEN")

# ==========================================
# 1. Activation Hook Class
# ==========================================
class ActivationProfiler:
    def __init__(self, model):
        self.model = model
        self.hooks = []
        self.stats = {
            "layer_idx": [],
            "module_type": [],
            "layer_name": [],
            "max_val": [],
            "min_val": [],
            "mean_val": [],
            "kurtosis": [],
            "tensor_sample": []
        }

    def _get_hook(self, layer_idx, module_type, layer_name):
        def hook(module, args, kwargs, output):
            inp = None
            # 1. 위치 인자 (MLP 등)
            if args and isinstance(args, tuple) and len(args) > 0:
                inp = args[0]
            # 2. 키워드 인자 (Attention 등)
            elif kwargs and 'hidden_states' in kwargs:
                inp = kwargs['hidden_states']

            if inp is None:
                return

            # 분석을 위해 float32 변환 및 Detach
            act = inp.detach().float()
            flat_act = act.reshape(-1).cpu().numpy()

            # 통계 수집
            self.stats["layer_idx"].append(layer_idx)
            self.stats["module_type"].append(module_type)
            self.stats["layer_name"].append(layer_name)
            self.stats["max_val"].append(np.max(flat_act))
            self.stats["min_val"].append(np.min(flat_act))
            self.stats["mean_val"].append(np.mean(np.abs(flat_act)))
            
            k_val = kurtosis(flat_act)
            self.stats["kurtosis"].append(k_val)

            # 시각화용 샘플링 (최대 2000개)
            sample_size = min(len(flat_act), 2000)
            sampled_data = np.random.choice(flat_act, sample_size, replace=False)
            self.stats["tensor_sample"].append(sampled_data)

        return hook

    def register_hooks(self):
        print("🔗 Registering Hooks (Layer-wise iteration)...")
        
        # LG Aimers 모델 아키텍처에 따른 레이어 접근
        layers = self.model.model.layers
        
        for i, layer in enumerate(layers):
            # 해당 레이어 내부의 Linear 모듈만 타겟팅
            target_modules = {name: mod for name, mod in layer.named_modules() if isinstance(mod, nn.Linear)}
            
            for name, module in target_modules.items():
                # 모듈 타입 구분 (시각화 색상 구분용)
                if 'self_attn' in name:
                    module_type = "Attention"
                elif 'mlp' in name:
                    module_type = "MLP"
                else:
                    module_type = "Other"

                # Hook 등록
                self.hooks.append(
                    module.register_forward_hook(
                        self._get_hook(i, module_type, name),
                        with_kwargs=True
                    )
                )

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []


# ==========================================
# 2. 데이터 및 모델 로드 함수
# ==========================================
def load_model(model_id, hf_token, device):
    print(f"🚀 Loading Model: {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token, trust_remote_code=True)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        token=hf_token,
        dtype=torch.bfloat16, # 메모리 효율 및 최신 모델 호환성을 위해 bfloat16 권장
        device_map=None,
    ).to(device) 
    
    return model, tokenizer

def prepare_calibration_data(tokenizer, num_samples, max_length, device):
    dataset_name = "LGAI-EXAONE/MANTA-1M"
    print(f"📚 Loading Dataset ({dataset_name})...")
    
    # 데이터셋 로드 (학습 데이터의 앞부분만 사용)
    data = load_dataset(dataset_name, split=f"train[:{num_samples}]")
    
    batch_text = []
    for example in data:
        # Chat Template을 적용하여 하나의 문자열로 변환
        # [{"role": "user", "content": "안녕, 너는 누구니?"}, {"role": "assistant", "content": "저는 EXAONE입니다."}]
        # -> 하나의 str으로 변환
        text = tokenizer.apply_chat_template(
            example["conversations"],
            tokenize=False,
            add_generation_prompt=True
        )
        batch_text.append(text)

    # 토크나이징 및 텐서 변환
    inputs = tokenizer(
        batch_text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length
    ).to(device)
    
    return inputs

# ==========================================
# 3. 시각화 및 결과 저장 함수
# ==========================================
def save_visualizations(profiler, output_dir):
    print("🎨 Generating Plots...")
    os.makedirs(output_dir, exist_ok=True)
    
    df = pd.DataFrame({k: v for k, v in profiler.stats.items() if k != "tensor_sample"})
    df.to_csv(f"{output_dir}/activation_stats.csv", index=False)

    # 1. Max Magnitude Plot
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df, x="layer_idx", y="max_val", hue="module_type", marker="o")
    plt.title("Max Activation Magnitude per Layer")
    plt.ylabel("Max Absolute Value")
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{output_dir}/01_max_magnitude.png")
    plt.close()

    # 2. Kurtosis Plot
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df, x="layer_idx", y="kurtosis", hue="module_type", marker="s")
    plt.title("Activation Kurtosis (Outlier Severity)")
    plt.ylabel("Kurtosis (Higher = More Outliers)")
    plt.axhline(y=3.0, color='r', linestyle='--', label="Normal Distribution")
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(f"{output_dir}/02_kurtosis.png")
    plt.close()

    # 3. Boxplot Preparation
    plot_data = []
    for i in range(len(profiler.stats["layer_idx"])):
        lid = profiler.stats["layer_idx"][i]
        mtype = profiler.stats["module_type"][i]
        samples = profiler.stats["tensor_sample"][i]
        for val in samples:
            plot_data.append({"Layer": lid, "Value": val, "Module": mtype})
    
    df_plot = pd.DataFrame(plot_data)

    # 3. Distribution Boxplot
    plt.figure(figsize=(16, 8))
    sns.boxplot(data=df_plot, x="Layer", y="Value", hue="Module", showfliers=False)
    plt.title("Activation Distribution per Layer (Without Extreme Outliers)")
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{output_dir}/03_distribution_boxplot.png")
    plt.close()

    # 4. Detailed Boxen Plot
    plt.figure(figsize=(16, 8))
    sns.boxenplot(data=df_plot, x="Layer", y="Value", hue="Module")
    plt.title("Activation Distribution (Detailed Boxen Plot)")
    plt.ylim(-50, 50)
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{output_dir}/04_distribution_detailed.png")
    plt.close()

    print(f"✅ All analysis saved in: {output_dir}")

if __name__ == "__main__":
    # --- Configuration ---
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_SAMPLES = 128
    MAX_LENGTH = 512
    BATCH_SIZE = 4
    
    # 현재 파일의 위치를 기준으로 결과 폴더 설정 (src/activation_distribution)
    OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "activation_distribution")

    # 1. 모델 및 데이터 로드
    model, tokenizer = load_model(MODEL_ID, HF_TOKEN, DEVICE)
    # [수정됨] MANTA 데이터셋 로드 함수 호출
    inputs = prepare_calibration_data(tokenizer, NUM_SAMPLES, MAX_LENGTH, DEVICE)

    # 2. Profiler 설정 및 Hook 등록
    profiler = ActivationProfiler(model)
    profiler.register_hooks()

    # 3. Inference 실행 (통계 수집)
    print("🏃 Running Inference to collect statistics...")

    total_samples = len(inputs.input_ids)
    num_batches = math.ceil(total_samples / BATCH_SIZE)
    with torch.no_grad():        
        # tqdm으로 감싸서 진행률 바 생성
        for i in tqdm(range(0, NUM_SAMPLES, BATCH_SIZE), total=num_batches, desc="Inference Progress"):
            # 데이터 슬라이싱 (Mini-batch 생성)
            batch_input_ids = inputs.input_ids[i : i + BATCH_SIZE]
            batch_attention_mask = inputs.attention_mask[i : i + BATCH_SIZE]

            # 모델 실행 (결과는 Hook이 가로채므로 반환값 무시)
            model(batch_input_ids, attention_mask=batch_attention_mask)
            
            torch.cuda.empty_cache()
    
    profiler.remove_hooks()
    print("✅ Data Collection Complete.")

    # 4. 결과 시각화 및 저장
    save_visualizations(profiler, OUTPUT_DIR)