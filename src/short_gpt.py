import os
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm

# ==========================================
# 1. 설정 (노트북의 Setting 섹션 값 반영)
# ==========================================
MODEL_ID = "LGAI-EXAONE/EXAONE-4.0-1.2B"
DATASET_ID = "LGAI-EXAONE/MANTA-1M"
DATASET_SPLIT = "train"
NUM_CALIBRATION_SAMPLES = 100  # 측정 정확도를 위해 100개 권장
MAX_SEQUENCE_LENGTH = 512      # 노트북의 MAX_SEQUENCE_LENGTH 반영
NUM_LAYERS = 30               # EXAONE-4.0-1.2B의 Decoder 블록 수

device = "cuda" if torch.cuda.is_available() else "cpu"

# ==========================================
# 2. 모델 및 토크나이저 로드
# ==========================================
print(f"[INFO] 모델 로드 중: {MODEL_ID}")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
)

# 노트북에서 사용한 torch.bfloat16 또는 fp16 권장
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16, 
    device_map="auto",
    trust_remote_code=True
)
model.eval()
print("[INFO] 모델/토크나이저 로드 완료")

# ==========================================
# 3. 데이터 로드 및 전처리 (노트북 로직 반영)
# ==========================================
print(f"[INFO] 캘리브레이션 데이터 로드 중...")
ds = load_dataset(
    DATASET_ID,
    split=f"{DATASET_SPLIT}[:{NUM_CALIBRATION_SAMPLES}]",
)

def preprocess(example):
    # 노트북의 전처리 함수와 동일하게 구성
    return {
        "text": tokenizer.apply_chat_template(
            example["conversations"],
            add_generation_prompt=True,
            tokenize=False)
    }

ds = ds.map(preprocess)
print("[INFO] 데이터 전처리 완료")

# ==========================================
# 4. BI Score 측정 함수 (ShortGPT 알고리즘)
# ==========================================
def calculate_bi_scores(model, tokenizer, dataset):
    all_bi_scores = []
    
    with torch.no_grad():
        for i in tqdm(range(len(dataset)), desc="BI Score 계산 중"):
            text = dataset[i]["text"]
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_SEQUENCE_LENGTH).to(model.device)
            
            # hidden_states=True를 설정하여 각 층의 출력을 수집
            outputs = model(**inputs, output_hidden_states=True)
            h = outputs.hidden_states 
            
            layer_bi_scores = []
            
            # 30개 레이어 순회
            # h[0]: Embedding 출력 (Layer 0의 입력)
            # h[1]: Layer 0의 출력
            # ...
            # h[j]와 h[j+1]을 비교하여 Layer j의 영향력 측정 
            for j in range(NUM_LAYERS):
                x_in = h[j].float()      # 입력값
                x_out = h[j+1].float()   # 출력값
                
                # 코사인 유사도 계산 [cite: 726]
                cos_sim = F.cosine_similarity(x_in, x_out, dim=-1)
                
                # BI = 1 - Cosine Similarity [cite: 726]
                # 값이 낮을수록 해당 레이어가 변화를 거의 주지 않는 '중복'임을 의미 
                bi_val = 1 - cos_sim.mean().item()
                layer_bi_scores.append(bi_val)
            
            all_bi_scores.append(layer_bi_scores)

    # 샘플 전체의 레이어별 평균 BI 계산
    avg_bi_scores = torch.tensor(all_bi_scores).mean(dim=0).tolist()
    return avg_bi_scores

# ==========================================
# 5. 결과 도출 및 프루닝 추천
# ==========================================
bi_results = calculate_bi_scores(model, tokenizer, ds)

print("\n" + "="*40)
print(f"{'Layer ID':<10} | {'BI Score':<20}")
print("-" * 35)
for idx, score in enumerate(bi_results):
    print(f"Layer {idx:02d}   | {score:.8f}")
print("="*40)

# 중요도가 낮은(BI가 낮은) 순서로 정렬하여 삭제 우선순위 도출 
sorted_layers = sorted(range(len(bi_results)), key=lambda k: bi_results[k])
print(f"\n[추천] 중복도가 높은 레이어 순위: {sorted_layers}")

# 논문 기준 약 25% 삭제 권장 (30개 중 7~8개) [cite: 651, 960]
pruning_count = int(NUM_LAYERS * 0.25)
top_k_redundant = sorted_layers[:pruning_count]
print(f"[알림] 성능 하락 최소화를 위한 {pruning_count}개 삭제 대상: {top_k_redundant}")