import csv
import gc
import json
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import torch
from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier


MODEL_ID = "LGAI-EXAONE/EXAONE-4.0-1.2B"
DATASET_ID = "LGAI-EXAONE/MANTA-1M"
IGNORE_MODULES = ["embed_tokens", "lm_head"]
ALL_SUBMODULES = [
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
]

# Colab-friendly runtime configuration (edit values directly instead of CLI flags).
OUT_DIR = "content/drive/MyDrive/model"
FINAL_MODEL_DIR = "artifacts/final_model"
SUBMISSION_ZIP = "submit.zip"
EVAL_SAMPLES = 96
DTYPE = "bfloat16"  # "bfloat16" | "float16"
SKIP_EXP03_ANCHOR = False
EXP03_PUBLIC_SCORE = 0.605
EXP03_MODEL_DIR = "artifacts/exp03_anchor_model"
PERF_SOURCE = "external"  # "external" | "ce_proxy" | "token_acc"
# External perf evaluator command template. Use {model_dir} placeholder.
PERF_CMD_TEMPLATE = ""
BASE_MODEL_DIR = ""
# Optional precomputed baseline values to skip base model evaluation.
# Set all three values as floats to reuse a previously measured baseline.
# Example:
# BASE_CE_LOSS = 1.768302
# BASE_SPEED_SEC_PER_TOKEN = 0.044679
# BASE_PERF = 0.565432
BASE_CE_LOSS = None
BASE_SPEED_SEC_PER_TOKEN = None
BASE_PERF = None


@dataclass(frozen=True)
class Experiment:
    name: str
    layer_start: int
    layer_end: int
    scheme: str = "W4A16"
    n_calib: int = 256
    max_seq_len: int = 512
    submodules: tuple = tuple(ALL_SUBMODULES)
    dampening_frac: float = 0.01
    block_size: int = 128
    actorder: str = "weight"
    offload_hessians: bool = False
    sequential_targets: tuple = ()

    def signature(self) -> str:
        return json.dumps(
            {
                "layer_start": self.layer_start,
                "layer_end": self.layer_end,
                "scheme": self.scheme,
                "n_calib": self.n_calib,
                "max_seq_len": self.max_seq_len,
                "submodules": list(self.submodules),
                "dampening_frac": self.dampening_frac,
                "block_size": self.block_size,
                "actorder": self.actorder,
                "offload_hessians": self.offload_hessians,
                "sequential_targets": list(self.sequential_targets),
            },
            ensure_ascii=True,
            sort_keys=True,
        )


KNOWN_EXPERIMENTS = {
    # User-provided completed runs, to avoid duplicates.
    "exp_03": json.dumps(
        {
            "layer_start": 5,
            "layer_end": 24,
            "scheme": "W4A16",
            "n_calib": 256,
            "max_seq_len": 512,
            "submodules": ALL_SUBMODULES,
            "dampening_frac": 0.01,
            "block_size": 128,
            "actorder": "weight",
            "offload_hessians": False,
            "sequential_targets": [],
        },
        ensure_ascii=True,
        sort_keys=True,
    ),
    "exp_27": json.dumps(
        {
            "layer_start": 5,
            "layer_end": 24,
            "scheme": "W4A16",
            "n_calib": 512,
            "max_seq_len": 1024,
            "submodules": ALL_SUBMODULES,
            "dampening_frac": 0.01,
            "block_size": 128,
            "actorder": "weight",
            "offload_hessians": False,
            "sequential_targets": [],
        },
        ensure_ascii=True,
        sort_keys=True,
    ),
    "exp_30": json.dumps(
        {
            "layer_start": 5,
            "layer_end": 24,
            "scheme": "W4A16",
            "n_calib": 256,
            "max_seq_len": 512,
            "submodules": ALL_SUBMODULES,
            "dampening_frac": 0.01,
            "block_size": 64,
            "actorder": "weight",
            "offload_hessians": False,
            "sequential_targets": [],
        },
        ensure_ascii=True,
        sort_keys=True,
    ),
    "exp_22_like": json.dumps(
        {
            "layer_start": 5,
            "layer_end": 24,
            "scheme": "W4A16",
            "n_calib": 256,
            "max_seq_len": 512,
            "submodules": [
                "self_attn.q_proj",
                "self_attn.k_proj",
                "self_attn.v_proj",
                "self_attn.o_proj",
                "mlp.down_proj",
            ],
            "dampening_frac": 0.01,
            "block_size": 128,
            "actorder": "weight",
            "offload_hessians": False,
            "sequential_targets": [],
        },
        ensure_ascii=True,
        sort_keys=True,
    ),
}

HISTORICAL_RESULTS = [
    {"name": "Exp_03", "score": 0.605, "time": "9m36s"},
    {"name": "Exp_27", "score": 0.603, "time": "10m28s"},
    {"name": "Exp_30", "score": 0.596, "time": "9m54s"},
    {"name": "Exp_23", "score": 0.592, "time": "10m56s"},
    {"name": "Exp_22", "score": 0.587, "time": "10m24s"},
    {"name": "Exp_06", "score": 0.578, "time": "10m27s"},
    {"name": "Exp_02", "score": 0.575, "time": "11m02s"},
    {"name": "Exp_04", "score": 0.538, "time": "12m19s"},
    {"name": "Exp_21", "score": 0.504, "time": "13m35s"},
    {"name": "Exp_09", "score": 0.503, "time": "12m57s"},
    {"name": "Exp_11", "score": 0.501, "time": "13m03s"},
    {"name": "Exp_07", "score": 0.500, "time": "12m58s"},
    {"name": "Exp_17", "score": 0.498, "time": "12m25s"},
    {"name": "Exp_13", "score": 0.486, "time": "12m56s"},
    {"name": "Exp_01", "score": 0.476, "time": "13m31s"},
    {"name": "Exp_19", "score": 0.474, "time": "14m06s"},
    {"name": "Exp_20", "score": 0.471, "time": "13m05s"},
    {"name": "Exp_10", "score": 0.448, "time": "14m08s"},
    {"name": "Exp_05", "score": 0.136, "time": "12m44s"},
    {"name": "Exp_32", "score": None, "time": "20m (timeout)"},
]


def build_targets(layer_start: int, layer_end: int, submodules: Iterable[str]) -> List[str]:
    targets = []
    for i in range(layer_start, layer_end + 1):
        for sub in submodules:
            targets.append(f"model.layers.{i}.{sub}")
    return targets


def to_chat_text(tokenizer, conversations):
    return tokenizer.apply_chat_template(
        conversations,
        add_generation_prompt=True,
        tokenize=False,
    )


def make_text_dataset(tokenizer, split_expr: str) -> Dataset:
    print(f"[DATASET] loading split: {split_expr}")
    ds = load_dataset(DATASET_ID, split=split_expr)
    text_ds = ds.map(
        lambda x: {"text": to_chat_text(tokenizer, x["conversations"])},
        remove_columns=ds.column_names,
    )
    print(f"[DATASET] prepared text dataset: {len(text_ds)} rows")
    return text_ds


def mean_ce_loss(model, tokenizer, ds, max_len: int) -> float:
    model.eval()
    total_nll = 0.0
    total_tokens = 0
    total_rows = len(ds) if hasattr(ds, "__len__") else None
    progress_step = max(total_rows // 10, 1) if total_rows else None
    if total_rows:
        print(f"[EVAL-CE] start: {total_rows} samples")
    with torch.no_grad():
        for i, row in enumerate(ds, start=1):
            enc = tokenizer(
                row["text"],
                return_tensors="pt",
                truncation=True,
                max_length=max_len,
            )
            enc = {k: v.to(model.device) for k, v in enc.items()}
            if enc["input_ids"].shape[1] < 2:
                continue
            out = model(**enc, labels=enc["input_ids"])
            n_tokens = int(enc["input_ids"].shape[1])
            total_nll += float(out.loss.item()) * n_tokens
            total_tokens += n_tokens
            if total_rows and (i % progress_step == 0 or i == total_rows):
                pct = (i / total_rows) * 100.0
                print(f"[EVAL-CE] progress: {i}/{total_rows} ({pct:.0f}%)")
    ce = total_nll / max(total_tokens, 1)
    print(f"[EVAL-CE] done: tokens={total_tokens}, loss={ce:.6f}")
    return ce


def mean_token_accuracy(model, tokenizer, ds, max_len: int) -> float:
    model.eval()
    total_correct = 0
    total_count = 0
    total_rows = len(ds) if hasattr(ds, "__len__") else None
    progress_step = max(total_rows // 10, 1) if total_rows else None
    if total_rows:
        print(f"[EVAL-ACC] start: {total_rows} samples")
    with torch.no_grad():
        for i, row in enumerate(ds, start=1):
            enc = tokenizer(
                row["text"],
                return_tensors="pt",
                truncation=True,
                max_length=max_len,
            )
            enc = {k: v.to(model.device) for k, v in enc.items()}
            if enc["input_ids"].shape[1] < 2:
                continue
            logits = model(**enc).logits
            pred_ids = logits[:, :-1, :].argmax(dim=-1)
            tgt_ids = enc["input_ids"][:, 1:]
            total_correct += int((pred_ids == tgt_ids).sum().item())
            total_count += int(tgt_ids.numel())
            if total_rows and (i % progress_step == 0 or i == total_rows):
                pct = (i / total_rows) * 100.0
                print(f"[EVAL-ACC] progress: {i}/{total_rows} ({pct:.0f}%)")
    acc = total_correct / max(total_count, 1)
    print(f"[EVAL-ACC] done: tokens={total_count}, accuracy={acc:.6f}")
    return acc


def sec_per_token(model, tokenizer, ds, prompt_max_len: int, max_new_tokens: int = 64) -> float:
    model.eval()
    total_time = 0.0
    total_new_tokens = 0
    total_rows = len(ds) if hasattr(ds, "__len__") else None
    progress_step = max(total_rows // 10, 1) if total_rows else None
    if total_rows:
        print(f"[EVAL-SPD] start: {total_rows} samples, max_new_tokens={max_new_tokens}")
    with torch.no_grad():
        for i, row in enumerate(ds, start=1):
            enc = tokenizer(
                row["text"],
                return_tensors="pt",
                truncation=True,
                max_length=prompt_max_len,
            )
            enc = {k: v.to(model.device) for k, v in enc.items()}
            prompt_len = int(enc["input_ids"].shape[1])
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = model.generate(
                **enc,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.eos_token_id,
            )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0
            new_tokens = int(out.shape[1] - prompt_len)
            if new_tokens > 0:
                total_time += elapsed
                total_new_tokens += new_tokens
            if total_rows and (i % progress_step == 0 or i == total_rows):
                pct = (i / total_rows) * 100.0
                print(f"[EVAL-SPD] progress: {i}/{total_rows} ({pct:.0f}%)")
    spt = total_time / max(total_new_tokens, 1)
    print(f"[EVAL-SPD] done: new_tokens={total_new_tokens}, sec/token={spt:.6f}")
    return spt


def compute_score(perf_model: float, perf_base: float, spt_model: float, spt_base: float):
    # Official competition formula:
    # PerfNorm = Perf_model / Perf_base_model
    # SpeedNorm = 1 - (sec_per_tok_model / sec_per_tok_base)
    perf_norm = perf_model / max(perf_base, 1e-12)
    speed_norm = 1.0 - (spt_model / max(spt_base, 1e-12))
    score = max(0.5 * perf_norm + 0.5 * speed_norm, 0.0)
    return perf_norm, speed_norm, score


def load_model(dtype: torch.dtype, device_map=None):
    return AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map=device_map,
    )


def free_memory(*objs):
    for obj in objs:
        del obj
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_experiment(exp: Experiment, tokenizer, eval_ds, dtype):
    print(f"[RUN] {exp.name}")
    print(f"[RUN] loading model: {MODEL_ID}")
    model = load_model(dtype=dtype, device_map="auto")
    targets = build_targets(exp.layer_start, exp.layer_end, exp.submodules)
    print(f"[RUN] target modules: {len(targets)}")
    print(f"[RUN] preparing calibration dataset: n_calib={exp.n_calib}, max_seq_len={exp.max_seq_len}")
    calib_ds = make_text_dataset(tokenizer, f"train[:{exp.n_calib}]")
    modifier_kwargs = {
        "scheme": exp.scheme,
        "targets": targets,
        "ignore": IGNORE_MODULES,
        "dampening_frac": exp.dampening_frac,
        "block_size": exp.block_size,
        "actorder": exp.actorder,
        "offload_hessians": exp.offload_hessians,
    }
    if exp.sequential_targets:
        modifier_kwargs["sequential_targets"] = list(exp.sequential_targets)
    recipe = [GPTQModifier(**modifier_kwargs)]
    print("[RUN] quantization started")
    oneshot(
        model=model,
        dataset=calib_ds,
        recipe=recipe,
        max_seq_length=exp.max_seq_len,
        num_calibration_samples=exp.n_calib,
    )
    print("[RUN] quantization finished")
    print("[RUN] evaluating CE loss")
    ce_loss = mean_ce_loss(model, tokenizer, eval_ds, max_len=512)
    print("[RUN] evaluating speed proxy")
    spt = sec_per_token(model, tokenizer, eval_ds, prompt_max_len=512, max_new_tokens=64)
    print(f"[RUN] {exp.name} done")
    return model, {
        "name": exp.name,
        "model_source": "fresh_quantized",
        "signature": exp.signature(),
        "ce_loss": ce_loss,
        "speed_proxy_sec_per_token": spt,
    }


def save_results_csv(rows, out_csv: Path):
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def ensure_clean_dir(path: Path):
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def parse_perf_from_output(text: str) -> float:
    # Accept either plain float output or lines containing a float.
    for line in reversed(text.splitlines()):
        m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", line)
        if m:
            return float(m.group(0))
    raise RuntimeError("Could not parse perf value from external evaluator output.")


def evaluate_perf_external(perf_cmd_template: str, model_dir: Path) -> float:
    cmd = perf_cmd_template.format(model_dir=model_dir.as_posix())
    print(f"[PERF] evaluating external metric for: {model_dir}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"Perf evaluation command failed (code={result.returncode}).\n"
            f"cmd: {cmd}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    perf = parse_perf_from_output(f"{result.stdout}\n{result.stderr}")
    print(f"[PERF] done: {perf:.6f}")
    return perf


def resolve_perf(
    perf_source: str,
    perf_cmd_template: str,
    model_dir: Path,
    ce_loss: float,
) -> tuple[float, str]:
    if perf_source == "external":
        if not perf_cmd_template:
            print(
                "[PERF] PERF_SOURCE='external' but PERF_CMD_TEMPLATE is empty; "
                "falling back to ce_proxy."
            )
            return 1.0 / max(ce_loss, 1e-12), "ce_proxy"
        perf = evaluate_perf_external(perf_cmd_template=perf_cmd_template, model_dir=model_dir)
        return perf, "external"
    # Fallback debug mode only.
    return 1.0 / max(ce_loss, 1e-12), "ce_proxy"


def print_score_breakdown(row):
    msg = (
        f"[SCORE] {row['name']} | "
        f"Perf={row['Perf']:.6f} | "
        f"PerfNorm={row['PerfNorm']:.6f} | "
        f"sec/token={row['speed_proxy_sec_per_token']:.6f} | "
        f"SpeedNorm={row['SpeedNorm']:.6f} | "
        f"Score={row['Score']:.6f} | "
        f"CE_loss={row['ce_loss']:.6f}"
    )
    if "ScorePred_AnchoredToExp03" in row:
        msg += f" | ScorePred={row['ScorePred_AnchoredToExp03']:.6f}"
    print(msg)


def apply_exp03_anchor(rows, exp03_public_score: float):
    anchor_row = None
    for row in rows:
        if row["name"] == "exp03_anchor":
            anchor_row = row
            break
    if anchor_row is None:
        return None

    anchor_proxy = float(anchor_row["Score"])
    # Additive calibration: keeps relative gaps and guarantees Exp_03 => target score.
    for row in rows:
        calibrated = exp03_public_score + (float(row["Score"]) - anchor_proxy)
        row["ScorePred_AnchoredToExp03"] = max(calibrated, 0.0)
    anchor_row["ScorePred_AnchoredToExp03"] = exp03_public_score
    return {
        "anchor_name": "exp03_anchor",
        "anchor_proxy_score": anchor_proxy,
        "anchor_public_score": exp03_public_score,
        "method": "additive_shift",
    }


def main():
    if DTYPE not in {"bfloat16", "float16"}:
        raise ValueError(f"DTYPE must be 'bfloat16' or 'float16', got: {DTYPE}")
    if PERF_SOURCE not in {"external", "ce_proxy", "token_acc"}:
        raise ValueError(
            f"PERF_SOURCE must be 'external', 'ce_proxy', or 'token_acc', got: {PERF_SOURCE}"
        )

    dtype = torch.bfloat16 if DTYPE == "bfloat16" else torch.float16
    output_dir = Path(OUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[INIT] loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    print(f"[INIT] preparing eval dataset: {EVAL_SAMPLES} samples")
    eval_ds = make_text_dataset(tokenizer, f"train[20000:{20000 + EVAL_SAMPLES}]")

    perf_tmp_root = output_dir / "perf_models"
    perf_tmp_root.mkdir(parents=True, exist_ok=True)

    use_precomputed_base = all(v is not None for v in (BASE_CE_LOSS, BASE_SPEED_SEC_PER_TOKEN, BASE_PERF))
    if use_precomputed_base:
        base_ce = float(BASE_CE_LOSS)
        base_spt = float(BASE_SPEED_SEC_PER_TOKEN)
        base_perf = float(BASE_PERF)
        base_perf_source = "precomputed_constant"
        print("[BASE] using precomputed constants (BASE_CE_LOSS/BASE_SPEED_SEC_PER_TOKEN/BASE_PERF)")
    else:
        # Baseline model: speed and diagnostic CE-loss.
        print("[BASE] loading baseline model")
        base_model = load_model(dtype=dtype, device_map="auto")
        print("[BASE] computing CE loss")
        base_ce = mean_ce_loss(base_model, tokenizer, eval_ds, max_len=512)
        print("[BASE] computing speed proxy")
        base_spt = sec_per_token(base_model, tokenizer, eval_ds, prompt_max_len=512, max_new_tokens=64)

        if PERF_SOURCE == "token_acc":
            base_perf = mean_token_accuracy(base_model, tokenizer, eval_ds, max_len=512)
            base_perf_source = "token_acc"
        else:
            if BASE_MODEL_DIR:
                base_model_dir = Path(BASE_MODEL_DIR)
                if not base_model_dir.exists():
                    raise FileNotFoundError(f"BASE_MODEL_DIR does not exist: {base_model_dir}")
            else:
                base_model_dir = perf_tmp_root / "base_model"
                ensure_clean_dir(base_model_dir)
                base_model.save_pretrained(base_model_dir, safe_serialization=True)
                tokenizer.save_pretrained(base_model_dir)
            base_perf, base_perf_source = resolve_perf(
                perf_source=PERF_SOURCE,
                perf_cmd_template=PERF_CMD_TEMPLATE,
                model_dir=base_model_dir,
                ce_loss=base_ce,
            )
        free_memory(base_model)

    print(
        f"[BASE] Perf={base_perf:.6f} (source={base_perf_source}) | "
        f"sec/token={base_spt:.6f} | CE_loss={base_ce:.6f}"
    )

    candidates = [
        Experiment(
            name="exp03_anchor",
            layer_start=5,
            layer_end=24,
        ),
        Experiment(
            name="new_01_layer5_24_actorder_group",
            layer_start=5,
            layer_end=24,
            actorder="group",
        ),
        Experiment(
            name="new_02_layer5_24_damp_low",
            layer_start=5,
            layer_end=24,
            dampening_frac=0.005,
        ),
        Experiment(
            name="new_03_layer5_24_seq_attn_mlp",
            layer_start=5,
            layer_end=24,
            sequential_targets=(
                "self_attn.q_proj",
                "self_attn.k_proj",
                "self_attn.v_proj",
                "self_attn.o_proj",
                "mlp.gate_proj",
                "mlp.up_proj",
                "mlp.down_proj",
            ),
        ),
        Experiment(
            name="new_04_layer5_24_calib320",
            layer_start=5,
            layer_end=24,
            n_calib=320,
        ),
        Experiment(
            name="new_05_layer5_24_maxseq768",
            layer_start=5,
            layer_end=24,
            max_seq_len=768,
        ),
        Experiment(
            name="new_06_layer5_23_actorder_group",
            layer_start=5,
            layer_end=23,
            actorder="group",
        ),
    ]

    if SKIP_EXP03_ANCHOR:
        candidates = [c for c in candidates if c.name != "exp03_anchor"]

    known_signatures = set(KNOWN_EXPERIMENTS.values())
    rows = []
    best_model = None
    best_row = None

    total_candidates = len(candidates)
    for idx, exp in enumerate(candidates, start=1):
        print(f"[LOOP] candidate {idx}/{total_candidates}: {exp.name}")
        # exp03_anchor is intentionally rerun as calibration anchor.
        if exp.name != "exp03_anchor" and exp.signature() in known_signatures:
            print(f"[SKIP] duplicate signature: {exp.name}")
            continue
        exp03_model_dir = Path(EXP03_MODEL_DIR)
        if exp.name == "exp03_anchor" and exp03_model_dir.exists():
            print(f"[RUN] {exp.name} (load prebuilt model: {exp03_model_dir})")
            model = AutoModelForCausalLM.from_pretrained(
                exp03_model_dir.as_posix(),
                trust_remote_code=True,
                torch_dtype=dtype,
                device_map="auto",
            )
            ce_loss = mean_ce_loss(model, tokenizer, eval_ds, max_len=512)
            spt = sec_per_token(model, tokenizer, eval_ds, prompt_max_len=512, max_new_tokens=64)
            if PERF_SOURCE == "token_acc":
                perf, perf_source = mean_token_accuracy(model, tokenizer, eval_ds, max_len=512), "token_acc"
            else:
                perf, perf_source = resolve_perf(
                    perf_source=PERF_SOURCE,
                    perf_cmd_template=PERF_CMD_TEMPLATE,
                    model_dir=exp03_model_dir,
                    ce_loss=ce_loss,
                )
            perf_norm, speed_norm, score = compute_score(perf, base_perf, spt, base_spt)
            row = {
                "name": exp.name,
                "model_source": f"loaded_from:{exp03_model_dir.as_posix()}",
                "signature": exp.signature(),
                "perf_source": perf_source,
                "Perf": perf,
                "ce_loss": ce_loss,
                "speed_proxy_sec_per_token": spt,
                "PerfNorm": perf_norm,
                "SpeedNorm": speed_norm,
                "Score": score,
            }
        else:
            model, row = run_experiment(
                exp=exp,
                tokenizer=tokenizer,
                eval_ds=eval_ds,
                dtype=dtype,
            )
            exp_model_dir = perf_tmp_root / exp.name
            ensure_clean_dir(exp_model_dir)
            model.save_pretrained(exp_model_dir, safe_serialization=True, save_compressed=True)
            tokenizer.save_pretrained(exp_model_dir)
            if PERF_SOURCE == "token_acc":
                perf, perf_source = mean_token_accuracy(model, tokenizer, eval_ds, max_len=512), "token_acc"
            else:
                perf, perf_source = resolve_perf(
                    perf_source=PERF_SOURCE,
                    perf_cmd_template=PERF_CMD_TEMPLATE,
                    model_dir=exp_model_dir,
                    ce_loss=row["ce_loss"],
                )
            perf_norm, speed_norm, score = compute_score(
                perf,
                base_perf,
                row["speed_proxy_sec_per_token"],
                base_spt,
            )
            row["perf_source"] = perf_source
            row["Perf"] = perf
            row["PerfNorm"] = perf_norm
            row["SpeedNorm"] = speed_norm
            row["Score"] = score
        rows.append(row)
        print_score_breakdown(row)
        if best_row is None or row["Score"] > best_row["Score"]:
            if best_model is not None:
                free_memory(best_model)
            best_model = model
            best_row = row
        else:
            free_memory(model)

        save_results_csv(rows, output_dir / "results.csv")
        (output_dir / "results.json").write_text(
            json.dumps(rows, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    calibration = apply_exp03_anchor(rows, EXP03_PUBLIC_SCORE)
    if calibration is not None:
        print(f"[CALIBRATION] {calibration}")
        for row in rows:
            print_score_breakdown(row)
        save_results_csv(rows, output_dir / "results.csv")
        (output_dir / "results.json").write_text(
            json.dumps(rows, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    if best_model is None:
        raise RuntimeError("No new experiment left after duplicate filtering.")

    final_model_dir = Path(FINAL_MODEL_DIR)
    if final_model_dir.exists():
        shutil.rmtree(final_model_dir)
    final_model_dir.mkdir(parents=True, exist_ok=True)
    best_model.save_pretrained(final_model_dir, safe_serialization=True, save_compressed=True)
    tokenizer.save_pretrained(final_model_dir)

    # Submission format: submit.zip/model/*
    model_dir = Path("model")
    if model_dir.exists():
        shutil.rmtree(model_dir)
    shutil.copytree(final_model_dir, model_dir)
    zip_no_ext = Path(SUBMISSION_ZIP).with_suffix("")
    if Path(SUBMISSION_ZIP).exists():
        Path(SUBMISSION_ZIP).unlink()
    shutil.make_archive(zip_no_ext.as_posix(), "zip", root_dir=".", base_dir="model")

    summary = {
        "best": best_row,
        "historical_baseline": {"name": "Exp_03", "public_score": 0.605, "elapsed": "9m36s"},
        "historical_results": HISTORICAL_RESULTS,
        "perf_metric_note": "Score uses official formula shape with Perf from external/token_acc/ce_proxy source.",
        "base": {"Perf": base_perf, "perf_source": base_perf_source, "ce_loss": base_ce, "speed_sec_per_token": base_spt},
        "calibration": calibration,
        "saved_model_dir": str(final_model_dir),
        "submission_zip": f"{zip_no_ext}.zip",
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("[DONE]")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
