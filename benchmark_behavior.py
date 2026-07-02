# benchmark_behavior.py
import os
import sys
import time
import datetime
import gc
import json
import numpy as np

try:
    import pynvml
    pynvml.nvmlInit()
    gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    HAS_GPU_TRACKING = True
except Exception:
    HAS_GPU_TRACKING = False

try:
    from llama_cpp import Llama
except ImportError:
    print("❌ Error: 'llama-cpp-python' not installed.")
    sys.exit(1)

def get_gpu_vram():
    if not HAS_GPU_TRACKING:
        return 0.0
    try:
        mem = pynvml.nvmlDeviceGetMemoryInfo(gpu_handle)
        return mem.used / (1024**3)
    except:
        return 0.0

# --- SIMULATED MOCK TENSOR ERROR DATA FROM YOUR WEIGHT ANALYSIS ---
# (In a production setup, parse the text file or dictionary your weight analysis script saved)
TENSOR_DIAGNOSTICS = {
    "blk.0.ffn_norm.weight": {"mse": 0.712, "cosine": 0.984, "sparsity": 0.00},
    "blk.0.attn_q.weight":   {"mse": 1.423, "cosine": 0.912, "sparsity": 12.4},
    "blk.0.attn_k.weight":   {"mse": 1.115, "cosine": 0.935, "sparsity": 10.1},
    "blk.1.ffn_down.weight": {"mse": 3.841, "cosine": 0.742, "sparsity": 45.2},
    "blk.2.ffn_up.weight":   {"mse": 2.104, "cosine": 0.829, "sparsity": 31.8}
}

PROMPTS = [
    {"id": "GSM8K_Math", "prompt": "Solve for x: 3x + 7 = 22. Show your work step-by-step."},
    {"id": "Logic_Riddle", "prompt": "A man has 4 sons. Each son has one sister. How many children does the man have?"},
    {"id": "MMLU_Concept", "prompt": "What are the primary structural differences between a standard Transformer LLM block and an isolated Feed-Forward Network (FFN) layer?"},
    {"id": "Creative_Expl", "prompt": "Explain quantum computing to a 10 year old using an analogy."},
    {"id": "Philosophy", "prompt": "If a tree falls in a forest and no one is around to hear it, does it make a sound? Answer philosophically."}
]

def benchmark_model(model_path, name):
    print(f"\n🏗️ Loading Graph: {name} ({model_path})")
    if not os.path.exists(model_path):
        print(f"⚠️ Path missing: {model_path}")
        return None

    base_vram = get_gpu_vram()
    llm = Llama(model_path=model_path, n_ctx=512, n_gpu_layers=-1, verbose=False)
    time.sleep(2)
    loaded_vram = get_gpu_vram()
    vram_overhead = max(0.0, loaded_vram - base_vram)
    
    prompt_records = []
    for idx, item in enumerate(PROMPTS, 1):
        p_id = item["id"]
        prompt_text = item["prompt"]
        formatted_prompt = f"<|start_header_id|>user<|end_header_id|>\n\n{prompt_text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        
        start_time = time.time()
        output = llm(formatted_prompt, max_tokens=128, temperature=0.1, stop=["<|eot_id|>"])
        total_time = time.time() - start_time
        
        tokens = output["usage"]["completion_tokens"]
        tps = tokens / total_time if total_time > 0 else 0
        
        prompt_records.append({
            "id": p_id,
            "prompt": prompt_text,
            "tokens_per_sec": round(tps, 2),
            "output_text": output["choices"][0]["text"].strip()
        })
        
    del llm
    gc.collect()
    return {
        "model_name": name,
        "vram_gb": round(vram_overhead, 2),
        "avg_tps": round(float(np.mean([x["tokens_per_sec"] for x in prompt_records])), 2),
        "runs": prompt_records
    }

if __name__ == "__main__":
    ORIGINAL_MODEL = "./models/Llama-3.2-3B-Instruct-BF16.gguf"
    MUTANT_MODEL = "./models/Llama-3.2-3B-Instruct-Mutant.gguf"
    
    original_data = benchmark_model(ORIGINAL_MODEL, "Baseline_BF16")
    mutant_data = benchmark_model(MUTANT_MODEL, "Hybrid_Mutant")
    
    if not original_data or not mutant_data:
        print("❌ Script aborted: One or both model paths returned invalid parameters.")
        sys.exit(1)

    # ---------------------------------------------------------
    # WRITE ENHANCED MARKDOWN SUMMARY REPOSITORY
    # ---------------------------------------------------------
    with open("benchmark_report.md", "w", encoding="utf-8") as f:
        f.write("# 📊 Comprehensive Hybridization Architecture & Behavior Report\n\n")
        
        # Section 1: Low-level weight structural corruption diagnostics
        f.write("## 🧬 1. Tensor Reconstruction Matrix (Weight Distortions)\n")
        f.write("> Shows how much the mutant weights strayed from the baseline layers.\n\n")
        f.write("| LAYER NAME | RECON. MSE ERROR | COSINE SIMILARITY | SPARSITY (0s %) | STATUS |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- |\n")
        for layer, metrics in TENSOR_DIAGNOSTICS.items():
            status = "🚨 HIGH DISTORTION" if metrics["mse"] > 2.0 else "🟢 STABLE"
            f.write(f"| `{layer}` | {metrics['mse']:.4f} | {metrics['cosine']:.4f} | {metrics['sparsity']}% | **{status}** |\n")
            
        # Section 2: Macro Hardware Aggregates
        f.write("\n## 🎛️ 2. Macro Resource & Velocity Allocation\n\n")
        f.write("| Hardware Performance Attribute | Baseline BF16 Model | Hybrid Mutant Model | Net Delta |\n")
        f.write("| :--- | :--- | :--- | :--- |\n")
        vram_diff = mutant_data['vram_gb'] - original_data['vram_gb']
        speed_pct = ((mutant_data['avg_tps'] - original_data['avg_tps']) / original_data['avg_tps']) * 100
        f.write(f"| **Model Graph VRAM Footprint** | {original_data['vram_gb']:.2f} GB | {mutant_data['vram_gb']:.2f} GB | {vram_diff:+.2f} GB |\n")
        f.write(f"| **Average Token Velocity** | {original_data['avg_tps']:.2f} t/s | {mutant_data['avg_tps']:.2f} t/s | {speed_pct:+.1f}% |\n\n")

        # Section 3: Side-by-Side Prompt Responses
        f.write("---\n\n## 🧩 3. Side-by-Side Behavioral Output Check\n\n")
        for orig, mut in zip(original_data["runs"], mutant_data["runs"]):
            f.write(f"### 📌 Test Case Axis: {orig['id']}\n")
            f.write(f"**Prompt:** *\"{orig['prompt']}\"*\n\n")
            f.write(f"* **Baseline Speed:** {orig['tokens_per_sec']} t/s | **Mutant Speed:** {mut['tokens_per_sec']} t/s\n\n")
            f.write(f"#### 🍏 Baseline Expected Output:\n> {orig['output_text']}\n\n")
            f.write(f"#### 🧪 Mutant Realized Output:\n> {mut['output_text']}\n\n")
            f.write("<br>\n\n")

    print("\n💾 Complete side-by-side weight alignment & behavior matrix saved to 'benchmark_report.md'")