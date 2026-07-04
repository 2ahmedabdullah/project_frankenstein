# benchmark_behavior.py 
import os
import sys
import time
import datetime
import gc
import json
import numpy as np

# 🎛️ MANUALLY TOGGLE YOUR VARIABLE HERE: "baseline", "simulated", OR "packed_mutant"
MODEL_MODE = "simulated"  

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

TENSOR_DIAGNOSTICS = {
    "FFN Core (Grafted Ternary)":       {"mse": 0.000143, "cosine": 0.898801, "status": "MUTATED (2.62 bpw)"},
    "Attention Projections (Preserved)":  {"mse": 0.000000, "cosine": 1.000000, "status": "PRESERVED"},
    "Layer Norms & Embeddings":           {"mse": 0.000000, "cosine": 1.000000, "status": "PRESERVED"}
}

PROMPTS = [
    {"id": "GSM8K_Math", "prompt": "Solve for x: 3x + 7 = 22. Show your work step-by-step."},
    {"id": "Logic_Riddle", "prompt": "A man has 4 sons. Each son has one sister. How many children does the man have?"},
    {"id": "MMLU_Concept", "prompt": "What are the primary structural differences between a standard Transformer LLM block and an isolated Feed-Forward Network (FFN) layer?"}
]

def benchmark_model(model_path, name):
    print(f"\n🏗️ Loading Graph: {name} ({model_path})")
    if not os.path.exists(model_path):
        print(f"⚠️ Path missing: {model_path}")
        return None

    base_vram = get_gpu_vram()
    try:
        # Note: If testing 'packed_mutant' without custom C++ builds, this will throw an exception
        llm = Llama(model_path=model_path, n_ctx=512, n_gpu_layers=-1, verbose=True)
        time.sleep(1)
        loaded_vram = get_gpu_vram()
        vram_overhead = max(0.0, loaded_vram - base_vram)
    except Exception as e:
        print(f"❌ ENGINE REJECTION: {name} failed initialization.")
        print(f"   Reason: {e}")
        return {
            "model_name": name, "vram_gb": 0.0, "avg_tps": 0.0, 
            "runs": [{"id": p["id"], "prompt": p["prompt"], "tokens_per_sec": 0.0, "output_text": f"REJECTED BY RUNTIME ENGINE: {e}"} for p in PROMPTS]
        }

    prompt_records = []
    for item in PROMPTS:
        p_id = item["id"]
        prompt_text = item["prompt"]
        formatted_prompt = f"<|start_header_id|>user<|end_header_id|>\n\n{prompt_text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        
        print(f"⚡ Processing Test Case: {p_id}...")
        start_time = time.time()
        
        # 🟥 REMOVED THE TRY/EXCEPT BLOCK TO FORCE THE EXCEPTION TO SURFACE
        output = llm(formatted_prompt, max_tokens=64, temperature=0.1, stop=["<|eot_id|>"])
        
        total_time = time.time() - start_time
        tokens = output["usage"]["completion_tokens"]
        tps = tokens / total_time if total_time > 0 else 0
        text_out = output["choices"][0]["text"].strip()
        
        print(f"✅ Generated {tokens} tokens at {tps:.2f} t/s")
        
        prompt_records.append({
            "id": p_id, "prompt": prompt_text, "tokens_per_sec": round(tps, 2), "output_text": text_out or "[EMPTY]"
        })
        
    del llm
    gc.collect()
    return {
        "model_name": name, "vram_gb": round(vram_overhead, 2),
        "avg_tps": round(float(np.mean([x["tokens_per_sec"] for x in prompt_records])), 2) if prompt_records else 0.0,
        "runs": prompt_records
    }

def update_report(run_data, mode):
    report_file = "benchmark_report.md"
    state_file = "benchmark_state.json"
    state = {"baseline": None, "simulated": None, "packed_mutant": None}
    
    if os.path.exists(state_file):
        try:
            with open(state_file, "r") as sf:
                state = json.load(sf)
        except:
            pass

    state[mode] = run_data
    with open(state_file, "w") as sf:
        json.dump(state, sf, indent=4)

    with open(report_file, "w", encoding="utf-8") as f:
        f.write("# 📊 Comprehensive Hybridization Architecture & Behavior Report\n\n")
        
        f.write("## 🧬 1. Tensor Reconstruction Matrix (Weight Distortions)\n\n")
        f.write("| LAYER COMPONENT | RECON. MSE ERROR | COSINE SIMILARITY | STATUS |\n")
        f.write("| :--- | :--- | :--- | :--- |\n")
        for layer, metrics in TENSOR_DIAGNOSTICS.items():
            f.write(f"| **{layer}** | {metrics['mse']:.6f} | {metrics['cosine']:.6f} | `{metrics['status']}` |\n")
            
        f.write("\n## 🎛️ 2. Macro Resource & Velocity Allocation\n\n")
        f.write("| Hardware Performance Attribute | Baseline BF16 | Simulated Hybrid | Packed Mutant (Radix) |\n")
        f.write("| :--- | :--- | :--- | :--- |\n")
        
        b = state.get("baseline")
        s = state.get("simulated")
        m = state.get("packed_mutant")
        
        # 🛠️ Extract formatting logic out of the f-string to prevent SyntaxErrors
        b_vram_str = f"{b['vram_gb']:.2f} GB" if b else "WAITING"
        s_vram_str = f"{s['vram_gb']:.2f} GB" if s else "WAITING"
        m_vram_str = f"{m['vram_gb']:.2f} GB" if m else "WAITING"
        
        b_tps_str = f"{b['avg_tps']:.2f} t/s" if b else "WAITING"
        s_tps_str = f"{s['avg_tps']:.2f} t/s" if s else "WAITING"
        m_tps_str = f"{m['avg_tps']:.2f} t/s" if m else "WAITING"
        
        # Write clean, uncomplicated rows
        f.write(f"| **Model Graph VRAM Footprint** | {b_vram_str} | {s_vram_str} | {m_vram_str} |\n")
        f.write(f"| **Average Token Velocity**     | {b_tps_str} | {s_tps_str} | {m_tps_str} |\n\n")

        f.write("---\n\n## 3. Side-by-Side Behavioral Output Check\n\n")
        for i, item in enumerate(PROMPTS):
            f.write(f"### 📌 Test Case Axis: {item['id']}\n")
            f.write(f"**Prompt:** *\"{item['prompt']}\"*\n\n")
            
            b_run = b["runs"][i] if b else None
            s_run = s["runs"][i] if s else None
            m_run = m["runs"][i] if m else None
            
            f.write(f"#### 🍏 Baseline Expected Output:\n> {b_run['output_text'] if b_run else 'PENDING'}\n\n")
            f.write(f"#### 🟡 Simulated Realized Output:\n> {s_run['output_text'] if s_run else 'PENDING'}\n\n")
            f.write(f"#### 🧪 Packed Mutant Realized Output:\n> {m_run['output_text'] if m_run else 'PENDING'}\n\n")
            f.write("<br>\n\n")

if __name__ == "__main__":
    valid_modes = ["baseline", "simulated", "packed_mutant"]
    if MODEL_MODE not in valid_modes:
        print(f"❌ Invalid MODEL_MODE execution target. Choose from {valid_modes}")
        sys.exit(1)

    if MODEL_MODE == "baseline":
        data = benchmark_model("./models/Llama-3.2-3B-Instruct-BF16.gguf", "Baseline_BF16")
        if data: update_report(data, "baseline")
    elif MODEL_MODE == "simulated":
        data = benchmark_model("./models/Llama-3.2-3B-Instruct-Simulated.gguf", "Hybrid_Simulated")
        if data: update_report(data, "simulated")
    elif MODEL_MODE == "packed_mutant":
        data = benchmark_model("./models/Llama-3.2-3B-Instruct-HeaderPatched.gguf", "Packed_Mutant_True158")
        if data: update_report(data, "packed_mutant")

    print(f"\n💾 State updated. 'benchmark_report.md' modified for mode: [{MODEL_MODE}]")