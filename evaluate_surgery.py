# evaluate_surgery.py

import numpy as np
from gguf import GGUFReader

ORIGINAL_PATH = "./models/Llama-3.2-3B-Instruct-BF16.gguf"
MUTANT_PATH = "./models/Llama-3.2-3B-Instruct-Mutant.gguf"

def calculate_similarity_metrics(orig, mutant):
    # Flatten tensors for global array comparisons
    o_flat = orig.flatten().astype(np.float32)
    m_flat = mutant.flatten().astype(np.float32)
    
    # 1. Mean Squared Error (MSE)
    mse = np.mean((o_flat - m_flat) ** 2)
    
    # 2. Cosine Similarity
    norm_o = np.linalg.norm(o_flat)
    norm_m = np.linalg.norm(m_flat)
    if norm_o == 0 or norm_m == 0:
        cosine_sim = 0.0
    else:
        cosine_sim = np.dot(o_flat, m_flat) / (norm_o * norm_m)
        
    return mse, cosine_sim

def run_diagnostic_report():
    print("🔬 Extracting model footprints for degradation report...")
    
    orig_reader = GGUFReader(ORIGINAL_PATH)
    mutant_reader = GGUFReader(MUTANT_PATH)
    
    # Map out mutant tensors for quick structural matching
    mutant_tensors = {t.name: t for t in mutant_reader.tensors}
    
    print("\n" + "="*90)
    print(f"{'LAYER NAME':<40} | {'MSE ERROR':<12} | {'COSINE SIM':<12} | {'SPARSITY (0s)':<12}")
    print("="*90)
    
    total_layers_evaluated = 0
    running_mse = 0.0
    running_cosine = 0.0
    
    for o_tensor in orig_reader.tensors:
        name = o_tensor.name
        
        # We only care about tracking the modified FFN blocks
        if "ffn" in name or "mlp" in name:
            if name not in mutant_tensors:
                continue
                
            m_tensor = mutant_tensors[name]
            
            orig_data = np.array(o_tensor.data)
            mutant_data = np.array(m_tensor.data)
            
            # If shapes don't align or data is empty, skip gracefully
            if orig_data.shape != mutant_data.shape or orig_data.size == 0:
                continue
                
            # Compute stats
            mse, cosine_sim = calculate_similarity_metrics(orig_data, mutant_data)
            
            # Calculate actual weight sparsity (how many elements are exactly 0)
            zeros = np.count_nonzero(mutant_data == 0)
            sparsity_pct = (zeros / mutant_data.size) * 100
            
            print(f"{name:<40} | {mse:<12.6f} | {cosine_sim:<12.6f} | {sparsity_pct:<11.2f}%")
            
            running_mse += mse
            running_cosine += cosine_sim
            total_layers_evaluated += 1
            
    print("="*90)
    if total_layers_evaluated > 0:
        avg_mse = running_mse / total_layers_evaluated
        avg_cos = running_cosine / total_layers_evaluated
        print(f"{'AVERAGE FFN DISTORTION METRICS':<40} | {avg_mse:<12.6f} | {avg_cos:<12.6f} |")
    print("="*90)

if __name__ == "__main__":
    run_diagnostic_report()