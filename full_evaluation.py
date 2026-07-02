import numpy as np
from gguf import GGUFReader

ORIGINAL_PATH = "./models/Llama-3.2-3B-Instruct-BF16.gguf"
MUTANT_PATH = "./models/Llama-3.2-3B-Instruct-Mutant.gguf"

def get_metrics(orig, mutant):
    o_flat = orig.flatten().astype(np.float32)
    m_flat = mutant.flatten().astype(np.float32)
    
    mse = np.mean((o_flat - m_flat) ** 2)
    norm_o, norm_m = np.linalg.norm(o_flat), np.linalg.norm(m_flat)
    cosine_sim = np.dot(o_flat, m_flat) / (norm_o * norm_m) if norm_o and norm_m else 0.0
    return mse, cosine_sim

def run_comprehensive_evaluation():
    print("🔬 INITIALIZING FULL ARCHITECTURE MUTATION REPORT...")
    
    orig_reader = GGUFReader(ORIGINAL_PATH)
    mutant_reader = GGUFReader(MUTANT_PATH)
    mutant_tensors = {t.name: t for t in mutant_reader.tensors}
    
    # Track categories for structural summaries
    categories = {
        "FFN Core (Grafted Ternary)": ["ffn_up", "ffn_gate", "ffn_down"],
        "Attention Projections (Preserved)": ["attn_q", "attn_k", "attn_v", "attn_output"],
        "Layer Norms & Embeddings": ["norm", "token_embd", "output.weight"]
    }
    
    stats = {cat: {"mse": [], "cos": [], "count": 0} for cat in categories}
    
    print("\n" + "="*105)
    print(f"{'GROUP / TENSOR COMPONENT':<45} | {'RECON. MSE':<15} | {'COSINE SIMILARITY':<18} | {'STATUS':<15}")
    print("="*105)
    
    for o_tensor in orig_reader.tensors:
        name = o_tensor.name
        if name not in mutant_tensors:
            continue
            
        orig_data = np.array(o_tensor.data)
        mutant_data = np.array(mutant_tensors[name].data)
        
        if orig_data.shape != mutant_data.shape or orig_data.size == 0:
            continue
            
        mse, cosine_sim = get_metrics(orig_data, mutant_data)
        
        # Classify the tensor structural group
        matched_cat = "Layer Norms & Embeddings" # Default fallback
        for cat, keywords in categories.items():
            if any(kw in name for kw in keywords):
                matched_cat = cat
                break
                
        stats[matched_cat]["mse"].append(mse)
        stats[matched_cat]["cos"].append(cosine_sim)
        stats[matched_cat]["count"] += 1
        
        # Print representative snapshots or high-impact layers explicitly
        if "blk.0" in name or "blk.14" in name or "output" in name:
            status = "MUTATED" if "FFN Core" in matched_cat else "PRESERVED"
            print(f"👉 {name:<42} | {mse:<15.6f} | {cosine_sim:<18.6f} | {status:<15}")
            
    print("="*105)
    print("📊 COMPREHENSIVE ARCHITECTURE MUTATION SUMMARY")
    print("="*105)
    for cat, data in stats.items():
        if data["count"] > 0:
            avg_mse = np.mean(data["mse"])
            avg_cos = np.mean(data["cos"])
            print(f"🔹 {cat:<42} | Avg MSE: {avg_mse:.6f} | Avg Cos Sim: {avg_cos:.6f} | Items: {data['count']}")
    print("="*105)

if __name__ == "__main__":
    run_comprehensive_evaluation()