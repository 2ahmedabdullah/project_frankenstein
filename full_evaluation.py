# full_evaluation.py

import numpy as np
from gguf import GGUFReader

ORIGINAL_PATH = "./models/Llama-3.2-3B-Instruct-BF16.gguf"
MUTANT_PATH = "./models/Llama-3.2-3B-Instruct-Mutant.gguf"
# MUTANT_PATH = "./models/Llama-3.2-3B-Instruct-HeaderPatched.gguf"
SIMULATED_PATH = "./models/Llama-3.2-3B-Instruct-Simulated.gguf"

# 🎛️ TOGGLE THIS TO SWITCH EVALUATION TARGETS
MODEL_MODE = "mutant"  # "mutant" or "simulated"

def decode_bf16_buffer(tensor_data):
    """
    Safely unpacks a raw GGUF BF16 byte buffer into standard NumPy Float32 
    by mapping the 16-bit structural layout directly to 32-bit floats.
    """
    raw_bytes = memoryview(tensor_data).tobytes()
    uint16_data = np.frombuffer(raw_bytes, dtype=np.uint16)
    
    fp32_data = np.zeros(len(uint16_data), dtype=np.float32)
    uint32_view = fp32_data.view(np.uint32)
    uint32_view[:] = uint16_data.astype(np.uint32) << 16
    
    return fp32_data

def unpack_radix3_mutant_block(tensor_data, expected_elements):
    raw_bytes = memoryview(tensor_data).tobytes()
    block_size = 9
    num_blocks = len(raw_bytes) // block_size
    
    total_packed_capacity = num_blocks * 32
    unpacked_weights = np.empty(total_packed_capacity, dtype=np.float32)
    
    # 🤫 Suppress the NaN/Inf warnings from trailing padding bytes
    with np.errstate(invalid='ignore'):
        for i in range(num_blocks):
            offset = i * block_size
            block = raw_bytes[offset:offset + block_size]
            
            scale = np.frombuffer(block[0:2], dtype=np.float16)[0].astype(np.float32)
            payload = np.frombuffer(block[2:9], dtype=np.uint8)
            
            digits = np.zeros(32, dtype=np.int8)
            for b in range(6):
                byte_val = int(payload[b])
                idx = b * 5
                digits[idx]     = byte_val // 81
                digits[idx + 1] = (byte_val % 81) // 27
                digits[idx + 2] = (byte_val % 27) // 9
                digits[idx + 3] = (byte_val % 9)  // 3
                digits[idx + 4] = byte_val % 3

            digits[30] = int(payload[6]) // 3
            digits[31] = int(payload[6]) % 3

            unpacked_weights[i*32 : (i+1)*32] = (digits.astype(np.float32) - 1.0) * scale
        
    return unpacked_weights[:expected_elements]


def extract_clean_weights(gguf_tensor, is_ffn_layer=False):
    """
    Decodes tensor weights cleanly based on structural shapes and model expectations.
    """
    expected_elements = int(np.prod(gguf_tensor.shape))
    
    # Force access to the exact raw underlying byte buffer from GGUF
    raw_bytes = memoryview(gguf_tensor.data).tobytes()
    
    # 🚨 FORCED MUTANT ROUTING: If we are in mutant mode and this is an FFN layer,
    # decode using the 9-byte radix unpacker regardless of what dtype metadata claims.
    if MODEL_MODE == "mutant" and is_ffn_layer:
        return unpack_radix3_mutant_block(raw_bytes, expected_elements)
            
    # Standard floating point fallback handling
    if len(raw_bytes) == expected_elements * 2:
        return decode_bf16_buffer(gguf_tensor.data)
        
    return np.array(gguf_tensor.data, dtype=np.float32).flatten()

def get_metrics(orig_flat, eval_flat):
    min_len = min(orig_flat.size, eval_flat.size)
    o = orig_flat[:min_len]
    e = eval_flat[:min_len]
    
    valid_mask = np.isfinite(o) & np.isfinite(e)
    if not np.any(valid_mask):
        return 0.0, 0.0
        
    o = o[valid_mask]
    e = e[valid_mask]
    
    mse = np.mean((o - e) ** 2)
    norm_o, norm_e = np.linalg.norm(o), np.linalg.norm(e)
    cosine_sim = np.dot(o, e) / (norm_o * norm_e) if norm_o and norm_e else 0.0
    return mse, cosine_sim

def run_comprehensive_evaluation():
    target_path = MUTANT_PATH if MODEL_MODE == "mutant" else SIMULATED_PATH
    print(f"🔬 INITIALIZING FULL ARCHITECTURE MUTATION REPORT [{MODEL_MODE.upper()} MODE]...")
    print(f"   Target File: {target_path}")
    
    orig_reader = GGUFReader(ORIGINAL_PATH)
    eval_reader = GGUFReader(target_path)
    eval_tensors = {t.name: t for t in eval_reader.tensors}
    
    categories = {
        "FFN Core (Grafted Ternary)": ["ffn_up", "ffn_gate", "ffn_down"],
        "Attention Projections (Preserved)": ["attn_q", "attn_k", "attn_v", "attn_output"],
        "Layer Norms & Embeddings": ["norm", "token_embd", "output.weight"]
    }
    
    stats = {cat: {"mse": [], "cos": [], "count": 0} for cat in categories}
    
    print("\n" + "="*115)
    print(f"{'GROUP / TENSOR COMPONENT':<45} | {'RECON. MSE':<15} | {'COSINE SIMILARITY':<18} | {'STATUS':<15}")
    print("="*115)
    
    for o_tensor in orig_reader.tensors:
        name = o_tensor.name
        if name not in eval_tensors:
            continue
            
        e_tensor = eval_tensors[name]
        
        # 1. Classify tensor group
        matched_cat = "Layer Norms & Embeddings"
        is_ffn_layer = False
        for cat, keywords in categories.items():
            if any(kw in name for kw in keywords):
                matched_cat = cat
                if cat == "FFN Core (Grafted Ternary)":
                    is_ffn_layer = True
                break
        
        try:
            orig_flat = extract_clean_weights(o_tensor)
            eval_flat = extract_clean_weights(e_tensor, is_ffn_layer=is_ffn_layer)
            
            if orig_flat.size == 0 or eval_flat.size == 0:
                continue
                
            # 🔍 DIAGNOSTIC PRINT: Inspecting targeted states
            if "blk.0.ffn_gate.weight" in name:
                print(f"\n🔍 DEBUG SNAPSHOT FOR {name}:")
                print(f"  -> Original: Type={type(orig_flat)}, Dtype={orig_flat.dtype}, Shape={orig_flat.shape}")
                print(f"  -> Original sample values: {orig_flat[:5]}")
                print(f"  -> Eval Array: Type={type(eval_flat)}, Dtype={eval_flat.dtype}, Shape={eval_flat.shape}")
                print(f"  -> Eval sample values: {eval_flat[:5]}")
                print(f"  -> GGUF Tensor Types: Original Type ID={getattr(o_tensor, 'type', 'N/A')}, Eval Type ID={getattr(e_tensor, 'type', 'N/A')}\n")
                
            mse, cosine_sim = get_metrics(orig_flat, eval_flat)
            
            stats[matched_cat]["mse"].append(mse)
            stats[matched_cat]["cos"].append(cosine_sim)
            stats[matched_cat]["count"] += 1
            
            if "blk.0" in name or "blk.14" in name or "output_norm" in name:
                status = "MUTATED" if is_ffn_layer else "PRESERVED"
                print(f"👉 {name:<42} | {mse:<15.6f} | {cosine_sim:<18.6f} | {status:<15}")
                
        except Exception as e:
            print(f"❌ CRASH ON LAYER {name}: {str(e)}")
            continue
            
    print("="*115)
    print("📊 COMPREHENSIVE ARCHITECTURE MUTATION SUMMARY")
    print("=" * 115)
    for cat, data in stats.items():
        if data["count"] > 0:
            avg_mse = np.mean(data["mse"])
            avg_cos = np.mean(data["cos"])
            print(f"🔹 {cat:<42} | Avg MSE: {avg_mse:.6f} | Avg Cos Sim: {avg_cos:.6f} | Items: {data['count']}")
    print("="*115)

if __name__ == "__main__":
    run_comprehensive_evaluation()