# surgery.py

import os
import sys
import numpy as np
from gguf import GGUFReader, GGUFWriter

def ternary_158_quantize(tensor):
    """
    Surgically compresses a weight tensor into a 1.58-bit ternary format {-1, 0, 1}.
    """
    # Safeguard: Convert data cleanly depending on source quantization layout
    if tensor.data.dtype == np.float32:
        weights = tensor.data.copy()
    else:
        # Fallback handling for packed formats: view as float32 if data matches
        try:
            weights = np.frombuffer(tensor.data, dtype=np.float32).copy()
        except Exception:
            # If data array is non-standard bytes, cast to float32 array
            weights = tensor.data.astype(np.float32)

    if weights.size == 0:
        return weights, 1.0

    # Calculate Project README Formula: Delta = 0.7 * Mean(|W|)
    abs_mean = np.mean(np.abs(weights))
    threshold = 0.7 * abs_mean
    
    scale = np.max(np.abs(weights)) if np.max(np.abs(weights)) > 0 else 1.0
    
    # Apply ternary mapping
    ternary_weights = np.zeros_like(weights, dtype=np.float32)
    ternary_weights[weights > threshold] = 1.0
    ternary_weights[weights < -threshold] = -1.0
    
    return ternary_weights, scale

def perform_model_surgery(src_path, dest_path):
    print("🏥 Starting Model Surgery Phase...")
    print(f"📖 Reading original blueprint: {src_path}")
    
    reader = GGUFReader(src_path)
    writer = GGUFWriter(dest_path, arch="llama")

    # 1. Clone original metadata fields to preserve architectural geometry
    print("🧬 Replicating hyperparameter metadata...")
    for field in reader.fields.values():
        writer.add_key_value(field.name, field.data, field.type)

    # Add custom surgical metadata tracking keys
    writer.add_key_value("surgery.split_execution", True, type=3) # Type 3 = Boolean
    writer.add_key_value("surgery.ffn_format", "ternary_1.58b", type=3)

    # 2. Iterate through weights and apply the split-brain routing
    print("\n⚡ Beginning tensor splitting and grafting...")
    print("-" * 70)
    
    for tensor in reader.tensors:
        name = tensor.name
        shape = tensor.shape
        
        if "ffn" in name or "mlp" in name:
            print(f"🪓 [GRAFTING TERNARY 1.58-BIT]  -> {name} (System RAM)")
            
            # Compute ternary weights and scale vector
            ternary_array, scale = ternary_158_quantize(tensor)
            
            # Add mutated tensor cleanly using native array structures
            writer.add_tensor(name, ternary_array, raw_shape=shape)
            
            # Weld the tracking scale tensor right next to it
            scale_name = f"{name}.scale"
            writer.add_tensor(scale_name, np.array([scale], dtype=np.float32))
            
        else:
            # Pass attention blocks and embedding arrays straight through
            print(f"💎 [PRESERVING TRACK]           -> {name}")
            writer.add_tensor(name, tensor.data, raw_shape=shape)

    print("-" * 70)
    print("💾 Committing surgical changes to final file payload...")
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    
    print(f"\n🎉 Surgery Complete! Mutant Architecture Saved at: {dest_path}")

if __name__ == "__main__":
    SRC_MODEL = "./models/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf"
    OUTPUT_MODEL = "./models/Meta-Llama-3-8B-Surgically-Split-1.58b.gguf"
    
    if not os.path.exists(SRC_MODEL):
        print(f"❌ Error: Source model missing at {SRC_MODEL}")
        sys.exit(1)
        
    perform_model_surgery(SRC_MODEL, OUTPUT_MODEL)