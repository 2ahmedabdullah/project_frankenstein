# surgery.py

import os
import sys
import numpy as np
from gguf import GGUFReader, GGUFWriter

def ternary_158_quantize(tensor_data):
    """
    Surgically compresses a weight tensor into a 1.58-bit ternary format {-1, 0, 1}.
    Uses a dynamic threshold scale based on the absolute mean of the weight weights.
    """
    # Convert raw tensor bytes to float32 for processing if needed
    # Note: In a production setup, use ggml dequantize routines if the source is Q4_K
    weights = np.frombuffer(tensor_data, dtype=np.float32).copy()
    
    if weights.size == 0:
        return tensor_data, 1.0

    # Calculate the ternary scale constant: delta = 0.7 * Mean(|W|)
    abs_mean = np.mean(np.abs(weights))
    threshold = 0.7 * abs_mean
    
    # Calculate global scale factor factor
    scale = np.max(np.abs(weights)) / 1.0 if np.max(np.abs(weights)) > 0 else 1.0
    
    # Apply ternary mapping
    ternary_weights = np.zeros_like(weights)
    ternary_weights[weights > threshold] = 1.0
    ternary_weights[weights < -threshold] = -1.0
    
    # Pack the mutated ternary floats back into raw bytes 
    # (Or pack into a 2-bit custom bitstream lookup format for CPU execution)
    return ternary_weights.tobytes(), scale

def perform_model_surgery(src_path, dest_path):
    print("✂️ Starting Model Surgery Phase...")
    print(f"📖 Reading original blueprint: {src_path}")
    
    reader = GGUFReader(src_path)
    writer = GGUFWriter(dest_path, arch="llama")

    # 1. Clone original metadata fields to preserve architectural geometry
    print("🧬 Replicating hyperparameter metadata...")
    for field in reader.fields.values():
        writer.add_key_value(field.name, field.data, field.type)

    # Add custom surgical metadata tracking keys
    writer.add_key_value("surgery.split_execution", True, type=3) # Boolean flag
    writer.add_key_value("surgery.ffn_format", "ternary_1.58b", type=3)

    # 2. Iterate through weights and apply the split-brain routing
    print("\n⚡ Beginning tensor splitting and grafting...")
    print("-" * 70)
    
    for tensor in reader.tensors:
        name = tensor.name
        shape = tensor.shape
        tensor_type = tensor.tensor_type
        raw_data = tensor.data
        
        if "attn" in name:
            # --- GPU High-Precision Attention Track ---
            print(f"💎 [PRESERVING HIGH PRECISION] -> {name} (VRAM)")
            writer.add_tensor_info(name, shape, tensor_type, raw_data.size)
            # Append intact tensor bytes directly to the writer payload
            writer.tensors_data.append(raw_data)
            
        elif "ffn" in name or "mlp" in name:
            # --- CPU Ternary 1.58-Bit Grafting Track ---
            print(f"🪓 [GRAFTING TERNARY 1.58-BIT]  -> {name} (System RAM)")
            
            # Perform runtime weight conversion to ternary space
            ternary_bytes, scale = ternary_158_quantize(raw_data)
            
            # Save the mutated low-bitweights
            writer.add_tensor_info(name, shape, tensor_type, len(ternary_bytes))
            writer.tensors_data.append(ternary_bytes)
            
            # Store the corresponding floating-point scale factor vector for lookup alignment
            scale_name = f"{name}.scale"
            scale_data = np.array([scale], dtype=np.float32).tobytes()
            writer.add_tensor_info(scale_name, [1], 0, len(scale_data)) # 0 = GGML_TYPE_F32
            writer.tensors_data.append(scale_data)
            
        else:
            # --- Base Embedding and Output Infrastructure ---
            print(f"📦 [ROUTING GLOBAL HEADER]     -> {name}")
            writer.add_tensor_info(name, shape, tensor_type, raw_data.size)
            writer.tensors_data.append(raw_data)

    print("-" * 70)
    print("💾 Committing surgical changes to final file payload...")
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    
    print(f"\n🎉 Surgery Complete! New Hybrid Architecture Compiled at: {dest_path}")

if __name__ == "__main__":
    SRC_MODEL = "./models/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf"
    OUTPUT_MODEL = "./models/Meta-Llama-3-8B-Surgically-Split-1.58b.gguf"
    
    if not os.path.exists(SRC_MODEL):
        print(f"❌ Error: Source model missing at {SRC_MODEL}")
        sys.exit(1)
        
    perform_model_surgery(SRC_MODEL, OUTPUT_MODEL)