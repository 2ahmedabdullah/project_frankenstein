# surgery.py

import os
import sys
import numpy as np
from gguf import GGUFReader, GGUFWriter, GGUFValueType

def ternary_158_quantize(tensor):
    """
    Surgically compresses a raw float weight tensor into a 1.58-bit ternary format {-1, 0, 1}.
    """
    if tensor.data.dtype == np.float16:
        weights = tensor.data.astype(np.float32)
    elif tensor.data.dtype == np.float32:
        weights = tensor.data.copy()
    else:
        try:
            weights = np.frombuffer(tensor.data, dtype=np.float16).astype(np.float32)
        except Exception:
            weights = tensor.data.astype(np.float32)

    if weights.size == 0:
        return weights, 1.0

    # Calculate Formula: Delta = 0.7 * Mean(|W|)
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

    # 1. Cleanly un-nest and clone original metadata fields
    print("🧬 Replicating hyperparameter metadata...")
    for field in reader.fields.values():
        name = field.name
        
        # Avoid writing duplicate headers or our explicit tracking tags
        if name in ["GGUF.version", "GGUF.tensor_count", "GGUF.kv_count", 
                    "general.architecture", "surgery.split_execution", "surgery.ffn_format"]:
            continue
            
        try:
            if len(field.parts) > 0 and len(field.types) > 0:
                val_type = field.types[0]
                raw_val = field.parts[-1]
                
                # Pre-process NumPy arrays safely
                if isinstance(raw_val, np.ndarray):
                    # ONLY flatten to a scalar if it's not supposed to be an array field
                    if raw_val.size == 1 and val_type != GGUFValueType.ARRAY:
                        raw_val = raw_val.item()
                    else:
                        raw_val = raw_val.tolist()

                # If it's an ARRAY type or became a list after processing
                if val_type == GGUFValueType.ARRAY or isinstance(raw_val, list):
                    # Ensure raw_val is wrapped as an iterable if it somehow escaped as a scalar
                    if not isinstance(raw_val, list):
                        raw_val = [raw_val]
                        
                    clean_list = []
                    for item in raw_val:
                        if isinstance(item, bytes):
                            clean_list.append(str(item, encoding="utf-8"))
                        elif hasattr(item, "item"):
                            clean_list.append(item.item())
                        else:
                            clean_list.append(item)
                            
                    writer.add_array(name, clean_list)
                    continue

                # Handle regular non-array scalar values
                if isinstance(raw_val, bytes):
                    raw_val = str(raw_val, encoding="utf-8")

                writer.add_key_value(name, raw_val, val_type)
                
        except Exception as e:
            print(f"  ⚠️ Skipping metadata key '{name}' due to parsing friction: {e}")

    # Add custom surgical tracking metadata using strict raw types
    writer.add_key_value("surgery.split_execution", True, GGUFValueType.BOOL)  
    writer.add_key_value("surgery.ffn_format", "ternary_1.58b", GGUFValueType.STRING) 

    # 2. Iterate through weights and apply the split-brain routing
    print("\n⚡ Beginning tensor splitting and grafting...")
    print("-" * 70)
    
    for tensor in reader.tensors:
        name = tensor.name
        shape = tensor.shape
        
        # Convert any raw memoryviews or byte objects into a structured numpy array
        tensor_data = np.array(tensor.data)
        
        if "ffn" in name or "mlp" in name:
            print(f"🪓 [GRAFTING TERNARY 1.58-BIT]  -> {name} (System RAM)")
            
            # Compute ternary weights and scale vector
            ternary_array, scale = ternary_158_quantize(tensor)
            
            # Write out explicitly as standard float16
            writer.add_tensor(name, ternary_array.astype(np.float16), raw_shape=shape)
            
            # Weld the tracking scale tensor right next to it
            scale_name = f"{name}.scale"
            writer.add_tensor(scale_name, np.array([scale], dtype=np.float16))
            
        else:
            # Pass attention blocks and embedding arrays straight through, 
            # safely converting any unsupported BF16 formats to standard F16
            print(f"💎 [PRESERVING TRACK]           -> {name}")
            
            if tensor_data.dtype == np.float32 or tensor_data.dtype == np.float16:
                safe_data = tensor_data
            else:
                try:
                    safe_data = tensor_data.astype(np.float16)
                except Exception:
                    safe_data = np.frombuffer(tensor.data, dtype=np.float16)
            
            writer.add_tensor(name, safe_data, raw_shape=shape)

    print("-" * 70)
    print("💾 Committing surgical changes to final file payload...")
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    
    print(f"\n🎉 Surgery Complete! Mutant Architecture Saved at: {dest_path}")

if __name__ == "__main__":
    SRC_MODEL = "./models/Llama-3.2-3B-Instruct-BF16.gguf"
    OUTPUT_MODEL = "./models/Llama-3.2-3B-Instruct-Mutant.gguf"
    
    if not os.path.exists(SRC_MODEL):
        print(f"❌ Error: Source model missing at {SRC_MODEL}")
        sys.exit(1)
        
    perform_model_surgery(SRC_MODEL, OUTPUT_MODEL)