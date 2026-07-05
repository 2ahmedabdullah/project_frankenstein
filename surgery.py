# surgery.py

import os
import sys
import numpy as np
from gguf import GGUFReader, GGUFWriter, GGUFValueType

# ==============================================================================
# CONFIGURATION VARIABLE
# ==============================================================================
RUN_MODE = "packed"  

# ==============================================================================
# 1. HELPERS AND QUANTIZATION LOGIC
# ==============================================================================

def encode_fp32_to_bf16_bytes(fp32_array):
    """
    Converts a standard float32 numpy array into raw BF16 bytes
    by isolating the upper 16 bits of the 32-bit float structure.
    """
    uint32_view = fp32_array.flatten().view(np.uint32)
    uint16_bf16 = (uint32_view >> 16).astype(np.uint16)
    return uint16_bf16.tobytes()

def encode_ternary_q8_block(weights_chunk):
    """
    Processes a block of 32 weights for standard validation checks.
    Returns a 2-byte scale (max abs val) + 32-byte hard int8 (-1, 0, 1).
    """
    abs_mean = np.mean(np.abs(weights_chunk))
    threshold = 0.7 * abs_mean
    
    ternary = np.zeros(32, dtype=np.int8)
    ternary[weights_chunk > threshold] = 1
    ternary[weights_chunk < -threshold] = -1

    scale = np.max(np.abs(weights_chunk)) if np.max(np.abs(weights_chunk)) > 0 else 1.0
    scale_bytes = np.array([scale], dtype=np.float16).tobytes()
    return scale_bytes + ternary.tobytes()

def custom_ternary_quantize_q8(tensor_data):
    """
    Packs a continuous weight matrix into a standard GGUF Q8_0 physical container.
    """
    flat_weights = tensor_data.flatten()
    remainder = len(flat_weights) % 32
    if remainder != 0:
        flat_weights = np.concatenate([flat_weights, np.zeros(32 - remainder, dtype=np.float32)])
        
    num_blocks = len(flat_weights) // 32
    packed_tensor_bytes = bytearray()
    
    for b in range(num_blocks):
        block_chunk = flat_weights[b*32 : (b+1)*32]
        packed_tensor_bytes.extend(encode_ternary_q8_block(block_chunk))
        
    return np.frombuffer(packed_tensor_bytes, dtype=np.int8)


# ==============================================================================
# 1.58-BIT MUTANT TRACK (PADDED TO NATIVE Q4_0 SHELL)
# ==============================================================================

def encode_mutant_true_158_radix(weights_chunk):
    """
    Packs a 32-weight ternary chunk into a true, hardware-compliant 
    18-byte Q4_0 block (2-byte scale delta + 16-byte packed nibble payload).
    """
    # 🔍 Fix 1: Calculate a scientifically accurate delta scale factor
    max_abs = float(np.max(np.abs(weights_chunk)))
    scale = max_abs / 7.0 if max_abs > 1e-5 else 0.0
    
    # Ternary Thresholding 
    abs_mean = np.mean(np.abs(weights_chunk))
    threshold = 0.7 * abs_mean
    
    # Map values directly into true Q4_0 hardware expected integer states
    # Standard Q4_0 uses symmetric mapping: 0 is neutral, positive/negative map to integers
    quant_digits = np.zeros(32, dtype=np.uint8)
    if scale > 0:
        quant_digits[weights_chunk > threshold] = 7   # +1 State mapped to positive max
        quant_digits[weights_chunk < -threshold] = 8  # -1 State mapped via two's complement / offset signed boundary
    
    # 🔍 Fix 2: Pack 32 elements into 16 bytes using true 4-bit low/high nibble splitting
    # GGUF Q4_0 layout places element i and element i+16 into the same byte split
    packed_payload = np.zeros(16, dtype=np.uint8)
    for i in range(16):
        low_nibble = quant_digits[i] & 0x0F
        high_nibble = (quant_digits[i + 16] & 0x0F) << 4
        packed_payload[i] = low_nibble | high_nibble

    # Convert delta scale into the required 2-byte FP16 field
    scale_bytes = np.array([scale], dtype=np.float16).tobytes()
    
    # Combine directly into a perfectly aligned 18-byte block
    return scale_bytes + packed_payload.tobytes()


def custom_mutant_quantize_true_158(tensor_data):
    """
    Lightning-fast vectorized driver. Quantizes the entire matrix at once 
    using numpy vector operations instead of slow python block loops.
    """
    flat_weights = tensor_data.flatten()
    remainder = len(flat_weights) % 32
    if remainder != 0:
        flat_weights = np.concatenate([flat_weights, np.zeros(32 - remainder, dtype=np.float32)])
        
    num_blocks = len(flat_weights) // 32
    
    # Reshape into a 2D grid where every single row is a block of 32 weights
    blocks = flat_weights.reshape(num_blocks, 32)
    
    # 1. Parallelize Scale Calculation (Shape: [num_blocks, 1])
    max_abs = np.max(np.abs(blocks), axis=1, keepdims=True)
    scales = np.where(max_abs > 1e-5, max_abs / 7.0, 0.0).astype(np.float16)
    
    # 2. Parallelize Thresholding (Shape: [num_blocks, 1])
    abs_mean = np.mean(np.abs(blocks), axis=1, keepdims=True)
    thresholds = 0.7 * abs_mean
    
    # 3. Create Ternary Quant Digit Grid (Shape: [num_blocks, 32])
    quant_digits = np.zeros_like(blocks, dtype=np.uint8)
    
    # Mask out the zones over the entire matrix instantly
    valid_scale_mask = (scales > 0)
    quant_digits[(blocks > thresholds) & valid_scale_mask] = 7  # +1 Zone
    quant_digits[(blocks < -thresholds) & valid_scale_mask] = 8 # -1 Zone
    
    # 4. Vectorized Hardware Nibble Packing
    # Splitting element i and i+16 for all blocks simultaneously
    low_nibbles = quant_digits[:, :16] & 0x0F
    high_nibbles = (quant_digits[:, 16:] & 0x0F) << 4
    packed_payloads = low_nibbles | high_nibbles  # Shape: [num_blocks, 16]
    
    # 5. Interleave Scales and Payloads into physical Q4_0 Layout
    # Every block must be [Scale (2 bytes)][Payload (16 bytes)]
    scale_bytes_view = scales.view(np.uint8) # Interpret FP16 scales as 2 bytes
    
    # Stitch them together block by block natively in memory
    final_packed_bytes = np.empty((num_blocks, 18), dtype=np.uint8)
    final_packed_bytes[:, :2] = scale_bytes_view
    final_packed_bytes[:, 2:] = packed_payloads
    
    return final_packed_bytes.flatten()

# ==============================================================================
# 2. MASTER SURGERY RUNNER
# ==============================================================================
def perform_model_surgery(src_path, dest_path, mode="simulated"):
    print(f"\n🏥 Starting Surgery Phase | MODE: {mode.upper()}")
    
    reader = GGUFReader(src_path)
    writer = GGUFWriter(dest_path, arch="llama")
    
    writer.alignment = 256
    writer.data_alignment = 256
    writer.add_uint32("general.alignment", 256)
    
    if hasattr(writer, "pad_to_alignment"):
        writer.pad_to_alignment = 256
        
    print(f"🎯 Adjusted GGUFWriter alignment grid to 256 bytes to match physical layout.")

    print("✨ Replicating metadata parameters...")
    for key, field in reader.fields.items():
        if key in ["GGUF.version", "GGUF.tensor_count", "GGUF.kv_count", "general.architecture", "general.alignment"]:
            continue 
            
        try:
            val_type = field.types[0] if field.types else GGUFValueType.UINT32
            if val_type == GGUFValueType.ARRAY:
                arr_type = field.types[1] if len(field.types) > 1 else GGUFValueType.UINT32
                data_parts = [field.parts[i] for i in field.data]
                if arr_type == GGUFValueType.STRING:
                    clean_list = [p.tobytes().decode('utf-8', errors='ignore') for p in data_parts]
                    writer.add_array(key, clean_list)
                else:
                    combined = np.concatenate(data_parts) if len(data_parts) > 1 else data_parts[0]
                    writer.add_array(key, combined.tolist())
            elif val_type == GGUFValueType.STRING:
                # 🛠️ Fix scalar string extraction directly
                val = field.parts[-1].tobytes().decode('utf-8', errors='ignore')
                writer.add_key_value(key, val, val_type)
            else:
                raw_part = field.parts[-1]
                if hasattr(raw_part, "item"):
                    val = raw_part.item()
                elif isinstance(raw_part, (bytes, bytearray)):
                    val = raw_part
                else:
                    val = raw_part[0]
                    
                if val_type in [GGUFValueType.UINT8, GGUFValueType.INT8, GGUFValueType.UINT16, 
                                  GGUFValueType.INT16, GGUFValueType.UINT32, GGUFValueType.INT32]:
                    val = int(val)
                elif val_type in [GGUFValueType.FLOAT32, GGUFValueType.FLOAT64]:
                    val = float(val)
                elif val_type == GGUFValueType.BOOL:
                    val = bool(val)
                    
                writer.add_key_value(key, val, val_type)
                
        except Exception as e:
            print(f"⚠️ Warning: Skipping key '{key}': {e}")

    # ==========================================================================
    # 🎯 HARDCODED OVERRIDES FOR LLAMA 3.2 3B
    # ==========================================================================
    print("🚀 Forcing explicit Llama 3.2 3B architectural settings...")
    writer.add_uint32("llama.block_count", 28)
    writer.add_uint32("llama.embedding_length", 3072)
    writer.add_uint32("llama.feed_forward_length", 8192)  # <-- This forces n_ff to 8192
    writer.add_uint32("llama.attention.head_count", 24)
    writer.add_uint32("llama.attention.head_count_kv", 8)
    writer.add_uint32("llama.rope.dimension_count", 128)
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    # ==========================================================================
    # ⚡ 3. FULL TENSOR PROCESSING TRACK
    # ==========================================================================
    total_tensors = len(reader.tensors)
    print(f"⚡ Processing {total_tensors} total tensors...")
    
    for idx, tensor in enumerate(reader.tensors, 1):
        name = tensor.name
            
        if tensor.data.dtype in [np.float16, np.float32]:
            orig_data = tensor.data.astype(np.float32)
        else:
            try:
                raw_bytes = memoryview(tensor.data).tobytes()
                uint16_data = np.frombuffer(raw_bytes, dtype=np.uint16)
                fp32_data = np.zeros(len(uint16_data), dtype=np.float32)
                fp32_data.view(np.uint32)[:] = uint16_data.astype(np.uint32) << 16
                orig_data = fp32_data
            except Exception:
                orig_data = tensor.data.astype(np.float32)

        if ("ffn" in name or "mlp" in name) and not "norm" in name:
            if mode == "packed":
                # 1. Run our vectorized packing function to get the raw uint8 bytes
                packed_bytes = custom_mutant_quantize_true_158(orig_data)
                
                # 🧠 THE STRATEGY: Provide the exact physical byte array size.
                # This ensures (total_bytes % 18 == 0) holds true, clearing the internal check.
                total_bytes_count = len(packed_bytes) # This equals 14155776
                modified_shape_gguf = [total_bytes_count]
                
                from gguf import GGMLQuantizationType
                target_dtype = GGMLQuantizationType.Q4_0
                
                writer.add_tensor(
                    name=name, 
                    tensor=packed_bytes,  # Clean uint8 numpy array
                    raw_shape=modified_shape_gguf, 
                    raw_dtype=target_dtype
                )
                print(f"✂️ Wrapped Mutant into Q4_0 Shell: {name} | Total Bytes: {total_bytes_count}")

            else:
                flat = orig_data.flatten()
                sim_flat = np.zeros_like(flat, dtype=np.float32)
                chunk_size = 32
                num_blocks = len(flat) // chunk_size
                
                for b in range(num_blocks):
                    start_idx = b * chunk_size
                    end_idx = start_idx + chunk_size
                    chunk = flat[start_idx:end_idx]
                    abs_mean = float(np.mean(np.abs(chunk)))
                    threshold = 0.7 * abs_mean
                    if threshold < 1e-7: threshold = 0.0
                    
                    ternary = np.zeros(chunk_size, dtype=np.float32)
                    ternary[chunk > threshold] = 1.0
                    ternary[chunk < -threshold] = -1.0
                    sim_flat[start_idx:end_idx] = ternary * threshold
                
                if len(flat) % chunk_size != 0:
                    remainder_start = num_blocks * chunk_size
                    chunk = flat[remainder_start:]
                    abs_mean = float(np.mean(np.abs(chunk)))
                    threshold = 0.7 * abs_mean if (0.7 * abs_mean) >= 1e-7 else 0.0
                    ternary = np.zeros(len(chunk), dtype=np.float32)
                    ternary[chunk > threshold] = 1.0
                    ternary[chunk < -threshold] = -1.0
                    sim_flat[remainder_start:] = ternary * threshold

                sim_flat = np.nan_to_num(sim_flat, nan=0.0)
                bf16_raw_bytes = encode_fp32_to_bf16_bytes(sim_flat)
                bf16_array = np.frombuffer(bf16_raw_bytes, dtype=np.uint16).reshape(tensor.shape)
                writer.add_tensor(name, bf16_array, raw_shape=tensor.shape[::-1], raw_dtype=int(tensor.tensor_type))
        else:
            from gguf import GGML_QUANT_SIZES
            
            start_offset = int(tensor.data_offset)
            ttype = int(tensor.tensor_type)
            
            if ttype in GGML_QUANT_SIZES:
                block_size, type_size = GGML_QUANT_SIZES[ttype]
                num_elements = int(np.prod(tensor.shape))
                exact_disk_bytes = (num_elements // block_size) * type_size
            else:
                exact_disk_bytes = int(tensor.n_bytes)
            
            with open(src_path, "rb") as src_f:
                src_f.seek(start_offset)
                clean_bytes = src_f.read(exact_disk_bytes)
            
            num_elements = int(np.prod(tensor.shape))
            element_size = exact_disk_bytes // num_elements
            
            if element_size == 2:
                clean_data = np.frombuffer(clean_bytes, dtype=np.uint16)
            elif element_size == 4:
                clean_data = np.frombuffer(clean_bytes, dtype=np.float32)
            else:
                clean_data = np.frombuffer(clean_bytes, dtype=np.uint8)
            
            writer.add_tensor(
                name=name, 
                tensor=clean_data, 
                raw_shape=tensor.shape[::-1],
                raw_dtype=ttype
            )

    sys.stdout.write("\n") 
    print("💾 Committing surgical changes to disk...")
    
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    
    print(f"🎉 Saved successfully at: {dest_path}")


if __name__ == "__main__":
    SRC_MODEL = "./models/Llama-3.2-3B-Instruct-BF16.gguf"
    SIM_OUTPUT = "./models/Llama-3.2-3B-Instruct-Simulated.gguf"
    MUTANT_OUTPUT = "./models/Llama-3.2-3B-Instruct-Mutant.gguf"
    
    if not os.path.exists(SRC_MODEL):
        print(f"❌ Error: Source model missing at {SRC_MODEL}")
        sys.exit(1)
        
    valid_modes = ["simulated", "packed"]
    mode_selection = RUN_MODE.strip().lower()
    
    if mode_selection not in valid_modes:
        print(f"❌ Error: Invalid RUN_MODE value '{RUN_MODE}'. Use 'simulated' or 'packed'.")
        sys.exit(1)

    if mode_selection == "simulated":
        perform_model_surgery(SRC_MODEL, SIM_OUTPUT, mode="simulated")
    elif mode_selection == "packed":
        perform_model_surgery(SRC_MODEL, MUTANT_OUTPUT, mode="packed")
    
    print("\n🚀 Run finished. Target file updated successfully.")