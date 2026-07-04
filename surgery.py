# surgery.py

import os
import sys
import numpy as np
from gguf import GGUFReader, GGUFWriter, GGUFValueType

# ==============================================================================
# CONFIGURATION VARIABLE
# Set to "simulated" to generate the FP16/BF16 Float Baseline with ternary deltas.
# Set to "packed" to generate the 1.58-bit deployment file.
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
# 1.58-BIT MUTANT TRACK
# ==============================================================================

def encode_mutant_true_158_radix(weights_chunk):
    """
    True 1.58-bit Ternary Radix Packing (Base-3).
    Packs a 32-weight chunk into exactly 9 bytes (2-byte scale + 7-byte payload).
    """
    abs_mean = np.mean(np.abs(weights_chunk))
    threshold = 0.7 * abs_mean
    
    digits = np.ones(32, dtype=np.uint8)  # Default to 1 (0 state)
    digits[weights_chunk > threshold] = 2  # +1 state
    digits[weights_chunk < -threshold] = 0 # -1 state

    packed_bytes = np.zeros(7, dtype=np.uint8)

    for b in range(6):
        idx = b * 5
        packed_bytes[b] = (
            int(digits[idx])     * 81 +
            int(digits[idx + 1]) * 27 +
            int(digits[idx + 2]) * 9  +
            int(digits[idx + 3]) * 3  +
            int(digits[idx + 4])
        )

    packed_bytes[6] = int(digits[30]) * 3 + int(digits[31])

    scale = threshold if threshold > 0 else 1.0
    scale_bytes = np.array([scale], dtype=np.float16).tobytes()
    
    return scale_bytes + packed_bytes.tobytes()


def custom_mutant_quantize_true_158(tensor_data):
    """
    Quantization driver streaming the weight matrix into the true 1.58-bit radix layout.
    """
    flat_weights = tensor_data.flatten()
    remainder = len(flat_weights) % 32
    if remainder != 0:
        flat_weights = np.concatenate([flat_weights, np.zeros(32 - remainder, dtype=np.float32)])
        
    num_blocks = len(flat_weights) // 32
    packed_tensor_bytes = bytearray()
    
    for b in range(num_blocks):
        block_chunk = flat_weights[b*32 : (b+1)*32]
        packed_tensor_bytes.extend(encode_mutant_true_158_radix(block_chunk))
        
    return np.frombuffer(packed_tensor_bytes, dtype=np.int8)

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
            else:
                raw_part = field.parts[-1].tobytes() if field.parts else b""
                if val_type == GGUFValueType.STRING:
                    val = raw_part.decode('utf-8', errors='ignore')
                elif val_type in [GGUFValueType.UINT8, GGUFValueType.INT8, GGUFValueType.UINT16, 
                                  GGUFValueType.INT16, GGUFValueType.UINT32, GGUFValueType.INT32]:
                    val = int(field.parts[-1][0])
                elif val_type in [GGUFValueType.FLOAT32, GGUFValueType.FLOAT64]:
                    val = float(field.parts[-1][0])
                elif val_type == GGUFValueType.BOOL:
                    val = bool(field.parts[-1][0])
                else:
                    val = raw_part
                    
                writer.add_key_value(key, val, val_type)
        except Exception as e:
            print(f"⚠️ Warning: Skipping key '{key}': {e}")

    # ==========================================================================
    # ⚡ 3. FULL TENSOR PROCESSING TRACK (FAST FEEDBACK VERSION)
    # ==========================================================================
    total_tensors = len(reader.tensors)
    print(f"⚡ Processing {total_tensors} total tensors...")
    
    for idx, tensor in enumerate(reader.tensors, 1):
        name = tensor.name
        
        # 🏎️ FAST FEEDBACK HACK: Skip layer 1 and above to speed up tests
        # if any(f"blk.{i}." in name for i in range(1, 32)):
        #     continue
            
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
                # 1. Compress via your Base-3 Radix engine
                packed_bytes = custom_mutant_quantize_true_158(orig_data).flatten()
                
                d0 = tensor.shape[0]
                d1 = tensor.shape[1]
                
                # 2. Determine base compressed dimensions
                if "ffn_down" in name:
                    modified_d0 = (d0 // 32) * 9
                    modified_shape_gguf = [modified_d0, d1]
                else:
                    modified_d1 = (d1 // 32) * 9
                    modified_shape_gguf = [d0, modified_d1]
                
                # 3. Swap layout orientation to match GGUF's column-major validation rules
                modified_shape_gguf = [modified_shape_gguf[1], modified_shape_gguf[0]]
                
                from gguf import GGMLQuantizationType
                target_dtype = int(GGMLQuantizationType.I8)
                
                writer.add_tensor(
                    name=name, 
                    tensor=packed_bytes, 
                    raw_shape=modified_shape_gguf, 
                    raw_dtype=target_dtype
                )
                print(f"✂️ Inverted-Spliced: {name} | Layout: {modified_shape_gguf}")

            else:
                # [UNTOUCHED SIMULATED TRACK]
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
            # ✨ FIXED FALLBACK BRANCH
            # Casts the underlying disk array back to a properly matching element width 
            # to keep the GGUFWriter structure aligned with the original tensor metadata.
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
            
            # Fixed: Inverted raw_shape format to match GGUF column-major expectation [::-1]
            writer.add_tensor(
                name=name, 
                tensor=clean_data, 
                raw_shape=tensor.shape[::-1],
                raw_dtype=ttype
            )

    # ==========================================================================
    # 💾 4. UNIFIED FILE WRITING BLOCK
    # ==========================================================================
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