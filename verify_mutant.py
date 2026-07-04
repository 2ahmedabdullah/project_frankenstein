# verify_mutant.py
import numpy as np
from gguf import GGUFReader

ORIGINAL_PATH = "./models/Llama-3.2-3B-Instruct-BF16.gguf"
MUTANT_PATH = "./models/Llama-3.2-3B-Instruct-Mutant.gguf"
SIMULATED_PATH = "./models/Llama-3.2-3B-Instruct-Simulated.gguf"
HEADERPATCHED_PATH = "./models/Llama-3.2-3B-Instruct-HeaderPatched.gguf"

target_layer = "blk.0.ffn_gate.weight"

def get_first_block_raw(file_path, tensor_name):
    try:
        reader = GGUFReader(file_path)
        for tensor in reader.tensors:
            if tensor.name == tensor_name:
                raw_bytes = memoryview(tensor.data).tobytes()
                uint16_data = np.frombuffer(raw_bytes, dtype=np.uint16)[:32]
                fp32_data = np.zeros(len(uint16_data), dtype=np.float32)
                fp32_data.view(np.uint32)[:] = uint16_data.astype(np.uint32) << 16
                return fp32_data
    except Exception:
        pass
    return None


def decode_mutant_true_158_radix(packed_block_bytes):
    scale = np.frombuffer(packed_block_bytes[0:2], dtype=np.float16)[0]
    payload = np.frombuffer(packed_block_bytes[2:9], dtype=np.uint8)
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

    ternary = digits.astype(np.float32) - 1.0
    return ternary * scale


def get_first_block_mutant_packed(file_path, tensor_name):
    try:
        reader = GGUFReader(file_path)
        for tensor in reader.tensors:
            if tensor.name == tensor_name:
                raw_bytes = memoryview(tensor.data).tobytes()
                if len(raw_bytes) >= 9:
                    block = raw_bytes[0:9]
                    return decode_mutant_true_158_radix(block)
    except Exception:
        pass
    return None


# Gather first block data
print("🎯 Extracting Block 0 (First 32 Weights)...")
models_config = [
    {"label": "ORIGINAL BLOCK (BF16 View)", "path": ORIGINAL_PATH, "fn": get_first_block_raw},
    {"label": "MUTANT BLOCK (De-quantized Bits)", "path": MUTANT_PATH, "fn": get_first_block_mutant_packed},
    {"label": "HEADERPATCHED BLOCK (De-quantized Bits)", "path": HEADERPATCHED_PATH, "fn": get_first_block_mutant_packed},
]

columns_lines = []

for cfg in models_config:
    lines = []
    block_w = cfg["fn"](cfg["path"], target_layer)
    
    if block_w is None:
        lines.append("❌ Error loading block")
        columns_lines.append(lines)
        continue
        
    unique_vals = np.unique(np.round(block_w, 6))
    lines.append(f"Unique values ({len(unique_vals)} found):")
    lines.append("-" * 40)
    
    # ✂️ Truncate if there are more than 5 unique values to keep layout readable
    max_display = 5
    display_vals = unique_vals[:max_display]
    
    for val in display_vals:
        count = np.sum(np.round(block_w, 6) == val)
        pct = (count / 32) * 100
        bar = "█" * int(pct / 4)
        lines.append(f"  {val: .6f} -> {count:2d} weights {bar}")
        
    if len(unique_vals) > max_display:
        remaining = len(unique_vals) - max_display
        lines.append(f"  ... and {remaining} more unique values")
        
    columns_lines.append(lines)

# -------------------------------------------------------------------------
# Print Horizontal Block View
# -------------------------------------------------------------------------
max_lines = max(len(col) for col in columns_lines)
col_width = 46

print("\n" + "=" * 146)
print(f"{models_config[0]['label']:<{col_width}} | {models_config[1]['label']:<{col_width}} | {models_config[2]['label']:<{col_width}}")
print("=" * 146)

for i in range(max_lines):
    row_str = []
    for col in columns_lines:
        line_text = col[i] if i < len(col) else ""
        row_str.append(f"{line_text:<{col_width}}")
    print(" | ".join(row_str))

print("=" * 146)