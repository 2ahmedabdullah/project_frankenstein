# verify_mutant.py

import numpy as np
from gguf import GGUFReader

ORIGINAL_PATH = "./models/Llama-3.2-3B-Instruct-BF16.gguf"
MUTANT_PATH = "./models/Llama-3.2-3B-Instruct-Mutant.gguf"
target_layer = "blk.0.ffn_gate.weight"

def get_tensor_data(file_path, tensor_name):
    reader = GGUFReader(file_path)
    for tensor in reader.tensors:
        if tensor.name == tensor_name:
            # Safely cast buffer to a standard numpy array
            return np.array(tensor.data)
    return None

print("📖 Reading and analyzing layers...")
orig_w = get_tensor_data(ORIGINAL_PATH, target_layer)
mutant_w = get_tensor_data(MUTANT_PATH, target_layer)

if orig_w is not None and mutant_w is not None:
    print(f"\n📊 SIDE-BY-SIDE DISTRIBUTION PROFILE FOR: {target_layer}")
    print("=" * 115)
    print(f"{'ORIGINAL MODEL (Continuous)':<55} | {'MUTANT MODEL (Ternary 1.58b)':<55}")
    print("=" * 115)
    
    # 1. Process Original (Bin it to show a snapshot of the distribution)
    # Filter out small values or extreme outliers to make a clean histogram
    orig_flat = orig_w.flatten()
    counts_orig, bin_edges = np.histogram(orig_flat, bins=5)
    total_orig = orig_flat.size
    
    orig_lines = []
    for i in range(len(counts_orig)):
        pct = (counts_orig[i] / total_orig) * 100
        bar = "█" * int(pct / 2)
        bin_label = f"[{bin_edges[i]:.2f} to {bin_edges[i+1]:.2f}]"
        orig_lines.append(f"{bin_label:<18} {pct:6.2f}% {bar:<28}")

    # 2. Process Mutant (Exact unique value match since it's discrete)
    mutant_flat = mutant_w.flatten()
    unique_vals, counts_mutant = np.unique(np.round(mutant_flat, 4), return_counts=True)
    total_mutant = mutant_flat.size
    
    mutant_lines = []
    for val, count in zip(unique_vals, counts_mutant):
        pct = (count / total_mutant) * 100
        bar = "█" * int(pct / 2)
        val_label = f"Value: {val:5.1f}"
        mutant_lines.append(f"{val_label:<18} {pct:6.2f}% {bar:<28}")
        
    # Pad lists if they have different line lengths so zip doesn't truncate
    max_len = max(len(orig_lines), len(mutant_lines))
    orig_lines += [""] * (max_len - len(orig_lines))
    mutant_lines += [""] * (max_len - len(mutant_lines))
    
    # Print side-by-side
    for o_line, m_line in zip(orig_lines, mutant_lines):
        print(f"{o_line:<55} | {m_line:<55}")
        
    print("=" * 115)
else:
    print("❌ Error: Could not extract target layer from one or both models.")