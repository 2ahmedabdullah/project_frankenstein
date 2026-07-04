# check_bytes.py

import numpy as np
from gguf import GGUFReader

# Paths to your models
ORIGINAL_PATH = "./models/Llama-3.2-3B-Instruct-BF16.gguf"
MUTANT_PATH = "./models/Llama-3.2-3B-Instruct-Mutant.gguf"
SIMULATED_PATH = "./models/Llama-3.2-3B-Instruct-Simulated.gguf"
HEADERPATCHED_PATH = "./models/Llama-3.2-3B-Instruct-HeaderPatched.gguf"

target_layer = "blk.0.ffn_gate.weight"

# Feel free to add, remove, or rename any models here:
models_to_check = {
    "Original Model": ORIGINAL_PATH,
    "Mutant Model": MUTANT_PATH,
    "HeaderPatched Model": HEADERPATCHED_PATH,
    # "Simulated Model": SIMULATED_PATH
}

results = {}

# 1. Gather data from all models
for label, path in models_to_check.items():
    try:
        reader = GGUFReader(path)
        found = False
        for tensor in reader.tensors:
            if tensor.name == target_layer:
                gguf_type = tensor.tensor_type
                num_elements = np.prod(tensor.shape)
                actual_bytes = tensor.data.nbytes
                bits_per_weight = (actual_bytes * 8) / num_elements
                
                # Convert bytes to Megabytes (1 MB = 1024^2 bytes)
                mb_size = actual_bytes / (1024 ** 2)
                
                # Determine status message
                if bits_per_weight == 16.0:
                    status = "🚨 FP16 (16 bits/w)"
                elif bits_per_weight == 32.0:
                    status = "🚨 FP32 (32 bits/w)"
                elif bits_per_weight <= 2.1: # handling potential float precision
                    status = "🎉 Packed Ternary (<=2 bits/w)"
                else:
                    status = f"ℹ️  Custom ({bits_per_weight:.1f} bits/w)"

                results[label] = {
                    "type": str(gguf_type),
                    "shape": f"{num_elements:,}",
                    "bytes": f"{actual_bytes:,} bytes",
                    "mb": f"{mb_size:.2f} MB",
                    "bpw": f"{bits_per_weight:.2f} bits",
                    "status": status
                }
                found = True
                break
        if not found:
            results[label] = {"error": f"Tensor '{target_layer}' not found"}
    except Exception as e:
        results[label] = {"error": f"Failed to read: {str(e)}"}

# 2. Print Dynamic Side-by-Side Comparison Report
print(f"\n📊 SIDE-BY-SIDE REPORT FOR TENSOR: {target_layer}")

# Dynamic column sizing configuration
metric_col_width = 28
model_col_width = max(max(len(name) for name in models_to_check.keys()), 25)
total_width = metric_col_width + 3 + (len(models_to_check) * (model_col_width + 3))

print("=" * total_width)

# Print Header Row
headers = [f"{'Metric / Property':<{metric_col_width}}"]
for label in models_to_check.keys():
    headers.append(f"{label:<{model_col_width}}")
print(" | ".join(headers))
print("-" * total_width)

# Define rows to print dynamically (updated metric key to 'mb')
rows_to_print = [
    ("GGUF Type ID", "type"),
    ("Tensor Elements (Shape)", "shape"),
    ("Physical Size on Disk", "bytes"),
    ("Physical Size (MB)", "mb"),
    ("Bits Per Weight (bpw)", "bpw"),
    ("Conclusion", "status")
]

# Iterate and build rows dynamically
for row_label, data_key in rows_to_print:
    row_cells = [f"{row_label:<{metric_col_width}}"]
    
    for model_label in models_to_check.keys():
        model_data = results.get(model_label, {})
        
        # If the model hit an error, output the error message instead
        if "error" in model_data:
            cell_value = f"❌ {model_data['error']}"
        else:
            cell_value = model_data.get(data_key, "N/A")
            
        row_cells.append(f"{cell_value:<{model_col_width}}")
    
    print(" | ".join(row_cells))
    
    # Visual break before the conclusion row
    if row_label == "Bits Per Weight (bpw)":
        print("-" * total_width)

print("=" * total_width)