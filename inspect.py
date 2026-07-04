# inspect.py

import gguf

def audit_model(path):
    print(f"\n=== Auditing: {path} ===")
    try:
        reader = gguf.GGUFReader(path)
        print(f"✅ GGUF Structure valid.")
        print(f"Total Tensors: {len(reader.tensors)}")
        print(f"Total Metadata KV pairs: {len(reader.fields)}")
        
        # Check a couple of specific layers to see if they are broken
        for i, tensor in enumerate(reader.tensors):
            if "ffn_down" in tensor.name or "ffn_gate" in tensor.name:
                print(f"👉 Tensor: {tensor.name} | Shape: {tensor.shape} | Type: {tensor.tensor_type}")
                if i > 20: break # Just print a few sample layers
    except Exception as e:
        print(f"❌ CRITICAL GGUF CORRUPTION: {e}")

audit_model("./models/Llama-3.2-3B-Instruct-BF16.gguf")
audit_model("./models/Llama-3.2-3B-Instruct-Simulated.gguf")
audit_model("./models/Llama-3.2-3B-Instruct-Mutant.gguf")
audit_model("./models/Llama-3.2-3B-Instruct-HeaderPatched.gguf")

