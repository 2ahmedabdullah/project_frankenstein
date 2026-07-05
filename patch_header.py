# patch_header.py

import os
import struct

mutant_path = "./models/Llama-3.2-3B-Instruct-Mutant.gguf"
patched_path = "./models/Llama-3.2-3B-Instruct-HeaderPatched.gguf"

print("📂 Rebuilding GGUF container headers to execute the Q4_0 dimension illusion...")

if os.path.exists(patched_path):
    os.remove(patched_path)

# Target layers affected by our flat 1D bypass strategy
mlp_targets = [b"ffn_gate.weight", b"ffn_up.weight", b"ffn_down.weight"]

with open(mutant_path, "rb") as src, open(patched_path, "wb") as dst:
    content = src.read()
    
    # --------------------------------------------------------------------------
    # STEP 1: Reconstruct tensor dictionary headers sequentially (DO NOT TOUCH GLOBAL METADATA)
    # --------------------------------------------------------------------------
    print("🕵️ Expanding target MLP tensor descriptors from 1D -> 2D...")
    
    write_cursor = 0
    patched_count = 0
    
    # Find all target strings and their exact positions
    targets_to_patch = {}
    for block_idx in range(28):
        for target in mlp_targets:
            t_name = f"blk.{block_idx}.{target.decode()}".encode()
            offset = content.find(t_name)
            if offset != -1:
                # Store the name token so we know which layer type it is later
                targets_to_patch[offset + len(t_name)] = target

    # Scan through the file and inject correct 2D dimensions sequentially
    offsets = sorted(targets_to_patch.keys())
    
    for geom_offset in offsets:
        dst.write(content[write_cursor:geom_offset])
        
        n_dims = struct.unpack("<I", content[geom_offset:geom_offset+4])[0]
        
        if n_dims == 1:
            # Determine correct shape based on the layer type to match 3B architecture natively
            target_type = targets_to_patch[geom_offset]
            if target_type == b"ffn_down.weight":
                d0, d1 = 8192, 3072
            else:  # ffn_gate.weight or ffn_up.weight
                d0, d1 = 3072, 8192
                
            new_n_dims = struct.pack("<I", 2)
            dim0_bytes = struct.pack("<Q", d0)
            dim1_bytes = struct.pack("<Q", d1)
            
            dst.write(new_n_dims + dim0_bytes + dim1_bytes)
            
            write_cursor = geom_offset + 12  
            patched_count += 1
        else:
            write_cursor = geom_offset

    # Write out the remainder of the file (including all raw tensor weights data)
    dst.write(content[write_cursor:])
    print(f"🧱 Tensor Geometry Success! Expanded {patched_count} layer fields cleanly to 2D matrices.")

print("\n✅ Rebuild complete. Run your model suite now!")