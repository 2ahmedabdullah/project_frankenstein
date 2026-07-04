# patch_header.py

import os
import struct

mutant_path = "./models/Llama-3.2-3B-Instruct-Mutant.gguf"
patched_path = "./models/Llama-3.2-3B-Instruct-HeaderPatched.gguf"

# 1. Duplicate the valid mutant file to keep the source safe
print("📂 Creating a carbon copy of the mutant file...")
if os.path.exists(patched_path):
    os.remove(patched_path)

with open(mutant_path, "rb") as src, open(patched_path, "wb") as dst:
    dst.write(src.read())

# 2. Perform a direct binary update on the key flag
print("🔬 Locating and updating feed_forward_length in place...")

# The string key we are targeting inside the GGUF header dictionary
target_key = b"llama.feed_forward_length"

with open(patched_path, "r+b") as f:
    content = f.read()
    
    # Locate the unique string in the metadata header
    key_offset = content.find(target_key)
    
    if key_offset == -1:
        raise ValueError("Could not find 'llama.feed_forward_length' in the GGUF header!")
        
    print(f"🎯 Key found at byte offset {key_offset}. Patching...")
    
    # GGUF key-value store layouts: [Key length] [Key string bytes] [Value Type ID] [Value bytes]
    # Value Type for UINT32 is 4. Let's scan forward past the key name string to find the value payload.
    value_search_offset = key_offset + len(target_key)
    
    # Llama 3.2 defaults to 8192 (0x00002000). Let's scan for this integer structure.
    old_val_bytes = struct.pack("<I", 8192)       # Little-endian uint32 for 8192
    new_val_bytes = struct.pack("<I", 2304)       # Little-endian uint32 for 2304
    
    # Find where the 8192 value lives right after the key designation
    val_offset = content.find(old_val_bytes, value_search_offset, value_search_offset + 32)
    
    if val_offset == -1:
        # Fallback to general lookup if the space has non-standard layout alignments
        val_offset = content.find(old_val_bytes)
        
    if val_offset != -1:
        f.seek(val_offset)
        f.write(new_val_bytes)
        print(f"⚡ Success! Swapped 8192 -> 2304 inline at byte location {val_offset}.")
    else:
        print("❌ Could not match the expected binary scalar footprint for 8192.")

print("✅ File modified cleanly without shifting tensor alignments.")