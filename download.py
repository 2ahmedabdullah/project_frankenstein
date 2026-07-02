# download.py

import os
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download, snapshot_download
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# 🔌 Load environmental variables from the local .env file
load_dotenv()

# Fetch the token from the environment configuration matrix
HF_ACCESS_TOKEN = os.getenv("HF_TOKEN")

# --- High-Performance Network Configuration Matrix ---
if HF_ACCESS_TOKEN is not None:
    os.environ["HF_TOKEN"] = HF_ACCESS_TOKEN
    print("🔑 Hugging Face token loaded successfully from environment configuration.")
else:
    print("⚠️ WARNING: No 'HF_TOKEN' detected in your .env file. Attempting public unauthenticated download...")

os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"


def secure_download(repo_id, filename=None):
    if filename:
        # Download a single specific file (e.g., GGUF)
        print(f"\n⚡ Initiating authenticated chunk transfer for file: {filename}")
        print(f"📂 Destination: ./models/{filename}")
        try:
            hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir="./models",
            )
            print(f"✅ Successfully downloaded and verified file: {filename}")
        except Exception as e:
            print(f"❌ Download failed for {filename}: {e}")
    else:
        # Download the entire full-precision repository (Safetensors shards + configs)
        repo_folder_name = repo_id.split("/")[-1]
        destination = f"./models/{repo_folder_name}"
        
        print(f"\n⚡ Initiating full repository snapshot transfer for: {repo_id}")
        print(f"📂 Destination: {destination}")
        try:
            snapshot_download(
                repo_id=repo_id,
                local_dir=destination,
            )
            print(f"✅ Successfully downloaded and verified full repository: {repo_id}")
        except Exception as e:
            print(f"❌ Full repository download failed for {repo_id}: {e}")

if __name__ == "__main__":
    # Define our exact target pair for the RTX 3050 matrix
    models_to_fetch = [
        # --- TARGET: Meta Llama 3 8B Instruct Q4_K_M ---
        {
            "repo": "bartowski/Meta-Llama-3-8B-Instruct-GGUF", 
            "file": "Meta-Llama-3-8B-Instruct-Q4_K_M.gguf"
        },
    ]
    
    print("🚀 Initializing Authenticated Pre-Download Matrix Run...")
    
    # Executing safe model matrix download loops
    for item in models_to_fetch:
        secure_download(item["repo"], item["file"])
        
    print("\n🏁 Master local model cache populated. System primed for benchmarking!")