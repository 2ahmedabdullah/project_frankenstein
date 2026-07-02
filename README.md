# 🧬 Project Frankenstein: Hybrid-Precision Model Surgery

This repository contains the tooling and research code required to perform architectural "surgery" on standard Transformer models (e.g., Llama, Qwen). Instead of training an LLM from scratch, this project surgically splits a pre-trained model down the center of its transformer blocks—preserving high-precision Attention layers for execution in GPU VRAM while grafting ultra-low 1.58-bit ternary Feed-Forward Networks (FFNs) for execution in host system RAM via CPU lookup tables.

---

## 🏗️ Architecture Overview

Standard local AI inference engines offload models *horizontally* (by entire layer blocks). This project introduces **Vertical Precision Splitting**, dividing the workload based on hardware-specific computing strengths:

                        [ USER PROMPT ]
                               │
                               ▼
              ┌──────────────────────────────────┐
              │  Embedding Layer (FP16) - VRAM   │
              └────────────────┬─────────────────┘
                               │
                  ┌────────────┴────────────┐
                  ▼                         ▼
        ┌───────────────────┐     ┌───────────────────┐
        │  Layer Attention  │     │   Layer FFN       │
        │    (Q4 / FP16)    │     │   (1.58-bit)      │
        ├───────────────────┤     ├───────────────────┤
        │   RUNS ON GPU     │     │    RUNS ON CPU    │
        │  (Tensor Cores)   │     │  (Lookup Tables)  │
        └─────────┬─────────┘     └─────────┬─────────┘
                  │                         │
                  └────────────┬────────────┘
                               │
                               ▼
              ┌──────────────────────────────────┐
              │    LM Head Output (FP16) - VRAM  │
              └────────────────┬─────────────────┘
                               │
                               ▼
                       [ GENERATED TOKEN ]




### Target Hardware Allocation (Optimized for 6GB VRAM / 24GB RAM)

| Component Layer | Parameter Weight | Targeted Precision | Target Hardware | Memory Footprint |
| :--- | :--- | :--- | :--- | :--- |
| **Embeddings & Head** | ~5% | FP16 / BF16 | GPU (VRAM) | ~0.5 GB |
| **Self-Attention Blocks** | ~35% | FP16 or Q4_K | GPU (VRAM) | ~1.5 GB |
| **Feed-Forward (FFN)** | ~60% | 1.58-bit Ternary | CPU (System RAM) | ~2.5 GB |

---

## ⚡ Quick Start: Execution Pipeline

### Step 1: Prerequisites & Environment Setup
Ensure you have PyTorch installed with CUDA support, alongside your system's hardware configurations.

```bash
pip install torch transformers accelerate datasets
```

### Step 2: Performing the Model Surgery (surgery.py)
This script loads a pre-trained model, freezes the Attention mechanics, and surgically replaces the target FFN matrices (gate_proj, up_proj, down_proj) with custom BitLinear layers initializing ternary states.

```
// modify the graph builder
for (auto & tensor : model.tensors) {
    if (tensor.name.contains("self_attn")) {
        // Force attention mechanisms onto your 6GB RTX GPU
        tensor.backend = GGML_BACKEND_GPU; 
    } 
    else if (tensor.name.contains("mlp") || tensor.name.contains("ffn")) {
        // Force the massive factual layers onto your 24GB System RAM
        tensor.backend = GGML_BACKEND_CPU; 
    }
}
```

### 🏥 Step 3: Post-Op Healing (Distillation / Fine-Tuning)
Directly after surgery, the model will output gibberish due to a language mismatch between the smooth floating-point attention layers and the blocky ternary FFN layers.

To bridge this gap, run the training pipeline with Quantization-Aware Fine-Tuning (QAFT). We pass an open-source instructional dataset through the model for 3–5 epochs, keeping the attention weights strictly locked. This forces the new 1.58-bit FFN parameters to adapt to their new host ecosystem.


```
python train_healing.py --dataset "Open-Orca/OpenOrca" --epochs 3 --lr 2e-4
```

### 🚀 Step 4: Split-Hardware Inference Pipeline
To run the finalized model on consumer hardware (like a laptop with 6GB VRAM and 24GB System RAM), navigate to the /inference subdirectory to build our custom execution loop compiler:

```
cd inference && make hybrid_runner
```

The runtime executable uses the following memory strategy:

Device 0 (CUDA): Allocates the static KV Cache arrays, input context embedding tables, and all standard floating-point Attention matrices directly to GPU VRAM blocks.

Device 1 (CPU): Maps the large ternary FFN weights as unmultiplied tensor arrays within system memory.

The Loop: During generation execution steps, layers process attention workflows via CUDA, ping tokens over the PCIe lanes to system RAM for integer addition processing inside the ternary FFN layers, and pull the activation tensor blocks back to the GPU to complete the loop cycle.

## ⚠️ Known Implementation Limits
The PCIe Bottleneck: Due to structural constraints on standard consumer motherboards, routing step data back and forth between VRAM and system memory introduces data traffic stalls. Average generation ranges between 5 to 12 tokens per second over typical PCIe 4.0 slots.

Quantization Noise: Smaller models (sub-3B parameters) exhibit higher vulnerability to logical degradation post-surgery. Targets sized at 7B parameters or higher demonstrate the most stable post-op recoveries.

## 📄 License & Attribution
This architecture framework builds heavily upon concepts outlined in Microsoft Research's The Era of 1.58-bit Large Language Models (BitNet) and the open-source acceleration tooling inside llama.cpp. Licensed under the MIT Research License.
