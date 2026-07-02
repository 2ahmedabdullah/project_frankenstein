# 🧬 Project Frankenstein: Hybrid-Precision Model Surgery

This repository contains the tooling and research code required to perform architectural "surgery" on standard Transformer models (e.g., Llama, Qwen). Instead of training an LLM from scratch, this project surgically splits a pre-trained model down the center of its transformer blocks—preserving high-precision Attention layers for execution in GPU VRAM while grafting ultra-low 1.58-bit ternary Feed-Forward Networks (FFNs) for execution in host system RAM via CPU lookup tables.

---

## 🏗️ Architecture Overview

Standard local AI inference engines offload models *horizontally* (by entire layer blocks). 
This project introduces **Vertical Precision Splitting**, dividing the workload based on hardware-specific computing strengths:


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


## 🏥 Post-Op Model Surgery: 291 Tensor Architecture Breakdown

The **Meta-Llama-3-8B-Instruct** GGUF model layout. 

The model blueprint is composed of two primary sections: **Global Setup Layers** and a deep stack of identical, repeating **Decoder Layer Blocks**. 



### 💡 Part 1: The Global Parts (3 Tensors)

Before the model even starts processing language layer-by-layer, it needs a few global components:

1) token_embd.weight: The "translator" that turns the text words into math numbers.

2) output_norm.weight: A tool that cleans up and stabilizes the math at the very end.

3) output.weight: The final translator that turns the math numbers back into readable text words.


### 💡 Part 2: The Attention Block (5 Tensors)

This block is responsible for looking at a word and figuring out which other words in the sentence it relates to. To do this mathematically, it needs 5 distinct tensors:

1) attn_norm.weight: An RMS (Root Mean Square) Normalization tensor. It cleans up, scales, and stabilizes the data entering the layer so the math doesn't spiral out of control.

2) attn_q.weight (Query): Represents what the current word is "searching" for.

3) attn_k.weight (Key): Represents what characteristics this word offers to other words.

4) attn_v.weight (Value): Holds the actual semantic meaning of the word.

5) attn_output.weight: After Q, K, and V interact, this tensor projects the combined result back into the model's main data highway.

Advanced Note: In standard Transformers, Q, K, and V usually have the exact same size. However, Llama 3 uses Grouped-Query Attention (GQA). It uses 32 heads for Queries but scales down to just 8 heads for Keys and Values. Even though the sizes are smaller to save memory, they still require their own individual tensors!


### 💡 Part 3: The Feed-Forward Neural Network / FFNN (4 Tensors)

Once the attention block figures out how the words relate to each other, it passes the data to the FFN. The FFN acts like a massive local encyclopedia lookup. It uses 4 tensors:

1) ffnn_norm.weight: Another normalization tensor that stabilizes the data right before it hits the heavy fact-checking math.

2) ffnn_gate.weight

3) ffnn_up.weight

4) ffnn_down.weight


            [ Input Data ] (Size: 4,096)
                                │
                    ┌───────────┴───────────┐
                    ▼                       ▼
            1. ffnn_up               2. ffnn_gate
            (Dense Layer)           (Dense Layer)
            (Size: 14,336)          (Size: 14,336)
                    │                       │
                    │                       ▼
                    │               [ Swish Activation ]
                    │                       │
                    └───────────┬───────────┘
                                ▼
                    [ Element-wise Multi ] (The Gate Filter)
                                │
                                ▼
                            3. ffnn_down
                        (Dense Layer)
                        (Size: 4,096)
                                │
                                ▼
                        [ Next Layer ]



$$\text{Layer} = \text{Attention Block} + \text{FFNN Block}$$

$$\text{5 Attention Tensors} + \text{4 FFNN Tensors} = \mathbf{9\text{ Tensors per Layer}}$$

$$\text{Total Tensors} = \text{Global Tensors} + (\text{Number of Layers} \times \text{Tensors per Layer})$$

$$\text{Total Tensors} = 3 + (32 \times 9) = \mathbf{291}$$


## ⚡ Quick Start: Execution Pipeline

### Step 1: Prerequisites & Environment Setup
Ensure you have PyTorch installed with CUDA support, alongside the system's hardware configurations.

```bash
pip install torch transformers accelerate datasets
```

### Step 2: Performing the Model Surgery (surgery.py)
This script loads a pre-trained model, freezes the Attention mechanics, and surgically replaces the target FFN matrices (gate_proj, up_proj, down_proj) with custom BitLinear layers initializing ternary states.


```
// modify the graph builder
for (auto & tensor : model.tensors) {
    if (tensor.name.contains("self_attn")) {
        // Force attention mechanisms onto the 6GB RTX GPU
        tensor.backend = GGML_BACKEND_GPU; 
    } 
    else if (tensor.name.contains("mlp") || tensor.name.contains("ffn")) {
        // Force the massive factual layers onto the 24GB System RAM
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



---

## 🪓 Step-by-Step Layer Anatomy

### 1. Global Components (3 Tensors)
These elements manage global input-output translations and token stabilization across the entire execution loop.
*   `token_embd.weight`: Maps input text tokens into high-dimensional mathematical vectors.
*   `output_norm.weight`: A final RMSNorm layer that stabilizes numeric values at the exit point.
*   `output.weight`: Predicts and decodes raw mathematics back into text tokens.

### 2. The 32 Repeating Decoder Layers (9 Tensors per Layer)
Each individual block (`blk.0` through `blk.31`) is split into a **Pre-LN Attention Block** and a **Pre-LN SwiGLU Feed-Forward Block**:


## ⚠️ Known Implementation Limits
The PCIe Bottleneck: Due to structural constraints on standard consumer motherboards, routing step data back and forth between VRAM and system memory introduces data traffic stalls. Average generation ranges between 5 to 12 tokens per second over typical PCIe 4.0 slots.

Quantization Noise: Smaller models (sub-3B parameters) exhibit higher vulnerability to logical degradation post-surgery. Targets sized at 7B parameters or higher demonstrate the most stable post-op recoveries.

## 📄 License & Attribution
This architecture framework builds heavily upon concepts outlined in Microsoft Research's The Era of 1.58-bit Large Language Models (BitNet) and the open-source acceleration tooling inside llama.cpp. Licensed under the MIT Research License.
