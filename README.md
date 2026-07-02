# 🧬 Project Frankenstein: Hybrid-Precision Model Surgery

This repository contains the tooling and research code required to perform architectural "surgery" on standard Transformer models (e.g., Llama, Qwen). Instead of training an LLM from scratch, this project surgically splits a pre-trained model down the center of its transformer blocks—preserving high-precision Attention layers for execution in GPU VRAM while grafting ultra-low 1.58-bit ternary Feed-Forward Networks (FFNNs) for execution in host system RAM via CPU lookup tables.


## ⚙️ Target Hardware (The Constraints)

This suite is deliberately designed to profile budget, edge, and consumer-grade hardware configurations:


```toml
[Hardware Profile]
Device       = "RTX 3060 Laptop (6GB VRAM) / RTX 4050 (6GB VRAM)"
Compute      = "Intel i7-12700H / Ryzen 7"
System RAM   = "16GB / 32GB DDR4/DDR5"
OS           = "Windows 11"

2024 Lenovo LOQ 15IRX9 (Type 83DV) gaming laptop

```


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
| **Feed-Forward (FFNN)** | ~60% | 1.58-bit Ternary | CPU (System RAM) | ~2.5 GB |

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


## 🗺️  DYNAMIC ARCHITECTURE MAP & TOPOLOGY GIST:

          [ INPUT HIGHWAY VECTOR ] (Size: 4,096)
                │
                ▼
    ┌──────────────────────────────────────────────────────┐
    │  STAGE 1: ATTENTION BLOCK (Routed to GPU VRAM)       │
    ├──────────────────────────────────────────────────────┤
    │  • Input Norm   -> attn_norm.weight                  │
    │  • Projections  -> Context Weights Matrix (Q, K, V)  │
    │  • Output Mix   -> attn_output.weight                │
    └──────────────────────────────────────────────────────┘
                │
                ▼
         [ UPDATED DATA ] (Size: 4,096)
                │
                ├───────────────────────────┐
                ▼                           ▼
      ┌──────────────────────┐    ┌──────────────────────┐
      │ STAGE 2A: ffn_up     │    │ STAGE 2B: ffn_gate   │
      │ (Dense Fact Lookup)  │    │ (Routing Filter)     │
      │ (Size: 14,336)       │    │ (Size: 14,336)       │
      └──────────────────────┘    └──────────────────────┘
                │                           │
                │                           ▼
                │                   [ Swish Activation ]
                │                           │
                └───────────┬───────────────┘
                            ▼
                [ Element-wise Multi ] (The Gate Filter)
                            │
                            ▼
                ┌──────────────────────────┐
                │ STAGE 2C: ffn_down       │
                │ (Highway Compressor)     │
                │ (Size: 4,096)            │
                └──────────────────────────┘
                            │
                            ▼
                  [ TO NEXT LAYER / BLOCK ]



$$\text{Layer} = \text{Attention Block} + \text{FFNN Block}$$

$$\text{5 Attention Tensors} + \text{4 FFNN Tensors} = \mathbf{9\text{ Tensors per Layer}}$$

$$\text{Total Tensors} = \text{Global Tensors} + (\text{Number of Layers} \times \text{Tensors per Layer})$$

$$\text{Total Tensors} = 3 + (32 \times 9) = \mathbf{291}$$


## ⚡ Quick Start: Execution Pipeline

### Step 1: Prerequisites & Environment Setup
Ensure one have PyTorch installed with CUDA support, alongside the system's hardware configurations.

```bash
pip install torch transformers accelerate datasets
```

### Step 2: Performing the Model Surgery (surgery.py)
This script loads a pre-trained model, freezes the Attention mechanics, and surgically replaces the target FFN matrices (gate_proj, up_proj, down_proj) with custom BitLinear layers initializing ternary states.


                       [ Original Q4_K_M Model ]
                                   │
                                   ▼
                 [ GGUFReader loops through Tensors ]
                                   │
                 ┌─────────────────┴─────────────────┐
                 │ (Is it an Attention Tensor?)      │(Is it an FFN Tensor?)
                 ▼                                   ▼ 
         ┌───────────────┐                   ┌───────────────────────────┐
         │ Pass Through  │                   │ ternary_158_quantize()    │
         │  Unchanged    │                   ├───────────────────────────┤
         └───────┬───────┘                   │ 1. Compute Threshold      │
                 │                           │ 2. Clamp to -1, 0, 1      │
                 │                           │ 3. Calculate Scale Factor │
                 │                           └─────────────┬─────────────┘
                 │                                         │
                 ▼                                         ▼
         ┌───────────────────────────────────────────────────────────────┐
         │                 perform_model_surgery()                       │
         │  Assembles & Writes Final Custom GGUF File to Disk            │
         └───────────────────────────────────────────────────────────────┘
                                    │     
                                    ▼
                          [ Frankenstein GGUF ] (Mutant GGUF file)
                                    │
                                    ▼
                            ┌───────────────┐
                            │ C++ Execution │  ◄── Routes Attn -> GPU VRAM
                            │    Engine     │  ◄── Routes FFN  -> System RAM (CPU)
                            └───────┬───────┘
                                    │
                  ┌─────────────────┴─────────────────┐
                  ▼ (If name contains "attn")         ▼ (If name contains "mlp"/"ffn")
        ┌──────────────────────┐         ┌───────────────────────────┐
        │  GGML_BACKEND_GPU    │         │     GGML_BACKEND_CPU      │
        ├──────────────────────┤         ├───────────────────────────┤
        │ Routes to 6GB VRAM   │         │ Routes to 24GB System RAM │
        │ Smooth FP16/Q4 Math  │         │ Pure Addition/Subtraction │
        └──────────┬───────────┘         └─────────────┬─────────────┘
                   │                                   │
                   └────────────────┬──────────────────┘
                                    ▼
                    [ Active Inference Token Loop ]
        


### The "Cell-Level Transformer"
This function handles the raw math. Its single goal is to take a high-precision layer matrix and crush it into a ternary state ($-1$, $0$, or $+1$).

Step 1: Raw Data Extraction 

It checks if the incoming tensor is readable as floating-point math (np.float32). If it is still compressed or in bytes, it converts it to clean, raw floats so it can perform calculations.

Step 2: Finding the Dynamic Threshold Line ($\Delta$)

Most weights in a neural network layer follow a Gaussian (Normal) Distribution—meaning a huge cluster of weights sit very close to zero, while fewer, more powerful weights stretch out to the positive and negative sides.

$$\Delta = 0.7 \times \text{Mean}(|W|)$$

When multiplied the absolute mean by 0.7, one draws two sharp lines right in the middle of that bell curve:

The Dead Zone (Set to 0): Any weight that falls inside the $-\Delta$ to $+\Delta$ window is deemed "background noise." The script aggressively zeroes them out. This accounts for roughly 30% to 40% of the matrix, creating sparsity which makes CPU memory streaming incredibly efficient.

The Positive Charge (Set to +1): Any weight resting safely above $+\Delta$ is a strong positive signal.

The Negative Charge (Set to -1): Any weight resting safely below $-\Delta$ is a strong negative inhibitor.


Standard Weight Distribution Curve
             
                                 ▲
                                ╱█╲
                               ╱███╲  ◄── High density of weights near 0
                              ╱█████╲
                            _╱███████╲_
        -1 Zone            ╱███████████╲           +1 Zone
     ◄────────────│███████│      0      │███████├─────────────►
                  │  -Δ   │The Dead Zone│  +Δ   │ 
     (Strong Neg) │       │◄───────────►│       │ (Strong Pos)
                  ▼       ▼             ▼       ▼
                          (Squeezed to 0)        
                        
⚖️ Why exactly 0.7? 

The number 0.7 was found through structural optimization to minimize Quantization Noise (the loss of intelligence when one crush numbers).

If the number was too low (e.g., 0.1): Almost no weights would become 0. The script would force almost everything to $+1$ or $-1$. one lose the ability to have "neutral" weights, completely breaking the model's factual retention.

If the number was too high (e.g., 1.5): The threshold would be too wide. The script would wipe out 80% of the weights to 0. One destroy too much information, and the model goes braindead (outputs total gibberish that even fine-tuning cannot heal).

By using 0.7, onemathematically guarantee that the structural variance of the newly squeezed 1.58-bit ternary matrix matches the variance of the original high-precision matrix as closely as possible, allowing the laptop CPU to process a highly optimized, clean database.
            

Step 4: Keeping the Magnitude (scale)

Because turning numbers into raw $-1$, $0$, and $1$ destroys the scaling magnitude of the model's knowledge, it calculates a single scale float (the maximum absolute value of the original tensor). It returns both the newly flattened matrix and this scale multiplier.


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

## Repo Structure

```text
project-frankenstein/
│
├── models/
│   └── Meta-Llama-3-8B-Instruct-Q4_K_M.gguf         <--      [Raw model]
│   └── Meta-Llama-3-8B-Surgically-Split-1.58b.gguf  <--  🧬  [THE MUTANT]
│ 
├── diagnosis.py                                  
├── surgery.py                                     
├── healing.py  
│
└── inference/
    ├── Makefile                                   
    └── hybrid_runner.cpp                            <-- (C++ engine)
```

```text
1. THE PATIENT (download.py)
 └── Downloads Meta-Llama-3-8B-Instruct-Q4_K_M.gguf straight to the /models directory.
       │
       ▼
 2. THE DIAGNOSTIC (diagnosis.py)
 └── Scans the file layout, mapping out the 291 total tensors across 32 structural blocks.
       │
       ▼
 3. THE SCALPEL (surgery.py)
 └── Feeds the FFN tensors into the ternary_158_quantize() filter using the 0.7 Δ threshold.
 └── Leaves Attention tensors pristine. Welds custom ".scale" tracking vectors to the FFN blocks.
       │
       ▼
 4. THE MUTANT (Meta-Llama-3-8B-Surgically-Split-1.58b.gguf)
 └── Saved to disk. It features high-precision context layers and ultra-lean database layers.
       │
       ▼
 5. THE HEALING (healing.py)
 └── Fine-tunes the mutant with Quantization-Aware Fine-Tuning (QAFT) so the layers synchronize.
       │
       ▼
 6. THE NERVOUS SYSTEM (inference/hybrid_runner.cpp via Makefile)
 └── Reads the mutant file. Pushes Attention to the laptop's 6GB VRAM (GPU).
 └── Pushes the 1.58-bit FFN arrays to the 24GB System RAM (CPU) for blazingly fast integer additions.
 ```

## ⚠️ Known Implementation Limits
The PCIe Bottleneck: Due to structural constraints on standard consumer motherboards, routing step data back and forth between VRAM and system memory introduces data traffic stalls. Average generation ranges between 5 to 12 tokens per second over typical PCIe 4.0 slots.

Quantization Noise: Smaller models (sub-3B parameters) exhibit higher vulnerability to logical degradation post-surgery. Targets sized at 7B parameters or higher demonstrate the most stable post-op recoveries.

## 📄 License & Attribution
This architecture framework builds heavily upon concepts outlined in Microsoft Research's The Era of 1.58-bit Large Language Models (BitNet) and the open-source acceleration tooling inside llama.cpp. Licensed under the MIT Research License.
