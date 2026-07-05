# 🧬 Project Frankenstein: Hybrid-Precision Model Surgery

This repository contains the tooling and research code required to perform architectural "surgery" on standard Transformer models (e.g., Llama, Qwen). Instead of training an LLM from scratch, this project surgically splits a pre-trained model down the center of its transformer blocks. The approach retains high-precision attention layers while replacing feed-forward layers with ternary representations inspired by BitNet.


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
## Models

```text

[1. Baseline FP16]  ──► Smooth, continuous decimal numbers (e.g., -0.0029, 0.0198)
       │
       ├─► [2. Simulated FP16] ──► Clean ternary steps (e.g., -0.0094, 0, +0.0094)
       │
       └─► [3. Mutant 1.58-bit] ──► Packed raw binary bits (1D array, broken layout)
                 │
                 └─►   [4. HeaderPatched] ──► Packed raw binary bits (2D matrix layout compatible with Llama cpp)
                       

```

## Physical Layout Comparison

```text

        [ Original Track: Pure FP16 ]
        Shape: 32 elements  |  Size: 32 × 2 bytes = 64 bytes total
        ┌───┬───┬───┬───┐ ... ┌───┐
        │F16│F16│F16│F16│     │F16│  <- Every single weight is 2 raw bytes.
        └───┴───┴───┴───┘     └───┘


        [ Simulated Track: Blocked Ternary in Float Container ]
        Shape: 32 elements  |  Size: 32 × 2 bytes = 64 bytes total
        ┌───┬───┬───┬───┐ ... ┌───┐
        │ 0 │-Δ │+Δ │ 0 │     │-Δ │  <- Still 64 bytes on disk, but values are snapped
        └───┴───┴───┴───┘     └───┘     to ternary boundaries multiplied by block scale.


        [ Mutant Track: True 1.58-Bit Radix Pack ]
        Shape: 1 Flat Compressed Chunk  |  Size: Exactly 9 bytes total
        ┌───────────────┬──────────────────────────────────────────┐
        │ Scale (FP16)  │ Packed Base-3 Radix Byte Array Payload   │
        │    2 Bytes    │                 7 Bytes                  │
        └───────────────┴──────────────────────────────────────────┘
```


## 🏗️ Architecture Overview

Standard local AI inference engines offload models *horizontally* (by entire layer blocks). 
This project introduces **Vertical Precision Splitting**, dividing the workload based on hardware-specific computing strengths:


                           [ Original Model ]
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
                                [ Mutant ] (Mutant GGUF file saved)
                                    │
                                    ▼
                        ┌───────────────────────┐
                        │    Fixing the GGUF    │ (HeaderPatched GGUF file saved)
                        │     Compatibility     │ 
                        └───────────┬───────────┘
                                    │
                                    ▼
                     [ Active Inference Token Loop ]
        
---


## 🏥 Post-Op Model Surgery: 291 Tensor Architecture Breakdown

The **Meta-Llama-3-8B-Instruct** GGUF model layout. 

The model blueprint is composed of two primary sections: **Global Setup Layers** and a deep stack of identical, repeating **Decoder Layer Blocks**. 


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
This script loads a pre-trained model, preserves the attention layers, and replaces selected FFNN matrices with custom BitLinear modules initialized using ternary weights (gate_proj, up_proj, down_proj).


### The "Ternary Quantization Pipeline": The Mathematical Transformation

Deliberately leaving the Attention Layers completely untouched because they handle the contextual relationships between tokens. Instead, surgically targeting the Feed-Forward Network (FFN) layers. In a standard Llama-style block, the FFN operation is typically defined using SwiGLU activation:

$$\text{FFN}(X) = (\text{Swish}(X \cdot W_{\text{gate}}) \odot (X \cdot W_{\text{up}})) \cdot W_{\text{down}}$$

Each of these three weight matrices ($W_{\text{gate}}, W_{\text{up}}, W_{\text{down}}$) is a massive high-precision high-dimensional tensor. Here is how the quantization pipeline alters them mathematically step-by-step:

Step A: The Abs-Mean Threshold ($\Delta$)

Instead of using a fixed static threshold, the code calculates a dynamic threshold $\Delta$ tailored to the variance of each discrete 32-element chunk (block) inside the weight matrix:

$$\Delta = 0.7 \times \frac{1}{32}\sum_{i=1}^{32} |W_i|$$

$$\Delta = 0.7 \times \text{Mean}(|W|)$$

Step B: The Squeeze (Ternary Mapping)

Every continuous floating-point value inside that 32-element weight chunk is mapped into an idealized, discrete ternary state space $D \in \{-1, 0, 1\}$ based on the dynamic boundaries:

$$D_i = \begin{cases} 
+1 & \text{if } W_i > \Delta \\
0 & \text{if } -\Delta \le W_i \le \Delta \\
-1 & \text{if } W_i < -\Delta 
\end{cases}$$


Standard Weight Distribution Curve
             
                                                        ▲
                                                       ╱█╲
                                                      ╱███╲  ◄── High density of weights near 0
                                                     ╱█████╲
                                                   _╱███████╲_
                                -1 Zone           ╱███████████╲           +1 Zone
                            ◄────────────│███████│      0      │███████├─────────────►
                                         │  -Δ   │The Dead Zone│  +Δ   │ 
                            (Strong Neg) │       │◄───────────►│       │ (Strong Pos)
                                         ▼       ▼             ▼       ▼
                                                 (Squeezed to 0)        

This function handles the raw math. The quantization routine converts floating-point weight tensors into ternary representations while retaining a scaling factor for reconstruction. ($-1$, $0$, or $+1$).

The Dead Zone (Set to 0): Any weight that falls inside the $-\Delta$ to $+\Delta$ window is deemed "background noise." Weights within the threshold are mapped to zero, increasing sparsity while reducing storage requirements. This accounts for roughly 30% to 40% of the matrix, increasing sparsity, which may improve memory efficiency depending on the execution backend.

The Positive Charge (Set to +1): Any weight resting safely above $+\Delta$ is a strong positive signal.

The Negative Charge (Set to -1): Any weight resting safely below $-\Delta$ is a strong negative inhibitor.


Step C: Scaled Reconstruction

To track performance metrics during validation (Simulated Mode), the numbers are projected back into floating-point approximations ($W'$) by scaling the discrete charges by the dynamic block boundary:

$$W'_i = D_i \times \Delta$$

Because roughly 30% to 40% of the weights naturally fall into the "Dead Zone" ($0$), the matrix becomes highly sparse. When evaluating $W'$, the Mean Squared Error (MSE) is incredibly low because $\Delta$ precisely preserves the underlying variance of the original distribution.



                        
⚖️ Why exactly 0.7? 

The threshold coefficient (0.7) was selected empirically during preliminary experiments to balance sparsity and reconstruction quality.

If the number was too low (e.g., 0.1): Almost no weights would become 0. The script would force almost everything to $+1$ or $-1$. one lose the ability to have "neutral" weights, completely breaking the model's factual retention.

If the number was too high (e.g., 1.5): The threshold would be too wide. The script would wipe out 80% of the weights to 0. One destroy too much information, and the model goes braindead (outputs total gibberish that even fine-tuning cannot heal).

By using 0.7, it aims to preserve that the structural variance of the newly squeezed 1.58-bit ternary matrix matches the variance of the original high-precision matrix as closely as possible, allowing the laptop CPU to process a compact ternary representation.
            

### 🛠️ The HeaderPatch Hack: Why and What

The Problem (Why it was necessary)
During an upstream processing or quantization phase, the model's architecture suffered a critical metadata mutation. Specifically, the multi-dimensional Feed-Forward Network (FFN) layers (such as blk.n.ffn_down.weight) had their 2D shape dimensions (e.g., [8192, 3072]) completely flattened into an identical 1D array ([25165824]).

While the underlying binary weight data and file sizes remained intact, this structural corruption made the file illegal to GGUF parsers. Attempting to run the raw Mutant file caused execution engines (llama.cpp, Ollama) to immediately reject the tensor layout and crash on load.

The Intervention (What the Author did)
To bypass the hard crash and make the model execution-ready, the author performed low-level surgery directly on the GGUF metadata header:

Dimension Restoration: Forced the flattened 1D tensor shape attributes back into their native 2D shapes ([8192, 3072]), restoring architectural compatibility.

Byte-Offset Realignment: Recalculated the GGUF dictionary string lengths and shifted the internal tensor data padding down the binary stream (resulting in a precise +512 byte alignment shift from the original mutant offsets).

Current Status & Next Steps
Thanks to the HeaderPatched hack, the model now successfully passes structural validation, allocates memory correctly, and loads without crashing.

⚠️ Note on Output: While structurally sound, the internal weights are currently experiencing layout disorientation (Row-Major vs. Column-Major mismatch from the original flattening process), resulting in incoherent/gibberish textual responses. Do not attempt to finetune or QAFT this build yet. The next phase of development requires a python-level matrix reshape/stride correction to realign the underlying binary data to match the newly patched header maps.


### Results and Analysis

#### 📊 SIDE-BY-SIDE REPORT FOR TENSOR Check: blk.0.ffn_gate.weight

```
===============================================================================================================================================
Metric / Property            | Original Model            | Mutant Model              | HeaderPatched Model       | Simulated Model          
-----------------------------------------------------------------------------------------------------------------------------------------------
GGUF Type ID                 | 30                        | 2                         | 2                         | 30                       
Tensor Elements (Shape)      | 25,165,824                | 25,165,824                | 25,165,824                | 25,165,824               
Physical Size on Disk        | 50,331,648 bytes          | 14,155,776 bytes          | 14,155,776 bytes          | 50,331,648 bytes         
Physical Size (MB)           | 48.00 MB                  | 13.50 MB                  | 13.50 MB                  | 48.00 MB                 
Bits Per Weight (bpw)        | 16.00 bits                | 4.50 bits                 | 4.50 bits                 | 16.00 bits               
-----------------------------------------------------------------------------------------------------------------------------------------------
Conclusion                   | 🚨 FP16 (16 bits/w)        | ℹ️  Custom (4.5 bits/w)   | ℹ️  Custom (4.5 bits/w)   | 🚨 FP16 (16 bits/w)       
===============================================================================================================================================
```

#### 📊 SIDE-BY-SIDE DISTRIBUTION PROFILE FOR: blk.0.ffn_gate.weight

```

===============================================================================================================================================================================================
ORIGINAL BLOCK (BF16 View)                     | MUTANT BLOCK (De-quantized Bits)               | HEADERPATCHED BLOCK (De-quantized Bits)        | SIMULATED BLOCK           
===============================================================================================================================================================================================
Unique values (31 found):                      | Unique values (4 found):                       | Unique values (4 found):                       | Unique values (3 found):                      
----------------------------------------       | ----------------------------------------       | ----------------------------------------       | ----------------------------------------      
  -0.030762 ->  1 weights                      |   -0.004395 -> 11 weights ████████             |   -0.004673 ->  1 weights                      |   -0.009460 -> 11 weights ████████            
  -0.029297 ->  1 weights                      |    0.000000 -> 15 weights ███████████          |   -0.000114 ->  4 weights ███                  |    0.000000 ->  9 weights ███████             
  -0.020020 ->  1 weights                      |    0.004395 ->  5 weights ███                  |   -0.000000 -> 16 weights ████████████         |    0.009460 -> 12 weights █████████           
  -0.018799 ->  1 weights                      |    0.193359 ->  1 weights                      |    0.000114 -> 11 weights ████████             |                                               
  -0.014954 ->  1 weights                      |                                                |                                                |                                               
  ... and 26 more unique values                |                                                |                                                |                                               
===============================================================================================================================================================================================

```


### 📊 MUTATION REPORT

```

🔍 DEBUG SNAPSHOT FOR blk.0.ffn_gate.weight:
  -> Original: Type=<class 'numpy.ndarray'>, Dtype=float32, Shape=(25165824,)
  -> Original sample values: [-0.00296021 -0.00479126  0.01989746 -0.01086426  0.02075195]
  -> Eval Array: Type=<class 'numpy.ndarray'>, Dtype=float32, Shape=(25165824,)
  -> Eval sample values: [ 0.          0.          0.00946045 -0.00946045  0.00946045]
  -> GGUF Tensor Types: Original Type ID=N/A, Eval Type ID=N/A


FULL ARCHITECTURE MUTATION REPORT [SIMULATED MODEL]
===================================================================================================================
GROUP / TENSOR COMPONENT                      | RECON. MSE      | COSINE SIMILARITY  | STATUS         
===================================================================================================================
👉 blk.0.attn_norm.weight                     | 0.000000        | 1.000000           | PRESERVED      
👉 blk.0.ffn_down.weight                      | 0.000129        | 0.896367           | MUTATED        
👉 blk.0.ffn_gate.weight                      | 0.000145        | 0.898919           | MUTATED        
👉 blk.0.ffn_up.weight                        | 0.000130        | 0.899407           | MUTATED        
👉 blk.0.ffn_norm.weight                      | 0.000000        | 1.000000           | PRESERVED      
👉 blk.0.attn_k.weight                        | 0.000000        | 1.000000           | PRESERVED      
👉 blk.0.attn_output.weight                   | 0.000000        | 1.000000           | PRESERVED      
👉 blk.0.attn_q.weight                        | 0.000000        | 1.000000           | PRESERVED      
👉 blk.0.attn_v.weight                        | 0.000000        | 1.000000           | PRESERVED      
👉 blk.14.attn_norm.weight                    | 0.000000        | 1.000000           | PRESERVED      
👉 blk.14.ffn_down.weight                     | 0.000131        | 0.896715           | MUTATED        
👉 blk.14.ffn_gate.weight                     | 0.000154        | 0.896514           | MUTATED        
👉 blk.14.ffn_up.weight                       | 0.000137        | 0.897944           | MUTATED        
👉 blk.14.ffn_norm.weight                     | 0.000000        | 1.000000           | PRESERVED      
👉 blk.14.attn_k.weight                       | 0.000000        | 1.000000           | PRESERVED      
👉 blk.14.attn_output.weight                  | 0.000000        | 1.000000           | PRESERVED      
👉 blk.14.attn_q.weight                       | 0.000000        | 1.000000           | PRESERVED      
👉 blk.14.attn_v.weight                       | 0.000000        | 1.000000           | PRESERVED      
👉 output_norm.weight                         | 0.000000        | 1.000000           | PRESERVED      
===================================================================================================================
📊 COMPREHENSIVE ARCHITECTURE MUTATION SUMMARY
===================================================================================================================
🔹 FFN Core (Grafted Ternary)                 | Avg MSE: 0.000143 | Avg Cos Sim: 0.898547 | Items: 84
🔹 Attention Projections (Preserved)          | Avg MSE: 0.000000 | Avg Cos Sim: 1.000000 | Items: 112
🔹 Layer Norms & Embeddings                   | Avg MSE: 0.000000 | Avg Cos Sim: 1.000000 | Items: 59
===================================================================================================================

```

### Result Analysis

#### 📌 Test Case Axis: GSM8K_Math
**Prompt:** *"Solve for x: 3x + 7 = 22. Show the  work step-by-step."*

#### 🍏 Baseline Expected Output:
> Here's the step-by-step solution to solve for x:

**Equation:** 3x + 7 = 22

**Step 1: Subtract 7 from both sides**

Our goal is to isolate the term with x. To do this, we'll subtract 7 from both sides of the equation:

#### 🟡 Simulated Realized Output:
> gemgemgemgemgemgemgemgemgemgemgemgemgemgemgemgemgemgemgemgemgemgemgemgemgemgemgemgemgemgemgemgemgemgemgemgemgemgemgemgemgemgemgemgemgemgemgemgemgemgemgemgemgemgemgemgemgemgemgemgemgemgemgemgem

#### 🧪 Packed Mutant Realized Output:
> ????????????????????????????????????????????????????????????????

<br>

#### 📌 Test Case Axis: Logic_Riddle
**Prompt:** *"A man has 4 sons. Each son has one sister. How many children does the man have?"*

#### 🍏 Baseline Expected Output:
> The answer is... 5!

The man has 4 sons, and each son has one sister, which means the man also has one daughter. So, in total, the man has 4 sons and 1 daughter, making a total of 5 children.

#### 🟡 Simulated Realized Output:
> cols cols cols cols cols cols cols cols cols cols cols cols cols cols cols cols cols cols cols cols cols cols cols cols cols cols cols cols cols cols cols cols cols cols cols cols cols cols cols cols cols cols cols cols cols cols cols cols cols cols cols cols cols cols cols cols cols cols cols cols cols cols cols cols

#### 🧪 Packed Mutant Realized Output:
> ????????????????????????????????????????????????????????????????

<br>

#### 📌 Test Case Axis: MMLU_Concept
**Prompt:** *"What are the primary structural differences between a standard Transformer LLM block and an isolated Feed-Forward Network (FFN) layer?"*

#### 🍏 Baseline Expected Output:
> Transformer Language Models (LLMs) and Feed-Forward Networks (FFNs) are two different architectures used in natural language processing (NLP) tasks. While both can be used for language modeling, they have distinct structural differences.

**Standard Transformer LLM Block:**

A standard Transformer LLM block consists of several layers

#### 🟡 Simulated Realized Output:
> innoc innoc innoc innoc innoc innoc innoc innoc innoc innoc innoc innoc innoc innoc innoc innoc innoc innoc innoc innoc innoc innoc innoc innoc innoc innoc innoc innoc innoc innoc innoc innoc innoc innoc innoc innoc innoc innoc innoc innoc innoc innoc innoc innoc innoc innoc innoc innoc innoc innoc innoc innoc innoc innoc innoc innoc innoc innoc innoc innoc innoc innoc innoc innoc

#### 🧪 Packed Mutant Realized Output:
> ????????????????????????????????????????????????????????????????


### 🏥 Summary


#### 🔍 Why the Simulated Model is Giving Bad Responses

It was noticed that the Simulated Model has a much better MSE (0.000143) and Cosine Similarity (0.898) than the HeaderPatched model. Mathematically, it is much closer to the original Llama model.

So why does it output repeating words like gemgemgem?

Even though a cosine similarity of 0.89 looks decent on paper, it represents a massive structural shock to a living neural network.

Llama’s attention layers are incredibly sensitive. They send precise fractional numbers into the FFN layers. The original FFN layers had a smooth, continuous bell curve of millions of precise values to process them.

In the Simulated model, the model was chopped that smooth curve into just 3 rigid flat steps (+delta, 0, -delta).

Because the model was never trained to handle this, the math inside the hidden states gets slightly corrupted at Layer 0. That corruption multiplies at Layer 1, compounds at Layer 2, and by Layer 32, the model's brain is totally scrambled. It completely loses its ability to form sentences, causing it to get stuck in a loop printing the exact same word over and over.


#### 🔍 Why the HeaderPatched Model is Giving Even Worse Responses (????)

If the Simulated model is experiencing a math shock, the HeaderPatched model is experiencing total sensory deprivation. It is outputting ??????? because llama.cpp is misinterpreting the binary data.

The author lied to llama.cpp. the author took the  custom 1.58-bit radix-packed bytes and told llama.cpp, "Hey, this is a standard Q4_0 4-bit tensor!"

llama.cpp accepted the file without crashing, but when it runs the model, it uses its built-in Q4_0 calculation formula to read the  bytes.

A standard Q4_0 block expects a 16-bit scale factor followed by 4-bit signed integers ranging from -8 to +7.

the  packing code grouped numbers completely differently based on the  custom ternary radix math.

Because llama.cpp is using a standard Q4_0 formula to read a custom 1.58-bit binary layout, it decodes the weights into wildly incorrect, astronomical numbers. The activations explode instantly to infinity, causing the engine to give up and print ? characters.


## Repo Structure

```text
project-frankenstein/
│
├── models/
│   └── Llama-3.2-3B-Instruct-BF16.gguf           <--       [Raw model]
│   └── Llama-3.2-3B-Instruct-Simulated.gguf      <--  🧬  [SIMULATED]
│   └── Llama-3.2-3B-Instruct-Mutant.gguf         <--  🧬  [THE MUTANT]
│   └── Llama-3.2-3B-Instruct-HeaderPatched.gguf  <--  🧬  [THE HEADERPATCHED]
│ 
├── diagnosis.py                                  
├── surgery.py                                     
├── patch_header.py  
│
└── evaluation/
    ├── inspect.py                                   
    ├── verify_mutant.py                                   
    ├── benchmark_behavior.py  
    ├── full_evaluation.py  
    └── check_bytes.py           

 
 ```

## 📄 License & Attribution
This architecture framework builds heavily upon concepts outlined in Microsoft Research's The Era of 1.58-bit Large Language Models (BitNet) and the open-source acceleration tooling inside llama.cpp. Licensed under the MIT Research License.
