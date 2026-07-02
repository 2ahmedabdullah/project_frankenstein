# 📊 Comprehensive Hybridization Architecture & Behavior Report

**Generated:** 2026-07-03 03:31:01

## 🧬 1. Tensor Reconstruction Matrix (Weight Distortions)
> Structural metrics generated from target surgery layers.

| LAYER NAME | RECON. MSE ERROR | COSINE SIMILARITY | SPARSITY (0s %) | STATUS |
| :--- | :--- | :--- | :--- | :--- |
| `blk.0.ffn_norm.weight` | 0.7120 | 0.9840 | 0.0% | **🟢 STABLE** |
| `blk.0.ffn_gate.weight` | 3.4210 | 0.7120 | 34.2% | **🚨 HIGH QUANT LOSS** |
| `blk.0.ffn_up.weight` | 3.1150 | 0.7350 | 31.1% | **🚨 HIGH QUANT LOSS** |
| `blk.0.ffn_down.weight` | 3.8410 | 0.6920 | 45.2% | **🚨 HIGH QUANT LOSS** |

## 🎛️ 2. Macro Resource & Velocity Allocation

| Hardware Performance Attribute | Baseline BF16 Model | Hybrid Mutant Model | Net Delta |
| :--- | :--- | :--- | :--- |
| **Model Graph VRAM Footprint** | 5.83 GB | 0.00 GB | -5.83 GB |
| **Average Token Velocity** | 11.40 t/s | 0.00 t/s | -100.0% |

---

## 🧩 3. Side-by-Side Behavioral Output Check

### 📌 Test Case Axis: GSM8K_Math
**Prompt:** *"Solve for x: 3x + 7 = 22. Show your work step-by-step."*

* **Baseline Speed:** 11.13 t/s | **Mutant Speed:** 0.0 t/s

#### 🍏 Baseline Expected Output:
> Here's the step-by-step solution to solve for x:

**Equation:** 3x + 7 = 22

**Step 1: Subtract 7 from both sides**

Our goal is to isolate the term with x. To do this, we'll subtract 7 from both sides of the equation:

3x + 7 - 7 = 22 - 7
3x = 15

**Step 2: Divide both sides by 3**

Now that we have 3x by itself, we'll divide both sides of the equation by 3 to solve for x:

(3x) /

#### 🧪 Mutant Realized Output:
> Execution Error: Failed to load model from file: ./models/Llama-3.2-3B-Instruct-Mutant.gguf

<br>

### 📌 Test Case Axis: Logic_Riddle
**Prompt:** *"A man has 4 sons. Each son has one sister. How many children does the man have?"*

* **Baseline Speed:** 11.25 t/s | **Mutant Speed:** 0.0 t/s

#### 🍏 Baseline Expected Output:
> The answer is... 5!

The man has 4 sons, and each son has one sister, which means the man also has one daughter. So, in total, the man has 4 sons and 1 daughter, making a total of 5 children.

#### 🧪 Mutant Realized Output:
> Execution Error: Failed to load model from file: ./models/Llama-3.2-3B-Instruct-Mutant.gguf

<br>

### 📌 Test Case Axis: MMLU_Concept
**Prompt:** *"What are the primary structural differences between a standard Transformer LLM block and an isolated Feed-Forward Network (FFN) layer?"*

* **Baseline Speed:** 11.54 t/s | **Mutant Speed:** 0.0 t/s

#### 🍏 Baseline Expected Output:
> Transformer Language Models (LLMs) and Feed-Forward Networks (FFNs) are two different architectures used in natural language processing (NLP) tasks. While both can be used for language modeling, they have distinct structural differences.

**Standard Transformer LLM Block:**

A standard Transformer LLM block consists of several layers, each containing:

1. **Self-Attention Mechanism**: This is the core component of the Transformer architecture. It allows the model to attend to different parts of the input sequence simultaneously and weigh their importance.
2. **Feed-Forward Network (FFN)**: This is a fully connected neural network that transforms the

#### 🧪 Mutant Realized Output:
> Execution Error: Failed to load model from file: ./models/Llama-3.2-3B-Instruct-Mutant.gguf

<br>

### 📌 Test Case Axis: Creative_Expl
**Prompt:** *"Explain quantum computing to a 10 year old using an analogy."*

* **Baseline Speed:** 11.54 t/s | **Mutant Speed:** 0.0 t/s

#### 🍏 Baseline Expected Output:
> Imagine you have a huge library with an infinite number of books, and each book has a different combination of letters and words. Now, imagine you want to find a specific book with a certain word on it, like "cat".

A regular computer would look through the books one by one, checking each book until it finds the one with the word "cat". This would take a very long time, because there are so many books!

A quantum computer is like a super-smart librarian who can look at ALL the books at the same time. It can check every single book simultaneously, and it can even look at the books in a special

#### 🧪 Mutant Realized Output:
> Execution Error: Failed to load model from file: ./models/Llama-3.2-3B-Instruct-Mutant.gguf

<br>

### 📌 Test Case Axis: Philosophy
**Prompt:** *"If a tree falls in a forest and no one is around to hear it, does it make a sound? Answer philosophically."*

* **Baseline Speed:** 11.52 t/s | **Mutant Speed:** 0.0 t/s

#### 🍏 Baseline Expected Output:
> The classic philosophical thought experiment! This question has been debated by philosophers, physicists, and linguists for centuries, and there's no straightforward answer. However, let's dive into the philosophical aspects of this conundrum.

**The Physical Perspective**

From a physical standpoint, when a tree falls, it creates a disturbance in the air particles around it, which propagates outward in all directions as a pressure wave. This pressure wave, or sound wave, is what we perceive as sound. So, from this perspective, the answer is: yes, the falling tree would still make a sound, even if no one is present to hear it.

#### 🧪 Mutant Realized Output:
> Execution Error: Failed to load model from file: ./models/Llama-3.2-3B-Instruct-Mutant.gguf

<br>

