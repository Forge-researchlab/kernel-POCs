# Embedding Kernel - Visual Explanation

## 1. What is a "Weight Matrix" in Embeddings?

The "weight matrix" IS the embedding table. There's no math happening — it's
just a big lookup table that the model LEARNS during training.

Before training, it's initialized randomly. During training, backpropagation
updates these numbers so that similar words end up with similar vectors.

```
WEIGHT MATRIX (aka Embedding Table)
====================================
This is a 2D tensor of shape (vocab_size x embedding_dim)
Each ROW is one word's learned vector representation.

                  dim_0   dim_1   dim_2   dim_3    (embedding_dim = 4)
                ┌───────┬───────┬───────┬───────┐
  index 0 "the" │  0.12 │ -0.34 │  0.56 │  0.78 │  <- row 0
                ├───────┼───────┼───────┼───────┤
  index 1 "cat" │  0.91 │  0.23 │ -0.45 │  0.67 │  <- row 1
                ├───────┼───────┼───────┼───────┤
  index 2 "sat" │ -0.11 │  0.89 │  0.32 │ -0.54 │  <- row 2
                ├───────┼───────┼───────┼───────┤
  index 3 "on"  │  0.43 │ -0.67 │  0.21 │  0.88 │  <- row 3
                ├───────┼───────┼───────┼───────┤
  index 4 "mat" │ -0.33 │  0.55 │ -0.77 │  0.11 │  <- row 4
                └───────┴───────┴───────┴───────┘

  These numbers are NOT hand-picked.
  They are LEARNED parameters, just like weights in a linear layer.
  PyTorch calls them "weight" because they're learnable parameters.
```

**Why "weight"?** In PyTorch, any learnable parameter in a layer is called a
"weight". `nn.Embedding.weight` is a `(vocab_size, embedding_dim)` Parameter
tensor. It's the ONLY thing inside an embedding layer — there's no bias, no
activation, just this table.


## 2. The Forward Pass: Table Lookup (Gather)

Given input indices, just grab the corresponding rows. That's it.

```
INPUT INDICES: [1, 3, 1, 4]
(These are token IDs — "cat", "on", "cat", "mat")

WEIGHT MATRIX                          OUTPUT
┌───────────────────────────┐
│ row 0:  0.12 -0.34  0.56  0.78 │
│                                 │
│ row 1:  0.91  0.23 -0.45  0.67 │──┐
│                                 │  │    ┌─────────────────────────────┐
│ row 2: -0.11  0.89  0.32 -0.54 │  ├──> │  0.91  0.23 -0.45  0.67   │ index 1
│                                 │  │    │  0.43 -0.67  0.21  0.88   │ index 3
│ row 3:  0.43 -0.67  0.21  0.88 │──┘    │  0.91  0.23 -0.45  0.67   │ index 1 (same!)
│                                 │  ┌──> │ -0.33  0.55 -0.77  0.11   │ index 4
│ row 4: -0.33  0.55 -0.77  0.11 │──┘    └─────────────────────────────┘
└───────────────────────────┘
                                          shape: (4, 4) = (n_tokens, embedding_dim)

Notice: index 1 appears TWICE → same row is copied twice.
No multiplication. No addition. Just COPY rows.
```


## 3. Why is This a GPU Problem?

In real models:
- vocab_size = 32,000 to 256,000
- embedding_dim = 768 to 8,192
- batch of indices = thousands of tokens

That's millions of memory accesses. GPUs are great at this because they have
massive memory bandwidth and can do thousands of lookups in parallel.


## 4. How the Triton Kernel Tiles the Work

The kernel splits the 2D output (n_tokens x embedding_dim) into blocks:

```
OUTPUT TENSOR (n_tokens=6, embedding_dim=8)
Each cell is one float that needs to be looked up and written.

              dim 0-3          dim 4-7
           ┌──────────────┬──────────────┐
  token 0  │              │              │
  token 1  │  Block(0,0)  │  Block(0,1)  │
  token 2  │              │              │
           ├──────────────┼──────────────┤
  token 3  │              │              │
  token 4  │  Block(1,0)  │  Block(1,1)  │
  token 5  │              │              │
           └──────────────┴──────────────┘

           BLOCK_SIZE_M=3     BLOCK_SIZE_N=4

  Grid = (ceil(6/3), ceil(8/4)) = (2, 2) = 4 GPU thread blocks

  Each block runs independently on the GPU!
  Block(0,0): handles tokens 0-2, dims 0-3
  Block(0,1): handles tokens 0-2, dims 4-7
  Block(1,0): handles tokens 3-5, dims 0-3
  Block(1,1): handles tokens 3-5, dims 4-7
```


## 5. Inside One Block of the Forward Kernel

Let's trace Block(0,0) step by step:

```
pid_m = 0, pid_n = 0
BLOCK_SIZE_M = 3, BLOCK_SIZE_N = 4

Step 1: Compute which tokens we handle
────────────────────────────────────────
  offsets_m = [0, 1, 2]          (token positions in the input)
  mask_m    = [T, T, T]          (all within bounds)

Step 2: Load the actual vocabulary indices
────────────────────────────────────────
  indices_ptr:  [1, 3, 1, 4, 0, 2]  (the full input)
                 ^  ^  ^
  indices = tl.load(indices_ptr + [0,1,2]) = [1, 3, 1]

  These are the ROW numbers in the weight matrix!

Step 3: Compute where to READ from in the weight matrix
────────────────────────────────────────
  offsets_n = [0, 1, 2, 3]      (which dimensions)

  embedding_offsets = indices[:, None] * embedding_dim + offsets_n[None, :]

     indices = [1, 3, 1]  (column vector)
     offsets = [0, 1, 2, 3] (row vector)

     Broadcasting:
     ┌─────────────────────────────────┐
     │ 1*8+0  1*8+1  1*8+2  1*8+3     │   = [8,  9,  10, 11]  <- row 1
     │ 3*8+0  3*8+1  3*8+2  3*8+3     │   = [24, 25, 26, 27]  <- row 3
     │ 1*8+0  1*8+1  1*8+2  1*8+3     │   = [8,  9,  10, 11]  <- row 1
     └─────────────────────────────────┘

  These are SCATTERED addresses — jumping around in memory!

Step 4: Load the actual embedding values
────────────────────────────────────────
  embeddings = tl.load(embeddings_ptr + embedding_offsets)

     Reads from weight matrix memory positions:
     [8,9,10,11], [24,25,26,27], [8,9,10,11]

Step 5: Compute where to WRITE in the output
────────────────────────────────────────
  output_offsets = offsets_m[:, None] * embedding_dim + offsets_n[None, :]

     offsets_m = [0, 1, 2]
     ┌─────────────────────────────────┐
     │ 0*8+0  0*8+1  0*8+2  0*8+3     │   = [0, 1, 2, 3]    <- output row 0
     │ 1*8+0  1*8+1  1*8+2  1*8+3     │   = [8, 9, 10, 11]  <- output row 1
     │ 2*8+0  2*8+1  2*8+2  2*8+3     │   = [16,17,18,19]   <- output row 2
     └─────────────────────────────────┘

  These are CONTIGUOUS — nice sequential addresses!

Step 6: Store
────────────────────────────────────────
  tl.store(output_ptr + output_offsets, embeddings)

  DONE. Scattered read → Contiguous write.
```


## 6. The Backward Pass: Why atomic_add?

During training, we need gradients for the weight matrix.
The gradient says "how should each embedding vector change?"

```
PROBLEM: Multiple tokens can have the SAME index

  Input indices:  [1, 3, 1, 4]
                   ^     ^
                   Both point to row 1!

  grad_output (from next layer):
    token 0 (index 1): [0.1, 0.2, -0.3, 0.4]
    token 1 (index 3): [0.5, -0.1, 0.2, 0.3]
    token 2 (index 1): [0.3, 0.1, -0.2, 0.5]   <- also for row 1!
    token 3 (index 4): [-0.2, 0.4, 0.1, -0.3]

  grad_weight for row 1 must be the SUM of both gradients:
    = [0.1+0.3, 0.2+0.1, -0.3+(-0.2), 0.4+0.5]
    = [0.4, 0.3, -0.5, 0.9]


  WHY atomic_add?
  ═══════════════
  Multiple GPU threads run IN PARALLEL. If two threads both try to
  write to row 1 at the same time:

    Thread A: reads  grad_weight[1] = 0.0
    Thread B: reads  grad_weight[1] = 0.0      <- STALE!
    Thread A: writes grad_weight[1] = 0.0 + 0.1 = 0.1
    Thread B: writes grad_weight[1] = 0.0 + 0.3 = 0.3   <- OVERWRITES A's work!

    Result: 0.3 (WRONG! Should be 0.4)

  atomic_add guarantees the read-modify-write is ONE indivisible operation:

    Thread A: atomic_add(grad_weight[1], 0.1) → now 0.1
    Thread B: atomic_add(grad_weight[1], 0.3) → now 0.4  ✓ CORRECT
```


## 7. Full Forward + Backward Flow

```
                         FORWARD PASS
                         ════════════

  ┌─────────────┐     ┌────────────────────┐     ┌─────────────┐
  │   indices    │     │   Weight Matrix     │     │   output    │
  │  [1, 3, 1]  │────>│  (learned table)    │────>│ (3 vectors) │
  │             │     │                    │     │             │
  │ "which rows" │     │  row 0: [........] │     │ row1: [...] │
  │             │     │  row 1: [........] │     │ row3: [...] │
  │             │     │  row 2: [........] │     │ row1: [...] │
  │             │     │  row 3: [........] │     │             │
  └─────────────┘     └────────────────────┘     └──────┬──────┘
                                                        │
                       ... rest of model ...            │
                                                        │
                         BACKWARD PASS                  │
                         ═════════════                  │
                                                        ▼
  ┌─────────────┐     ┌────────────────────┐     ┌─────────────┐
  │   indices    │     │   grad_weight      │     │ grad_output │
  │  [1, 3, 1]  │────>│  (zeros initially) │<────│ (from loss) │
  │             │     │                    │     │             │
  │ "which rows  │     │  row 0: [0,0,0,0] │     │ g0: [....] ─┐
  │  to update" │     │  row 1: [g0 + g2]  │<────│ g1: [....] │ │
  │             │     │  row 2: [0,0,0,0] │     │ g2: [....] ─┘
  │             │     │  row 3: [g1]       │<────│             │  atomic_add
  └─────────────┘     └────────────────────┘     └─────────────┘  (because row 1
                                                                   gets 2 gradients)
                       This grad_weight is used by
                       the optimizer (Adam, SGD, etc.)
                       to UPDATE the weight matrix.
```


## 8. The Autograd Wrapper

```
LigerEmbeddingFunction (torch.autograd.Function)
│
├── forward(ctx, embeddings, indices)
│   │
│   ├── 1. Flatten indices: [2,3] shape → [6] (1D)
│   │
│   ├── 2. Allocate output: empty(n_tokens, embedding_dim)
│   │
│   ├── 3. Compute grid:
│   │      grid = (ceil(n_tokens/BLOCK_M), ceil(emb_dim/BLOCK_N))
│   │
│   ├── 4. Launch forward kernel on GPU
│   │      embedding_forward_kernel[grid](...)
│   │
│   ├── 5. Save for backward:
│   │      ctx.save_for_backward(indices, embeddings)
│   │
│   └── 6. Reshape output: (6, dim) → (2, 3, dim)
│
└── backward(ctx, grad_output)
    │
    ├── 1. Retrieve saved: indices, embedding_table
    │
    ├── 2. Allocate grad_weight: zeros_like(embedding_table)
    │      (MUST be zeros — we accumulate into it)
    │
    ├── 3. Launch backward kernel on GPU
    │      embedding_backward_kernel[grid](...)
    │
    └── 4. Return (grad_weight, None)
                              ^^^^
                    indices have no gradient
                    (they're integers, not floats)
```


## 9. Why Triton Over PyTorch's Built-in?

```
PyTorch nn.Embedding:
  indices ──> Python overhead ──> CUDA kernel launch ──> lookup ──> output
                    ^
                    Extra dispatch, possible extra copies

Triton Embedding:
  indices ──> Single fused kernel ──> output
                    ^
                    Direct control over memory access patterns,
                    block sizes, and parallelism
```

Benefits:
- **One kernel launch** instead of potentially multiple internal ops
- **Explicit tiling** — you control how work maps to GPU threads
- **Better memory coalescing** — adjacent threads access adjacent memory
- **No Python/C++ dispatch overhead** for the core operation


---


## 10. What is a Gradient? (From Scratch)

A gradient answers one question: **"If I nudge this number a tiny bit, how
much does the final loss change?"**

### 10.1 The Big Picture: Why Do We Need Gradients?

```
The GOAL of training: minimize the loss (a single number that says
                      "how wrong is the model?")

The TOOL: gradients tell us which direction to nudge each parameter
          to make the loss go DOWN.

  loss = 3.5        "model is wrong"
    │
    │  compute gradients
    ▼
  gradient of weight[1][2] = +0.3
    │
    │  meaning: "if weight[1][2] goes UP, loss goes UP by 0.3"
    │           so we should make weight[1][2] go DOWN
    ▼
  weight[1][2] -= learning_rate * 0.3
    │
    │  now the model is slightly less wrong
    ▼
  loss = 3.47       "model is less wrong"
```

### 10.2 Chain Rule: How Gradients Flow Backward

A neural network is a chain of operations. Gradients flow BACKWARD
through this chain using the **chain rule** from calculus.

```
FORWARD (left to right):
═══════════════════════

  input    Embedding     Linear      ReLU       Linear      Loss
  tokens ──────────> h1 ──────> h2 ──────> h3 ──────> h4 ──────> loss
  [1,3,1]                                                     = 3.5


BACKWARD (right to left):
═════════════════════════

  The loss is a single number. We ask: "how does each thing
  before it affect the loss?"

                                                          d(loss)
  loss = 3.5                                             ─────── = 1.0
                                                          d(loss)
         ◄── Linear ◄── ReLU ◄── Linear ◄── Embedding
                                                │
                                                ▼
                                          "how does each
                                           embedding value
                                           affect the loss?"
                                           = grad_output for
                                             the embedding layer

  Each layer receives grad_output from the layer AFTER it.
  Each layer computes grad_weight (gradient for its own parameters).
  Each layer passes grad_input to the layer BEFORE it.
```

### 10.3 What is grad_output for the Embedding Layer?

By the time the backward pass reaches the embedding layer, all the
layers after it have already computed their gradients. The embedding
layer receives **grad_output** — a tensor the SAME SHAPE as the
embedding output.

```
Forward output shape:   (n_tokens, embedding_dim)
grad_output shape:      (n_tokens, embedding_dim)    ← SAME SHAPE!

  Forward output (what embedding produced):
  ┌─────────────────────────────┐
  │  token 0:  0.91  0.23 -0.45  0.67  │   (looked up from row 1)
  │  token 1:  0.43 -0.67  0.21  0.88  │   (looked up from row 3)
  │  token 2:  0.91  0.23 -0.45  0.67  │   (looked up from row 1)
  └─────────────────────────────┘

  grad_output (received from next layer):
  ┌─────────────────────────────┐
  │  token 0:  0.10  0.20 -0.30  0.40  │   "how token 0's embedding affected loss"
  │  token 1:  0.50 -0.10  0.20  0.30  │   "how token 1's embedding affected loss"
  │  token 2:  0.30  0.10 -0.20  0.50  │   "how token 2's embedding affected loss"
  └─────────────────────────────┘

  Each number means:
    grad_output[0][2] = -0.30
    → "if output[0][2] increases by 1, the loss DECREASES by 0.30"
```


## 11. Backward Pass with Duplicate Tokens: Full Walkthrough

This is the KEY scenario. Let's trace what happens when "cat" (index 1)
appears at position 0 AND position 2 in the input.

### 11.1 Setup

```
Vocabulary:   0="the", 1="cat", 2="sat", 3="on"
Input:        "cat on cat"  →  indices = [1, 3, 1]
                                          ^     ^
                                          SAME TOKEN (index 1)

Weight Matrix (embedding_dim=4):
  row 0 "the": [0.12, -0.34,  0.56,  0.78]
  row 1 "cat": [0.91,  0.23, -0.45,  0.67]  ← used TWICE
  row 2 "sat": [-0.11, 0.89,  0.32, -0.54]
  row 3 "on":  [0.43, -0.67,  0.21,  0.88]  ← used once
```

### 11.2 Forward Pass

```
indices = [1, 3, 1]

  Position 0 → look up row 1 → [0.91,  0.23, -0.45,  0.67]  (cat)
  Position 1 → look up row 3 → [0.43, -0.67,  0.21,  0.88]  (on)
  Position 2 → look up row 1 → [0.91,  0.23, -0.45,  0.67]  (cat, SAME vector)

Output:
  ┌──────────────────────────────┐
  │ pos 0: [0.91,  0.23, -0.45,  0.67] │  ← copy of row 1
  │ pos 1: [0.43, -0.67,  0.21,  0.88] │  ← copy of row 3
  │ pos 2: [0.91,  0.23, -0.45,  0.67] │  ← copy of row 1 (SAME data!)
  └──────────────────────────────┘

  NOTE: Even though pos 0 and pos 2 have the same data,
  they are INDEPENDENT copies in the output. They will go through
  the rest of the network separately and produce DIFFERENT gradients
  because they are at different positions in the sequence.
```

### 11.3 What Happens in the Rest of the Network

```
Output from embedding goes through many layers...

  pos 0 "cat" embedding ──┐
  pos 1 "on"  embedding ──┼──> Attention, FFN, ... ──> prediction ──> loss
  pos 2 "cat" embedding ──┘

  Even though pos 0 and pos 2 started with identical vectors,
  attention treats them differently because:
  - They attend to different positions
  - Positional encoding distinguishes them
  - They contribute to different parts of the prediction

  So grad_output will be DIFFERENT for pos 0 and pos 2!
```

### 11.4 Backward Pass Arrives at Embedding Layer

```
grad_output (received from the layers above):
  ┌──────────────────────────────────┐
  │ pos 0: [0.10,  0.20, -0.30,  0.40] │  gradient for "cat" at position 0
  │ pos 1: [0.50, -0.10,  0.20,  0.30] │  gradient for "on"  at position 1
  │ pos 2: [0.30,  0.10, -0.20,  0.50] │  gradient for "cat" at position 2
  └──────────────────────────────────┘
              ▲               ▲
              │               │
        DIFFERENT gradients even though same token!
        (because the network used them differently)
```

### 11.5 The Core Question: How to Update the Weight Matrix?

```
We need grad_weight: same shape as the weight matrix (vocab_size x embedding_dim)
It starts as ALL ZEROS:

  grad_weight:
  row 0 "the": [0.00, 0.00, 0.00, 0.00]   ← not in input, stays zero
  row 1 "cat": [0.00, 0.00, 0.00, 0.00]   ← needs update from pos 0 AND pos 2
  row 2 "sat": [0.00, 0.00, 0.00, 0.00]   ← not in input, stays zero
  row 3 "on":  [0.00, 0.00, 0.00, 0.00]   ← needs update from pos 1


THE RULE: If a weight matrix row was used N times in the forward pass,
          its gradient is the SUM of all N grad_outputs that came from it.

WHY SUM? Because in the forward pass, the SAME row was COPIED to N places.
Calculus chain rule says: if one variable feeds into multiple paths,
the total gradient = sum of gradients from all paths.

  Think of it like a river splitting:

         ┌──> pos 0 (used in attention, etc.) ──> contributes to loss
  row 1 ─┤
         └──> pos 2 (used in attention, etc.) ──> contributes to loss

  Row 1 affects the loss through BOTH paths.
  Total effect = effect through path 0 + effect through path 2


COMPUTATION:
────────────

  For row 1 "cat" (appeared at pos 0 and pos 2):
    grad_weight[1] = grad_output[0] + grad_output[2]
                   = [0.10, 0.20, -0.30, 0.40] + [0.30, 0.10, -0.20, 0.50]
                   = [0.40, 0.30, -0.50, 0.90]

  For row 3 "on" (appeared at pos 1 only):
    grad_weight[3] = grad_output[1]
                   = [0.50, -0.10, 0.20, 0.30]

  For rows 0, 2 (not in input):
    grad_weight[0] = [0.00, 0.00, 0.00, 0.00]   ← zero, untouched
    grad_weight[2] = [0.00, 0.00, 0.00, 0.00]   ← zero, untouched


FINAL grad_weight:
  row 0 "the": [0.00,  0.00,  0.00,  0.00]
  row 1 "cat": [0.40,  0.30, -0.50,  0.90]  ← SUM of two gradients
  row 2 "sat": [0.00,  0.00,  0.00,  0.00]
  row 3 "on":  [0.50, -0.10,  0.20,  0.30]  ← single gradient
```

### 11.6 Then the Optimizer Uses This

```
After backward gives us grad_weight, the optimizer updates the weight matrix:

  Simple SGD (learning_rate = 0.01):

  weight[1] = weight[1] - 0.01 * grad_weight[1]
            = [0.91, 0.23, -0.45, 0.67] - 0.01 * [0.40, 0.30, -0.50, 0.90]
            = [0.91 - 0.004, 0.23 - 0.003, -0.45 + 0.005, 0.67 - 0.009]
            = [0.906, 0.227, -0.445, 0.661]

  The "cat" embedding just got slightly adjusted to make the model's
  prediction better. This happens for EVERY training step.
```


## 12. What is atomic_add and Why is it Needed?

### 12.1 The Problem: Race Conditions on a GPU

A GPU runs THOUSANDS of threads at the same time. In the backward kernel,
different threads handle different tokens. When two tokens have the same
index, their threads both need to write to the SAME row in grad_weight.

```
NORMAL ADD (not atomic) — THREE separate steps:
═══════════════════════════════════════════════

  Step 1: READ   the current value from memory
  Step 2: ADD    your value to it
  Step 3: WRITE  the result back to memory


THE RACE CONDITION:
═══════════════════

  grad_weight[1] starts at [0.0, 0.0, 0.0, 0.0]

  Thread A (handling pos 0, index 1):  wants to add [0.10, 0.20, -0.30, 0.40]
  Thread B (handling pos 2, index 1):  wants to add [0.30, 0.10, -0.20, 0.50]

  Time ──────────────────────────────────────────────────────>

  Thread A                          Thread B
  ────────                          ────────
  1. READ:  val = [0,0,0,0]
                                    1. READ:  val = [0,0,0,0]   ← STALE!!
                                       (Thread A hasn't written yet)
  2. ADD:   val = [0.1,0.2,-0.3,0.4]
                                    2. ADD:   val = [0.3,0.1,-0.2,0.5]
  3. WRITE: grad_weight[1] = [0.1,0.2,-0.3,0.4]
                                    3. WRITE: grad_weight[1] = [0.3,0.1,-0.2,0.5]
                                       ^^^ OVERWRITES Thread A's result!

  RESULT: grad_weight[1] = [0.3, 0.1, -0.2, 0.5]  ← WRONG!
  SHOULD BE:                [0.4, 0.3, -0.5, 0.9]  ← sum of both

  Thread A's contribution is COMPLETELY LOST.
```

### 12.2 The Solution: atomic_add

atomic_add makes the READ + ADD + WRITE happen as ONE indivisible operation.
No other thread can interfere in the middle.

```
ATOMIC ADD — one INDIVISIBLE step:
═══════════════════════════════════

  atomic_add(address, value):
    "read what's at address, add value, write back — ALL AT ONCE,
     NO other thread can touch this address until I'm done"


WITH atomic_add:
════════════════

  grad_weight[1] starts at [0.0, 0.0, 0.0, 0.0]

  Thread A                          Thread B
  ────────                          ────────
  atomic_add(gw[1], [0.1,...])
    ┌─ LOCKED ─────────────┐
    │ read:  [0.0,0.0,0.0,0.0] │
    │ add:   [0.1,0.2,-0.3,0.4]│       (Thread B WAITS — can't access gw[1])
    │ write: [0.1,0.2,-0.3,0.4]│
    └──────────────────────┘
                                    atomic_add(gw[1], [0.3,...])
                                      ┌─ LOCKED ─────────────┐
                                      │ read:  [0.1,0.2,-0.3,0.4] │ ← sees A's write!
                                      │ add:   [0.3,0.1,-0.2,0.5] │
                                      │ write: [0.4,0.3,-0.5,0.9] │
                                      └──────────────────────┘

  RESULT: grad_weight[1] = [0.4, 0.3, -0.5, 0.9]  ← CORRECT!
```

### 12.3 Why Not Just Use atomic_add Everywhere?

```
atomic_add is SLOWER than normal add because:

  1. It forces threads to take turns (serialization)
  2. The hardware has to coordinate between threads
  3. If many threads hit the same address, they queue up

  Normal add:     ~4 nanoseconds
  atomic_add:     ~30-100 nanoseconds (depending on contention)

  For the FORWARD pass: every output position is UNIQUE
    → no two threads write to the same place
    → normal tl.store is fine and FAST

  For the BACKWARD pass: multiple positions can map to the SAME row
    → threads MIGHT write to the same place
    → MUST use atomic_add for correctness, even though slower
```

### 12.4 Visual Summary

```
  FORWARD: output[i] = weight[indices[i]]

    Each output position i is unique → no conflicts → use normal store
    ┌───────────┐        ┌───────────┐
    │ weight    │        │ output    │
    │           │        │           │
    │ row 1 ────┼───────>│ pos 0     │  one-to-one
    │           │        │           │
    │ row 3 ────┼───────>│ pos 1     │  one-to-one
    │           │        │           │
    │ row 1 ────┼───────>│ pos 2     │  one-to-one (same source, different dest)
    └───────────┘        └───────────┘


  BACKWARD: grad_weight[indices[i]] += grad_output[i]

    Multiple positions can target the SAME row → conflicts → use atomic_add
    ┌───────────┐        ┌───────────┐
    │grad_weight│        │grad_output│
    │           │        │           │
    │ row 1 <═══╪════════╪═ pos 0    │  ╗
    │     ▲     │        │           │  ║ BOTH write to row 1
    │     ╚═════╪════════╪═ pos 2    │  ╝ must use atomic_add!
    │           │        │           │
    │ row 3 <───┼────────┼─ pos 1    │  (only one writer, but we
    └───────────┘        └───────────┘   use atomic_add anyway —
                                         we can't know at compile
                                         time which rows collide)
```
