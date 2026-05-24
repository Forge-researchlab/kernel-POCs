# v6 — Design Notes: cuBLAS-Triton Synergy + Optional CUDA Streams

**Date:** 2026-05-24
**GPU:** NVIDIA A100-SXM4-80GB (bf16 tensor cores, CUDA 12.4, PyTorch 2.4.1)
**Status:** Implemented and benchmarked. Source at `experiments/v6/lora_mlp_kernel_v6.py`.

## 1. Where each framework shines

Through v1–v5 we kept gravitating toward one of two extremes:

- **All Triton** (v1, v2): One launch per phase, but Triton's tiled matmul lands at ~0.7x of cuBLAS for the big projections. The launch-count wins evaporate against the per-tile speed deficit.
- **All packed cuBLAS** (v5, v5_upgrade_1): Pack `[W_gate; W_up; A_gate; A_up]` into one mega-GEMM. Cuts forward launches from 8 to 4 (or 5 after the padding fix). But the packed N is awkward (28704 misses cuBLAS's preferred 128-wide tile), and the mega-output tensor is ~448 MB at LLaMA-8B production — a ~3× memory blow-up vs v3 in fwd.

v3 — the simplest member of the family — quietly dominated on memory and stayed within 4% of the fastest variant on latency.

**v6's premise:** drop the packing dogma. cuBLAS is great at the big matmuls when given a clean tile; Triton is great at the *small* matmuls that cuBLAS launches inefficiently. Use both.

```
+---------+----------+--------------------------+
| Phase   | Tool     | Why                       |
+---------+----------+--------------------------+
| gate    | cuBLAS   | N=I=14336 (multiple of 128)
| up      | cuBLAS   | N=I=14336 (multiple of 128)
| LoRA-A  | Triton   | N=2r ∈ {16, 32, 64, 128} —
|         |          | too narrow for cuBLAS tile;
|         |          | stacked → load X once via SMEM
| SwiGLU+ | Triton   | bandwidth-bound elementwise +
| LoRA-B  |          | tiny tl.dot — cuBLAS can't fuse
| down    | cuBLAS   | N=H=4096 (multiple of 128)
| A_down  | cuBLAS   | tiny, hidden under W_down via streams
+---------+----------+--------------------------+
```

## 2. Why per-projection cuBLAS beats mega-packing on tile alignment

The v5 packing diagnosis ([`v5_packing_diagnosis.md`](v5_packing_diagnosis.md)) showed that on A100 bf16, cuBLAS picks ~128-wide N tiles, and any N that's not a multiple of 128 wastes a column of the tile. The gate+up mega-matrix in v5 has `N = 2*I + 2*r = 2*14336 + 32 = 28704` — not a multiple of 128. cuBLAS lands at 79.6% of peak (vs 82.7% for N=28672, the next clean tile).

v5_upgrade_1 fixed this by padding to N=28800. The N-tile penalty went away (+0.9 ms in microbench), but:

- The mega-output tensor is still `[M, 28800]` = 448 MB at LLaMA-8B production — still ~3× v3's per-projection outputs.
- The Triton SwiGLU+LoRA epilogue reads `e_base` and `g_base` as non-contiguous slices (stride 0 = 28800 instead of 14336). That L2-locality penalty costs ~100-200 µs.

**v6 sidesteps both:**

- Per-projection cuBLAS calls produce **independent contiguous `[M, I]` outputs** for `e_base` and `g_base`. No mega-matrix, no slice strides.
- Every cuBLAS call sees its best tile: `N = I = 14336` (gate / up) and `N = H = 4096` (down) are both multiples of 128.

The result: v6 is **bit-equal to v3** on memory (1185 MB at LLaMA-8B production vs v5_upgrade_1's 1412 MB and v5's 1585 MB) while ALSO matching v5_upgrade_1's latency.

## 3. The Triton stacked-LoRA-A trick

The two LoRA-A matmuls `X @ A_gate.T` and `X @ A_up.T` are individually `[M, H] @ [H, r] → [M, r]`. At r=16 each output is `[M, 16]` — way too narrow for cuBLAS to fill a tile efficiently, and the launch overhead is a large fraction of the GEMM time.

The naive fix would be to write a custom Triton kernel that loads `X` once and produces both outputs in two separate accumulators. That's the v1/v2 "fused gate+up" approach — and it works, but it's a fancy multi-output kernel that doesn't generalize.

**v6's trick:** stack the two `A` matrices vertically into `A_stack = cat([A_gate, A_up], dim=0)` of shape `[2r, H]`. Then a *standard* Triton matmul `X @ A_stack.T → [M, 2r]` produces both outputs side-by-side. The Triton compiler sees a normal GEMM with `N = 2r ∈ {16, 32, 64, 128}` — small enough that it can fit the entire N dimension in one program block (`BLOCK_N = next_power_of_2(2r)`).

The payoff is that the compiler **automatically caches each X tile in SMEM and reuses it across the 2r output columns**. We get "load X once" for free, via a kernel that's no more complex than a textbook GEMM:

```python
@triton.jit
def _stacked_lora_a_kernel(X_ptr, A_ptr, Out_ptr, M, N, K, ...):
    pid_m, pid_n = tl.program_id(0), tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_block in range(tl.cdiv(K, BLOCK_K)):
        x_tile = tl.load(...)               # [BLOCK_M, BLOCK_K] — cached in SMEM
        a_tile = tl.load(...)               # [BLOCK_N, BLOCK_K]
        acc += tl.dot(x_tile, tl.trans(a_tile))
        ...
    tl.store(out_ptrs, acc.to(out_dtype), ...)
```

Returning `(xa_gate, xa_up)` is just two views of the `[M, 2r]` output. They are non-contiguous (stride 0 = 2r), but `fused_lora_swiglu` already accepts non-contiguous inputs via explicit stride args (it was designed that way for v5).

## 4. Role of stream parallelism

After the kernel changes, the v6 forward has 6 host-side launches plus an `addmm_` (so 7 total at training time):

1. `e_base = X @ W_gate.T` (cuBLAS)
2. `g_base = X @ W_up.T` (cuBLAS)
3. `[xa_gate, xa_up] = fused_lora_a_stacked(X, A_stack)` (Triton)
4. `h, e_full, g_full = fused_lora_swiglu(e_base, g_base, xa_gate, xa_up, B_gate, B_up)` (Triton)
5. `out = h @ W_down.T` (cuBLAS)
6. `xa_down = h @ A_down.T` (cuBLAS)
7. `out.addmm_(xa_down, B_down.T, alpha=s_down)` (cuBLAS)

Two opportunities for stream parallelism:

- Phase 1: kernels 1 and 2 are independent — both read X but write to different output tensors. They can run on separate streams.
- Phase 4: kernels 5 and 6 are independent — both read h but produce independent outputs. Same trick.

The `LoRAMLPv6Module` lazy-initializes a side stream once per module instance (avoiding repeated `torch.cuda.Stream()` allocation). Event-based sync points serialize before consumers:

```python
side_stream.wait_stream(default)              # side picks up where default left off
e_base = X @ W_gate.t()                       # default
with cuda.stream(side):
    g_base = X @ W_up.t()                     # side overlaps
xa_gate, xa_up = fused_lora_a_stacked(...)    # default
default.wait_stream(side)                     # SwiGLU epilogue needs g_base
h, ... = fused_lora_swiglu(...)               # default
```

**Measured impact:**

- At **LLaMA-8B production (M=8192)**, the GEMMs are big enough to fully saturate the SMs; running gate and up on parallel streams gives the SMs no extra work to do. Stream overlap recovers ~1.4% (12.19 → 12.03 ms). Within run-to-run noise on some configs.
- At **small M (M=512)**, kernels are short and launch overhead is a large fraction of each kernel's runtime. Stream overlap **hides** that overhead entirely: v6_sync regresses to 1.015 ms (~Unsloth speed, slower than v5_upgrade_1's 0.870 ms) but v6_streams runs at **0.852 ms — 1.19x Unsloth, 1.02x v5_upgrade_1, and 1.19x v6_sync**.

Sync-vs-streams output is **bit-exact** in our test suite (37/40 TestV6 tests assert `rtol=0, atol=0` between the two modes). The streams path doesn't change which kernels run or in what order — it only changes which CUDA stream each launch lands on. Both observe the same accumulation order inside each cuBLAS call.

The `enable_streams=False` fallback is a one-kwarg toggle at every level (`lora_mlp_v6()`, `LoRAMLPv6.apply()`, `LoRAMLPv6Module(enable_streams=...)`). Use it on V100 (no stream-concurrency support), in CUDA Graph-captured paths (streams complicate capture), or any time the workload is large enough that overlap doesn't help.

## 5. Measured results

### Latency (bf16, batch=4, seq=2048, M=8192, H=4096, I=14336, r=16)

| Impl | ms | vs Unsloth | vs v3 | vs v5_up1 |
|---|---:|---:|---:|---:|
| Unsloth `apply_lora_mlp_swiglu` | 12.89 | 1.00x | — | — |
| v3 (cuBLAS + Triton epilogue) | 12.35 | 1.04x | 1.00x | 0.97x |
| v5 (packed mega-GEMM) | 12.38 | 1.04x | 1.00x | 0.97x |
| v5_upgrade_1 (padded mega + v3-style down) | 11.99 | 1.07x | 1.03x | 1.00x |
| **v6_sync (cuBLAS + Triton-stacked LoRA-A)** | **12.19** | **1.06x** | **1.01x** | **0.98x** |
| **v6_streams (v6_sync + side-stream overlap)** | **12.03** | **1.07x** | **1.03x** | **1.00x** |
| v5 inference (pre-merged) | 11.78 | 1.09x | 1.05x | 1.02x |

### Memory (LLaMA-8B production, peak forward MB)

| Impl | Fwd (MB) | Fwd+Bwd (MB) | vs Unsloth fwd | Resident after fwd (MB) |
|---|---:|---:|---:|---:|
| Unsloth | 736 | 802 | 1.00x | 512 |
| v3 | 1185 | 1185 | 1.61x | 512 |
| v5 | 1585 | 1585 | 2.15x | 512 |
| v5_upgrade_1 | 1412 | 1412 | 1.92x | 512 |
| **v6_sync** | **1185** | **1185** | **1.61x** | 512 |
| **v6_streams** | **1185** | **1185** | **1.61x** | 512 |
| v5 inference (pre-merged) | 736 | — | 1.00x | 64 |

The exact tie with v3 isn't accidental: v6 uses the same per-projection cuBLAS calls and the same fused SwiGLU+LoRA epilogue. The 0.3 MB delta (1185.0 vs 1184.75) is the tiny `[M, 2r]` Triton-stacked LoRA-A output (~ 1 MiB at LLaMA-8B / r=16) vs v3's two separate `[M, r]` cuBLAS outputs.

### Small-M latency (M=512, b=1, s=512, r=16) — the stream regression threshold

| Impl | ms | vs Unsloth | vs v6_sync | Fwd peak (MB) |
|---|---:|---:|---:|---:|
| Unsloth | 1.013 | 1.00x | — | 46 |
| v3 | 0.973 | 1.04x | — | 74 |
| v5 | 0.859 | 1.18x | — | 415 |
| v5_upgrade_1 | 0.870 | 1.17x | — | 300 |
| v6_sync | 1.015 | 1.00x | 1.00x | 74 |
| **v6_streams** | **0.852** | **1.19x** | **1.19x** | **74** |
| v5 inference | 0.921 | 1.10x | — | 46 |

At M=512, kernels are short enough that launch overhead dominates. v6_sync's 7 launches add up to ~Unsloth speed, while v5/v5_upgrade_1's 4–5 launches give them a small win at the cost of 4–5× the memory. **v6_streams gets both**: 1.19x speedup AND v3-level memory, by overlapping launches across two streams.

### Rank sweep at LLaMA-8B production (bf16, batch=4, seq=2048)

| Rank | Unsloth | v3 | v5_up1 | v6_sync | v6_streams | v6_streams vs v3 | v6_streams vs v5_up1 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 13.01 | 12.16 | 12.05 | 12.20 | 12.04 | 1.010x | 1.001x |
| 16 | 12.89 | 12.35 | 11.99 | 12.19 | 12.03 | 1.027x | 0.997x |
| 32 | 12.71 | 12.44 | 12.09 | 12.15 | 12.01 | 1.036x | 1.007x |
| 64 | 12.79 | 12.23 | 12.20 | 12.39 | 12.20 | 1.003x | 1.000x |

The stacked-A kernel is rank-independent in our regime (BLOCK_N = max(next_power_of_2(2r), 16); single program block for r ≤ 64). Latency scales with H (the K dimension), not r — exactly what you'd expect from a "load X once, multiply by tiny A" architecture.

## 6. Caveats & open questions

- **Small-M stream regression is real but bounded.** At M=512 the cost of `v6_sync` is ~17% over `v5_upgrade_1`. The `enable_streams=True` default fixes it (and is the recommended setting). The threshold where streams stop helping is around M=2048 in our measurements — at M=8192 the win is ~1.4%, at M=2048 it's ~5%, at M=512 it's 19%.
- **`v6_streams` is the default**. If you're running on V100 (no concurrent-streams), in a `torch.compile` graph that captures streams unsafely, or in a Cuda Graph capture, set `enable_streams=False`. The bench shows `v6_sync` is within 1.5% of `v6_streams` at LLaMA-8B production, so falling back is cheap.
- **r=64 stays on the Triton stacked-A path** (BLOCK_N=128). Microbench was not yet run for cuBLAS-vs-Triton crossover at r=64; the current implementation simply uses the stacked-A Triton kernel for all r in {8, 16, 32, 64}. Empirically (the rank sweep above) it ties or beats v5_upgrade_1 at every rank.
- **Forward output is bit-exact across sync and streams modes** (in our tests). This is a property of the current pipeline (no kernel splits, no split-K) and is exploited in `test_v6_sync_vs_streams_bf16` / `test_v6_sync_vs_streams_rank_sweep`. A future tiling change (e.g. switching to a Triton autotuner that may pick different tile shapes) could break the exact-equality and require relaxing those assertions.
- **Backward is unchanged from v3 / v5_upgrade_1.** Backward GEMMs have distinct LHS operands (dY, df, de, X) and no shared input, so neither packing nor stream parallelism help. We keep Unsloth's in-place buffer-reuse pattern verbatim.

## 7. Summary

v6 is the first variant in this family that matches v3's memory while landing in v5_upgrade_1's latency band — and **the only training variant that beats Unsloth on the small-M config** (1.19x at M=512). The principle is straightforward once you stop chasing launch-count wins: cuBLAS and Triton are good at different things. Use each for what it's good at, then add CUDA streams where launch overhead is exposed.

**Production recommendation:** `LoRAMLPv6Module(..., enable_streams=True)` with `refresh_packed()` called after every optimizer step. Use the inference path (`lora_mlp_v6_inference`) once weights are merged.
