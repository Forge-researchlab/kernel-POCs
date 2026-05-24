"""
Run bench_lora_mlp.py 10 times for the LLaMA-8B production config
and report run-to-run variance + post-warmup averages.
"""
import subprocess
import re
import sys
import numpy as np

NUM_RUNS = 10
CMD = [
    sys.executable,
    "benchmarks/bench_lora_mlp.py",
    "--mode", "mlp",
    "--hidden", "4096",
    "--intermediate", "14336",
    "--rank", "16",
    "--seq", "2048",
    "--batch", "4",
]

KEYS = ["unsloth", "v3", "v5_train", "v5_up1_train", "v6_sync", "v6_streams", "v5_inf"]
PATTERN = re.compile(
    r"unsloth=([\d.]+)ms\s+"
    r"v3=([\d.]+)ms\s+"
    r"v5_train=([\d.]+)ms\s+"
    r"v5_up1_train=([\d.]+)ms\s+"
    r"v6_sync=([\d.]+)ms\s+"
    r"v6_streams=([\d.]+)ms\s+"
    r"v5_inf=([\d.]+)ms"
)

results = {k: [] for k in KEYS}

print(f"Running benchmark {NUM_RUNS} times...")
print(f"Config: batch=4, seq=2048, hidden=4096, intermediate=14336, rank=16, bf16")
print(f"Each run: warmup={10}, rep={50} (internal triton.testing.do_bench)")
print("=" * 90)

for i in range(NUM_RUNS):
    print(f"\n--- Run {i+1}/{NUM_RUNS} ---")
    proc = subprocess.run(
        CMD,
        capture_output=True,
        text=True,
        cwd="/workspace/kernel-POCs/kernels/lora_mlp",
    )
    output = proc.stdout + proc.stderr
    match = PATTERN.search(output)
    if match:
        vals = [float(match.group(j+1)) for j in range(len(KEYS))]
        for k, v in zip(KEYS, vals):
            results[k].append(v)
        print(f"  unsloth={vals[0]:.3f}  v3={vals[1]:.3f}  v6_sync={vals[4]:.3f}  v6_streams={vals[5]:.3f} ms")
    else:
        print(f"  ERROR: Could not parse output!")
        print(f"  stdout: {proc.stdout[-500:]}")
        print(f"  stderr: {proc.stderr[-500:]}")
        sys.exit(1)

print("\n" + "=" * 90)
print("\nFULL RESULTS TABLE (all 10 runs, ms):")
print(f"{'Run':<5} {'Unsloth':<10} {'v3':<10} {'v5_train':<10} {'v5_up1':<10} {'v6_sync':<10} {'v6_streams':<12} {'v5_inf':<10}")
print("-" * 87)
for i in range(NUM_RUNS):
    print(f"{i+1:<5} {results['unsloth'][i]:<10.3f} {results['v3'][i]:<10.3f} "
          f"{results['v5_train'][i]:<10.3f} {results['v5_up1_train'][i]:<10.3f} "
          f"{results['v6_sync'][i]:<10.3f} {results['v6_streams'][i]:<12.3f} "
          f"{results['v5_inf'][i]:<10.3f}")

print("\n" + "=" * 90)
print("\nSUMMARY STATISTICS (all 10 runs):")
print(f"{'Impl':<14} {'Mean':<10} {'Std':<10} {'Min':<10} {'Max':<10} {'Median':<10}")
print("-" * 64)
for k in KEYS:
    arr = np.array(results[k])
    print(f"{k:<14} {arr.mean():<10.3f} {arr.std():<10.3f} {arr.min():<10.3f} {arr.max():<10.3f} {np.median(arr):<10.3f}")

print("\n" + "=" * 90)
print("\nPOST-WARMUP STATISTICS (runs 3-10, discarding runs 1-2):")
print(f"{'Impl':<14} {'Mean':<10} {'Std':<10} {'Min':<10} {'Max':<10} {'Median':<10}")
print("-" * 64)
post_warmup = {}
for k in KEYS:
    arr = np.array(results[k][2:])  # discard runs 1-2
    post_warmup[k] = arr.mean()
    print(f"{k:<14} {arr.mean():<10.3f} {arr.std():<10.3f} {arr.min():<10.3f} {arr.max():<10.3f} {np.median(arr):<10.3f}")

print("\n" + "=" * 90)
print("\nSPEEDUPS (post-warmup means):")
unsloth_mean = post_warmup["unsloth"]
for k in KEYS:
    if k != "unsloth":
        speedup = unsloth_mean / post_warmup[k]
        print(f"  {k:<14} vs Unsloth: {speedup:.3f}x  ({post_warmup[k]:.3f} ms vs {unsloth_mean:.3f} ms)")

v6s = post_warmup["v6_streams"]
v6sync = post_warmup["v6_sync"]
print(f"\n  v6_streams vs v6_sync: {v6sync/v6s:.3f}x")
print(f"  v6_streams vs Unsloth: {unsloth_mean/v6s:.3f}x")
