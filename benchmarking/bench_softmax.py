import os
import sys

# Add parent directory to path so we can import from sibling packages
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from online_softmax.online_softmax import softmax, online_softmax_triton
from triton.compiler.errors import CompileTimeAssertionFailure

import torch
import math
import pandas as pd
import argparse

device = "cuda"
dtype = torch.float32
batch_size = 8
nb_warmup = 10
nb_forward_passes = 100

d1 = 2048
d2 = [64, 128, 256, 512, 1024, 2048, 4096, 8192]
B = [8, 16, 32, 64, 128]



def time_loop(fn, iters):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    torch.cuda.synchronize()
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()

    return start.elapsed_time(end) / iters  # ms/iter


def _create_inputs(batch_size, d1, d2, device, dtype):
    x = torch.randn(batch_size, d1, d2, device=device, dtype=dtype)
    return x

def _get_softmax(triton, BLOCK=16):
    if triton:
        return lambda x: online_softmax_triton(x, BLOCK_1=BLOCK, BLOCK_2=BLOCK)
    else:
        return softmax

def _warmup(x, fn, nb_warmup):
    for _ in range(nb_warmup):
        y = fn(x)
    return y

def _benchmark_forward(x, fn, n_iters):
    torch.cuda.reset_peak_memory_stats()
    fwd_ms = time_loop(lambda: fn(x), n_iters)
    fwd_peak = torch.cuda.max_memory_allocated()

    return {
        "forward_ms": fwd_ms,
        "forward_peak_MiB": fwd_peak / 1024**2,
    }

def run_config(batch_size, d1, d2, BLOCK, device, dtype, snapshot_name, triton=False):
    config_str = f"bs={batch_size}_d1={d1}_d2={d2}_triton={triton}_BLOCK={BLOCK}"

    print(f"Running config: {config_str}")
    x = _create_inputs(batch_size, d1, d2, device, dtype)

    softmax_fn = _get_softmax(triton, BLOCK=BLOCK)

    # Warmup
    _warmup(x, softmax_fn, nb_warmup)

    results = _benchmark_forward(x, softmax_fn, nb_forward_passes)


    return { "batch_size": batch_size, "d1": d1, "d2": d2, "triton": triton, "BLOCK": BLOCK, **results }

def run_benchmark():
    # Build all configs: (d2_val, triton, BLOCK)
    configs = []
    for d2_val in d2:
        configs.append((d2_val, False, None))  # Standard softmax
        for BLOCK in B:
            configs.append((d2_val, True, BLOCK))  # Triton variants


    results = []
    for i, (d2_val, triton, BLOCK) in enumerate(configs, 1):
        print(f"\n{'='*60}")
        print(f"Config {i}/{len(configs)}")
        print(f"{'='*60}")

        # Generate snapshot filename
        impl = f"triton_BLOCK{BLOCK}" if triton else "standard"
        snapshot_name = f"outputs/snapshots/softmax_d2_{d2_val}_{impl}.pickle"

        try:
            res = run_config(batch_size, d1, d2_val, BLOCK, device, dtype, snapshot_name, triton=triton)
            results.append(res)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"  OOM in config: d2={d2_val}, triton={triton}, BLOCK={BLOCK}")
                torch.cuda.empty_cache()
            else:
                raise
            results.append({
                "batch_size": batch_size, "d1": d1, "d2": d2_val,
                "triton": triton, "BLOCK": BLOCK,
                "forward_ms": None, "forward_peak_MiB": None,
            })
        except CompileTimeAssertionFailure:
            print(f"  Skipping: BLOCK={BLOCK} incompatible with d1={d1} or d2={d2_val}")
            results.append({
                "batch_size": batch_size, "d1": d1, "d2": d2_val,
                "triton": triton, "BLOCK": BLOCK,
                "forward_ms": None, "forward_peak_MiB": None,
            })

    print(f"\n{'='*60}")
    print("BENCHMARK RESULTS")
    print(f"{'='*60}\n")
    df = pd.DataFrame(results)
    df["BLOCK"] = df["BLOCK"].astype("Int64")  # Nullable integer type
    print(df.to_string())
    # Save results
    out_csv = "outputs/softmax_benchmark.csv"
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"\nResults saved to {out_csv}")

if __name__ == "__main__":
    run_benchmark()