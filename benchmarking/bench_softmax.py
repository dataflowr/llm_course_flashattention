import os
import sys

# Add parent directory to path so we can import from sibling packages
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from online_softmax.online_softmax import softmax, online_softmax_triton

import torch
import math
import pandas as pd

from torch import Tensor
from jaxtyping import Float, Bool, Int

device = "cuda"
dtype = torch.float32
batch_size = 8
nb_warmup = 10
nb_forward_passes = 100

d1 = [2048] # [16, 32, 64 , 128, 256, 512, 1024, 2048]
d2 = [64, 128, 256, 512, 1024, 2048, 4096, 8192]
B = [8, 16, 32, 64]


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

def run_config(batch_size, d1, d2, BLOCK, device, dtype, triton=False):
    config_str = f"bs={batch_size}_d1={d1}_d2={d2}_triton={triton}_BLOCK={BLOCK}"

    print(f"Running config: {config_str}")
    x = _create_inputs(batch_size, d1, d2, device, dtype)

    softmax_fn = _get_softmax(triton, BLOCK=BLOCK)

    # Warmup
    _warmup(x, softmax_fn, nb_warmup)
    results = _benchmark_forward(x, softmax_fn, nb_forward_passes)

    return { "batch_size": batch_size, "d1": d1, "d2": d2, "triton": triton, "BLOCK": BLOCK, **results }

if __name__ == "__main__":
    results = []
    total_configs = len(d1) * len(d2) * (1+len(B))  # triton and non-triton cases
    current_config = 0
    for i in range(len(d1)):
        for j in range(len(d2)):
            for triton in [False, True]:
                if triton:
                    for BLOCK in B:
                        current_config += 1
                        print(f"\n{'='*60}")
                        print(f"Config {current_config}/{total_configs}")
                        print(f"{'='*60}")

                        try:
                            res = run_config(
                                batch_size,
                                d1[i],
                                d2[j],
                                BLOCK,
                                device,
                                dtype,
                                triton=triton,
                        )
                            results.append(res)
                        except RuntimeError as e:
                            if "out of memory" in str(e).lower():
                                print(
                            f"  ⚠️  OOM in config: batch_size={batch_size}, d1={d1[i]}, d2={d2[j]}, triton={triton}, BLOCK={BLOCK}"
                        )
                                torch.cuda.empty_cache()
                                results.append(
                            {
                                "batch_size": batch_size,
                                "d1": d1[i],
                                "d2": d2[j],
                                "triton": triton,
                                "BLOCK": BLOCK, 
                                "forward_ms": None,
                                "forward_peak_MiB": None,
                            }
                        )
                            else:
                                raise
                else:
                    current_config += 1
                    print(f"\n{'='*60}")
                    print(f"Config {current_config}/{total_configs}")
                    print(f"{'='*60}")

                    try:
                        res = run_config(
                            batch_size, d1[i], d2[j], None, device, dtype, triton)
                        results.append(res)
                    except RuntimeError as e:
                        if "out of memory" in str(e).lower():
                            print(
                        f"  ⚠️  OOM in config: batch_size={batch_size}, d1={d1[i]}, d2={d2[j]}, triton={triton}"
                    )
                            torch.cuda.empty_cache()
                            results.append(
                        {
                            "batch_size": batch_size,
                            "d1": d1[i],
                            "d2": d2[j],
                            "triton": triton,
                            "BLOCK": None, 
                            "forward_ms": None,
                            "forward_peak_MiB": None,
                        }
                    )
                        else:
                            raise
    
    # Print results
    print(f"\n{'='*60}")
    print("BENCHMARK RESULTS")
    print(f"{'='*60}\n")

    df = pd.DataFrame(results)
    print(df.to_string())