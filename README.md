# llm_course_flashattention
Assignment for the LLM course on FlashAttention

## Installation

Create and activate the conda environment:

```bash
conda env create -f environment.yml
conda activate flashattention
```

## Project Structure

```
.
├── online_softmax/
│   ├── online_softmax.py    # Online softmax implementation (Triton + PyTorch)
│   └── fused_softmax.py     # Fused softmax with matrix multiplication
├── benchmarking/
│   ├── bench_softmax.py     # Benchmark for online softmax
│   └── bench_fused.py       # Benchmark for fused softmax
├── tests/
│   ├── test_online_softmax.py
│   └── test_fused_softmax.py
└── environment.yml
```

## Running Tests

```bash
pytest tests/ -v
```

## Running Benchmarks

```bash
python benchmarking/bench_softmax.py
python benchmarking/bench_fused.py
```

Results are saved to `outputs/` directory.

## Notes

- The fused softmax kernel uses `tl.dot()` which requires all dimensions to be >= 16 (tensor core constraint)
- Block sizes must be >= 16 for the fused softmax implementation
- Numerical tolerance for fused softmax tests is 1e-3 (vs 1e-5 for simple softmax) due to error accumulation from online algorithm + TF32 tensor core operations
