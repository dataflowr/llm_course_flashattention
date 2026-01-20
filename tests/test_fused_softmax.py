import os
import sys

# Add parent directory to path so we can import from sibling packages
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import torch
from online_softmax.fused_softmax import (
    softmax,
    softmax_mult,
    fused_softmax_triton,
)


# Test configurations
# Note: Block sizes must be >= 16 for tl.dot, and d3 must also be >= 16
BATCH_SIZES = [1, 2, 8]
D1_SIZES = [16, 32, 64]
D2_SIZES = [32, 64, 128, 256]
D3_SIZES = [16, 32, 64]
BLOCK_SIZES = [16, 32]  # Must be >= 16 for tl.dot
DTYPES = [torch.float32]
ATOL = 1e-3
RTOL = 1e-3


@pytest.fixture
def device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return "cuda"


class TestFusedSoftmaxTritonForward:
    """Test that Triton fused softmax produces correct results compared to reference."""

    @pytest.mark.parametrize("batch_size", BATCH_SIZES)
    @pytest.mark.parametrize("d1", D1_SIZES)
    @pytest.mark.parametrize("d2", D2_SIZES)
    @pytest.mark.parametrize("d3", D3_SIZES)
    @pytest.mark.parametrize("block_size", BLOCK_SIZES)
    @pytest.mark.parametrize("dtype", DTYPES)
    def test_forward_correctness(self, device, batch_size, d1, d2, d3, block_size, dtype):
        """Test that Triton fused softmax matches reference softmax_mult."""
        # Skip if dimensions not divisible by block size
        if d1 % block_size != 0 or d2 % block_size != 0:
            pytest.skip(f"d1={d1} or d2={d2} not divisible by block_size={block_size}")

        # d3 must be >= 16 for tl.dot
        if d3 < 16:
            pytest.skip(f"d3={d3} must be >= 16 for tl.dot")

        x = torch.randn(batch_size, d1, d2, device=device, dtype=dtype)
        V = torch.randn(batch_size, d2, d3, device=device, dtype=dtype)

        # Reference implementation
        y_ref = softmax_mult(x, V)

        # Triton implementation
        y_triton = fused_softmax_triton(x, V, BLOCK_1=block_size, BLOCK_2=block_size)

        torch.testing.assert_close(y_triton, y_ref, atol=ATOL, rtol=RTOL)

    @pytest.mark.parametrize("batch_size", BATCH_SIZES)
    @pytest.mark.parametrize("d1", D1_SIZES)
    @pytest.mark.parametrize("d2", D2_SIZES)
    @pytest.mark.parametrize("d3", D3_SIZES)
    def test_forward_vs_pytorch(self, device, batch_size, d1, d2, d3):
        """Test that Triton fused softmax matches PyTorch softmax @ V."""
        block_size = 16
        if d1 % block_size != 0 or d2 % block_size != 0:
            pytest.skip(f"d1={d1} or d2={d2} not divisible by block_size={block_size}")

        if d3 < 16:
            pytest.skip(f"d3={d3} must be >= 16 for tl.dot")

        x = torch.randn(batch_size, d1, d2, device=device, dtype=torch.float32)
        V = torch.randn(batch_size, d2, d3, device=device, dtype=torch.float32)

        # PyTorch reference
        y_pytorch = torch.softmax(x, dim=-1) @ V

        # Triton implementation
        y_triton = fused_softmax_triton(x, V, BLOCK_1=block_size, BLOCK_2=block_size)

        torch.testing.assert_close(y_triton, y_pytorch, atol=ATOL, rtol=RTOL)

    def test_output_shape(self, device):
        """Test that output has correct shape (batch_size, d1, d3)."""
        batch_size, d1, d2, d3 = 4, 32, 64, 32
        x = torch.randn(batch_size, d1, d2, device=device)
        V = torch.randn(batch_size, d2, d3, device=device)

        y = fused_softmax_triton(x, V, BLOCK_1=16, BLOCK_2=16)

        assert y.shape == (batch_size, d1, d3), f"Expected shape {(batch_size, d1, d3)}, got {y.shape}"

    def test_numerical_stability_large_values(self, device):
        """Test numerical stability with large input values."""
        x = torch.randn(2, 32, 64, device=device) * 100  # Large values
        V = torch.randn(2, 64, 32, device=device)

        y_ref = softmax_mult(x, V)
        y_triton = fused_softmax_triton(x, V, BLOCK_1=16, BLOCK_2=16)

        torch.testing.assert_close(y_triton, y_ref, atol=1e-3, rtol=1e-3)

    def test_numerical_stability_small_values(self, device):
        """Test numerical stability with small input values."""
        x = torch.randn(2, 32, 64, device=device) * 0.001  # Small values
        V = torch.randn(2, 64, 32, device=device)

        y_ref = softmax_mult(x, V)
        y_triton = fused_softmax_triton(x, V, BLOCK_1=16, BLOCK_2=16)

        torch.testing.assert_close(y_triton, y_ref, atol=ATOL, rtol=RTOL)

    def test_different_block_sizes(self, device):
        """Test with different BLOCK_1 and BLOCK_2 values."""
        x = torch.randn(2, 64, 128, device=device)
        V = torch.randn(2, 128, 32, device=device)

        y_ref = softmax_mult(x, V)

        # Test with BLOCK_1=16, BLOCK_2=32
        y_triton = fused_softmax_triton(x, V, BLOCK_1=16, BLOCK_2=32)
        torch.testing.assert_close(y_triton, y_ref, atol=ATOL, rtol=RTOL)

        # Test with BLOCK_1=32, BLOCK_2=16
        y_triton = fused_softmax_triton(x, V, BLOCK_1=32, BLOCK_2=16)
        torch.testing.assert_close(y_triton, y_ref, atol=ATOL, rtol=RTOL)


class TestFusedSoftmaxEdgeCases:
    """Test edge cases for fused softmax."""

    def test_single_batch(self, device):
        """Test with batch size of 1."""
        x = torch.randn(1, 32, 64, device=device)
        V = torch.randn(1, 64, 32, device=device)

        y_ref = softmax_mult(x, V)
        y_triton = fused_softmax_triton(x, V, BLOCK_1=16, BLOCK_2=16)

        torch.testing.assert_close(y_triton, y_ref, atol=ATOL, rtol=RTOL)

    def test_square_dimensions(self, device):
        """Test with square x matrix."""
        x = torch.randn(2, 64, 64, device=device)
        V = torch.randn(2, 64, 32, device=device)

        y_ref = softmax_mult(x, V)
        y_triton = fused_softmax_triton(x, V, BLOCK_1=16, BLOCK_2=16)

        torch.testing.assert_close(y_triton, y_ref, atol=ATOL, rtol=RTOL)

    def test_minimum_dimensions(self, device):
        """Test with minimum valid dimensions (16x16 blocks)."""
        x = torch.randn(1, 16, 16, device=device)
        V = torch.randn(1, 16, 16, device=device)

        y_ref = softmax_mult(x, V)
        y_triton = fused_softmax_triton(x, V, BLOCK_1=16, BLOCK_2=16)

        torch.testing.assert_close(y_triton, y_ref, atol=ATOL, rtol=RTOL)


class TestFusedSoftmaxReference:
    """Test the reference softmax_mult implementation."""

    @pytest.mark.parametrize("batch_size", BATCH_SIZES)
    @pytest.mark.parametrize("d1", D1_SIZES)
    @pytest.mark.parametrize("d2", D2_SIZES)
    @pytest.mark.parametrize("d3", D3_SIZES)
    def test_softmax_mult_vs_pytorch(self, device, batch_size, d1, d2, d3):
        """Test that softmax_mult matches PyTorch implementation."""
        x = torch.randn(batch_size, d1, d2, device=device)
        V = torch.randn(batch_size, d2, d3, device=device)

        y_ref = softmax_mult(x, V)
        y_pytorch = torch.softmax(x, dim=-1) @ V

        torch.testing.assert_close(y_ref, y_pytorch, atol=1e-5, rtol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
