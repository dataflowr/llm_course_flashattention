import os
import sys

# Add parent directory to path so we can import from sibling packages
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import torch
from online_softmax.online_softmax import (
    softmax,
    online_softmax,
    online_softmax_triton,
    OnlineSoftmaxTriton,
)


# Test configurations
BATCH_SIZES = [1, 2, 8]
D1_SIZES = [16, 32, 64]
D2_SIZES = [32, 64, 128, 256]
BLOCK_SIZES = [8, 16, 32]
DTYPES = [torch.float32]
ATOL = 1e-5
RTOL = 1e-5


@pytest.fixture
def device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return "cuda"


class TestOnlineSoftmaxTritonForward:
    """Test that Triton online softmax produces correct results compared to reference."""

    @pytest.mark.parametrize("batch_size", BATCH_SIZES)
    @pytest.mark.parametrize("d1", D1_SIZES)
    @pytest.mark.parametrize("d2", D2_SIZES)
    @pytest.mark.parametrize("block_size", BLOCK_SIZES)
    @pytest.mark.parametrize("dtype", DTYPES)
    def test_forward_correctness(self, device, batch_size, d1, d2, block_size, dtype):
        """Test that Triton softmax matches reference softmax."""
        # Skip if dimensions not divisible by block size
        if d1 % block_size != 0 or d2 % block_size != 0:
            pytest.skip(f"d1={d1} or d2={d2} not divisible by block_size={block_size}")

        x = torch.randn(batch_size, d1, d2, device=device, dtype=dtype)

        # Reference implementation
        y_ref = softmax(x)

        # Triton implementation
        y_triton = online_softmax_triton(x, BLOCK_1=block_size, BLOCK_2=block_size)

        torch.testing.assert_close(y_triton, y_ref, atol=ATOL, rtol=RTOL)

    @pytest.mark.parametrize("batch_size", BATCH_SIZES)
    @pytest.mark.parametrize("d1", D1_SIZES)
    @pytest.mark.parametrize("d2", D2_SIZES)
    def test_forward_vs_pytorch(self, device, batch_size, d1, d2):
        """Test that Triton softmax matches PyTorch softmax."""
        block_size = 16
        if d1 % block_size != 0 or d2 % block_size != 0:
            pytest.skip(f"d1={d1} or d2={d2} not divisible by block_size={block_size}")

        x = torch.randn(batch_size, d1, d2, device=device, dtype=torch.float32)

        # PyTorch reference
        y_pytorch = torch.softmax(x, dim=-1)

        # Triton implementation
        y_triton = online_softmax_triton(x, BLOCK_1=block_size, BLOCK_2=block_size)

        torch.testing.assert_close(y_triton, y_pytorch, atol=ATOL, rtol=RTOL)

    def test_output_sums_to_one(self, device):
        """Test that softmax output sums to 1 along the last dimension."""
        x = torch.randn(4, 32, 64, device=device)
        y = online_softmax_triton(x, BLOCK_1=16, BLOCK_2=16)

        sums = y.sum(dim=-1)
        expected = torch.ones_like(sums)

        torch.testing.assert_close(sums, expected, atol=1e-5, rtol=1e-5)

    def test_output_non_negative(self, device):
        """Test that softmax output is non-negative."""
        x = torch.randn(4, 32, 64, device=device)
        y = online_softmax_triton(x, BLOCK_1=16, BLOCK_2=16)

        assert (y >= 0).all(), "Softmax output should be non-negative"

    def test_numerical_stability_large_values(self, device):
        """Test numerical stability with large input values."""
        x = torch.randn(2, 32, 64, device=device) * 100  # Large values

        y_ref = softmax(x)
        y_triton = online_softmax_triton(x, BLOCK_1=16, BLOCK_2=16)

        torch.testing.assert_close(y_triton, y_ref, atol=1e-4, rtol=1e-4)

    def test_numerical_stability_small_values(self, device):
        """Test numerical stability with small input values."""
        x = torch.randn(2, 32, 64, device=device) * 0.001  # Small values

        y_ref = softmax(x)
        y_triton = online_softmax_triton(x, BLOCK_1=16, BLOCK_2=16)

        torch.testing.assert_close(y_triton, y_ref, atol=ATOL, rtol=RTOL)


class TestOnlineSoftmaxTritonBackward:
    """Test backward pass of Triton online softmax."""

    @pytest.mark.parametrize("batch_size", BATCH_SIZES)
    @pytest.mark.parametrize("d1", D1_SIZES)
    @pytest.mark.parametrize("d2", D2_SIZES)
    @pytest.mark.parametrize("block_size", BLOCK_SIZES)
    def test_backward_correctness(self, device, batch_size, d1, d2, block_size):
        """Test that backward pass produces correct gradients."""
        if d1 % block_size != 0 or d2 % block_size != 0:
            pytest.skip(f"d1={d1} or d2={d2} not divisible by block_size={block_size}")

        # Create inputs with gradients
        x_ref = torch.randn(batch_size, d1, d2, device=device, requires_grad=True)
        x_triton = x_ref.detach().clone().requires_grad_(True)

        # Forward pass
        y_ref = torch.softmax(x_ref, dim=-1)
        y_triton = OnlineSoftmaxTriton.apply(x_triton, block_size, block_size)

        # Upstream gradient
        grad_output = torch.randn_like(y_ref)

        # Backward pass
        y_ref.backward(grad_output)
        y_triton.backward(grad_output)

        torch.testing.assert_close(x_triton.grad, x_ref.grad, atol=1e-4, rtol=1e-4)

    def test_backward_gradient_shape(self, device):
        """Test that backward pass produces gradients with correct shape."""
        x = torch.randn(2, 32, 64, device=device, requires_grad=True)
        y = OnlineSoftmaxTriton.apply(x, 16, 16)

        grad_output = torch.randn_like(y)
        y.backward(grad_output)

        assert x.grad.shape == x.shape, "Gradient shape should match input shape"


class TestOnlineSoftmaxPython:
    """Test the pure Python online softmax implementation."""

    @pytest.mark.parametrize("batch_size", BATCH_SIZES)
    @pytest.mark.parametrize("d1", D1_SIZES)
    @pytest.mark.parametrize("d2", D2_SIZES)
    def test_online_softmax_correctness(self, device, batch_size, d1, d2):
        """Test that Python online softmax matches reference."""
        x = torch.randn(batch_size, d1, d2, device=device)

        y_ref = softmax(x)
        y_online = online_softmax(x)

        torch.testing.assert_close(y_online, y_ref, atol=ATOL, rtol=RTOL)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
