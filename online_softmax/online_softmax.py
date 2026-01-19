import torch
import math
import triton
import triton.language as tl
import random


def softmax(x, dim=-1):
    rescaled_input = x - torch.max(x, dim=dim, keepdim=True)[0]
    exponentiated_rescaled_input = torch.exp(rescaled_input)
    return exponentiated_rescaled_input / torch.sum(
        exponentiated_rescaled_input, dim=dim, keepdim=True
    )


def online_softmax(x, B=16):
    *bs, d = x.shape
    device = x.device
    Td = math.ceil(d / B)
    m_prev = torch.full((*bs, 1), float("-inf"), device=device)  # current max
    l_prev = torch.zeros((*bs, 1), device=device)  # current sum of exps

    for i in range(Td):
        start = i * B
        end = min((i + 1) * B, d)
        x_block = x[..., start:end]

        block_max = x_block.max(dim=-1, keepdim=True).values
        m_curr = torch.maximum(m_prev, block_max)

        l_prev = l_prev * torch.exp(m_prev - m_curr) + torch.exp(x_block - m_curr).sum(
            dim=-1, keepdim=True
        )
        m_prev = m_curr

    softmax_output = torch.empty_like(x)
    for i in range(Td):
        start = i * B
        end = min((i + 1) * B, d)
        softmax_output[..., start:end] = torch.exp(x[..., start:end] - m_prev)

    return softmax_output / l_prev


def online_softmax_kernel(x_ptr, block_1, block_2, d_2, fake_pid):
    device = x_ptr.device
    Num_blocks = math.ceil(d_2 / block_2)

    m_prev = torch.full((block_1, 1), float("-inf"), device=device)  # current max
    l_prev = torch.zeros((block_1, 1), device=device)  # current sum of exps

    for i in range(Num_blocks):
        start = i * block_2
        end = min((i + 1) * block_2, d_2)
        x_block = x_ptr[
            fake_pid[0], fake_pid[1] * block_1 : (fake_pid[1] + 1) * block_1, start:end
        ]

        block_max = x_block.max(dim=-1, keepdim=True).values
        m_curr = torch.maximum(m_prev, block_max)

        l_prev = l_prev * torch.exp(m_prev - m_curr) + torch.exp(x_block - m_curr).sum(
            dim=-1, keepdim=True
        )
        m_prev = m_curr

    softmax_output = torch.empty_like(
        x_ptr[fake_pid[0], fake_pid[1] * block_1 : (fake_pid[1] + 1) * block_1, :]
    )
    for i in range(Num_blocks):
        start = i * block_2
        end = min((i + 1) * block_2, d_2)
        x_block = x_ptr[
            fake_pid[0], fake_pid[1] * block_1 : (fake_pid[1] + 1) * block_1, start:end
        ]
        softmax_output[..., start:end] = torch.exp(x_block - m_prev)

    return softmax_output / l_prev


def online_softmax_fake_triton(x, B=16):
    assert x.shape[1] % B == 0, "d1 must be a multiple of B for fake triton kernel"
    bs, d1, d2 = x.shape
    Num_tiles = math.ceil(d1 / B)
    softmax_output = torch.empty_like(x)

    # Create list of all (batch, tile) pairs and shuffle to show parallelism
    all_blocks = [(i, j) for i in range(bs) for j in range(Num_tiles)]
    random.shuffle(all_blocks)

    for i, j in all_blocks:
        fake_pid = (i, j)
        softmax_output[i, j * B : min((j + 1) * B, d1), :] = online_softmax_kernel(
            x, B, B, d2, fake_pid
        )
    return softmax_output


@triton.jit
def online_softmax_triton_kernel(
    x_ptr,
    softmax_ptr,
    stride_xbatch,
    stride_xrow,
    stride_xcol,
    stride_sbatch,
    stride_srow,
    stride_scol,
    d1: tl.constexpr,
    d2: tl.constexpr,
    BLOCK_1: tl.constexpr,
    BLOCK_2: tl.constexpr,
):

    tl.static_assert(d2 % BLOCK_2 == 0, "d2 must be divisible by BLOCK_2")
    tl.static_assert(d1 % BLOCK_1 == 0, "d1 must be divisible by BLOCK_1")

    # Each program handles one block of rows (BLOCK_1 rows)
    pid_batch = tl.program_id(0)
    pid_row = tl.program_id(1)

    x_block = tl.make_block_ptr(
        x_ptr + pid_batch * stride_xbatch,
        shape=(d1, d2),
        strides=(stride_xrow, stride_xcol),
        offsets=(pid_row * BLOCK_1, 0),
        block_shape=(BLOCK_1, BLOCK_2),
        order=(1, 0),
    )

    # Number of blocks in the column dimension
    Num_blocks = tl.cdiv(d2, BLOCK_2)

    # Initialize m_prev and l_prev for this block of rows
    m_prev = tl.full((BLOCK_1,), float("-inf"), dtype=tl.float32)
    l_prev = tl.zeros((BLOCK_1,), dtype=tl.float32)

    # First pass: compute global max and sum
    for _ in range(Num_blocks):
        x = tl.load(x_block, boundary_check=(0, 1), padding_option="zero")

        # Compute block max
        block_max = tl.max(x, axis=1)
        m_curr = tl.maximum(m_prev, block_max)

        # Update running sum with rescaling
        exp_x_block = tl.exp(x - m_curr[:, None])
        l_prev = l_prev * tl.exp(m_prev - m_curr) + tl.sum(exp_x_block, axis=1)
        m_prev = m_curr

        x_block = x_block.advance((0, BLOCK_2))

    # Second pass: compute and store softmax output
    softmax_block = tl.make_block_ptr(
        softmax_ptr + pid_batch * stride_sbatch,
        shape=(d1, d2),
        strides=(stride_srow, stride_scol),
        offsets=(pid_row * BLOCK_1, 0),
        block_shape=(BLOCK_1, BLOCK_2),
        order=(1, 0),
    )
    x_block = tl.make_block_ptr(
        x_ptr + pid_batch * stride_xbatch,
        shape=(d1, d2),
        strides=(stride_xrow, stride_xcol),
        offsets=(pid_row * BLOCK_1, 0),
        block_shape=(BLOCK_1, BLOCK_2),
        order=(1, 0),
    )
    # tl.device_print("m_prev:", m_prev)
    # tl.device_print("l_prev:", l_prev)

    for _ in range(Num_blocks):
        x = tl.load(x_block, boundary_check=(0, 1), padding_option="zero")
        # Compute softmax for this block
        tl.store(
            softmax_block,
            tl.exp(x - m_prev[:, None]) / l_prev[:, None],
            boundary_check=(0, 1),
        )

        x_block = x_block.advance((0, BLOCK_2))
        softmax_block = softmax_block.advance((0, BLOCK_2))


def online_softmax_triton(x, BLOCK_1=16, BLOCK_2=16):
    """
    Compute softmax using Triton kernel with online algorithm.

    Args:
        x: Input tensor of shape (batch_size, d1, d2)
        BLOCK_1: Block size for dimension d1 (rows)
        BLOCK_2: Block size for dimension d2 (columns, softmax dimension)
    """
    batch_size, d1, d2 = x.shape
    softmax_output = torch.empty_like(x)

    # Calculate grid dimensions
    grid = (batch_size, triton.cdiv(d1, BLOCK_1))

    # Launch kernel
    online_softmax_triton_kernel[grid](
        x,
        softmax_output,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        softmax_output.stride(0),
        softmax_output.stride(1),
        softmax_output.stride(2),
        d1,
        d2,
        BLOCK_1=BLOCK_1,
        BLOCK_2=BLOCK_2,
    )

    return softmax_output


def softmax_backward(grad_output, output):
    sum_grad_output = torch.sum(grad_output * output, dim=-1, keepdim=True)
    grad_input = output * (grad_output - sum_grad_output)
    return grad_input


class OnlineSoftmaxTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, BLOCK_1=16, BLOCK_2=16):
        y= online_softmax_triton(x, BLOCK_1=BLOCK_1, BLOCK_2=BLOCK_2)
        ctx.save_for_backward(y)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        y, = ctx.saved_tensors
        return softmax_backward(grad_output, y), None, None


if __name__ == "__main__":
    # Simple test
    x = torch.randn(1, 16, 32).cuda()
    B = 8
    print("Input:")
    print(x)
    print("Softmax:")
    print(softmax(x))
    print("Online Softmax:")
    print(online_softmax(x))
    print("Fake Triton Online Softmax:")
    print(online_softmax_fake_triton(x, B=B))
    print(
        "Difference 1:",
        torch.abs(softmax(x) - online_softmax(x)).max().item(),
    )
    print(
        "Difference 2:",
        torch.abs(softmax(x) - online_softmax_fake_triton(x, B=B)).max().item(),
    )
    print("Triton Online Softmax:")
    print(online_softmax_triton(x, BLOCK_1=B, BLOCK_2=B))
    print(
        "Difference 3:",
        torch.abs(softmax(x) - online_softmax_triton(x, BLOCK_1=B, BLOCK_2=B))
        .max()
        .item(),
    )

    # Test backward pass
    print("\n" + "=" * 50)
    print("BACKWARD PASS TEST")
    print("=" * 50)

    # Create input with gradients
    x_ref = torch.randn(2, 16, 32, device="cuda", requires_grad=True)
    x_triton = x_ref.detach().clone().requires_grad_(True)

    # Forward pass
    y_ref = torch.softmax(x_ref, dim=-1)
    y_triton = OnlineSoftmaxTriton.apply(x_triton, B, B)

    # Create upstream gradient
    grad_output = torch.randn_like(y_ref)

    # Backward pass
    y_ref.backward(grad_output)
    y_triton.backward(grad_output)

    # Compare gradients
    grad_diff = torch.abs(x_ref.grad - x_triton.grad).max().item()
    print(f"Max gradient difference: {grad_diff}")
    print(f"Backward pass test: {'PASSED' if grad_diff < 1e-5 else 'FAILED'}")
