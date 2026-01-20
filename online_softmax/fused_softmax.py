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

def softmax_mult(x, V, dim=-1):
    softmax_output = softmax(x, dim=dim)
    return softmax_output @ V

@triton.jit
def fused_softmax_triton_kernel(
    x_ptr,
    V_ptr,
    output_ptr,
    stride_xbatch,
    stride_xrow,
    stride_xcol,
    stride_Vbatch,
    stride_Vrow,
    stride_Vcol,
    stride_outbatch,
    stride_outrow,
    stride_outcol,
    d1: tl.constexpr,
    d2: tl.constexpr,
    d3: tl.constexpr,
    BLOCK_1: tl.constexpr,
    BLOCK_2: tl.constexpr,
):
    tl.static_assert(d2 % BLOCK_2 == 0, "d2 must be divisible by BLOCK_2")
    tl.static_assert(d1 % BLOCK_1 == 0, "d1 must be divisible by BLOCK_1")

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
    # V_block iterates over rows of V (d2 dimension), columns fixed to d3
    V_block = tl.make_block_ptr(
        V_ptr + pid_batch * stride_Vbatch,
        shape=(d2, d3),
        strides=(stride_Vrow, stride_Vcol),
        offsets=(0, 0),
        block_shape=(BLOCK_2, d3),
        order=(1, 0),
    )

    output_block = tl.make_block_ptr(
        output_ptr + pid_batch * stride_outbatch,
        shape=(d1, d3),
        strides=(stride_outrow, stride_outcol),
        offsets=(pid_row * BLOCK_1, 0),
        block_shape=(BLOCK_1, d3),
        order=(1, 0),
    )


    # Number of blocks in the column dimension (d2)
    Num_blocks = tl.cdiv(d2, BLOCK_2)

    # Initialize m_prev and l_prev for this block of rows
    m_prev = tl.full((BLOCK_1,), float("-inf"), dtype=tl.float32)
    l_prev = tl.zeros((BLOCK_1,), dtype=tl.float32)
    out_prev = tl.zeros((BLOCK_1, d3), dtype=tl.float32)

    for _ in range(Num_blocks):
        x = tl.load(x_block, boundary_check=(0, 1), padding_option="zero")
        v = tl.load(V_block, boundary_check=(0, 1), padding_option="zero")

        # Compute block max
        block_max = tl.max(x, axis=1)
        m_curr = tl.maximum(m_prev, block_max)

        # Update running sum with rescaling
        exp_x_block = tl.exp(x - m_curr[:, None])
        l_curr = l_prev * tl.exp(m_prev - m_curr) + tl.sum(exp_x_block, axis=1)

        # Rescale previous output and add contribution from current block
        
        scale = l_prev/l_curr * tl.exp(m_prev - m_curr)
        #tl.static_print("scale:", scale.shape)
        #tl.static_print("out_prev:", out_prev.shape)
        #tl.static_print("exp_x_block:", exp_x_block.shape)
        #tl.static_print("v:", v.shape)
        #tl.static_print("tl.dot(exp_x_block, v):", tl.dot(exp_x_block, v).shape)
        
        out_prev = out_prev * scale[:, None] + tl.dot(exp_x_block / l_curr[:, None], v)

        m_prev = m_curr
        l_prev = l_curr

        x_block = x_block.advance((0, BLOCK_2))
        V_block = V_block.advance((BLOCK_2, 0))

    # Final normalization by l_prev
    output = out_prev 
    tl.store(output_block, output, boundary_check=(0, 1))


    
def fused_softmax_triton(x, V, BLOCK_1=16, BLOCK_2=16):
    
    batch_size, d1, d2 = x.shape
    bs, d2 ,d3 = V.shape
    assert batch_size == bs, "Batch size of x and V must match"
    assert d2 == d2, "d2 of x and V must match"
    fused_softmax_output = torch.empty((batch_size, d1, d3), device=x.device, dtype=x.dtype)

    # Calculate grid dimensions
    grid = (batch_size, triton.cdiv(d1, BLOCK_1))

    # Launch kernel
    fused_softmax_triton_kernel[grid](
        x,
        V,
        fused_softmax_output,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        V.stride(0),
        V.stride(1),
        V.stride(2),
        fused_softmax_output.stride(0),
        fused_softmax_output.stride(1),
        fused_softmax_output.stride(2),
        d1,
        d2,
        d3,
        BLOCK_1=BLOCK_1,
        BLOCK_2=BLOCK_2,
    )

    return fused_softmax_output

if __name__ == "__main__":
    # Simple test
    x = torch.randn(1, 16, 64).cuda()
    V = torch.randn(1, 64, 16).cuda()

    B_1 = 16
    B_2 = 16
    print("Input:")
    print(x)
    print("Softmax mult:")
    print(softmax_mult(x, V))
    print("Fused Softmax mult:")
    print(fused_softmax_triton(x, V, BLOCK_1=B_1, BLOCK_2=B_2))
    print("Difference:")
    print(torch.abs(softmax_mult(x, V) - fused_softmax_triton(x, V, BLOCK_1=B_1, BLOCK_2=B_2)).max().item())