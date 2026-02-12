import torch
import triton
import triton.language as tl

DEVICE = triton.runtime.driver.active.get_active_torch_device()

@triton.jit
def softmax_kernel(
    output_ptr, input_ptr,
    input_row_stride, output_row_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr
):
    row_idx = tl.program_id(0)
    row_start_ptr = input_ptr + row_idx * input_row_stride
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start_ptr + col_offsets

    mask = col_offsets < n_cols
    row = tl.load(input_ptrs, mask=mask, other=-float('inf'))

    row_max = tl.max(row, axis=0)
    numerator = tl.exp(row - row_max)
    denominator = tl.sum(numerator, axis=0)
    softmax_output = numerator / denominator
    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    output_ptrs = output_row_start_ptr + col_offsets
    tl.store(output_ptrs, softmax_output, mask=mask)

def softmax(x):
    assert x.device == DEVICE, "Input must be on the active Triton device"
    assert x.ndim == 2, "Input must be a 2D tensor"
    assert x.is_contiguous(), "Input must be contiguous"
    n_rows, n_cols = x.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    y = torch.empty_like(x)
    grid = (n_rows, )
    softmax_kernel[grid](
        y, x,
        x.stride(0), y.stride(0),
        n_cols,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return y

x = torch.randn(182, 409, device=DEVICE)
output_triton = softmax(x)
output_torch = torch.softmax(x, dim=-1)
print(output_torch)
print(output_triton)
print(f'The maximum difference between torch and triton is '
      f'{torch.max(torch.abs(output_torch - output_triton))}')
print(torch.allclose(output_triton, output_torch, rtol=1e-2, atol=1e-5))
