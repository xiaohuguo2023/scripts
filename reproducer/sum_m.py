import pytest
import torch
import triton
import triton.language as tl
import struct

def fp32_to_binary(fp32_value):
    """
    Converts a floating-point (FP32) number to its equivalent 32-bit binary string.

    Parameters:
        fp32_value (float): A floating-point number.

    Returns:
        str: The 32-bit binary string representation of the floating-point number.
    """
    # Pack float into bytes and unpack as integer
    int_value = struct.unpack('!I', struct.pack('!f', fp32_value))[0]

    # Convert integer to binary string and pad to 32 bits
    binary_str = f"{int_value:032b}"
    return binary_str

def binary_to_fp32(binary_str):
    """
    Converts a 32-bit binary string to its equivalent floating-point (FP32) number.

    Parameters:
        binary_str (str): A 32-bit binary string.

    Returns:
        float: The floating-point equivalent of the binary string.
    """
    # Validate input
    if len(binary_str) != 32 or not all(bit in '01' for bit in binary_str):
        raise ValueError("Input must be a 32-bit binary string.")

    # Convert binary string to integer
    int_value = int(binary_str, 2)

    # Pack integer into bytes and unpack as float
    fp32_value = struct.unpack('!f', struct.pack('!I', int_value))[0]
    return fp32_value


@triton.jit
def add_constant_kernel(M, out_ptr):
    acc:tl.float32 = 0.
#    acc = M* 3.1415
    for i in tl.range(0, M):
        acc += 3.1

    # store final
    tl.store(out_ptr, acc.to(out_ptr.type.element_ty))


@pytest.mark.parametrize("out_dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("M", [10, 200, 4096, 8192])
def test_accum_constant(M, out_dtype):

    device = 'cuda'
    out = torch.zeros((1,), dtype=out_dtype, device=device)

    grid = (1,)
    add_constant_kernel[grid](
        M,
        out
    )

    triton_val = out.item()

    exact_val = torch.tensor(3.1415*M, dtype=out_dtype).item()
    # cast to float32 for a fair direct comparison
    ref_val = exact_val

    if out_dtype in (torch.float16, torch.bfloat16):
        atol, rtol = 1e-3, 1e-2
    else:
        # float32 typically can be tighter
        atol, rtol = 1e-5, 1e-5

    err = abs(triton_val - ref_val)
    print(type(exact_val))
    print(f"M={M}, Triton val={triton_val}, Ref val={ref_val}, abs error={err}, dtype = {out_dtype}")
    print(f"M={M}, Triton val={fp32_to_binary(triton_val)}, Ref val={ fp32_to_binary(ref_val)}, abs error={err}, dtype = {out_dtype}")
    print(f"M={M}, Triton val={binary_to_fp32(fp32_to_binary(triton_val))}, Ref val={ binary_to_fp32(fp32_to_binary(ref_val))}, abs error={err}, dtype = {out_dtype}")
    assert err < atol, f"Naive sum error too large: {err}, output dytpe = {out_dtype}, atol = {atol}"
