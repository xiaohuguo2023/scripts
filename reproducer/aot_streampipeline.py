from persistent_streamk_kernel import persistent_streamk_gemm
from triton.backends.compiler import GPUTarget
import triton
import argparse


signature = {'A': '*fp16', 'B': '*fp16', 'C': '*fp16', 'bias': '*fp16', 'P': '*fp32',
            'locks': '*i32', 'M': 'i32', 'N': 'i32', 'K': 'i32',
            'num_sms': 'i32', 'stride_am': 'i32', 'stride_ak': 'i32',
            'stride_bk': 'i32', 'stride_bn': 'i32', 'stride_cm': 'i32',
            'stride_cn': 'i32', 'stride_bias': 'i32'}                                                                                                                                                      

constants = {'stride_ak': 1, 'stride_bk': 1,
             'stride_cn': 1, 'stride_bias': 0, 'BLOCK_SIZE_M': 256,
             'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64,
             'GROUP_SIZE_M': 1, 'BIAS': False, 'EVEN_K': False}

parser = argparse.ArgumentParser(
  prog="persistent stream-k gemm",
  description="",
  allow_abbrev=False,
)

parser.add_argument("-b", "--backend", choices=['cuda', 'hip'],
  default='hip',
  help="backend")
args = parser.parse_args()

if args.backend == 'cuda':
  curr_target = GPUTarget("cuda", 90, 32)
elif args.backend == 'hip':
  curr_target = GPUTarget("hip", 'gfx942', 64)

print(f'{curr_target=}')
src = triton.compiler.ASTSource(fn=persistent_streamk_gemm, signature=signature, constants=constants)
k = triton.compile(src, target=curr_target)
