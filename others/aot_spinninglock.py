from  spinning_lock import spinning_lock_kernel
from triton.backends.compiler import GPUTarget
import triton
import argparse


signature = {'P': '*fp32', 'C': '*fp32',
            'locks': '*i32', 'num_sms': 'i32', 'M': 'i32', 'N': 'i32',
            'stride_cm': 'i32', 'stride_cn': 'i32'}                                                                                                                                                      

constants = {'stride_cn': 1, 'BLOCK_SIZE_M': 32,
             'BLOCK_SIZE_N': 32}

parser = argparse.ArgumentParser(
  prog="spinning lock in a loop",
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
src = triton.compiler.ASTSource(fn=spinning_lock_kernel, signature=signature, constants=constants)
k = triton.compile(src, target=curr_target)
