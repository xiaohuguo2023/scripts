import torch
import triton
import triton.language as tl
import argparse
import sys
import multiprocessing
from tune_streamk import gen_rotating_tensors
from myKernels import *


def matmul_BM16_BN16_BK16_GM1_nW2_nS0_EU0_kP1_mfma16(a, b, c, bias, P, locks, M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    grid = num_sms
    streamk_gemm_BM16_BN16_BK16_GM1_nW2_nS0_EU0_kP1_mfma16[grid,](
        a, b, c, bias, P, locks,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 16,
        BLOCK_SIZE_N = 16,
        BLOCK_SIZE_K = 16,
        GROUP_SIZE_M = 1,
        num_warps = 2,
        num_stages = 0,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 1,
        BIAS = False,
        EVEN_K = True
    )
    return c


def matmul_BM16_BN16_BK32_GM32_nW1_nS0_EU0_kP2_mfma16(a, b, c, bias, P, locks, M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    grid = num_sms
    streamk_gemm_BM16_BN16_BK32_GM32_nW1_nS0_EU0_kP2_mfma16[grid,](
        a, b, c, bias, P, locks,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 16,
        BLOCK_SIZE_N = 16,
        BLOCK_SIZE_K = 32,
        GROUP_SIZE_M = 32,
        num_warps = 1,
        num_stages = 0,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 2,
        BIAS = False,
        EVEN_K = True
    )
    return c


def matmul_BM16_BN16_BK32_GM4_nW8_nS0_EU0_kP2_mfma16(a, b, c, bias, P, locks, M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    grid = num_sms
    streamk_gemm_BM16_BN16_BK32_GM4_nW8_nS0_EU0_kP2_mfma16[grid,](
        a, b, c, bias, P, locks,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 16,
        BLOCK_SIZE_N = 16,
        BLOCK_SIZE_K = 32,
        GROUP_SIZE_M = 4,
        num_warps = 8,
        num_stages = 0,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 2,
        BIAS = False,
        EVEN_K = True
    )
    return c


def matmul_BM16_BN16_BK64_GM16_nW2_nS0_EU0_kP2_mfma16(a, b, c, bias, P, locks, M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    grid = num_sms
    streamk_gemm_BM16_BN16_BK64_GM16_nW2_nS0_EU0_kP2_mfma16[grid,](
        a, b, c, bias, P, locks,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 16,
        BLOCK_SIZE_N = 16,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 16,
        num_warps = 2,
        num_stages = 0,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 2,
        BIAS = False,
        EVEN_K = True
    )
    return c


def matmul_BM16_BN16_BK128_GM1_nW1_nS0_EU0_kP2_mfma16(a, b, c, bias, P, locks, M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    grid = num_sms
    streamk_gemm_BM16_BN16_BK128_GM1_nW1_nS0_EU0_kP2_mfma16[grid,](
        a, b, c, bias, P, locks,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 16,
        BLOCK_SIZE_N = 16,
        BLOCK_SIZE_K = 128,
        GROUP_SIZE_M = 1,
        num_warps = 1,
        num_stages = 0,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 2,
        BIAS = False,
        EVEN_K = True
    )
    return c


def matmul_BM16_BN16_BK128_GM8_nW4_nS0_EU0_kP2_mfma16(a, b, c, bias, P, locks, M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    grid = num_sms
    streamk_gemm_BM16_BN16_BK128_GM8_nW4_nS0_EU0_kP2_mfma16[grid,](
        a, b, c, bias, P, locks,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 16,
        BLOCK_SIZE_N = 16,
        BLOCK_SIZE_K = 128,
        GROUP_SIZE_M = 8,
        num_warps = 4,
        num_stages = 0,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 2,
        BIAS = False,
        EVEN_K = True
    )
    return c


def matmul_BM16_BN16_BK256_GM32_nW1_nS0_EU0_kP2_mfma16(a, b, c, bias, P, locks, M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    grid = num_sms
    streamk_gemm_BM16_BN16_BK256_GM32_nW1_nS0_EU0_kP2_mfma16[grid,](
        a, b, c, bias, P, locks,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 16,
        BLOCK_SIZE_N = 16,
        BLOCK_SIZE_K = 256,
        GROUP_SIZE_M = 32,
        num_warps = 1,
        num_stages = 0,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 2,
        BIAS = False,
        EVEN_K = True
    )
    return c


def matmul_BM16_BN16_BK256_GM4_nW8_nS0_EU0_kP2_mfma16(a, b, c, bias, P, locks, M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    grid = num_sms
    streamk_gemm_BM16_BN16_BK256_GM4_nW8_nS0_EU0_kP2_mfma16[grid,](
        a, b, c, bias, P, locks,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 16,
        BLOCK_SIZE_N = 16,
        BLOCK_SIZE_K = 256,
        GROUP_SIZE_M = 4,
        num_warps = 8,
        num_stages = 0,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 2,
        BIAS = False,
        EVEN_K = True
    )
    return c


def matmul_BM16_BN32_BK16_GM8_nW8_nS0_EU0_kP1_mfma16(a, b, c, bias, P, locks, M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    grid = num_sms
    streamk_gemm_BM16_BN32_BK16_GM8_nW8_nS0_EU0_kP1_mfma16[grid,](
        a, b, c, bias, P, locks,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 16,
        BLOCK_SIZE_N = 32,
        BLOCK_SIZE_K = 16,
        GROUP_SIZE_M = 8,
        num_warps = 8,
        num_stages = 0,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 1,
        BIAS = False,
        EVEN_K = True
    )
    return c


def matmul_BM16_BN32_BK32_GM1_nW4_nS0_EU0_kP2_mfma16(a, b, c, bias, P, locks, M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    grid = num_sms
    streamk_gemm_BM16_BN32_BK32_GM1_nW4_nS0_EU0_kP2_mfma16[grid,](
        a, b, c, bias, P, locks,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 16,
        BLOCK_SIZE_N = 32,
        BLOCK_SIZE_K = 32,
        GROUP_SIZE_M = 1,
        num_warps = 4,
        num_stages = 0,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 2,
        BIAS = False,
        EVEN_K = True
    )
    return c


def matmul_BM16_BN32_BK64_GM8_nW1_nS0_EU0_kP2_mfma16(a, b, c, bias, P, locks, M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    grid = num_sms
    streamk_gemm_BM16_BN32_BK64_GM8_nW1_nS0_EU0_kP2_mfma16[grid,](
        a, b, c, bias, P, locks,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 16,
        BLOCK_SIZE_N = 32,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 8,
        num_warps = 1,
        num_stages = 0,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 2,
        BIAS = False,
        EVEN_K = True
    )
    return c


def matmul_BM16_BN32_BK64_GM32_nW4_nS0_EU0_kP2_mfma16(a, b, c, bias, P, locks, M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    grid = num_sms
    streamk_gemm_BM16_BN32_BK64_GM32_nW4_nS0_EU0_kP2_mfma16[grid,](
        a, b, c, bias, P, locks,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 16,
        BLOCK_SIZE_N = 32,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 32,
        num_warps = 4,
        num_stages = 0,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 2,
        BIAS = False,
        EVEN_K = True
    )
    return c


def matmul_BM16_BN32_BK128_GM4_nW2_nS0_EU0_kP2_mfma16(a, b, c, bias, P, locks, M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    grid = num_sms
    streamk_gemm_BM16_BN32_BK128_GM4_nW2_nS0_EU0_kP2_mfma16[grid,](
        a, b, c, bias, P, locks,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 16,
        BLOCK_SIZE_N = 32,
        BLOCK_SIZE_K = 128,
        GROUP_SIZE_M = 4,
        num_warps = 2,
        num_stages = 0,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 2,
        BIAS = False,
        EVEN_K = True
    )
    return c


def matmul_BM16_BN32_BK128_GM16_nW8_nS0_EU0_kP2_mfma16(a, b, c, bias, P, locks, M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    grid = num_sms
    streamk_gemm_BM16_BN32_BK128_GM16_nW8_nS0_EU0_kP2_mfma16[grid,](
        a, b, c, bias, P, locks,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 16,
        BLOCK_SIZE_N = 32,
        BLOCK_SIZE_K = 128,
        GROUP_SIZE_M = 16,
        num_warps = 8,
        num_stages = 0,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 2,
        BIAS = False,
        EVEN_K = True
    )
    return c


def matmul_BM16_BN32_BK256_GM1_nW4_nS0_EU0_kP2_mfma16(a, b, c, bias, P, locks, M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    grid = num_sms
    streamk_gemm_BM16_BN32_BK256_GM1_nW4_nS0_EU0_kP2_mfma16[grid,](
        a, b, c, bias, P, locks,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 16,
        BLOCK_SIZE_N = 32,
        BLOCK_SIZE_K = 256,
        GROUP_SIZE_M = 1,
        num_warps = 4,
        num_stages = 0,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 2,
        BIAS = False,
        EVEN_K = True
    )
    return c


def matmul_BM16_BN64_BK16_GM1_nW2_nS0_EU0_kP1_mfma16(a, b, c, bias, P, locks, M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    grid = num_sms
    streamk_gemm_BM16_BN64_BK16_GM1_nW2_nS0_EU0_kP1_mfma16[grid,](
        a, b, c, bias, P, locks,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 16,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 16,
        GROUP_SIZE_M = 1,
        num_warps = 2,
        num_stages = 0,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 1,
        BIAS = False,
        EVEN_K = True
    )
    return c


def matmul_BM16_BN64_BK32_GM32_nW1_nS0_EU0_kP2_mfma16(a, b, c, bias, P, locks, M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    grid = num_sms
    streamk_gemm_BM16_BN64_BK32_GM32_nW1_nS0_EU0_kP2_mfma16[grid,](
        a, b, c, bias, P, locks,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 16,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 32,
        GROUP_SIZE_M = 32,
        num_warps = 1,
        num_stages = 0,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 2,
        BIAS = False,
        EVEN_K = True
    )
    return c


def matmul_BM16_BN64_BK32_GM4_nW8_nS0_EU0_kP2_mfma16(a, b, c, bias, P, locks, M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    grid = num_sms
    streamk_gemm_BM16_BN64_BK32_GM4_nW8_nS0_EU0_kP2_mfma16[grid,](
        a, b, c, bias, P, locks,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 16,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 32,
        GROUP_SIZE_M = 4,
        num_warps = 8,
        num_stages = 0,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 2,
        BIAS = False,
        EVEN_K = True
    )
    return c


def matmul_BM16_BN64_BK64_GM16_nW2_nS0_EU0_kP2_mfma16(a, b, c, bias, P, locks, M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    grid = num_sms
    streamk_gemm_BM16_BN64_BK64_GM16_nW2_nS0_EU0_kP2_mfma16[grid,](
        a, b, c, bias, P, locks,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 16,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 16,
        num_warps = 2,
        num_stages = 0,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 2,
        BIAS = False,
        EVEN_K = True
    )
    return c


def matmul_BM16_BN64_BK128_GM1_nW1_nS0_EU0_kP2_mfma16(a, b, c, bias, P, locks, M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    grid = num_sms
    streamk_gemm_BM16_BN64_BK128_GM1_nW1_nS0_EU0_kP2_mfma16[grid,](
        a, b, c, bias, P, locks,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 16,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 128,
        GROUP_SIZE_M = 1,
        num_warps = 1,
        num_stages = 0,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 2,
        BIAS = False,
        EVEN_K = True
    )
    return c


def matmul_BM16_BN64_BK128_GM8_nW4_nS0_EU0_kP2_mfma16(a, b, c, bias, P, locks, M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    grid = num_sms
    streamk_gemm_BM16_BN64_BK128_GM8_nW4_nS0_EU0_kP2_mfma16[grid,](
        a, b, c, bias, P, locks,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 16,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 128,
        GROUP_SIZE_M = 8,
        num_warps = 4,
        num_stages = 0,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 2,
        BIAS = False,
        EVEN_K = True
    )
    return c


def matmul_BM16_BN64_BK256_GM32_nW1_nS0_EU0_kP2_mfma16(a, b, c, bias, P, locks, M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    grid = num_sms
    streamk_gemm_BM16_BN64_BK256_GM32_nW1_nS0_EU0_kP2_mfma16[grid,](
        a, b, c, bias, P, locks,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 16,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 256,
        GROUP_SIZE_M = 32,
        num_warps = 1,
        num_stages = 0,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 2,
        BIAS = False,
        EVEN_K = True
    )
    return c


def matmul_BM16_BN64_BK256_GM4_nW8_nS0_EU0_kP2_mfma16(a, b, c, bias, P, locks, M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    grid = num_sms
    streamk_gemm_BM16_BN64_BK256_GM4_nW8_nS0_EU0_kP2_mfma16[grid,](
        a, b, c, bias, P, locks,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 16,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 256,
        GROUP_SIZE_M = 4,
        num_warps = 8,
        num_stages = 0,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 2,
        BIAS = False,
        EVEN_K = True
    )
    return c


def matmul_BM16_BN128_BK16_GM8_nW8_nS0_EU0_kP1_mfma16(a, b, c, bias, P, locks, M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    grid = num_sms
    streamk_gemm_BM16_BN128_BK16_GM8_nW8_nS0_EU0_kP1_mfma16[grid,](
        a, b, c, bias, P, locks,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 16,
        BLOCK_SIZE_N = 128,
        BLOCK_SIZE_K = 16,
        GROUP_SIZE_M = 8,
        num_warps = 8,
        num_stages = 0,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 1,
        BIAS = False,
        EVEN_K = True
    )
    return c


def matmul_BM16_BN128_BK32_GM1_nW4_nS0_EU0_kP2_mfma16(a, b, c, bias, P, locks, M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    grid = num_sms
    streamk_gemm_BM16_BN128_BK32_GM1_nW4_nS0_EU0_kP2_mfma16[grid,](
        a, b, c, bias, P, locks,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 16,
        BLOCK_SIZE_N = 128,
        BLOCK_SIZE_K = 32,
        GROUP_SIZE_M = 1,
        num_warps = 4,
        num_stages = 0,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 2,
        BIAS = False,
        EVEN_K = True
    )
    return c


def matmul_BM16_BN128_BK64_GM8_nW1_nS0_EU0_kP2_mfma16(a, b, c, bias, P, locks, M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    grid = num_sms
    streamk_gemm_BM16_BN128_BK64_GM8_nW1_nS0_EU0_kP2_mfma16[grid,](
        a, b, c, bias, P, locks,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 16,
        BLOCK_SIZE_N = 128,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 8,
        num_warps = 1,
        num_stages = 0,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 2,
        BIAS = False,
        EVEN_K = True
    )
    return c


def matmul_BM16_BN128_BK64_GM32_nW4_nS0_EU0_kP2_mfma16(a, b, c, bias, P, locks, M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    grid = num_sms
    streamk_gemm_BM16_BN128_BK64_GM32_nW4_nS0_EU0_kP2_mfma16[grid,](
        a, b, c, bias, P, locks,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 16,
        BLOCK_SIZE_N = 128,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 32,
        num_warps = 4,
        num_stages = 0,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 2,
        BIAS = False,
        EVEN_K = True
    )
    return c


def matmul_BM16_BN128_BK128_GM4_nW2_nS0_EU0_kP2_mfma16(a, b, c, bias, P, locks, M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    grid = num_sms
    streamk_gemm_BM16_BN128_BK128_GM4_nW2_nS0_EU0_kP2_mfma16[grid,](
        a, b, c, bias, P, locks,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 16,
        BLOCK_SIZE_N = 128,
        BLOCK_SIZE_K = 128,
        GROUP_SIZE_M = 4,
        num_warps = 2,
        num_stages = 0,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 2,
        BIAS = False,
        EVEN_K = True
    )
    return c


def matmul_BM16_BN128_BK128_GM16_nW8_nS0_EU0_kP2_mfma16(a, b, c, bias, P, locks, M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    grid = num_sms
    streamk_gemm_BM16_BN128_BK128_GM16_nW8_nS0_EU0_kP2_mfma16[grid,](
        a, b, c, bias, P, locks,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 16,
        BLOCK_SIZE_N = 128,
        BLOCK_SIZE_K = 128,
        GROUP_SIZE_M = 16,
        num_warps = 8,
        num_stages = 0,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 2,
        BIAS = False,
        EVEN_K = True
    )
    return c


def matmul_BM16_BN256_BK32_GM1_nW1_nS0_EU0_kP2_mfma16(a, b, c, bias, P, locks, M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    grid = num_sms
    streamk_gemm_BM16_BN256_BK32_GM1_nW1_nS0_EU0_kP2_mfma16[grid,](
        a, b, c, bias, P, locks,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 16,
        BLOCK_SIZE_N = 256,
        BLOCK_SIZE_K = 32,
        GROUP_SIZE_M = 1,
        num_warps = 1,
        num_stages = 0,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 2,
        BIAS = False,
        EVEN_K = True
    )
    return c


def matmul_BM16_BN256_BK32_GM8_nW4_nS0_EU0_kP2_mfma16(a, b, c, bias, P, locks, M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    grid = num_sms
    streamk_gemm_BM16_BN256_BK32_GM8_nW4_nS0_EU0_kP2_mfma16[grid,](
        a, b, c, bias, P, locks,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 16,
        BLOCK_SIZE_N = 256,
        BLOCK_SIZE_K = 32,
        GROUP_SIZE_M = 8,
        num_warps = 4,
        num_stages = 0,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 2,
        BIAS = False,
        EVEN_K = True
    )
    return c


def matmul_BM16_BN256_BK64_GM32_nW1_nS0_EU0_kP2_mfma16(a, b, c, bias, P, locks, M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    grid = num_sms
    streamk_gemm_BM16_BN256_BK64_GM32_nW1_nS0_EU0_kP2_mfma16[grid,](
        a, b, c, bias, P, locks,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 16,
        BLOCK_SIZE_N = 256,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 32,
        num_warps = 1,
        num_stages = 0,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 2,
        BIAS = False,
        EVEN_K = True
    )
    return c


def matmul_BM16_BN256_BK64_GM4_nW8_nS0_EU0_kP2_mfma16(a, b, c, bias, P, locks, M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    grid = num_sms
    streamk_gemm_BM16_BN256_BK64_GM4_nW8_nS0_EU0_kP2_mfma16[grid,](
        a, b, c, bias, P, locks,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 16,
        BLOCK_SIZE_N = 256,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 4,
        num_warps = 8,
        num_stages = 0,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 2,
        BIAS = False,
        EVEN_K = True
    )
    return c


def matmul_BM32_BN16_BK32_GM1_nW1_nS0_EU0_kP2_mfma16(a, b, c, bias, P, locks, M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    grid = num_sms
    streamk_gemm_BM32_BN16_BK32_GM1_nW1_nS0_EU0_kP2_mfma16[grid,](
        a, b, c, bias, P, locks,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 32,
        BLOCK_SIZE_N = 16,
        BLOCK_SIZE_K = 32,
        GROUP_SIZE_M = 1,
        num_warps = 1,
        num_stages = 0,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 2,
        BIAS = False,
        EVEN_K = True
    )
    return c


def matmul_BM32_BN16_BK32_GM1_nW8_nS0_EU0_kP2_mfma16(a, b, c, bias, P, locks, M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    grid = num_sms
    streamk_gemm_BM32_BN16_BK32_GM1_nW8_nS0_EU0_kP2_mfma16[grid,](
        a, b, c, bias, P, locks,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 32,
        BLOCK_SIZE_N = 16,
        BLOCK_SIZE_K = 32,
        GROUP_SIZE_M = 1,
        num_warps = 8,
        num_stages = 0,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 2,
        BIAS = False,
        EVEN_K = True
    )
    return c


def matmul_BM32_BN16_BK64_GM1_nW4_nS0_EU0_kP2_mfma16(a, b, c, bias, P, locks, M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    grid = num_sms
    streamk_gemm_BM32_BN16_BK64_GM1_nW4_nS0_EU0_kP2_mfma16[grid,](
        a, b, c, bias, P, locks,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 32,
        BLOCK_SIZE_N = 16,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 1,
        num_warps = 4,
        num_stages = 0,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 2,
        BIAS = False,
        EVEN_K = True
    )
    return c


def matmul_BM32_BN16_BK128_GM1_nW2_nS0_EU0_kP2_mfma16(a, b, c, bias, P, locks, M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    grid = num_sms
    streamk_gemm_BM32_BN16_BK128_GM1_nW2_nS0_EU0_kP2_mfma16[grid,](
        a, b, c, bias, P, locks,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 32,
        BLOCK_SIZE_N = 16,
        BLOCK_SIZE_K = 128,
        GROUP_SIZE_M = 1,
        num_warps = 2,
        num_stages = 0,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 2,
        BIAS = False,
        EVEN_K = True
    )
    return c


def matmul_BM32_BN16_BK256_GM1_nW1_nS0_EU0_kP2_mfma16(a, b, c, bias, P, locks, M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    grid = num_sms
    streamk_gemm_BM32_BN16_BK256_GM1_nW1_nS0_EU0_kP2_mfma16[grid,](
        a, b, c, bias, P, locks,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 32,
        BLOCK_SIZE_N = 16,
        BLOCK_SIZE_K = 256,
        GROUP_SIZE_M = 1,
        num_warps = 1,
        num_stages = 0,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 2,
        BIAS = False,
        EVEN_K = True
    )
    return c


def matmul_BM32_BN16_BK256_GM1_nW8_nS0_EU0_kP2_mfma16(a, b, c, bias, P, locks, M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    grid = num_sms
    streamk_gemm_BM32_BN16_BK256_GM1_nW8_nS0_EU0_kP2_mfma16[grid,](
        a, b, c, bias, P, locks,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 32,
        BLOCK_SIZE_N = 16,
        BLOCK_SIZE_K = 256,
        GROUP_SIZE_M = 1,
        num_warps = 8,
        num_stages = 0,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 2,
        BIAS = False,
        EVEN_K = True
    )
    return c


def matmul_BM32_BN32_BK16_GM4_nW2_nS0_EU0_kP2_mfma32(a, b, c, bias, P, locks, M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    grid = num_sms
    streamk_gemm_BM32_BN32_BK16_GM4_nW2_nS0_EU0_kP2_mfma32[grid,](
        a, b, c, bias, P, locks,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 32,
        BLOCK_SIZE_N = 32,
        BLOCK_SIZE_K = 16,
        GROUP_SIZE_M = 4,
        num_warps = 2,
        num_stages = 0,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 2,
        BIAS = False,
        EVEN_K = True
    )
    return c


def matmul_BM32_BN32_BK16_GM4_nW8_nS0_EU0_kP2_mfma32(a, b, c, bias, P, locks, M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    grid = num_sms
    streamk_gemm_BM32_BN32_BK16_GM4_nW8_nS0_EU0_kP2_mfma32[grid,](
        a, b, c, bias, P, locks,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 32,
        BLOCK_SIZE_N = 32,
        BLOCK_SIZE_K = 16,
        GROUP_SIZE_M = 4,
        num_warps = 8,
        num_stages = 0,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 2,
        BIAS = False,
        EVEN_K = True
    )
    return c


def matmul_BM32_BN32_BK32_GM1_nW2_nS0_EU0_kP2_mfma16(a, b, c, bias, P, locks, M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    grid = num_sms
    streamk_gemm_BM32_BN32_BK32_GM1_nW2_nS0_EU0_kP2_mfma16[grid,](
        a, b, c, bias, P, locks,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 32,
        BLOCK_SIZE_N = 32,
        BLOCK_SIZE_K = 32,
        GROUP_SIZE_M = 1,
        num_warps = 2,
        num_stages = 0,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 2,
        BIAS = False,
        EVEN_K = True
    )
    return c


def matmul_BM32_BN32_BK32_GM8_nW4_nS0_EU0_kP2_mfma16(a, b, c, bias, P, locks, M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    grid = num_sms
    streamk_gemm_BM32_BN32_BK32_GM8_nW4_nS0_EU0_kP2_mfma16[grid,](
        a, b, c, bias, P, locks,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 32,
        BLOCK_SIZE_N = 32,
        BLOCK_SIZE_K = 32,
        GROUP_SIZE_M = 8,
        num_warps = 4,
        num_stages = 0,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 2,
        BIAS = False,
        EVEN_K = True
    )
    return c


def matmul_BM32_BN32_BK64_GM1_nW1_nS0_EU0_kP2_mfma16(a, b, c, bias, P, locks, M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    grid = num_sms
    streamk_gemm_BM32_BN32_BK64_GM1_nW1_nS0_EU0_kP2_mfma16[grid,](
        a, b, c, bias, P, locks,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 32,
        BLOCK_SIZE_N = 32,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 1,
        num_warps = 1,
        num_stages = 0,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 2,
        BIAS = False,
        EVEN_K = True
    )
    return c


def matmul_BM32_BN32_BK64_GM8_nW2_nS0_EU0_kP2_mfma16(a, b, c, bias, P, locks, M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    grid = num_sms
    streamk_gemm_BM32_BN32_BK64_GM8_nW2_nS0_EU0_kP2_mfma16[grid,](
        a, b, c, bias, P, locks,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 32,
        BLOCK_SIZE_N = 32,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 8,
        num_warps = 2,
        num_stages = 0,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 2,
        BIAS = False,
        EVEN_K = True
    )
    return c


def matmul_BM32_BN32_BK64_GM1_nW8_nS0_EU0_kP2_mfma16(a, b, c, bias, P, locks, M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    grid = num_sms
    streamk_gemm_BM32_BN32_BK64_GM1_nW8_nS0_EU0_kP2_mfma16[grid,](
        a, b, c, bias, P, locks,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 32,
        BLOCK_SIZE_N = 32,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 1,
        num_warps = 8,
        num_stages = 0,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 2,
        BIAS = False,
        EVEN_K = True
    )
    return c


def matmul_BM32_BN32_BK128_GM8_nW1_nS0_EU0_kP2_mfma16(a, b, c, bias, P, locks, M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    grid = num_sms
    streamk_gemm_BM32_BN32_BK128_GM8_nW1_nS0_EU0_kP2_mfma16[grid,](
        a, b, c, bias, P, locks,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 32,
        BLOCK_SIZE_N = 32,
        BLOCK_SIZE_K = 128,
        GROUP_SIZE_M = 8,
        num_warps = 1,
        num_stages = 0,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 2,
        BIAS = False,
        EVEN_K = True
    )
    return c


def matmul_BM32_BN32_BK128_GM1_nW4_nS0_EU0_kP2_mfma16(a, b, c, bias, P, locks, M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    grid = num_sms
    streamk_gemm_BM32_BN32_BK128_GM1_nW4_nS0_EU0_kP2_mfma16[grid,](
        a, b, c, bias, P, locks,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 32,
        BLOCK_SIZE_N = 32,
        BLOCK_SIZE_K = 128,
        GROUP_SIZE_M = 1,
        num_warps = 4,
        num_stages = 0,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 2,
        BIAS = False,
        EVEN_K = True
    )
    return c


def matmul_BM32_BN32_BK128_GM8_nW8_nS0_EU0_kP2_mfma16(a, b, c, bias, P, locks, M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    grid = num_sms
    streamk_gemm_BM32_BN32_BK128_GM8_nW8_nS0_EU0_kP2_mfma16[grid,](
        a, b, c, bias, P, locks,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 32,
        BLOCK_SIZE_N = 32,
        BLOCK_SIZE_K = 128,
        GROUP_SIZE_M = 8,
        num_warps = 8,
        num_stages = 0,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 2,
        BIAS = False,
        EVEN_K = True
    )
    return c


def matmul_BM32_BN32_BK256_GM1_nW2_nS0_EU0_kP2_mfma16(a, b, c, bias, P, locks, M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    grid = num_sms
    streamk_gemm_BM32_BN32_BK256_GM1_nW2_nS0_EU0_kP2_mfma16[grid,](
        a, b, c, bias, P, locks,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 32,
        BLOCK_SIZE_N = 32,
        BLOCK_SIZE_K = 256,
        GROUP_SIZE_M = 1,
        num_warps = 2,
        num_stages = 0,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 2,
        BIAS = False,
        EVEN_K = True
    )
    return c


def matmul_BM32_BN32_BK256_GM8_nW4_nS0_EU0_kP2_mfma16(a, b, c, bias, P, locks, M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    grid = num_sms
    streamk_gemm_BM32_BN32_BK256_GM8_nW4_nS0_EU0_kP2_mfma16[grid,](
        a, b, c, bias, P, locks,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 32,
        BLOCK_SIZE_N = 32,
        BLOCK_SIZE_K = 256,
        GROUP_SIZE_M = 8,
        num_warps = 4,
        num_stages = 0,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 2,
        BIAS = False,
        EVEN_K = True
    )
    return c


def matmul_BM32_BN64_BK16_GM1_nW1_nS0_EU0_kP1_mfma32(a, b, c, bias, P, locks, M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    grid = num_sms
    streamk_gemm_BM32_BN64_BK16_GM1_nW1_nS0_EU0_kP1_mfma32[grid,](
        a, b, c, bias, P, locks,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 32,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 16,
        GROUP_SIZE_M = 1,
        num_warps = 1,
        num_stages = 0,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 1,
        BIAS = False,
        EVEN_K = True
    )
    return c


def matmul_BM32_BN64_BK16_GM1_nW4_nS0_EU0_kP1_mfma32(a, b, c, bias, P, locks, M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    grid = num_sms
    streamk_gemm_BM32_BN64_BK16_GM1_nW4_nS0_EU0_kP1_mfma32[grid,](
        a, b, c, bias, P, locks,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 32,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 16,
        GROUP_SIZE_M = 1,
        num_warps = 4,
        num_stages = 0,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 1,
        BIAS = False,
        EVEN_K = True
    )
    return c


def matmul_BM32_BN64_BK32_GM1_nW1_nS0_EU0_kP2_mfma16(a, b, c, bias, P, locks, M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    grid = num_sms
    streamk_gemm_BM32_BN64_BK32_GM1_nW1_nS0_EU0_kP2_mfma16[grid,](
        a, b, c, bias, P, locks,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 32,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 32,
        GROUP_SIZE_M = 1,
        num_warps = 1,
        num_stages = 0,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 2,
        BIAS = False,
        EVEN_K = True
    )
    return c


def matmul_BM32_BN64_BK32_GM8_nW2_nS0_EU0_kP2_mfma16(a, b, c, bias, P, locks, M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    grid = num_sms
    streamk_gemm_BM32_BN64_BK32_GM8_nW2_nS0_EU0_kP2_mfma16[grid,](
        a, b, c, bias, P, locks,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 32,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 32,
        GROUP_SIZE_M = 8,
        num_warps = 2,
        num_stages = 0,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 2,
        BIAS = False,
        EVEN_K = True
    )
    return c


def matmul_BM32_BN64_BK32_GM1_nW8_nS0_EU0_kP2_mfma16(a, b, c, bias, P, locks, M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    grid = num_sms
    streamk_gemm_BM32_BN64_BK32_GM1_nW8_nS0_EU0_kP2_mfma16[grid,](
        a, b, c, bias, P, locks,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 32,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 32,
        GROUP_SIZE_M = 1,
        num_warps = 8,
        num_stages = 0,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 2,
        BIAS = False,
        EVEN_K = True
    )
    return c


def matmul_BM32_BN64_BK64_GM8_nW1_nS0_EU0_kP2_mfma16(a, b, c, bias, P, locks, M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    grid = num_sms
    streamk_gemm_BM32_BN64_BK64_GM8_nW1_nS0_EU0_kP2_mfma16[grid,](
        a, b, c, bias, P, locks,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 32,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 8,
        num_warps = 1,
        num_stages = 0,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 2,
        BIAS = False,
        EVEN_K = True
    )
    return c


def matmul_BM32_BN64_BK64_GM1_nW4_nS0_EU0_kP2_mfma16(a, b, c, bias, P, locks, M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    grid = num_sms
    streamk_gemm_BM32_BN64_BK64_GM1_nW4_nS0_EU0_kP2_mfma16[grid,](
        a, b, c, bias, P, locks,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 32,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 1,
        num_warps = 4,
        num_stages = 0,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 2,
        BIAS = False,
        EVEN_K = True
    )
    return c


def matmul_BM32_BN64_BK64_GM8_nW8_nS0_EU0_kP2_mfma16(a, b, c, bias, P, locks, M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    grid = num_sms
    streamk_gemm_BM32_BN64_BK64_GM8_nW8_nS0_EU0_kP2_mfma16[grid,](
        a, b, c, bias, P, locks,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 32,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 8,
        num_warps = 8,
        num_stages = 0,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 2,
        BIAS = False,
        EVEN_K = True
    )
    return c


def matmul_BM32_BN64_BK128_GM1_nW2_nS0_EU0_kP2_mfma16(a, b, c, bias, P, locks, M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    grid = num_sms
    streamk_gemm_BM32_BN64_BK128_GM1_nW2_nS0_EU0_kP2_mfma16[grid,](
        a, b, c, bias, P, locks,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 32,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 128,
        GROUP_SIZE_M = 1,
        num_warps = 2,
        num_stages = 0,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 2,
        BIAS = False,
        EVEN_K = True
    )
    return c


def matmul_BM32_BN64_BK128_GM8_nW4_nS0_EU0_kP2_mfma16(a, b, c, bias, P, locks, M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    grid = num_sms
    streamk_gemm_BM32_BN64_BK128_GM8_nW4_nS0_EU0_kP2_mfma16[grid,](
        a, b, c, bias, P, locks,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 32,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 128,
        GROUP_SIZE_M = 8,
        num_warps = 4,
        num_stages = 0,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 2,
        BIAS = False,
        EVEN_K = True
    )
    return c


def matmul_BM32_BN64_BK256_GM1_nW1_nS0_EU0_kP2_mfma16(a, b, c, bias, P, locks, M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    grid = num_sms
    streamk_gemm_BM32_BN64_BK256_GM1_nW1_nS0_EU0_kP2_mfma16[grid,](
        a, b, c, bias, P, locks,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 32,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 256,
        GROUP_SIZE_M = 1,
        num_warps = 1,
        num_stages = 0,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 2,
        BIAS = False,
        EVEN_K = True
    )
    return c


def matmul_BM32_BN64_BK256_GM8_nW2_nS0_EU0_kP2_mfma16(a, b, c, bias, P, locks, M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    grid = num_sms
    streamk_gemm_BM32_BN64_BK256_GM8_nW2_nS0_EU0_kP2_mfma16[grid,](
        a, b, c, bias, P, locks,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 32,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 256,
        GROUP_SIZE_M = 8,
        num_warps = 2,
        num_stages = 0,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 2,
        BIAS = False,
        EVEN_K = True
    )
    return c


def matmul_BM32_BN64_BK256_GM1_nW8_nS0_EU0_kP2_mfma16(a, b, c, bias, P, locks, M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    grid = num_sms
    streamk_gemm_BM32_BN64_BK256_GM1_nW8_nS0_EU0_kP2_mfma16[grid,](
        a, b, c, bias, P, locks,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 32,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 256,
        GROUP_SIZE_M = 1,
        num_warps = 8,
        num_stages = 0,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 2,
        BIAS = False,
        EVEN_K = True
    )
    return c


def matmul_BM32_BN128_BK16_GM16_nW1_nS0_EU0_kP1_mfma16(a, b, c, bias, P, locks, M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    grid = num_sms
    streamk_gemm_BM32_BN128_BK16_GM16_nW1_nS0_EU0_kP1_mfma16[grid,](
        a, b, c, bias, P, locks,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 32,
        BLOCK_SIZE_N = 128,
        BLOCK_SIZE_K = 16,
        GROUP_SIZE_M = 16,
        num_warps = 1,
        num_stages = 0,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 1,
        BIAS = False,
        EVEN_K = True
    )
    return c


def matmul_BM32_BN128_BK16_GM16_nW4_nS0_EU0_kP1_mfma16(a, b, c, bias, P, locks, M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    grid = num_sms
    streamk_gemm_BM32_BN128_BK16_GM16_nW4_nS0_EU0_kP1_mfma16[grid,](
        a, b, c, bias, P, locks,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 32,
        BLOCK_SIZE_N = 128,
        BLOCK_SIZE_K = 16,
        GROUP_SIZE_M = 16,
        num_warps = 4,
        num_stages = 0,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 1,
        BIAS = False,
        EVEN_K = True
    )
    return c


def matmul_BM32_BN128_BK32_GM8_nW1_nS0_EU0_kP2_mfma16(a, b, c, bias, P, locks, M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    grid = num_sms
    streamk_gemm_BM32_BN128_BK32_GM8_nW1_nS0_EU0_kP2_mfma16[grid,](
        a, b, c, bias, P, locks,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 32,
        BLOCK_SIZE_N = 128,
        BLOCK_SIZE_K = 32,
        GROUP_SIZE_M = 8,
        num_warps = 1,
        num_stages = 0,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 2,
        BIAS = False,
        EVEN_K = True
    )
    return c


def matmul_BM32_BN128_BK32_GM1_nW4_nS0_EU0_kP2_mfma16(a, b, c, bias, P, locks, M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    grid = num_sms
    streamk_gemm_BM32_BN128_BK32_GM1_nW4_nS0_EU0_kP2_mfma16[grid,](
        a, b, c, bias, P, locks,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 32,
        BLOCK_SIZE_N = 128,
        BLOCK_SIZE_K = 32,
        GROUP_SIZE_M = 1,
        num_warps = 4,
        num_stages = 0,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 2,
        BIAS = False,
        EVEN_K = True
    )
    return c


def matmul_BM32_BN128_BK32_GM8_nW8_nS0_EU0_kP2_mfma16(a, b, c, bias, P, locks, M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    grid = num_sms
    streamk_gemm_BM32_BN128_BK32_GM8_nW8_nS0_EU0_kP2_mfma16[grid,](
        a, b, c, bias, P, locks,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 32,
        BLOCK_SIZE_N = 128,
        BLOCK_SIZE_K = 32,
        GROUP_SIZE_M = 8,
        num_warps = 8,
        num_stages = 0,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 2,
        BIAS = False,
        EVEN_K = True
    )
    return c


def matmul_BM32_BN128_BK64_GM1_nW2_nS0_EU0_kP2_mfma16(a, b, c, bias, P, locks, M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    grid = num_sms
    streamk_gemm_BM32_BN128_BK64_GM1_nW2_nS0_EU0_kP2_mfma16[grid,](
        a, b, c, bias, P, locks,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 32,
        BLOCK_SIZE_N = 128,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 1,
        num_warps = 2,
        num_stages = 0,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 2,
        BIAS = False,
        EVEN_K = True
    )
    return c


def matmul_BM32_BN128_BK64_GM8_nW4_nS0_EU0_kP2_mfma16(a, b, c, bias, P, locks, M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    grid = num_sms
    streamk_gemm_BM32_BN128_BK64_GM8_nW4_nS0_EU0_kP2_mfma16[grid,](
        a, b, c, bias, P, locks,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 32,
        BLOCK_SIZE_N = 128,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 8,
        num_warps = 4,
        num_stages = 0,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 2,
        BIAS = False,
        EVEN_K = True
    )
    return c


def matmul_BM32_BN128_BK128_GM1_nW1_nS0_EU0_kP2_mfma16(a, b, c, bias, P, locks, M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    grid = num_sms
    streamk_gemm_BM32_BN128_BK128_GM1_nW1_nS0_EU0_kP2_mfma16[grid,](
        a, b, c, bias, P, locks,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 32,
        BLOCK_SIZE_N = 128,
        BLOCK_SIZE_K = 128,
        GROUP_SIZE_M = 1,
        num_warps = 1,
        num_stages = 0,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 2,
        BIAS = False,
        EVEN_K = True
    )
    return c


def matmul_BM32_BN128_BK128_GM8_nW2_nS0_EU0_kP2_mfma16(a, b, c, bias, P, locks, M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    grid = num_sms
    streamk_gemm_BM32_BN128_BK128_GM8_nW2_nS0_EU0_kP2_mfma16[grid,](
        a, b, c, bias, P, locks,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 32,
        BLOCK_SIZE_N = 128,
        BLOCK_SIZE_K = 128,
        GROUP_SIZE_M = 8,
        num_warps = 2,
        num_stages = 0,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 2,
        BIAS = False,
        EVEN_K = True
    )
    return c


def matmul_BM32_BN128_BK128_GM1_nW8_nS0_EU0_kP2_mfma16(a, b, c, bias, P, locks, M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    grid = num_sms
    streamk_gemm_BM32_BN128_BK128_GM1_nW8_nS0_EU0_kP2_mfma16[grid,](
        a, b, c, bias, P, locks,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 32,
        BLOCK_SIZE_N = 128,
        BLOCK_SIZE_K = 128,
        GROUP_SIZE_M = 1,
        num_warps = 8,
        num_stages = 0,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 2,
        BIAS = False,
        EVEN_K = True
    )
    return c


def matmul_BM32_BN256_BK16_GM16_nW1_nS0_EU0_kP1_mfma16(a, b, c, bias, P, locks, M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    grid = num_sms
    streamk_gemm_BM32_BN256_BK16_GM16_nW1_nS0_EU0_kP1_mfma16[grid,](
        a, b, c, bias, P, locks,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 32,
        BLOCK_SIZE_N = 256,
        BLOCK_SIZE_K = 16,
        GROUP_SIZE_M = 16,
        num_warps = 1,
        num_stages = 0,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 1,
        BIAS = False,
        EVEN_K = True
    )
    return c


def matmul_BM32_BN256_BK16_GM16_nW4_nS0_EU0_kP1_mfma16(a, b, c, bias, P, locks, M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    grid = num_sms
    streamk_gemm_BM32_BN256_BK16_GM16_nW4_nS0_EU0_kP1_mfma16[grid,](
        a, b, c, bias, P, locks,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 32,
        BLOCK_SIZE_N = 256,
        BLOCK_SIZE_K = 16,
        GROUP_SIZE_M = 16,
        num_warps = 4,
        num_stages = 0,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 1,
        BIAS = False,
        EVEN_K = True
    )
    return c


def matmul_BM32_BN256_BK32_GM8_nW1_nS0_EU0_kP2_mfma16(a, b, c, bias, P, locks, M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    grid = num_sms
    streamk_gemm_BM32_BN256_BK32_GM8_nW1_nS0_EU0_kP2_mfma16[grid,](
        a, b, c, bias, P, locks,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 32,
        BLOCK_SIZE_N = 256,
        BLOCK_SIZE_K = 32,
        GROUP_SIZE_M = 8,
        num_warps = 1,
        num_stages = 0,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 2,
        BIAS = False,
        EVEN_K = True
    )
    return c


def matmul_BM32_BN256_BK32_GM1_nW4_nS0_EU0_kP2_mfma16(a, b, c, bias, P, locks, M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    grid = num_sms
    streamk_gemm_BM32_BN256_BK32_GM1_nW4_nS0_EU0_kP2_mfma16[grid,](
        a, b, c, bias, P, locks,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 32,
        BLOCK_SIZE_N = 256,
        BLOCK_SIZE_K = 32,
        GROUP_SIZE_M = 1,
        num_warps = 4,
        num_stages = 0,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 2,
        BIAS = False,
        EVEN_K = True
    )
    return c


def matmul_BM32_BN256_BK32_GM8_nW8_nS0_EU0_kP2_mfma16(a, b, c, bias, P, locks, M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    grid = num_sms
    streamk_gemm_BM32_BN256_BK32_GM8_nW8_nS0_EU0_kP2_mfma16[grid,](
        a, b, c, bias, P, locks,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 32,
        BLOCK_SIZE_N = 256,
        BLOCK_SIZE_K = 32,
        GROUP_SIZE_M = 8,
        num_warps = 8,
        num_stages = 0,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 2,
        BIAS = False,
        EVEN_K = True
    )
    return c


def matmul_BM32_BN256_BK64_GM1_nW2_nS0_EU0_kP2_mfma16(a, b, c, bias, P, locks, M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    grid = num_sms
    streamk_gemm_BM32_BN256_BK64_GM1_nW2_nS0_EU0_kP2_mfma16[grid,](
        a, b, c, bias, P, locks,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 32,
        BLOCK_SIZE_N = 256,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 1,
        num_warps = 2,
        num_stages = 0,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 2,
        BIAS = False,
        EVEN_K = True
    )
    return c


def matmul_BM32_BN256_BK64_GM8_nW4_nS0_EU0_kP2_mfma16(a, b, c, bias, P, locks, M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    grid = num_sms
    streamk_gemm_BM32_BN256_BK64_GM8_nW4_nS0_EU0_kP2_mfma16[grid,](
        a, b, c, bias, P, locks,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 32,
        BLOCK_SIZE_N = 256,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 8,
        num_warps = 4,
        num_stages = 0,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 2,
        BIAS = False,
        EVEN_K = True
    )
    return c


def matmul_BM64_BN16_BK16_GM4_nW1_nS0_EU0_kP1_mfma16(a, b, c, bias, P, locks, M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    grid = num_sms
    streamk_gemm_BM64_BN16_BK16_GM4_nW1_nS0_EU0_kP1_mfma16[grid,](
        a, b, c, bias, P, locks,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 16,
        BLOCK_SIZE_K = 16,
        GROUP_SIZE_M = 4,
        num_warps = 1,
        num_stages = 0,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 1,
        BIAS = False,
        EVEN_K = True
    )
    return c


def matmul_BM64_BN16_BK32_GM1_nW4_nS0_EU0_kP2_mfma16(a, b, c, bias, P, locks, M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    grid = num_sms
    streamk_gemm_BM64_BN16_BK32_GM1_nW4_nS0_EU0_kP2_mfma16[grid,](
        a, b, c, bias, P, locks,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 16,
        BLOCK_SIZE_K = 32,
        GROUP_SIZE_M = 1,
        num_warps = 4,
        num_stages = 0,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 2,
        BIAS = False,
        EVEN_K = True
    )
    return c


def matmul_BM64_BN16_BK64_GM1_nW4_nS0_EU0_kP2_mfma16(a, b, c, bias, P, locks, M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    grid = num_sms
    streamk_gemm_BM64_BN16_BK64_GM1_nW4_nS0_EU0_kP2_mfma16[grid,](
        a, b, c, bias, P, locks,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 16,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 1,
        num_warps = 4,
        num_stages = 0,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 2,
        BIAS = False,
        EVEN_K = True
    )
    return c


def matmul_BM64_BN16_BK128_GM1_nW4_nS0_EU0_kP2_mfma16(a, b, c, bias, P, locks, M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    grid = num_sms
    streamk_gemm_BM64_BN16_BK128_GM1_nW4_nS0_EU0_kP2_mfma16[grid,](
        a, b, c, bias, P, locks,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 16,
        BLOCK_SIZE_K = 128,
        GROUP_SIZE_M = 1,
        num_warps = 4,
        num_stages = 0,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 2,
        BIAS = False,
        EVEN_K = True
    )
    return c


def matmul_BM64_BN16_BK256_GM1_nW4_nS0_EU0_kP2_mfma16(a, b, c, bias, P, locks, M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    grid = num_sms
    streamk_gemm_BM64_BN16_BK256_GM1_nW4_nS0_EU0_kP2_mfma16[grid,](
        a, b, c, bias, P, locks,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 16,
        BLOCK_SIZE_K = 256,
        GROUP_SIZE_M = 1,
        num_warps = 4,
        num_stages = 0,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 2,
        BIAS = False,
        EVEN_K = True
    )
    return c


def matmul_BM64_BN32_BK16_GM4_nW2_nS0_EU0_kP1_mfma32(a, b, c, bias, P, locks, M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    grid = num_sms
    streamk_gemm_BM64_BN32_BK16_GM4_nW2_nS0_EU0_kP1_mfma32[grid,](
        a, b, c, bias, P, locks,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 32,
        BLOCK_SIZE_K = 16,
        GROUP_SIZE_M = 4,
        num_warps = 2,
        num_stages = 0,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 1,
        BIAS = False,
        EVEN_K = True
    )
    return c


def matmul_BM64_BN32_BK32_GM1_nW1_nS0_EU0_kP2_mfma16(a, b, c, bias, P, locks, M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    grid = num_sms
    streamk_gemm_BM64_BN32_BK32_GM1_nW1_nS0_EU0_kP2_mfma16[grid,](
        a, b, c, bias, P, locks,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 32,
        BLOCK_SIZE_K = 32,
        GROUP_SIZE_M = 1,
        num_warps = 1,
        num_stages = 0,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 2,
        BIAS = False,
        EVEN_K = True
    )
    return c


def matmul_BM64_BN32_BK32_GM1_nW4_nS0_EU0_kP2_mfma16(a, b, c, bias, P, locks, M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    grid = num_sms
    streamk_gemm_BM64_BN32_BK32_GM1_nW4_nS0_EU0_kP2_mfma16[grid,](
        a, b, c, bias, P, locks,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 32,
        BLOCK_SIZE_K = 32,
        GROUP_SIZE_M = 1,
        num_warps = 4,
        num_stages = 0,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 2,
        BIAS = False,
        EVEN_K = True
    )
    return c


def matmul_BM64_BN32_BK64_GM1_nW1_nS0_EU0_kP2_mfma16(a, b, c, bias, P, locks, M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    grid = num_sms
    streamk_gemm_BM64_BN32_BK64_GM1_nW1_nS0_EU0_kP2_mfma16[grid,](
        a, b, c, bias, P, locks,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 32,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 1,
        num_warps = 1,
        num_stages = 0,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 2,
        BIAS = False,
        EVEN_K = True
    )
    return c


def matmul_BM64_BN32_BK64_GM1_nW4_nS0_EU0_kP2_mfma16(a, b, c, bias, P, locks, M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    grid = num_sms
    streamk_gemm_BM64_BN32_BK64_GM1_nW4_nS0_EU0_kP2_mfma16[grid,](
        a, b, c, bias, P, locks,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 32,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 1,
        num_warps = 4,
        num_stages = 0,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 2,
        BIAS = False,
        EVEN_K = True
    )
    return c


def matmul_BM64_BN32_BK128_GM1_nW1_nS0_EU0_kP2_mfma16(a, b, c, bias, P, locks, M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    grid = num_sms
    streamk_gemm_BM64_BN32_BK128_GM1_nW1_nS0_EU0_kP2_mfma16[grid,](
        a, b, c, bias, P, locks,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 32,
        BLOCK_SIZE_K = 128,
        GROUP_SIZE_M = 1,
        num_warps = 1,
        num_stages = 0,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 2,
        BIAS = False,
        EVEN_K = True
    )
    return c


def matmul_BM64_BN32_BK128_GM1_nW4_nS0_EU0_kP2_mfma16(a, b, c, bias, P, locks, M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    grid = num_sms
    streamk_gemm_BM64_BN32_BK128_GM1_nW4_nS0_EU0_kP2_mfma16[grid,](
        a, b, c, bias, P, locks,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 32,
        BLOCK_SIZE_K = 128,
        GROUP_SIZE_M = 1,
        num_warps = 4,
        num_stages = 0,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 2,
        BIAS = False,
        EVEN_K = True
    )
    return c


def matmul_BM64_BN32_BK256_GM1_nW1_nS0_EU0_kP2_mfma16(a, b, c, bias, P, locks, M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    grid = num_sms
    streamk_gemm_BM64_BN32_BK256_GM1_nW1_nS0_EU0_kP2_mfma16[grid,](
        a, b, c, bias, P, locks,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 32,
        BLOCK_SIZE_K = 256,
        GROUP_SIZE_M = 1,
        num_warps = 1,
        num_stages = 0,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 2,
        BIAS = False,
        EVEN_K = True
    )
    return c


def matmul_BM64_BN32_BK256_GM1_nW4_nS0_EU0_kP2_mfma16(a, b, c, bias, P, locks, M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    grid = num_sms
    streamk_gemm_BM64_BN32_BK256_GM1_nW4_nS0_EU0_kP2_mfma16[grid,](
        a, b, c, bias, P, locks,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 32,
        BLOCK_SIZE_K = 256,
        GROUP_SIZE_M = 1,
        num_warps = 4,
        num_stages = 0,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 2,
        BIAS = False,
        EVEN_K = True
    )
    return c


def matmul_BM64_BN64_BK16_GM1_nW1_nS0_EU0_kP1_mfma32(a, b, c, bias, P, locks, M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    grid = num_sms
    streamk_gemm_BM64_BN64_BK16_GM1_nW1_nS0_EU0_kP1_mfma32[grid,](
        a, b, c, bias, P, locks,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 16,
        GROUP_SIZE_M = 1,
        num_warps = 1,
        num_stages = 0,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 1,
        BIAS = False,
        EVEN_K = True
    )
    return c


def matmul_BM64_BN64_BK16_GM8_nW4_nS0_EU0_kP1_mfma32(a, b, c, bias, P, locks, M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    grid = num_sms
    streamk_gemm_BM64_BN64_BK16_GM8_nW4_nS0_EU0_kP1_mfma32[grid,](
        a, b, c, bias, P, locks,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 16,
        GROUP_SIZE_M = 8,
        num_warps = 4,
        num_stages = 0,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 1,
        BIAS = False,
        EVEN_K = True
    )
    return c


def matmul_BM64_BN64_BK32_GM1_nW2_nS0_EU0_kP2_mfma16(a, b, c, bias, P, locks, M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    grid = num_sms
    streamk_gemm_BM64_BN64_BK32_GM1_nW2_nS0_EU0_kP2_mfma16[grid,](
        a, b, c, bias, P, locks,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 32,
        GROUP_SIZE_M = 1,
        num_warps = 2,
        num_stages = 0,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 2,
        BIAS = False,
        EVEN_K = True
    )
    return c


def matmul_BM64_BN64_BK32_GM1_nW8_nS0_EU0_kP2_mfma16(a, b, c, bias, P, locks, M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    grid = num_sms
    streamk_gemm_BM64_BN64_BK32_GM1_nW8_nS0_EU0_kP2_mfma16[grid,](
        a, b, c, bias, P, locks,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 32,
        GROUP_SIZE_M = 1,
        num_warps = 8,
        num_stages = 0,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 2,
        BIAS = False,
        EVEN_K = True
    )
    return c


def matmul_BM64_BN64_BK64_GM1_nW2_nS0_EU0_kP2_mfma16(a, b, c, bias, P, locks, M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    grid = num_sms
    streamk_gemm_BM64_BN64_BK64_GM1_nW2_nS0_EU0_kP2_mfma16[grid,](
        a, b, c, bias, P, locks,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 1,
        num_warps = 2,
        num_stages = 0,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 2,
        BIAS = False,
        EVEN_K = True
    )
    return c


def matmul_BM64_BN64_BK64_GM1_nW8_nS0_EU0_kP2_mfma16(a, b, c, bias, P, locks, M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    grid = num_sms
    streamk_gemm_BM64_BN64_BK64_GM1_nW8_nS0_EU0_kP2_mfma16[grid,](
        a, b, c, bias, P, locks,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 1,
        num_warps = 8,
        num_stages = 0,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 2,
        BIAS = False,
        EVEN_K = True
    )
    return c


def matmul_BM64_BN64_BK128_GM1_nW2_nS0_EU0_kP2_mfma16(a, b, c, bias, P, locks, M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    grid = num_sms
    streamk_gemm_BM64_BN64_BK128_GM1_nW2_nS0_EU0_kP2_mfma16[grid,](
        a, b, c, bias, P, locks,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 128,
        GROUP_SIZE_M = 1,
        num_warps = 2,
        num_stages = 0,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 2,
        BIAS = False,
        EVEN_K = True
    )
    return c


def matmul_BM64_BN64_BK128_GM1_nW8_nS0_EU0_kP2_mfma16(a, b, c, bias, P, locks, M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    grid = num_sms
    streamk_gemm_BM64_BN64_BK128_GM1_nW8_nS0_EU0_kP2_mfma16[grid,](
        a, b, c, bias, P, locks,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 128,
        GROUP_SIZE_M = 1,
        num_warps = 8,
        num_stages = 0,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 2,
        BIAS = False,
        EVEN_K = True
    )
    return c


def matmul_BM64_BN64_BK256_GM1_nW2_nS0_EU0_kP2_mfma16(a, b, c, bias, P, locks, M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    grid = num_sms
    streamk_gemm_BM64_BN64_BK256_GM1_nW2_nS0_EU0_kP2_mfma16[grid,](
        a, b, c, bias, P, locks,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 256,
        GROUP_SIZE_M = 1,
        num_warps = 2,
        num_stages = 0,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 2,
        BIAS = False,
        EVEN_K = True
    )
    return c


def matmul_BM64_BN64_BK256_GM1_nW8_nS0_EU0_kP2_mfma16(a, b, c, bias, P, locks, M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    grid = num_sms
    streamk_gemm_BM64_BN64_BK256_GM1_nW8_nS0_EU0_kP2_mfma16[grid,](
        a, b, c, bias, P, locks,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 256,
        GROUP_SIZE_M = 1,
        num_warps = 8,
        num_stages = 0,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 2,
        BIAS = False,
        EVEN_K = True
    )
    return c


def matmul_BM64_BN128_BK16_GM4_nW2_nS0_EU0_kP1_mfma32(a, b, c, bias, P, locks, M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    grid = num_sms
    streamk_gemm_BM64_BN128_BK16_GM4_nW2_nS0_EU0_kP1_mfma32[grid,](
        a, b, c, bias, P, locks,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 128,
        BLOCK_SIZE_K = 16,
        GROUP_SIZE_M = 4,
        num_warps = 2,
        num_stages = 0,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 1,
        BIAS = False,
        EVEN_K = True
    )
    return c


def matmul_BM64_BN128_BK32_GM1_nW1_nS0_EU0_kP2_mfma16(a, b, c, bias, P, locks, M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    grid = num_sms
    streamk_gemm_BM64_BN128_BK32_GM1_nW1_nS0_EU0_kP2_mfma16[grid,](
        a, b, c, bias, P, locks,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 128,
        BLOCK_SIZE_K = 32,
        GROUP_SIZE_M = 1,
        num_warps = 1,
        num_stages = 0,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 2,
        BIAS = False,
        EVEN_K = True
    )
    return c


def matmul_BM64_BN128_BK32_GM1_nW4_nS0_EU0_kP2_mfma16(a, b, c, bias, P, locks, M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    grid = num_sms
    streamk_gemm_BM64_BN128_BK32_GM1_nW4_nS0_EU0_kP2_mfma16[grid,](
        a, b, c, bias, P, locks,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 128,
        BLOCK_SIZE_K = 32,
        GROUP_SIZE_M = 1,
        num_warps = 4,
        num_stages = 0,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 2,
        BIAS = False,
        EVEN_K = True
    )
    return c


def matmul_BM64_BN128_BK64_GM1_nW1_nS0_EU0_kP2_mfma16(a, b, c, bias, P, locks, M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    grid = num_sms
    streamk_gemm_BM64_BN128_BK64_GM1_nW1_nS0_EU0_kP2_mfma16[grid,](
        a, b, c, bias, P, locks,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 128,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 1,
        num_warps = 1,
        num_stages = 0,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 2,
        BIAS = False,
        EVEN_K = True
    )
    return c


def matmul_BM64_BN128_BK64_GM1_nW4_nS0_EU0_kP2_mfma16(a, b, c, bias, P, locks, M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    grid = num_sms
    streamk_gemm_BM64_BN128_BK64_GM1_nW4_nS0_EU0_kP2_mfma16[grid,](
        a, b, c, bias, P, locks,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 128,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 1,
        num_warps = 4,
        num_stages = 0,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 2,
        BIAS = False,
        EVEN_K = True
    )
    return c


def matmul_BM64_BN128_BK128_GM1_nW1_nS0_EU0_kP2_mfma16(a, b, c, bias, P, locks, M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    grid = num_sms
    streamk_gemm_BM64_BN128_BK128_GM1_nW1_nS0_EU0_kP2_mfma16[grid,](
        a, b, c, bias, P, locks,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 128,
        BLOCK_SIZE_K = 128,
        GROUP_SIZE_M = 1,
        num_warps = 1,
        num_stages = 0,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 2,
        BIAS = False,
        EVEN_K = True
    )
    return c


def matmul_BM64_BN128_BK128_GM1_nW4_nS0_EU0_kP2_mfma16(a, b, c, bias, P, locks, M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    grid = num_sms
    streamk_gemm_BM64_BN128_BK128_GM1_nW4_nS0_EU0_kP2_mfma16[grid,](
        a, b, c, bias, P, locks,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 128,
        BLOCK_SIZE_K = 128,
        GROUP_SIZE_M = 1,
        num_warps = 4,
        num_stages = 0,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 2,
        BIAS = False,
        EVEN_K = True
    )
    return c


def matmul_BM64_BN256_BK16_GM1_nW1_nS0_EU0_kP1_mfma32(a, b, c, bias, P, locks, M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    grid = num_sms
    streamk_gemm_BM64_BN256_BK16_GM1_nW1_nS0_EU0_kP1_mfma32[grid,](
        a, b, c, bias, P, locks,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 256,
        BLOCK_SIZE_K = 16,
        GROUP_SIZE_M = 1,
        num_warps = 1,
        num_stages = 0,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 1,
        BIAS = False,
        EVEN_K = True
    )
    return c


def matmul_BM64_BN256_BK16_GM8_nW4_nS0_EU0_kP1_mfma32(a, b, c, bias, P, locks, M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    grid = num_sms
    streamk_gemm_BM64_BN256_BK16_GM8_nW4_nS0_EU0_kP1_mfma32[grid,](
        a, b, c, bias, P, locks,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 256,
        BLOCK_SIZE_K = 16,
        GROUP_SIZE_M = 8,
        num_warps = 4,
        num_stages = 0,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 1,
        BIAS = False,
        EVEN_K = True
    )
    return c


def matmul_BM64_BN256_BK32_GM1_nW2_nS0_EU0_kP2_mfma16(a, b, c, bias, P, locks, M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    grid = num_sms
    streamk_gemm_BM64_BN256_BK32_GM1_nW2_nS0_EU0_kP2_mfma16[grid,](
        a, b, c, bias, P, locks,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 256,
        BLOCK_SIZE_K = 32,
        GROUP_SIZE_M = 1,
        num_warps = 2,
        num_stages = 0,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 2,
        BIAS = False,
        EVEN_K = True
    )
    return c


def matmul_BM64_BN256_BK32_GM1_nW8_nS0_EU0_kP2_mfma16(a, b, c, bias, P, locks, M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    grid = num_sms
    streamk_gemm_BM64_BN256_BK32_GM1_nW8_nS0_EU0_kP2_mfma16[grid,](
        a, b, c, bias, P, locks,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 256,
        BLOCK_SIZE_K = 32,
        GROUP_SIZE_M = 1,
        num_warps = 8,
        num_stages = 0,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 2,
        BIAS = False,
        EVEN_K = True
    )
    return c


def matmul_BM64_BN256_BK64_GM1_nW2_nS0_EU0_kP2_mfma16(a, b, c, bias, P, locks, M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    grid = num_sms
    streamk_gemm_BM64_BN256_BK64_GM1_nW2_nS0_EU0_kP2_mfma16[grid,](
        a, b, c, bias, P, locks,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 256,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 1,
        num_warps = 2,
        num_stages = 0,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 2,
        BIAS = False,
        EVEN_K = True
    )
    return c


def matmul_BM64_BN256_BK64_GM1_nW8_nS0_EU0_kP2_mfma16(a, b, c, bias, P, locks, M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    grid = num_sms
    streamk_gemm_BM64_BN256_BK64_GM1_nW8_nS0_EU0_kP2_mfma16[grid,](
        a, b, c, bias, P, locks,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 256,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 1,
        num_warps = 8,
        num_stages = 0,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 2,
        BIAS = False,
        EVEN_K = True
    )
    return c


def matmul_BM128_BN16_BK32_GM1_nW2_nS0_EU0_kP2_mfma16(a, b, c, bias, P, locks, M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    grid = num_sms
    streamk_gemm_BM128_BN16_BK32_GM1_nW2_nS0_EU0_kP2_mfma16[grid,](
        a, b, c, bias, P, locks,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 128,
        BLOCK_SIZE_N = 16,
        BLOCK_SIZE_K = 32,
        GROUP_SIZE_M = 1,
        num_warps = 2,
        num_stages = 0,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 2,
        BIAS = False,
        EVEN_K = True
    )
    return c


def matmul_BM128_BN16_BK64_GM1_nW8_nS0_EU0_kP2_mfma16(a, b, c, bias, P, locks, M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    grid = num_sms
    streamk_gemm_BM128_BN16_BK64_GM1_nW8_nS0_EU0_kP2_mfma16[grid,](
        a, b, c, bias, P, locks,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 128,
        BLOCK_SIZE_N = 16,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 1,
        num_warps = 8,
        num_stages = 0,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 2,
        BIAS = False,
        EVEN_K = True
    )
    return c


def matmul_BM128_BN32_BK16_GM4_nW1_nS0_EU0_kP2_mfma32(a, b, c, bias, P, locks, M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    grid = num_sms
    streamk_gemm_BM128_BN32_BK16_GM4_nW1_nS0_EU0_kP2_mfma32[grid,](
        a, b, c, bias, P, locks,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 128,
        BLOCK_SIZE_N = 32,
        BLOCK_SIZE_K = 16,
        GROUP_SIZE_M = 4,
        num_warps = 1,
        num_stages = 0,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 2,
        BIAS = False,
        EVEN_K = True
    )
    return c


def matmul_BM128_BN32_BK32_GM4_nW1_nS0_EU0_kP2_mfma16(a, b, c, bias, P, locks, M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    grid = num_sms
    streamk_gemm_BM128_BN32_BK32_GM4_nW1_nS0_EU0_kP2_mfma16[grid,](
        a, b, c, bias, P, locks,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 128,
        BLOCK_SIZE_N = 32,
        BLOCK_SIZE_K = 32,
        GROUP_SIZE_M = 4,
        num_warps = 1,
        num_stages = 0,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 2,
        BIAS = False,
        EVEN_K = True
    )
    return c


def matmul_BM128_BN32_BK32_GM4_nW8_nS0_EU0_kP2_mfma16(a, b, c, bias, P, locks, M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    grid = num_sms
    streamk_gemm_BM128_BN32_BK32_GM4_nW8_nS0_EU0_kP2_mfma16[grid,](
        a, b, c, bias, P, locks,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 128,
        BLOCK_SIZE_N = 32,
        BLOCK_SIZE_K = 32,
        GROUP_SIZE_M = 4,
        num_warps = 8,
        num_stages = 0,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 2,
        BIAS = False,
        EVEN_K = True
    )
    return c


def matmul_BM128_BN32_BK64_GM4_nW4_nS0_EU0_kP2_mfma16(a, b, c, bias, P, locks, M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    grid = num_sms
    streamk_gemm_BM128_BN32_BK64_GM4_nW4_nS0_EU0_kP2_mfma16[grid,](
        a, b, c, bias, P, locks,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 128,
        BLOCK_SIZE_N = 32,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 4,
        num_warps = 4,
        num_stages = 0,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 2,
        BIAS = False,
        EVEN_K = True
    )
    return c


def matmul_BM128_BN32_BK128_GM4_nW2_nS0_EU0_kP2_mfma16(a, b, c, bias, P, locks, M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    grid = num_sms
    streamk_gemm_BM128_BN32_BK128_GM4_nW2_nS0_EU0_kP2_mfma16[grid,](
        a, b, c, bias, P, locks,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 128,
        BLOCK_SIZE_N = 32,
        BLOCK_SIZE_K = 128,
        GROUP_SIZE_M = 4,
        num_warps = 2,
        num_stages = 0,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 2,
        BIAS = False,
        EVEN_K = True
    )
    return c


def matmul_BM128_BN64_BK16_GM4_nW1_nS0_EU0_kP2_mfma32(a, b, c, bias, P, locks, M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    grid = num_sms
    streamk_gemm_BM128_BN64_BK16_GM4_nW1_nS0_EU0_kP2_mfma32[grid,](
        a, b, c, bias, P, locks,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 128,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 16,
        GROUP_SIZE_M = 4,
        num_warps = 1,
        num_stages = 0,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 2,
        BIAS = False,
        EVEN_K = True
    )
    return c


def matmul_BM128_BN64_BK32_GM4_nW1_nS0_EU0_kP2_mfma16(a, b, c, bias, P, locks, M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    grid = num_sms
    streamk_gemm_BM128_BN64_BK32_GM4_nW1_nS0_EU0_kP2_mfma16[grid,](
        a, b, c, bias, P, locks,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 128,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 32,
        GROUP_SIZE_M = 4,
        num_warps = 1,
        num_stages = 0,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 2,
        BIAS = False,
        EVEN_K = True
    )
    return c


def matmul_BM128_BN64_BK32_GM4_nW8_nS0_EU0_kP2_mfma16(a, b, c, bias, P, locks, M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    grid = num_sms
    streamk_gemm_BM128_BN64_BK32_GM4_nW8_nS0_EU0_kP2_mfma16[grid,](
        a, b, c, bias, P, locks,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 128,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 32,
        GROUP_SIZE_M = 4,
        num_warps = 8,
        num_stages = 0,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 2,
        BIAS = False,
        EVEN_K = True
    )
    return c


def matmul_BM128_BN64_BK64_GM4_nW4_nS0_EU0_kP2_mfma16(a, b, c, bias, P, locks, M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    grid = num_sms
    streamk_gemm_BM128_BN64_BK64_GM4_nW4_nS0_EU0_kP2_mfma16[grid,](
        a, b, c, bias, P, locks,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 128,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 4,
        num_warps = 4,
        num_stages = 0,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 2,
        BIAS = False,
        EVEN_K = True
    )
    return c


def matmul_BM128_BN64_BK128_GM4_nW2_nS0_EU0_kP2_mfma16(a, b, c, bias, P, locks, M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    grid = num_sms
    streamk_gemm_BM128_BN64_BK128_GM4_nW2_nS0_EU0_kP2_mfma16[grid,](
        a, b, c, bias, P, locks,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 128,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 128,
        GROUP_SIZE_M = 4,
        num_warps = 2,
        num_stages = 0,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 2,
        BIAS = False,
        EVEN_K = True
    )
    return c


def matmul_BM128_BN128_BK16_GM4_nW1_nS0_EU0_kP2_mfma32(a, b, c, bias, P, locks, M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    grid = num_sms
    streamk_gemm_BM128_BN128_BK16_GM4_nW1_nS0_EU0_kP2_mfma32[grid,](
        a, b, c, bias, P, locks,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 128,
        BLOCK_SIZE_N = 128,
        BLOCK_SIZE_K = 16,
        GROUP_SIZE_M = 4,
        num_warps = 1,
        num_stages = 0,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 2,
        BIAS = False,
        EVEN_K = True
    )
    return c


def matmul_BM128_BN128_BK32_GM4_nW1_nS0_EU0_kP2_mfma16(a, b, c, bias, P, locks, M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    grid = num_sms
    streamk_gemm_BM128_BN128_BK32_GM4_nW1_nS0_EU0_kP2_mfma16[grid,](
        a, b, c, bias, P, locks,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 128,
        BLOCK_SIZE_N = 128,
        BLOCK_SIZE_K = 32,
        GROUP_SIZE_M = 4,
        num_warps = 1,
        num_stages = 0,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 2,
        BIAS = False,
        EVEN_K = True
    )
    return c


def matmul_BM128_BN128_BK32_GM4_nW8_nS0_EU0_kP2_mfma16(a, b, c, bias, P, locks, M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    grid = num_sms
    streamk_gemm_BM128_BN128_BK32_GM4_nW8_nS0_EU0_kP2_mfma16[grid,](
        a, b, c, bias, P, locks,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 128,
        BLOCK_SIZE_N = 128,
        BLOCK_SIZE_K = 32,
        GROUP_SIZE_M = 4,
        num_warps = 8,
        num_stages = 0,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 2,
        BIAS = False,
        EVEN_K = True
    )
    return c


def matmul_BM128_BN128_BK64_GM4_nW4_nS0_EU0_kP2_mfma16(a, b, c, bias, P, locks, M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    grid = num_sms
    streamk_gemm_BM128_BN128_BK64_GM4_nW4_nS0_EU0_kP2_mfma16[grid,](
        a, b, c, bias, P, locks,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 128,
        BLOCK_SIZE_N = 128,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 4,
        num_warps = 4,
        num_stages = 0,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 2,
        BIAS = False,
        EVEN_K = True
    )
    return c


def matmul_BM128_BN128_BK128_GM1_nW4_nS0_EU0_kP2_mfma16(a, b, c, bias, P, locks, M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    grid = num_sms
    streamk_gemm_BM128_BN128_BK128_GM1_nW4_nS0_EU0_kP2_mfma16[grid,](
        a, b, c, bias, P, locks,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 128,
        BLOCK_SIZE_N = 128,
        BLOCK_SIZE_K = 128,
        GROUP_SIZE_M = 1,
        num_warps = 4,
        num_stages = 0,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 2,
        BIAS = False,
        EVEN_K = True
    )
    return c


def matmul_BM128_BN256_BK16_GM1_nW4_nS0_EU0_kP1_mfma32(a, b, c, bias, P, locks, M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    grid = num_sms
    streamk_gemm_BM128_BN256_BK16_GM1_nW4_nS0_EU0_kP1_mfma32[grid,](
        a, b, c, bias, P, locks,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 128,
        BLOCK_SIZE_N = 256,
        BLOCK_SIZE_K = 16,
        GROUP_SIZE_M = 1,
        num_warps = 4,
        num_stages = 0,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 1,
        BIAS = False,
        EVEN_K = True
    )
    return c


def matmul_BM128_BN256_BK32_GM4_nW2_nS0_EU0_kP2_mfma16(a, b, c, bias, P, locks, M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    grid = num_sms
    streamk_gemm_BM128_BN256_BK32_GM4_nW2_nS0_EU0_kP2_mfma16[grid,](
        a, b, c, bias, P, locks,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 128,
        BLOCK_SIZE_N = 256,
        BLOCK_SIZE_K = 32,
        GROUP_SIZE_M = 4,
        num_warps = 2,
        num_stages = 0,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 2,
        BIAS = False,
        EVEN_K = True
    )
    return c


def matmul_BM128_BN256_BK64_GM4_nW1_nS0_EU0_kP2_mfma16(a, b, c, bias, P, locks, M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    grid = num_sms
    streamk_gemm_BM128_BN256_BK64_GM4_nW1_nS0_EU0_kP2_mfma16[grid,](
        a, b, c, bias, P, locks,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 128,
        BLOCK_SIZE_N = 256,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 4,
        num_warps = 1,
        num_stages = 0,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 2,
        BIAS = False,
        EVEN_K = True
    )
    return c


def matmul_BM128_BN256_BK64_GM4_nW8_nS0_EU0_kP2_mfma16(a, b, c, bias, P, locks, M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    grid = num_sms
    streamk_gemm_BM128_BN256_BK64_GM4_nW8_nS0_EU0_kP2_mfma16[grid,](
        a, b, c, bias, P, locks,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 128,
        BLOCK_SIZE_N = 256,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 4,
        num_warps = 8,
        num_stages = 0,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 2,
        BIAS = False,
        EVEN_K = True
    )
    return c


def matmul_BM256_BN32_BK16_GM1_nW1_nS0_EU0_kP1_mfma32(a, b, c, bias, P, locks, M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    grid = num_sms
    streamk_gemm_BM256_BN32_BK16_GM1_nW1_nS0_EU0_kP1_mfma32[grid,](
        a, b, c, bias, P, locks,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 256,
        BLOCK_SIZE_N = 32,
        BLOCK_SIZE_K = 16,
        GROUP_SIZE_M = 1,
        num_warps = 1,
        num_stages = 0,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 1,
        BIAS = False,
        EVEN_K = True
    )
    return c


def matmul_BM256_BN32_BK32_GM1_nW8_nS0_EU0_kP2_mfma16(a, b, c, bias, P, locks, M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    grid = num_sms
    streamk_gemm_BM256_BN32_BK32_GM1_nW8_nS0_EU0_kP2_mfma16[grid,](
        a, b, c, bias, P, locks,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 256,
        BLOCK_SIZE_N = 32,
        BLOCK_SIZE_K = 32,
        GROUP_SIZE_M = 1,
        num_warps = 8,
        num_stages = 0,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 2,
        BIAS = False,
        EVEN_K = True
    )
    return c


def matmul_BM256_BN64_BK16_GM1_nW2_nS0_EU0_kP2_mfma32(a, b, c, bias, P, locks, M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    grid = num_sms
    streamk_gemm_BM256_BN64_BK16_GM1_nW2_nS0_EU0_kP2_mfma32[grid,](
        a, b, c, bias, P, locks,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 256,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 16,
        GROUP_SIZE_M = 1,
        num_warps = 2,
        num_stages = 0,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 2,
        BIAS = False,
        EVEN_K = True
    )
    return c


def matmul_BM256_BN64_BK64_GM1_nW1_nS0_EU0_kP2_mfma16(a, b, c, bias, P, locks, M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    grid = num_sms
    streamk_gemm_BM256_BN64_BK64_GM1_nW1_nS0_EU0_kP2_mfma16[grid,](
        a, b, c, bias, P, locks,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 256,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 1,
        num_warps = 1,
        num_stages = 0,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 2,
        BIAS = False,
        EVEN_K = True
    )
    return c


def matmul_BM256_BN128_BK16_GM1_nW8_nS0_EU0_kP1_mfma16(a, b, c, bias, P, locks, M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    grid = num_sms
    streamk_gemm_BM256_BN128_BK16_GM1_nW8_nS0_EU0_kP1_mfma16[grid,](
        a, b, c, bias, P, locks,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 256,
        BLOCK_SIZE_N = 128,
        BLOCK_SIZE_K = 16,
        GROUP_SIZE_M = 1,
        num_warps = 8,
        num_stages = 0,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 1,
        BIAS = False,
        EVEN_K = True
    )
    return c


def matmul_BM256_BN128_BK64_GM1_nW2_nS0_EU0_kP2_mfma16(a, b, c, bias, P, locks, M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    grid = num_sms
    streamk_gemm_BM256_BN128_BK64_GM1_nW2_nS0_EU0_kP2_mfma16[grid,](
        a, b, c, bias, P, locks,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 256,
        BLOCK_SIZE_N = 128,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 1,
        num_warps = 2,
        num_stages = 0,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 2,
        BIAS = False,
        EVEN_K = True
    )
    return c


def matmul_BM256_BN256_BK32_GM1_nW1_nS0_EU0_kP2_mfma16(a, b, c, bias, P, locks, M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    grid = num_sms
    streamk_gemm_BM256_BN256_BK32_GM1_nW1_nS0_EU0_kP2_mfma16[grid,](
        a, b, c, bias, P, locks,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 256,
        BLOCK_SIZE_N = 256,
        BLOCK_SIZE_K = 32,
        GROUP_SIZE_M = 1,
        num_warps = 1,
        num_stages = 0,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 2,
        BIAS = False,
        EVEN_K = True
    )
    return c


def matmul_BM256_BN256_BK64_GM1_nW4_nS0_EU0_kP2_mfma16(a, b, c, bias, P, locks, M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    grid = num_sms
    streamk_gemm_BM256_BN256_BK64_GM1_nW4_nS0_EU0_kP2_mfma16[grid,](
        a, b, c, bias, P, locks,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 256,
        BLOCK_SIZE_N = 256,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 1,
        num_warps = 4,
        num_stages = 0,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 2,
        BIAS = False,
        EVEN_K = True
    )
    return c

def test_gemm(M, N, K, num_sms, rotating_buffer_size, bias_size):
    tensors = gen_rotating_tensors(M, N, K, 'fp16', False, 'fp16', True, 'fp16',
                                   1, 'randn', rotating_buffer_size, bias_size, device='cuda')

    a = tensors['input_a'][0]
    b = tensors['input_b'][0]
    c = tensors['output_c'][0]
    assert bias_size == M or bias_size == 0

    stride_bias = tensors['bias'][0].stride(0) if bias_size > 0 else 0

    try:
        with open("/home/work/stream-k/tune_streamk/utils/../compile_driver.py.failed_configs", "r") as f:
            failed_configs = [cfg.strip() for cfg in f.readlines()]
    except Exception:
        failed_configs = []


    if 'BM16_BN16_BK16_GM1_nW2_nS0_EU0_kP1_mfma16' not in failed_configs:
        print(f"BM16_BN16_BK16_GM1_nW2_nS0_EU0_kP1_mfma16")
        rotating_num = tensors['rotating_num']
        locks = torch.zeros((200, num_sms), device = "cuda", dtype = torch.int32)
        P = torch.zeros((200, num_sms,  16*16), device="cuda", dtype=torch.float32)
        for i in range(200):
            a = tensors['input_a'][i % rotating_num]
            b = tensors['input_b'][i % rotating_num]
            c = tensors['output_c'][i % rotating_num]
            bias = tensors['bias'][i % rotating_num] if bias_size > 0 else None
            bias_stride = bias.stride(0) if bias_size > 0 else 0
            current_locks = locks[i]
            current_P = P[i]
            d = matmul_BM16_BN16_BK16_GM1_nW2_nS0_EU0_kP1_mfma16(a, b, c, bias, current_P, current_locks, M, N, K, num_sms, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1), bias_stride)

    if 'BM16_BN16_BK32_GM32_nW1_nS0_EU0_kP2_mfma16' not in failed_configs:
        print(f"BM16_BN16_BK32_GM32_nW1_nS0_EU0_kP2_mfma16")
        rotating_num = tensors['rotating_num']
        locks = torch.zeros((200, num_sms), device = "cuda", dtype = torch.int32)
        P = torch.zeros((200, num_sms,  16*16), device="cuda", dtype=torch.float32)
        for i in range(200):
            a = tensors['input_a'][i % rotating_num]
            b = tensors['input_b'][i % rotating_num]
            c = tensors['output_c'][i % rotating_num]
            bias = tensors['bias'][i % rotating_num] if bias_size > 0 else None
            bias_stride = bias.stride(0) if bias_size > 0 else 0
            current_locks = locks[i]
            current_P = P[i]
            d = matmul_BM16_BN16_BK32_GM32_nW1_nS0_EU0_kP2_mfma16(a, b, c, bias, current_P, current_locks, M, N, K, num_sms, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1), bias_stride)

    if 'BM16_BN16_BK32_GM4_nW8_nS0_EU0_kP2_mfma16' not in failed_configs:
        print(f"BM16_BN16_BK32_GM4_nW8_nS0_EU0_kP2_mfma16")
        rotating_num = tensors['rotating_num']
        locks = torch.zeros((200, num_sms), device = "cuda", dtype = torch.int32)
        P = torch.zeros((200, num_sms,  16*16), device="cuda", dtype=torch.float32)
        for i in range(200):
            a = tensors['input_a'][i % rotating_num]
            b = tensors['input_b'][i % rotating_num]
            c = tensors['output_c'][i % rotating_num]
            bias = tensors['bias'][i % rotating_num] if bias_size > 0 else None
            bias_stride = bias.stride(0) if bias_size > 0 else 0
            current_locks = locks[i]
            current_P = P[i]
            d = matmul_BM16_BN16_BK32_GM4_nW8_nS0_EU0_kP2_mfma16(a, b, c, bias, current_P, current_locks, M, N, K, num_sms, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1), bias_stride)

    if 'BM16_BN16_BK64_GM16_nW2_nS0_EU0_kP2_mfma16' not in failed_configs:
        print(f"BM16_BN16_BK64_GM16_nW2_nS0_EU0_kP2_mfma16")
        rotating_num = tensors['rotating_num']
        locks = torch.zeros((200, num_sms), device = "cuda", dtype = torch.int32)
        P = torch.zeros((200, num_sms,  16*16), device="cuda", dtype=torch.float32)
        for i in range(200):
            a = tensors['input_a'][i % rotating_num]
            b = tensors['input_b'][i % rotating_num]
            c = tensors['output_c'][i % rotating_num]
            bias = tensors['bias'][i % rotating_num] if bias_size > 0 else None
            bias_stride = bias.stride(0) if bias_size > 0 else 0
            current_locks = locks[i]
            current_P = P[i]
            d = matmul_BM16_BN16_BK64_GM16_nW2_nS0_EU0_kP2_mfma16(a, b, c, bias, current_P, current_locks, M, N, K, num_sms, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1), bias_stride)

    if 'BM16_BN16_BK128_GM1_nW1_nS0_EU0_kP2_mfma16' not in failed_configs:
        print(f"BM16_BN16_BK128_GM1_nW1_nS0_EU0_kP2_mfma16")
        rotating_num = tensors['rotating_num']
        locks = torch.zeros((200, num_sms), device = "cuda", dtype = torch.int32)
        P = torch.zeros((200, num_sms,  16*16), device="cuda", dtype=torch.float32)
        for i in range(200):
            a = tensors['input_a'][i % rotating_num]
            b = tensors['input_b'][i % rotating_num]
            c = tensors['output_c'][i % rotating_num]
            bias = tensors['bias'][i % rotating_num] if bias_size > 0 else None
            bias_stride = bias.stride(0) if bias_size > 0 else 0
            current_locks = locks[i]
            current_P = P[i]
            d = matmul_BM16_BN16_BK128_GM1_nW1_nS0_EU0_kP2_mfma16(a, b, c, bias, current_P, current_locks, M, N, K, num_sms, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1), bias_stride)

    if 'BM16_BN16_BK128_GM8_nW4_nS0_EU0_kP2_mfma16' not in failed_configs:
        print(f"BM16_BN16_BK128_GM8_nW4_nS0_EU0_kP2_mfma16")
        rotating_num = tensors['rotating_num']
        locks = torch.zeros((200, num_sms), device = "cuda", dtype = torch.int32)
        P = torch.zeros((200, num_sms,  16*16), device="cuda", dtype=torch.float32)
        for i in range(200):
            a = tensors['input_a'][i % rotating_num]
            b = tensors['input_b'][i % rotating_num]
            c = tensors['output_c'][i % rotating_num]
            bias = tensors['bias'][i % rotating_num] if bias_size > 0 else None
            bias_stride = bias.stride(0) if bias_size > 0 else 0
            current_locks = locks[i]
            current_P = P[i]
            d = matmul_BM16_BN16_BK128_GM8_nW4_nS0_EU0_kP2_mfma16(a, b, c, bias, current_P, current_locks, M, N, K, num_sms, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1), bias_stride)

    if 'BM16_BN16_BK256_GM32_nW1_nS0_EU0_kP2_mfma16' not in failed_configs:
        print(f"BM16_BN16_BK256_GM32_nW1_nS0_EU0_kP2_mfma16")
        rotating_num = tensors['rotating_num']
        locks = torch.zeros((200, num_sms), device = "cuda", dtype = torch.int32)
        P = torch.zeros((200, num_sms,  16*16), device="cuda", dtype=torch.float32)
        for i in range(200):
            a = tensors['input_a'][i % rotating_num]
            b = tensors['input_b'][i % rotating_num]
            c = tensors['output_c'][i % rotating_num]
            bias = tensors['bias'][i % rotating_num] if bias_size > 0 else None
            bias_stride = bias.stride(0) if bias_size > 0 else 0
            current_locks = locks[i]
            current_P = P[i]
            d = matmul_BM16_BN16_BK256_GM32_nW1_nS0_EU0_kP2_mfma16(a, b, c, bias, current_P, current_locks, M, N, K, num_sms, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1), bias_stride)

    if 'BM16_BN16_BK256_GM4_nW8_nS0_EU0_kP2_mfma16' not in failed_configs:
        print(f"BM16_BN16_BK256_GM4_nW8_nS0_EU0_kP2_mfma16")
        rotating_num = tensors['rotating_num']
        locks = torch.zeros((200, num_sms), device = "cuda", dtype = torch.int32)
        P = torch.zeros((200, num_sms,  16*16), device="cuda", dtype=torch.float32)
        for i in range(200):
            a = tensors['input_a'][i % rotating_num]
            b = tensors['input_b'][i % rotating_num]
            c = tensors['output_c'][i % rotating_num]
            bias = tensors['bias'][i % rotating_num] if bias_size > 0 else None
            bias_stride = bias.stride(0) if bias_size > 0 else 0
            current_locks = locks[i]
            current_P = P[i]
            d = matmul_BM16_BN16_BK256_GM4_nW8_nS0_EU0_kP2_mfma16(a, b, c, bias, current_P, current_locks, M, N, K, num_sms, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1), bias_stride)

    if 'BM16_BN32_BK16_GM8_nW8_nS0_EU0_kP1_mfma16' not in failed_configs:
        print(f"BM16_BN32_BK16_GM8_nW8_nS0_EU0_kP1_mfma16")
        rotating_num = tensors['rotating_num']
        locks = torch.zeros((200, num_sms), device = "cuda", dtype = torch.int32)
        P = torch.zeros((200, num_sms,  16*32), device="cuda", dtype=torch.float32)
        for i in range(200):
            a = tensors['input_a'][i % rotating_num]
            b = tensors['input_b'][i % rotating_num]
            c = tensors['output_c'][i % rotating_num]
            bias = tensors['bias'][i % rotating_num] if bias_size > 0 else None
            bias_stride = bias.stride(0) if bias_size > 0 else 0
            current_locks = locks[i]
            current_P = P[i]
            d = matmul_BM16_BN32_BK16_GM8_nW8_nS0_EU0_kP1_mfma16(a, b, c, bias, current_P, current_locks, M, N, K, num_sms, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1), bias_stride)

    if 'BM16_BN32_BK32_GM1_nW4_nS0_EU0_kP2_mfma16' not in failed_configs:
        print(f"BM16_BN32_BK32_GM1_nW4_nS0_EU0_kP2_mfma16")
        rotating_num = tensors['rotating_num']
        locks = torch.zeros((200, num_sms), device = "cuda", dtype = torch.int32)
        P = torch.zeros((200, num_sms,  16*32), device="cuda", dtype=torch.float32)
        for i in range(200):
            a = tensors['input_a'][i % rotating_num]
            b = tensors['input_b'][i % rotating_num]
            c = tensors['output_c'][i % rotating_num]
            bias = tensors['bias'][i % rotating_num] if bias_size > 0 else None
            bias_stride = bias.stride(0) if bias_size > 0 else 0
            current_locks = locks[i]
            current_P = P[i]
            d = matmul_BM16_BN32_BK32_GM1_nW4_nS0_EU0_kP2_mfma16(a, b, c, bias, current_P, current_locks, M, N, K, num_sms, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1), bias_stride)

    if 'BM16_BN32_BK64_GM8_nW1_nS0_EU0_kP2_mfma16' not in failed_configs:
        print(f"BM16_BN32_BK64_GM8_nW1_nS0_EU0_kP2_mfma16")
        rotating_num = tensors['rotating_num']
        locks = torch.zeros((200, num_sms), device = "cuda", dtype = torch.int32)
        P = torch.zeros((200, num_sms,  16*32), device="cuda", dtype=torch.float32)
        for i in range(200):
            a = tensors['input_a'][i % rotating_num]
            b = tensors['input_b'][i % rotating_num]
            c = tensors['output_c'][i % rotating_num]
            bias = tensors['bias'][i % rotating_num] if bias_size > 0 else None
            bias_stride = bias.stride(0) if bias_size > 0 else 0
            current_locks = locks[i]
            current_P = P[i]
            d = matmul_BM16_BN32_BK64_GM8_nW1_nS0_EU0_kP2_mfma16(a, b, c, bias, current_P, current_locks, M, N, K, num_sms, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1), bias_stride)

    if 'BM16_BN32_BK64_GM32_nW4_nS0_EU0_kP2_mfma16' not in failed_configs:
        print(f"BM16_BN32_BK64_GM32_nW4_nS0_EU0_kP2_mfma16")
        rotating_num = tensors['rotating_num']
        locks = torch.zeros((200, num_sms), device = "cuda", dtype = torch.int32)
        P = torch.zeros((200, num_sms,  16*32), device="cuda", dtype=torch.float32)
        for i in range(200):
            a = tensors['input_a'][i % rotating_num]
            b = tensors['input_b'][i % rotating_num]
            c = tensors['output_c'][i % rotating_num]
            bias = tensors['bias'][i % rotating_num] if bias_size > 0 else None
            bias_stride = bias.stride(0) if bias_size > 0 else 0
            current_locks = locks[i]
            current_P = P[i]
            d = matmul_BM16_BN32_BK64_GM32_nW4_nS0_EU0_kP2_mfma16(a, b, c, bias, current_P, current_locks, M, N, K, num_sms, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1), bias_stride)

    if 'BM16_BN32_BK128_GM4_nW2_nS0_EU0_kP2_mfma16' not in failed_configs:
        print(f"BM16_BN32_BK128_GM4_nW2_nS0_EU0_kP2_mfma16")
        rotating_num = tensors['rotating_num']
        locks = torch.zeros((200, num_sms), device = "cuda", dtype = torch.int32)
        P = torch.zeros((200, num_sms,  16*32), device="cuda", dtype=torch.float32)
        for i in range(200):
            a = tensors['input_a'][i % rotating_num]
            b = tensors['input_b'][i % rotating_num]
            c = tensors['output_c'][i % rotating_num]
            bias = tensors['bias'][i % rotating_num] if bias_size > 0 else None
            bias_stride = bias.stride(0) if bias_size > 0 else 0
            current_locks = locks[i]
            current_P = P[i]
            d = matmul_BM16_BN32_BK128_GM4_nW2_nS0_EU0_kP2_mfma16(a, b, c, bias, current_P, current_locks, M, N, K, num_sms, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1), bias_stride)

    if 'BM16_BN32_BK128_GM16_nW8_nS0_EU0_kP2_mfma16' not in failed_configs:
        print(f"BM16_BN32_BK128_GM16_nW8_nS0_EU0_kP2_mfma16")
        rotating_num = tensors['rotating_num']
        locks = torch.zeros((200, num_sms), device = "cuda", dtype = torch.int32)
        P = torch.zeros((200, num_sms,  16*32), device="cuda", dtype=torch.float32)
        for i in range(200):
            a = tensors['input_a'][i % rotating_num]
            b = tensors['input_b'][i % rotating_num]
            c = tensors['output_c'][i % rotating_num]
            bias = tensors['bias'][i % rotating_num] if bias_size > 0 else None
            bias_stride = bias.stride(0) if bias_size > 0 else 0
            current_locks = locks[i]
            current_P = P[i]
            d = matmul_BM16_BN32_BK128_GM16_nW8_nS0_EU0_kP2_mfma16(a, b, c, bias, current_P, current_locks, M, N, K, num_sms, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1), bias_stride)

    if 'BM16_BN32_BK256_GM1_nW4_nS0_EU0_kP2_mfma16' not in failed_configs:
        print(f"BM16_BN32_BK256_GM1_nW4_nS0_EU0_kP2_mfma16")
        rotating_num = tensors['rotating_num']
        locks = torch.zeros((200, num_sms), device = "cuda", dtype = torch.int32)
        P = torch.zeros((200, num_sms,  16*32), device="cuda", dtype=torch.float32)
        for i in range(200):
            a = tensors['input_a'][i % rotating_num]
            b = tensors['input_b'][i % rotating_num]
            c = tensors['output_c'][i % rotating_num]
            bias = tensors['bias'][i % rotating_num] if bias_size > 0 else None
            bias_stride = bias.stride(0) if bias_size > 0 else 0
            current_locks = locks[i]
            current_P = P[i]
            d = matmul_BM16_BN32_BK256_GM1_nW4_nS0_EU0_kP2_mfma16(a, b, c, bias, current_P, current_locks, M, N, K, num_sms, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1), bias_stride)

    if 'BM16_BN64_BK16_GM1_nW2_nS0_EU0_kP1_mfma16' not in failed_configs:
        print(f"BM16_BN64_BK16_GM1_nW2_nS0_EU0_kP1_mfma16")
        rotating_num = tensors['rotating_num']
        locks = torch.zeros((200, num_sms), device = "cuda", dtype = torch.int32)
        P = torch.zeros((200, num_sms,  16*64), device="cuda", dtype=torch.float32)
        for i in range(200):
            a = tensors['input_a'][i % rotating_num]
            b = tensors['input_b'][i % rotating_num]
            c = tensors['output_c'][i % rotating_num]
            bias = tensors['bias'][i % rotating_num] if bias_size > 0 else None
            bias_stride = bias.stride(0) if bias_size > 0 else 0
            current_locks = locks[i]
            current_P = P[i]
            d = matmul_BM16_BN64_BK16_GM1_nW2_nS0_EU0_kP1_mfma16(a, b, c, bias, current_P, current_locks, M, N, K, num_sms, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1), bias_stride)

    if 'BM16_BN64_BK32_GM32_nW1_nS0_EU0_kP2_mfma16' not in failed_configs:
        print(f"BM16_BN64_BK32_GM32_nW1_nS0_EU0_kP2_mfma16")
        rotating_num = tensors['rotating_num']
        locks = torch.zeros((200, num_sms), device = "cuda", dtype = torch.int32)
        P = torch.zeros((200, num_sms,  16*64), device="cuda", dtype=torch.float32)
        for i in range(200):
            a = tensors['input_a'][i % rotating_num]
            b = tensors['input_b'][i % rotating_num]
            c = tensors['output_c'][i % rotating_num]
            bias = tensors['bias'][i % rotating_num] if bias_size > 0 else None
            bias_stride = bias.stride(0) if bias_size > 0 else 0
            current_locks = locks[i]
            current_P = P[i]
            d = matmul_BM16_BN64_BK32_GM32_nW1_nS0_EU0_kP2_mfma16(a, b, c, bias, current_P, current_locks, M, N, K, num_sms, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1), bias_stride)

    if 'BM16_BN64_BK32_GM4_nW8_nS0_EU0_kP2_mfma16' not in failed_configs:
        print(f"BM16_BN64_BK32_GM4_nW8_nS0_EU0_kP2_mfma16")
        rotating_num = tensors['rotating_num']
        locks = torch.zeros((200, num_sms), device = "cuda", dtype = torch.int32)
        P = torch.zeros((200, num_sms,  16*64), device="cuda", dtype=torch.float32)
        for i in range(200):
            a = tensors['input_a'][i % rotating_num]
            b = tensors['input_b'][i % rotating_num]
            c = tensors['output_c'][i % rotating_num]
            bias = tensors['bias'][i % rotating_num] if bias_size > 0 else None
            bias_stride = bias.stride(0) if bias_size > 0 else 0
            current_locks = locks[i]
            current_P = P[i]
            d = matmul_BM16_BN64_BK32_GM4_nW8_nS0_EU0_kP2_mfma16(a, b, c, bias, current_P, current_locks, M, N, K, num_sms, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1), bias_stride)

    if 'BM16_BN64_BK64_GM16_nW2_nS0_EU0_kP2_mfma16' not in failed_configs:
        print(f"BM16_BN64_BK64_GM16_nW2_nS0_EU0_kP2_mfma16")
        rotating_num = tensors['rotating_num']
        locks = torch.zeros((200, num_sms), device = "cuda", dtype = torch.int32)
        P = torch.zeros((200, num_sms,  16*64), device="cuda", dtype=torch.float32)
        for i in range(200):
            a = tensors['input_a'][i % rotating_num]
            b = tensors['input_b'][i % rotating_num]
            c = tensors['output_c'][i % rotating_num]
            bias = tensors['bias'][i % rotating_num] if bias_size > 0 else None
            bias_stride = bias.stride(0) if bias_size > 0 else 0
            current_locks = locks[i]
            current_P = P[i]
            d = matmul_BM16_BN64_BK64_GM16_nW2_nS0_EU0_kP2_mfma16(a, b, c, bias, current_P, current_locks, M, N, K, num_sms, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1), bias_stride)

    if 'BM16_BN64_BK128_GM1_nW1_nS0_EU0_kP2_mfma16' not in failed_configs:
        print(f"BM16_BN64_BK128_GM1_nW1_nS0_EU0_kP2_mfma16")
        rotating_num = tensors['rotating_num']
        locks = torch.zeros((200, num_sms), device = "cuda", dtype = torch.int32)
        P = torch.zeros((200, num_sms,  16*64), device="cuda", dtype=torch.float32)
        for i in range(200):
            a = tensors['input_a'][i % rotating_num]
            b = tensors['input_b'][i % rotating_num]
            c = tensors['output_c'][i % rotating_num]
            bias = tensors['bias'][i % rotating_num] if bias_size > 0 else None
            bias_stride = bias.stride(0) if bias_size > 0 else 0
            current_locks = locks[i]
            current_P = P[i]
            d = matmul_BM16_BN64_BK128_GM1_nW1_nS0_EU0_kP2_mfma16(a, b, c, bias, current_P, current_locks, M, N, K, num_sms, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1), bias_stride)

    if 'BM16_BN64_BK128_GM8_nW4_nS0_EU0_kP2_mfma16' not in failed_configs:
        print(f"BM16_BN64_BK128_GM8_nW4_nS0_EU0_kP2_mfma16")
        rotating_num = tensors['rotating_num']
        locks = torch.zeros((200, num_sms), device = "cuda", dtype = torch.int32)
        P = torch.zeros((200, num_sms,  16*64), device="cuda", dtype=torch.float32)
        for i in range(200):
            a = tensors['input_a'][i % rotating_num]
            b = tensors['input_b'][i % rotating_num]
            c = tensors['output_c'][i % rotating_num]
            bias = tensors['bias'][i % rotating_num] if bias_size > 0 else None
            bias_stride = bias.stride(0) if bias_size > 0 else 0
            current_locks = locks[i]
            current_P = P[i]
            d = matmul_BM16_BN64_BK128_GM8_nW4_nS0_EU0_kP2_mfma16(a, b, c, bias, current_P, current_locks, M, N, K, num_sms, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1), bias_stride)

    if 'BM16_BN64_BK256_GM32_nW1_nS0_EU0_kP2_mfma16' not in failed_configs:
        print(f"BM16_BN64_BK256_GM32_nW1_nS0_EU0_kP2_mfma16")
        rotating_num = tensors['rotating_num']
        locks = torch.zeros((200, num_sms), device = "cuda", dtype = torch.int32)
        P = torch.zeros((200, num_sms,  16*64), device="cuda", dtype=torch.float32)
        for i in range(200):
            a = tensors['input_a'][i % rotating_num]
            b = tensors['input_b'][i % rotating_num]
            c = tensors['output_c'][i % rotating_num]
            bias = tensors['bias'][i % rotating_num] if bias_size > 0 else None
            bias_stride = bias.stride(0) if bias_size > 0 else 0
            current_locks = locks[i]
            current_P = P[i]
            d = matmul_BM16_BN64_BK256_GM32_nW1_nS0_EU0_kP2_mfma16(a, b, c, bias, current_P, current_locks, M, N, K, num_sms, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1), bias_stride)

    if 'BM16_BN64_BK256_GM4_nW8_nS0_EU0_kP2_mfma16' not in failed_configs:
        print(f"BM16_BN64_BK256_GM4_nW8_nS0_EU0_kP2_mfma16")
        rotating_num = tensors['rotating_num']
        locks = torch.zeros((200, num_sms), device = "cuda", dtype = torch.int32)
        P = torch.zeros((200, num_sms,  16*64), device="cuda", dtype=torch.float32)
        for i in range(200):
            a = tensors['input_a'][i % rotating_num]
            b = tensors['input_b'][i % rotating_num]
            c = tensors['output_c'][i % rotating_num]
            bias = tensors['bias'][i % rotating_num] if bias_size > 0 else None
            bias_stride = bias.stride(0) if bias_size > 0 else 0
            current_locks = locks[i]
            current_P = P[i]
            d = matmul_BM16_BN64_BK256_GM4_nW8_nS0_EU0_kP2_mfma16(a, b, c, bias, current_P, current_locks, M, N, K, num_sms, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1), bias_stride)

    if 'BM16_BN128_BK16_GM8_nW8_nS0_EU0_kP1_mfma16' not in failed_configs:
        print(f"BM16_BN128_BK16_GM8_nW8_nS0_EU0_kP1_mfma16")
        rotating_num = tensors['rotating_num']
        locks = torch.zeros((200, num_sms), device = "cuda", dtype = torch.int32)
        P = torch.zeros((200, num_sms,  16*128), device="cuda", dtype=torch.float32)
        for i in range(200):
            a = tensors['input_a'][i % rotating_num]
            b = tensors['input_b'][i % rotating_num]
            c = tensors['output_c'][i % rotating_num]
            bias = tensors['bias'][i % rotating_num] if bias_size > 0 else None
            bias_stride = bias.stride(0) if bias_size > 0 else 0
            current_locks = locks[i]
            current_P = P[i]
            d = matmul_BM16_BN128_BK16_GM8_nW8_nS0_EU0_kP1_mfma16(a, b, c, bias, current_P, current_locks, M, N, K, num_sms, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1), bias_stride)

    if 'BM16_BN128_BK32_GM1_nW4_nS0_EU0_kP2_mfma16' not in failed_configs:
        print(f"BM16_BN128_BK32_GM1_nW4_nS0_EU0_kP2_mfma16")
        rotating_num = tensors['rotating_num']
        locks = torch.zeros((200, num_sms), device = "cuda", dtype = torch.int32)
        P = torch.zeros((200, num_sms,  16*128), device="cuda", dtype=torch.float32)
        for i in range(200):
            a = tensors['input_a'][i % rotating_num]
            b = tensors['input_b'][i % rotating_num]
            c = tensors['output_c'][i % rotating_num]
            bias = tensors['bias'][i % rotating_num] if bias_size > 0 else None
            bias_stride = bias.stride(0) if bias_size > 0 else 0
            current_locks = locks[i]
            current_P = P[i]
            d = matmul_BM16_BN128_BK32_GM1_nW4_nS0_EU0_kP2_mfma16(a, b, c, bias, current_P, current_locks, M, N, K, num_sms, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1), bias_stride)

    if 'BM16_BN128_BK64_GM8_nW1_nS0_EU0_kP2_mfma16' not in failed_configs:
        print(f"BM16_BN128_BK64_GM8_nW1_nS0_EU0_kP2_mfma16")
        rotating_num = tensors['rotating_num']
        locks = torch.zeros((200, num_sms), device = "cuda", dtype = torch.int32)
        P = torch.zeros((200, num_sms,  16*128), device="cuda", dtype=torch.float32)
        for i in range(200):
            a = tensors['input_a'][i % rotating_num]
            b = tensors['input_b'][i % rotating_num]
            c = tensors['output_c'][i % rotating_num]
            bias = tensors['bias'][i % rotating_num] if bias_size > 0 else None
            bias_stride = bias.stride(0) if bias_size > 0 else 0
            current_locks = locks[i]
            current_P = P[i]
            d = matmul_BM16_BN128_BK64_GM8_nW1_nS0_EU0_kP2_mfma16(a, b, c, bias, current_P, current_locks, M, N, K, num_sms, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1), bias_stride)

    if 'BM16_BN128_BK64_GM32_nW4_nS0_EU0_kP2_mfma16' not in failed_configs:
        print(f"BM16_BN128_BK64_GM32_nW4_nS0_EU0_kP2_mfma16")
        rotating_num = tensors['rotating_num']
        locks = torch.zeros((200, num_sms), device = "cuda", dtype = torch.int32)
        P = torch.zeros((200, num_sms,  16*128), device="cuda", dtype=torch.float32)
        for i in range(200):
            a = tensors['input_a'][i % rotating_num]
            b = tensors['input_b'][i % rotating_num]
            c = tensors['output_c'][i % rotating_num]
            bias = tensors['bias'][i % rotating_num] if bias_size > 0 else None
            bias_stride = bias.stride(0) if bias_size > 0 else 0
            current_locks = locks[i]
            current_P = P[i]
            d = matmul_BM16_BN128_BK64_GM32_nW4_nS0_EU0_kP2_mfma16(a, b, c, bias, current_P, current_locks, M, N, K, num_sms, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1), bias_stride)

    if 'BM16_BN128_BK128_GM4_nW2_nS0_EU0_kP2_mfma16' not in failed_configs:
        print(f"BM16_BN128_BK128_GM4_nW2_nS0_EU0_kP2_mfma16")
        rotating_num = tensors['rotating_num']
        locks = torch.zeros((200, num_sms), device = "cuda", dtype = torch.int32)
        P = torch.zeros((200, num_sms,  16*128), device="cuda", dtype=torch.float32)
        for i in range(200):
            a = tensors['input_a'][i % rotating_num]
            b = tensors['input_b'][i % rotating_num]
            c = tensors['output_c'][i % rotating_num]
            bias = tensors['bias'][i % rotating_num] if bias_size > 0 else None
            bias_stride = bias.stride(0) if bias_size > 0 else 0
            current_locks = locks[i]
            current_P = P[i]
            d = matmul_BM16_BN128_BK128_GM4_nW2_nS0_EU0_kP2_mfma16(a, b, c, bias, current_P, current_locks, M, N, K, num_sms, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1), bias_stride)

    if 'BM16_BN128_BK128_GM16_nW8_nS0_EU0_kP2_mfma16' not in failed_configs:
        print(f"BM16_BN128_BK128_GM16_nW8_nS0_EU0_kP2_mfma16")
        rotating_num = tensors['rotating_num']
        locks = torch.zeros((200, num_sms), device = "cuda", dtype = torch.int32)
        P = torch.zeros((200, num_sms,  16*128), device="cuda", dtype=torch.float32)
        for i in range(200):
            a = tensors['input_a'][i % rotating_num]
            b = tensors['input_b'][i % rotating_num]
            c = tensors['output_c'][i % rotating_num]
            bias = tensors['bias'][i % rotating_num] if bias_size > 0 else None
            bias_stride = bias.stride(0) if bias_size > 0 else 0
            current_locks = locks[i]
            current_P = P[i]
            d = matmul_BM16_BN128_BK128_GM16_nW8_nS0_EU0_kP2_mfma16(a, b, c, bias, current_P, current_locks, M, N, K, num_sms, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1), bias_stride)

    if 'BM16_BN256_BK32_GM1_nW1_nS0_EU0_kP2_mfma16' not in failed_configs:
        print(f"BM16_BN256_BK32_GM1_nW1_nS0_EU0_kP2_mfma16")
        rotating_num = tensors['rotating_num']
        locks = torch.zeros((200, num_sms), device = "cuda", dtype = torch.int32)
        P = torch.zeros((200, num_sms,  16*256), device="cuda", dtype=torch.float32)
        for i in range(200):
            a = tensors['input_a'][i % rotating_num]
            b = tensors['input_b'][i % rotating_num]
            c = tensors['output_c'][i % rotating_num]
            bias = tensors['bias'][i % rotating_num] if bias_size > 0 else None
            bias_stride = bias.stride(0) if bias_size > 0 else 0
            current_locks = locks[i]
            current_P = P[i]
            d = matmul_BM16_BN256_BK32_GM1_nW1_nS0_EU0_kP2_mfma16(a, b, c, bias, current_P, current_locks, M, N, K, num_sms, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1), bias_stride)

    if 'BM16_BN256_BK32_GM8_nW4_nS0_EU0_kP2_mfma16' not in failed_configs:
        print(f"BM16_BN256_BK32_GM8_nW4_nS0_EU0_kP2_mfma16")
        rotating_num = tensors['rotating_num']
        locks = torch.zeros((200, num_sms), device = "cuda", dtype = torch.int32)
        P = torch.zeros((200, num_sms,  16*256), device="cuda", dtype=torch.float32)
        for i in range(200):
            a = tensors['input_a'][i % rotating_num]
            b = tensors['input_b'][i % rotating_num]
            c = tensors['output_c'][i % rotating_num]
            bias = tensors['bias'][i % rotating_num] if bias_size > 0 else None
            bias_stride = bias.stride(0) if bias_size > 0 else 0
            current_locks = locks[i]
            current_P = P[i]
            d = matmul_BM16_BN256_BK32_GM8_nW4_nS0_EU0_kP2_mfma16(a, b, c, bias, current_P, current_locks, M, N, K, num_sms, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1), bias_stride)

    if 'BM16_BN256_BK64_GM32_nW1_nS0_EU0_kP2_mfma16' not in failed_configs:
        print(f"BM16_BN256_BK64_GM32_nW1_nS0_EU0_kP2_mfma16")
        rotating_num = tensors['rotating_num']
        locks = torch.zeros((200, num_sms), device = "cuda", dtype = torch.int32)
        P = torch.zeros((200, num_sms,  16*256), device="cuda", dtype=torch.float32)
        for i in range(200):
            a = tensors['input_a'][i % rotating_num]
            b = tensors['input_b'][i % rotating_num]
            c = tensors['output_c'][i % rotating_num]
            bias = tensors['bias'][i % rotating_num] if bias_size > 0 else None
            bias_stride = bias.stride(0) if bias_size > 0 else 0
            current_locks = locks[i]
            current_P = P[i]
            d = matmul_BM16_BN256_BK64_GM32_nW1_nS0_EU0_kP2_mfma16(a, b, c, bias, current_P, current_locks, M, N, K, num_sms, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1), bias_stride)

    if 'BM16_BN256_BK64_GM4_nW8_nS0_EU0_kP2_mfma16' not in failed_configs:
        print(f"BM16_BN256_BK64_GM4_nW8_nS0_EU0_kP2_mfma16")
        rotating_num = tensors['rotating_num']
        locks = torch.zeros((200, num_sms), device = "cuda", dtype = torch.int32)
        P = torch.zeros((200, num_sms,  16*256), device="cuda", dtype=torch.float32)
        for i in range(200):
            a = tensors['input_a'][i % rotating_num]
            b = tensors['input_b'][i % rotating_num]
            c = tensors['output_c'][i % rotating_num]
            bias = tensors['bias'][i % rotating_num] if bias_size > 0 else None
            bias_stride = bias.stride(0) if bias_size > 0 else 0
            current_locks = locks[i]
            current_P = P[i]
            d = matmul_BM16_BN256_BK64_GM4_nW8_nS0_EU0_kP2_mfma16(a, b, c, bias, current_P, current_locks, M, N, K, num_sms, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1), bias_stride)

    if 'BM32_BN16_BK32_GM1_nW1_nS0_EU0_kP2_mfma16' not in failed_configs:
        print(f"BM32_BN16_BK32_GM1_nW1_nS0_EU0_kP2_mfma16")
        rotating_num = tensors['rotating_num']
        locks = torch.zeros((200, num_sms), device = "cuda", dtype = torch.int32)
        P = torch.zeros((200, num_sms,  32*16), device="cuda", dtype=torch.float32)
        for i in range(200):
            a = tensors['input_a'][i % rotating_num]
            b = tensors['input_b'][i % rotating_num]
            c = tensors['output_c'][i % rotating_num]
            bias = tensors['bias'][i % rotating_num] if bias_size > 0 else None
            bias_stride = bias.stride(0) if bias_size > 0 else 0
            current_locks = locks[i]
            current_P = P[i]
            d = matmul_BM32_BN16_BK32_GM1_nW1_nS0_EU0_kP2_mfma16(a, b, c, bias, current_P, current_locks, M, N, K, num_sms, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1), bias_stride)

    if 'BM32_BN16_BK32_GM1_nW8_nS0_EU0_kP2_mfma16' not in failed_configs:
        print(f"BM32_BN16_BK32_GM1_nW8_nS0_EU0_kP2_mfma16")
        rotating_num = tensors['rotating_num']
        locks = torch.zeros((200, num_sms), device = "cuda", dtype = torch.int32)
        P = torch.zeros((200, num_sms,  32*16), device="cuda", dtype=torch.float32)
        for i in range(200):
            a = tensors['input_a'][i % rotating_num]
            b = tensors['input_b'][i % rotating_num]
            c = tensors['output_c'][i % rotating_num]
            bias = tensors['bias'][i % rotating_num] if bias_size > 0 else None
            bias_stride = bias.stride(0) if bias_size > 0 else 0
            current_locks = locks[i]
            current_P = P[i]
            d = matmul_BM32_BN16_BK32_GM1_nW8_nS0_EU0_kP2_mfma16(a, b, c, bias, current_P, current_locks, M, N, K, num_sms, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1), bias_stride)

    if 'BM32_BN16_BK64_GM1_nW4_nS0_EU0_kP2_mfma16' not in failed_configs:
        print(f"BM32_BN16_BK64_GM1_nW4_nS0_EU0_kP2_mfma16")
        rotating_num = tensors['rotating_num']
        locks = torch.zeros((200, num_sms), device = "cuda", dtype = torch.int32)
        P = torch.zeros((200, num_sms,  32*16), device="cuda", dtype=torch.float32)
        for i in range(200):
            a = tensors['input_a'][i % rotating_num]
            b = tensors['input_b'][i % rotating_num]
            c = tensors['output_c'][i % rotating_num]
            bias = tensors['bias'][i % rotating_num] if bias_size > 0 else None
            bias_stride = bias.stride(0) if bias_size > 0 else 0
            current_locks = locks[i]
            current_P = P[i]
            d = matmul_BM32_BN16_BK64_GM1_nW4_nS0_EU0_kP2_mfma16(a, b, c, bias, current_P, current_locks, M, N, K, num_sms, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1), bias_stride)

    if 'BM32_BN16_BK128_GM1_nW2_nS0_EU0_kP2_mfma16' not in failed_configs:
        print(f"BM32_BN16_BK128_GM1_nW2_nS0_EU0_kP2_mfma16")
        rotating_num = tensors['rotating_num']
        locks = torch.zeros((200, num_sms), device = "cuda", dtype = torch.int32)
        P = torch.zeros((200, num_sms,  32*16), device="cuda", dtype=torch.float32)
        for i in range(200):
            a = tensors['input_a'][i % rotating_num]
            b = tensors['input_b'][i % rotating_num]
            c = tensors['output_c'][i % rotating_num]
            bias = tensors['bias'][i % rotating_num] if bias_size > 0 else None
            bias_stride = bias.stride(0) if bias_size > 0 else 0
            current_locks = locks[i]
            current_P = P[i]
            d = matmul_BM32_BN16_BK128_GM1_nW2_nS0_EU0_kP2_mfma16(a, b, c, bias, current_P, current_locks, M, N, K, num_sms, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1), bias_stride)

    if 'BM32_BN16_BK256_GM1_nW1_nS0_EU0_kP2_mfma16' not in failed_configs:
        print(f"BM32_BN16_BK256_GM1_nW1_nS0_EU0_kP2_mfma16")
        rotating_num = tensors['rotating_num']
        locks = torch.zeros((200, num_sms), device = "cuda", dtype = torch.int32)
        P = torch.zeros((200, num_sms,  32*16), device="cuda", dtype=torch.float32)
        for i in range(200):
            a = tensors['input_a'][i % rotating_num]
            b = tensors['input_b'][i % rotating_num]
            c = tensors['output_c'][i % rotating_num]
            bias = tensors['bias'][i % rotating_num] if bias_size > 0 else None
            bias_stride = bias.stride(0) if bias_size > 0 else 0
            current_locks = locks[i]
            current_P = P[i]
            d = matmul_BM32_BN16_BK256_GM1_nW1_nS0_EU0_kP2_mfma16(a, b, c, bias, current_P, current_locks, M, N, K, num_sms, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1), bias_stride)

    if 'BM32_BN16_BK256_GM1_nW8_nS0_EU0_kP2_mfma16' not in failed_configs:
        print(f"BM32_BN16_BK256_GM1_nW8_nS0_EU0_kP2_mfma16")
        rotating_num = tensors['rotating_num']
        locks = torch.zeros((200, num_sms), device = "cuda", dtype = torch.int32)
        P = torch.zeros((200, num_sms,  32*16), device="cuda", dtype=torch.float32)
        for i in range(200):
            a = tensors['input_a'][i % rotating_num]
            b = tensors['input_b'][i % rotating_num]
            c = tensors['output_c'][i % rotating_num]
            bias = tensors['bias'][i % rotating_num] if bias_size > 0 else None
            bias_stride = bias.stride(0) if bias_size > 0 else 0
            current_locks = locks[i]
            current_P = P[i]
            d = matmul_BM32_BN16_BK256_GM1_nW8_nS0_EU0_kP2_mfma16(a, b, c, bias, current_P, current_locks, M, N, K, num_sms, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1), bias_stride)

    if 'BM32_BN32_BK16_GM4_nW2_nS0_EU0_kP2_mfma32' not in failed_configs:
        print(f"BM32_BN32_BK16_GM4_nW2_nS0_EU0_kP2_mfma32")
        rotating_num = tensors['rotating_num']
        locks = torch.zeros((200, num_sms), device = "cuda", dtype = torch.int32)
        P = torch.zeros((200, num_sms,  32*32), device="cuda", dtype=torch.float32)
        for i in range(200):
            a = tensors['input_a'][i % rotating_num]
            b = tensors['input_b'][i % rotating_num]
            c = tensors['output_c'][i % rotating_num]
            bias = tensors['bias'][i % rotating_num] if bias_size > 0 else None
            bias_stride = bias.stride(0) if bias_size > 0 else 0
            current_locks = locks[i]
            current_P = P[i]
            d = matmul_BM32_BN32_BK16_GM4_nW2_nS0_EU0_kP2_mfma32(a, b, c, bias, current_P, current_locks, M, N, K, num_sms, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1), bias_stride)

    if 'BM32_BN32_BK16_GM4_nW8_nS0_EU0_kP2_mfma32' not in failed_configs:
        print(f"BM32_BN32_BK16_GM4_nW8_nS0_EU0_kP2_mfma32")
        rotating_num = tensors['rotating_num']
        locks = torch.zeros((200, num_sms), device = "cuda", dtype = torch.int32)
        P = torch.zeros((200, num_sms,  32*32), device="cuda", dtype=torch.float32)
        for i in range(200):
            a = tensors['input_a'][i % rotating_num]
            b = tensors['input_b'][i % rotating_num]
            c = tensors['output_c'][i % rotating_num]
            bias = tensors['bias'][i % rotating_num] if bias_size > 0 else None
            bias_stride = bias.stride(0) if bias_size > 0 else 0
            current_locks = locks[i]
            current_P = P[i]
            d = matmul_BM32_BN32_BK16_GM4_nW8_nS0_EU0_kP2_mfma32(a, b, c, bias, current_P, current_locks, M, N, K, num_sms, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1), bias_stride)

    if 'BM32_BN32_BK32_GM1_nW2_nS0_EU0_kP2_mfma16' not in failed_configs:
        print(f"BM32_BN32_BK32_GM1_nW2_nS0_EU0_kP2_mfma16")
        rotating_num = tensors['rotating_num']
        locks = torch.zeros((200, num_sms), device = "cuda", dtype = torch.int32)
        P = torch.zeros((200, num_sms,  32*32), device="cuda", dtype=torch.float32)
        for i in range(200):
            a = tensors['input_a'][i % rotating_num]
            b = tensors['input_b'][i % rotating_num]
            c = tensors['output_c'][i % rotating_num]
            bias = tensors['bias'][i % rotating_num] if bias_size > 0 else None
            bias_stride = bias.stride(0) if bias_size > 0 else 0
            current_locks = locks[i]
            current_P = P[i]
            d = matmul_BM32_BN32_BK32_GM1_nW2_nS0_EU0_kP2_mfma16(a, b, c, bias, current_P, current_locks, M, N, K, num_sms, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1), bias_stride)

    if 'BM32_BN32_BK32_GM8_nW4_nS0_EU0_kP2_mfma16' not in failed_configs:
        print(f"BM32_BN32_BK32_GM8_nW4_nS0_EU0_kP2_mfma16")
        rotating_num = tensors['rotating_num']
        locks = torch.zeros((200, num_sms), device = "cuda", dtype = torch.int32)
        P = torch.zeros((200, num_sms,  32*32), device="cuda", dtype=torch.float32)
        for i in range(200):
            a = tensors['input_a'][i % rotating_num]
            b = tensors['input_b'][i % rotating_num]
            c = tensors['output_c'][i % rotating_num]
            bias = tensors['bias'][i % rotating_num] if bias_size > 0 else None
            bias_stride = bias.stride(0) if bias_size > 0 else 0
            current_locks = locks[i]
            current_P = P[i]
            d = matmul_BM32_BN32_BK32_GM8_nW4_nS0_EU0_kP2_mfma16(a, b, c, bias, current_P, current_locks, M, N, K, num_sms, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1), bias_stride)

    if 'BM32_BN32_BK64_GM1_nW1_nS0_EU0_kP2_mfma16' not in failed_configs:
        print(f"BM32_BN32_BK64_GM1_nW1_nS0_EU0_kP2_mfma16")
        rotating_num = tensors['rotating_num']
        locks = torch.zeros((200, num_sms), device = "cuda", dtype = torch.int32)
        P = torch.zeros((200, num_sms,  32*32), device="cuda", dtype=torch.float32)
        for i in range(200):
            a = tensors['input_a'][i % rotating_num]
            b = tensors['input_b'][i % rotating_num]
            c = tensors['output_c'][i % rotating_num]
            bias = tensors['bias'][i % rotating_num] if bias_size > 0 else None
            bias_stride = bias.stride(0) if bias_size > 0 else 0
            current_locks = locks[i]
            current_P = P[i]
            d = matmul_BM32_BN32_BK64_GM1_nW1_nS0_EU0_kP2_mfma16(a, b, c, bias, current_P, current_locks, M, N, K, num_sms, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1), bias_stride)

    if 'BM32_BN32_BK64_GM8_nW2_nS0_EU0_kP2_mfma16' not in failed_configs:
        print(f"BM32_BN32_BK64_GM8_nW2_nS0_EU0_kP2_mfma16")
        rotating_num = tensors['rotating_num']
        locks = torch.zeros((200, num_sms), device = "cuda", dtype = torch.int32)
        P = torch.zeros((200, num_sms,  32*32), device="cuda", dtype=torch.float32)
        for i in range(200):
            a = tensors['input_a'][i % rotating_num]
            b = tensors['input_b'][i % rotating_num]
            c = tensors['output_c'][i % rotating_num]
            bias = tensors['bias'][i % rotating_num] if bias_size > 0 else None
            bias_stride = bias.stride(0) if bias_size > 0 else 0
            current_locks = locks[i]
            current_P = P[i]
            d = matmul_BM32_BN32_BK64_GM8_nW2_nS0_EU0_kP2_mfma16(a, b, c, bias, current_P, current_locks, M, N, K, num_sms, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1), bias_stride)

    if 'BM32_BN32_BK64_GM1_nW8_nS0_EU0_kP2_mfma16' not in failed_configs:
        print(f"BM32_BN32_BK64_GM1_nW8_nS0_EU0_kP2_mfma16")
        rotating_num = tensors['rotating_num']
        locks = torch.zeros((200, num_sms), device = "cuda", dtype = torch.int32)
        P = torch.zeros((200, num_sms,  32*32), device="cuda", dtype=torch.float32)
        for i in range(200):
            a = tensors['input_a'][i % rotating_num]
            b = tensors['input_b'][i % rotating_num]
            c = tensors['output_c'][i % rotating_num]
            bias = tensors['bias'][i % rotating_num] if bias_size > 0 else None
            bias_stride = bias.stride(0) if bias_size > 0 else 0
            current_locks = locks[i]
            current_P = P[i]
            d = matmul_BM32_BN32_BK64_GM1_nW8_nS0_EU0_kP2_mfma16(a, b, c, bias, current_P, current_locks, M, N, K, num_sms, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1), bias_stride)

    if 'BM32_BN32_BK128_GM8_nW1_nS0_EU0_kP2_mfma16' not in failed_configs:
        print(f"BM32_BN32_BK128_GM8_nW1_nS0_EU0_kP2_mfma16")
        rotating_num = tensors['rotating_num']
        locks = torch.zeros((200, num_sms), device = "cuda", dtype = torch.int32)
        P = torch.zeros((200, num_sms,  32*32), device="cuda", dtype=torch.float32)
        for i in range(200):
            a = tensors['input_a'][i % rotating_num]
            b = tensors['input_b'][i % rotating_num]
            c = tensors['output_c'][i % rotating_num]
            bias = tensors['bias'][i % rotating_num] if bias_size > 0 else None
            bias_stride = bias.stride(0) if bias_size > 0 else 0
            current_locks = locks[i]
            current_P = P[i]
            d = matmul_BM32_BN32_BK128_GM8_nW1_nS0_EU0_kP2_mfma16(a, b, c, bias, current_P, current_locks, M, N, K, num_sms, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1), bias_stride)

    if 'BM32_BN32_BK128_GM1_nW4_nS0_EU0_kP2_mfma16' not in failed_configs:
        print(f"BM32_BN32_BK128_GM1_nW4_nS0_EU0_kP2_mfma16")
        rotating_num = tensors['rotating_num']
        locks = torch.zeros((200, num_sms), device = "cuda", dtype = torch.int32)
        P = torch.zeros((200, num_sms,  32*32), device="cuda", dtype=torch.float32)
        for i in range(200):
            a = tensors['input_a'][i % rotating_num]
            b = tensors['input_b'][i % rotating_num]
            c = tensors['output_c'][i % rotating_num]
            bias = tensors['bias'][i % rotating_num] if bias_size > 0 else None
            bias_stride = bias.stride(0) if bias_size > 0 else 0
            current_locks = locks[i]
            current_P = P[i]
            d = matmul_BM32_BN32_BK128_GM1_nW4_nS0_EU0_kP2_mfma16(a, b, c, bias, current_P, current_locks, M, N, K, num_sms, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1), bias_stride)

    if 'BM32_BN32_BK128_GM8_nW8_nS0_EU0_kP2_mfma16' not in failed_configs:
        print(f"BM32_BN32_BK128_GM8_nW8_nS0_EU0_kP2_mfma16")
        rotating_num = tensors['rotating_num']
        locks = torch.zeros((200, num_sms), device = "cuda", dtype = torch.int32)
        P = torch.zeros((200, num_sms,  32*32), device="cuda", dtype=torch.float32)
        for i in range(200):
            a = tensors['input_a'][i % rotating_num]
            b = tensors['input_b'][i % rotating_num]
            c = tensors['output_c'][i % rotating_num]
            bias = tensors['bias'][i % rotating_num] if bias_size > 0 else None
            bias_stride = bias.stride(0) if bias_size > 0 else 0
            current_locks = locks[i]
            current_P = P[i]
            d = matmul_BM32_BN32_BK128_GM8_nW8_nS0_EU0_kP2_mfma16(a, b, c, bias, current_P, current_locks, M, N, K, num_sms, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1), bias_stride)

    if 'BM32_BN32_BK256_GM1_nW2_nS0_EU0_kP2_mfma16' not in failed_configs:
        print(f"BM32_BN32_BK256_GM1_nW2_nS0_EU0_kP2_mfma16")
        rotating_num = tensors['rotating_num']
        locks = torch.zeros((200, num_sms), device = "cuda", dtype = torch.int32)
        P = torch.zeros((200, num_sms,  32*32), device="cuda", dtype=torch.float32)
        for i in range(200):
            a = tensors['input_a'][i % rotating_num]
            b = tensors['input_b'][i % rotating_num]
            c = tensors['output_c'][i % rotating_num]
            bias = tensors['bias'][i % rotating_num] if bias_size > 0 else None
            bias_stride = bias.stride(0) if bias_size > 0 else 0
            current_locks = locks[i]
            current_P = P[i]
            d = matmul_BM32_BN32_BK256_GM1_nW2_nS0_EU0_kP2_mfma16(a, b, c, bias, current_P, current_locks, M, N, K, num_sms, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1), bias_stride)

    if 'BM32_BN32_BK256_GM8_nW4_nS0_EU0_kP2_mfma16' not in failed_configs:
        print(f"BM32_BN32_BK256_GM8_nW4_nS0_EU0_kP2_mfma16")
        rotating_num = tensors['rotating_num']
        locks = torch.zeros((200, num_sms), device = "cuda", dtype = torch.int32)
        P = torch.zeros((200, num_sms,  32*32), device="cuda", dtype=torch.float32)
        for i in range(200):
            a = tensors['input_a'][i % rotating_num]
            b = tensors['input_b'][i % rotating_num]
            c = tensors['output_c'][i % rotating_num]
            bias = tensors['bias'][i % rotating_num] if bias_size > 0 else None
            bias_stride = bias.stride(0) if bias_size > 0 else 0
            current_locks = locks[i]
            current_P = P[i]
            d = matmul_BM32_BN32_BK256_GM8_nW4_nS0_EU0_kP2_mfma16(a, b, c, bias, current_P, current_locks, M, N, K, num_sms, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1), bias_stride)

    if 'BM32_BN64_BK16_GM1_nW1_nS0_EU0_kP1_mfma32' not in failed_configs:
        print(f"BM32_BN64_BK16_GM1_nW1_nS0_EU0_kP1_mfma32")
        rotating_num = tensors['rotating_num']
        locks = torch.zeros((200, num_sms), device = "cuda", dtype = torch.int32)
        P = torch.zeros((200, num_sms,  32*64), device="cuda", dtype=torch.float32)
        for i in range(200):
            a = tensors['input_a'][i % rotating_num]
            b = tensors['input_b'][i % rotating_num]
            c = tensors['output_c'][i % rotating_num]
            bias = tensors['bias'][i % rotating_num] if bias_size > 0 else None
            bias_stride = bias.stride(0) if bias_size > 0 else 0
            current_locks = locks[i]
            current_P = P[i]
            d = matmul_BM32_BN64_BK16_GM1_nW1_nS0_EU0_kP1_mfma32(a, b, c, bias, current_P, current_locks, M, N, K, num_sms, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1), bias_stride)

    if 'BM32_BN64_BK16_GM1_nW4_nS0_EU0_kP1_mfma32' not in failed_configs:
        print(f"BM32_BN64_BK16_GM1_nW4_nS0_EU0_kP1_mfma32")
        rotating_num = tensors['rotating_num']
        locks = torch.zeros((200, num_sms), device = "cuda", dtype = torch.int32)
        P = torch.zeros((200, num_sms,  32*64), device="cuda", dtype=torch.float32)
        for i in range(200):
            a = tensors['input_a'][i % rotating_num]
            b = tensors['input_b'][i % rotating_num]
            c = tensors['output_c'][i % rotating_num]
            bias = tensors['bias'][i % rotating_num] if bias_size > 0 else None
            bias_stride = bias.stride(0) if bias_size > 0 else 0
            current_locks = locks[i]
            current_P = P[i]
            d = matmul_BM32_BN64_BK16_GM1_nW4_nS0_EU0_kP1_mfma32(a, b, c, bias, current_P, current_locks, M, N, K, num_sms, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1), bias_stride)

    if 'BM32_BN64_BK32_GM1_nW1_nS0_EU0_kP2_mfma16' not in failed_configs:
        print(f"BM32_BN64_BK32_GM1_nW1_nS0_EU0_kP2_mfma16")
        rotating_num = tensors['rotating_num']
        locks = torch.zeros((200, num_sms), device = "cuda", dtype = torch.int32)
        P = torch.zeros((200, num_sms,  32*64), device="cuda", dtype=torch.float32)
        for i in range(200):
            a = tensors['input_a'][i % rotating_num]
            b = tensors['input_b'][i % rotating_num]
            c = tensors['output_c'][i % rotating_num]
            bias = tensors['bias'][i % rotating_num] if bias_size > 0 else None
            bias_stride = bias.stride(0) if bias_size > 0 else 0
            current_locks = locks[i]
            current_P = P[i]
            d = matmul_BM32_BN64_BK32_GM1_nW1_nS0_EU0_kP2_mfma16(a, b, c, bias, current_P, current_locks, M, N, K, num_sms, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1), bias_stride)

    if 'BM32_BN64_BK32_GM8_nW2_nS0_EU0_kP2_mfma16' not in failed_configs:
        print(f"BM32_BN64_BK32_GM8_nW2_nS0_EU0_kP2_mfma16")
        rotating_num = tensors['rotating_num']
        locks = torch.zeros((200, num_sms), device = "cuda", dtype = torch.int32)
        P = torch.zeros((200, num_sms,  32*64), device="cuda", dtype=torch.float32)
        for i in range(200):
            a = tensors['input_a'][i % rotating_num]
            b = tensors['input_b'][i % rotating_num]
            c = tensors['output_c'][i % rotating_num]
            bias = tensors['bias'][i % rotating_num] if bias_size > 0 else None
            bias_stride = bias.stride(0) if bias_size > 0 else 0
            current_locks = locks[i]
            current_P = P[i]
            d = matmul_BM32_BN64_BK32_GM8_nW2_nS0_EU0_kP2_mfma16(a, b, c, bias, current_P, current_locks, M, N, K, num_sms, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1), bias_stride)

    if 'BM32_BN64_BK32_GM1_nW8_nS0_EU0_kP2_mfma16' not in failed_configs:
        print(f"BM32_BN64_BK32_GM1_nW8_nS0_EU0_kP2_mfma16")
        rotating_num = tensors['rotating_num']
        locks = torch.zeros((200, num_sms), device = "cuda", dtype = torch.int32)
        P = torch.zeros((200, num_sms,  32*64), device="cuda", dtype=torch.float32)
        for i in range(200):
            a = tensors['input_a'][i % rotating_num]
            b = tensors['input_b'][i % rotating_num]
            c = tensors['output_c'][i % rotating_num]
            bias = tensors['bias'][i % rotating_num] if bias_size > 0 else None
            bias_stride = bias.stride(0) if bias_size > 0 else 0
            current_locks = locks[i]
            current_P = P[i]
            d = matmul_BM32_BN64_BK32_GM1_nW8_nS0_EU0_kP2_mfma16(a, b, c, bias, current_P, current_locks, M, N, K, num_sms, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1), bias_stride)

    if 'BM32_BN64_BK64_GM8_nW1_nS0_EU0_kP2_mfma16' not in failed_configs:
        print(f"BM32_BN64_BK64_GM8_nW1_nS0_EU0_kP2_mfma16")
        rotating_num = tensors['rotating_num']
        locks = torch.zeros((200, num_sms), device = "cuda", dtype = torch.int32)
        P = torch.zeros((200, num_sms,  32*64), device="cuda", dtype=torch.float32)
        for i in range(200):
            a = tensors['input_a'][i % rotating_num]
            b = tensors['input_b'][i % rotating_num]
            c = tensors['output_c'][i % rotating_num]
            bias = tensors['bias'][i % rotating_num] if bias_size > 0 else None
            bias_stride = bias.stride(0) if bias_size > 0 else 0
            current_locks = locks[i]
            current_P = P[i]
            d = matmul_BM32_BN64_BK64_GM8_nW1_nS0_EU0_kP2_mfma16(a, b, c, bias, current_P, current_locks, M, N, K, num_sms, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1), bias_stride)

    if 'BM32_BN64_BK64_GM1_nW4_nS0_EU0_kP2_mfma16' not in failed_configs:
        print(f"BM32_BN64_BK64_GM1_nW4_nS0_EU0_kP2_mfma16")
        rotating_num = tensors['rotating_num']
        locks = torch.zeros((200, num_sms), device = "cuda", dtype = torch.int32)
        P = torch.zeros((200, num_sms,  32*64), device="cuda", dtype=torch.float32)
        for i in range(200):
            a = tensors['input_a'][i % rotating_num]
            b = tensors['input_b'][i % rotating_num]
            c = tensors['output_c'][i % rotating_num]
            bias = tensors['bias'][i % rotating_num] if bias_size > 0 else None
            bias_stride = bias.stride(0) if bias_size > 0 else 0
            current_locks = locks[i]
            current_P = P[i]
            d = matmul_BM32_BN64_BK64_GM1_nW4_nS0_EU0_kP2_mfma16(a, b, c, bias, current_P, current_locks, M, N, K, num_sms, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1), bias_stride)

    if 'BM32_BN64_BK64_GM8_nW8_nS0_EU0_kP2_mfma16' not in failed_configs:
        print(f"BM32_BN64_BK64_GM8_nW8_nS0_EU0_kP2_mfma16")
        rotating_num = tensors['rotating_num']
        locks = torch.zeros((200, num_sms), device = "cuda", dtype = torch.int32)
        P = torch.zeros((200, num_sms,  32*64), device="cuda", dtype=torch.float32)
        for i in range(200):
            a = tensors['input_a'][i % rotating_num]
            b = tensors['input_b'][i % rotating_num]
            c = tensors['output_c'][i % rotating_num]
            bias = tensors['bias'][i % rotating_num] if bias_size > 0 else None
            bias_stride = bias.stride(0) if bias_size > 0 else 0
            current_locks = locks[i]
            current_P = P[i]
            d = matmul_BM32_BN64_BK64_GM8_nW8_nS0_EU0_kP2_mfma16(a, b, c, bias, current_P, current_locks, M, N, K, num_sms, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1), bias_stride)

    if 'BM32_BN64_BK128_GM1_nW2_nS0_EU0_kP2_mfma16' not in failed_configs:
        print(f"BM32_BN64_BK128_GM1_nW2_nS0_EU0_kP2_mfma16")
        rotating_num = tensors['rotating_num']
        locks = torch.zeros((200, num_sms), device = "cuda", dtype = torch.int32)
        P = torch.zeros((200, num_sms,  32*64), device="cuda", dtype=torch.float32)
        for i in range(200):
            a = tensors['input_a'][i % rotating_num]
            b = tensors['input_b'][i % rotating_num]
            c = tensors['output_c'][i % rotating_num]
            bias = tensors['bias'][i % rotating_num] if bias_size > 0 else None
            bias_stride = bias.stride(0) if bias_size > 0 else 0
            current_locks = locks[i]
            current_P = P[i]
            d = matmul_BM32_BN64_BK128_GM1_nW2_nS0_EU0_kP2_mfma16(a, b, c, bias, current_P, current_locks, M, N, K, num_sms, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1), bias_stride)

    if 'BM32_BN64_BK128_GM8_nW4_nS0_EU0_kP2_mfma16' not in failed_configs:
        print(f"BM32_BN64_BK128_GM8_nW4_nS0_EU0_kP2_mfma16")
        rotating_num = tensors['rotating_num']
        locks = torch.zeros((200, num_sms), device = "cuda", dtype = torch.int32)
        P = torch.zeros((200, num_sms,  32*64), device="cuda", dtype=torch.float32)
        for i in range(200):
            a = tensors['input_a'][i % rotating_num]
            b = tensors['input_b'][i % rotating_num]
            c = tensors['output_c'][i % rotating_num]
            bias = tensors['bias'][i % rotating_num] if bias_size > 0 else None
            bias_stride = bias.stride(0) if bias_size > 0 else 0
            current_locks = locks[i]
            current_P = P[i]
            d = matmul_BM32_BN64_BK128_GM8_nW4_nS0_EU0_kP2_mfma16(a, b, c, bias, current_P, current_locks, M, N, K, num_sms, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1), bias_stride)

    if 'BM32_BN64_BK256_GM1_nW1_nS0_EU0_kP2_mfma16' not in failed_configs:
        print(f"BM32_BN64_BK256_GM1_nW1_nS0_EU0_kP2_mfma16")
        rotating_num = tensors['rotating_num']
        locks = torch.zeros((200, num_sms), device = "cuda", dtype = torch.int32)
        P = torch.zeros((200, num_sms,  32*64), device="cuda", dtype=torch.float32)
        for i in range(200):
            a = tensors['input_a'][i % rotating_num]
            b = tensors['input_b'][i % rotating_num]
            c = tensors['output_c'][i % rotating_num]
            bias = tensors['bias'][i % rotating_num] if bias_size > 0 else None
            bias_stride = bias.stride(0) if bias_size > 0 else 0
            current_locks = locks[i]
            current_P = P[i]
            d = matmul_BM32_BN64_BK256_GM1_nW1_nS0_EU0_kP2_mfma16(a, b, c, bias, current_P, current_locks, M, N, K, num_sms, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1), bias_stride)

    if 'BM32_BN64_BK256_GM8_nW2_nS0_EU0_kP2_mfma16' not in failed_configs:
        print(f"BM32_BN64_BK256_GM8_nW2_nS0_EU0_kP2_mfma16")
        rotating_num = tensors['rotating_num']
        locks = torch.zeros((200, num_sms), device = "cuda", dtype = torch.int32)
        P = torch.zeros((200, num_sms,  32*64), device="cuda", dtype=torch.float32)
        for i in range(200):
            a = tensors['input_a'][i % rotating_num]
            b = tensors['input_b'][i % rotating_num]
            c = tensors['output_c'][i % rotating_num]
            bias = tensors['bias'][i % rotating_num] if bias_size > 0 else None
            bias_stride = bias.stride(0) if bias_size > 0 else 0
            current_locks = locks[i]
            current_P = P[i]
            d = matmul_BM32_BN64_BK256_GM8_nW2_nS0_EU0_kP2_mfma16(a, b, c, bias, current_P, current_locks, M, N, K, num_sms, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1), bias_stride)

    if 'BM32_BN64_BK256_GM1_nW8_nS0_EU0_kP2_mfma16' not in failed_configs:
        print(f"BM32_BN64_BK256_GM1_nW8_nS0_EU0_kP2_mfma16")
        rotating_num = tensors['rotating_num']
        locks = torch.zeros((200, num_sms), device = "cuda", dtype = torch.int32)
        P = torch.zeros((200, num_sms,  32*64), device="cuda", dtype=torch.float32)
        for i in range(200):
            a = tensors['input_a'][i % rotating_num]
            b = tensors['input_b'][i % rotating_num]
            c = tensors['output_c'][i % rotating_num]
            bias = tensors['bias'][i % rotating_num] if bias_size > 0 else None
            bias_stride = bias.stride(0) if bias_size > 0 else 0
            current_locks = locks[i]
            current_P = P[i]
            d = matmul_BM32_BN64_BK256_GM1_nW8_nS0_EU0_kP2_mfma16(a, b, c, bias, current_P, current_locks, M, N, K, num_sms, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1), bias_stride)

    if 'BM32_BN128_BK16_GM16_nW1_nS0_EU0_kP1_mfma16' not in failed_configs:
        print(f"BM32_BN128_BK16_GM16_nW1_nS0_EU0_kP1_mfma16")
        rotating_num = tensors['rotating_num']
        locks = torch.zeros((200, num_sms), device = "cuda", dtype = torch.int32)
        P = torch.zeros((200, num_sms,  32*128), device="cuda", dtype=torch.float32)
        for i in range(200):
            a = tensors['input_a'][i % rotating_num]
            b = tensors['input_b'][i % rotating_num]
            c = tensors['output_c'][i % rotating_num]
            bias = tensors['bias'][i % rotating_num] if bias_size > 0 else None
            bias_stride = bias.stride(0) if bias_size > 0 else 0
            current_locks = locks[i]
            current_P = P[i]
            d = matmul_BM32_BN128_BK16_GM16_nW1_nS0_EU0_kP1_mfma16(a, b, c, bias, current_P, current_locks, M, N, K, num_sms, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1), bias_stride)

    if 'BM32_BN128_BK16_GM16_nW4_nS0_EU0_kP1_mfma16' not in failed_configs:
        print(f"BM32_BN128_BK16_GM16_nW4_nS0_EU0_kP1_mfma16")
        rotating_num = tensors['rotating_num']
        locks = torch.zeros((200, num_sms), device = "cuda", dtype = torch.int32)
        P = torch.zeros((200, num_sms,  32*128), device="cuda", dtype=torch.float32)
        for i in range(200):
            a = tensors['input_a'][i % rotating_num]
            b = tensors['input_b'][i % rotating_num]
            c = tensors['output_c'][i % rotating_num]
            bias = tensors['bias'][i % rotating_num] if bias_size > 0 else None
            bias_stride = bias.stride(0) if bias_size > 0 else 0
            current_locks = locks[i]
            current_P = P[i]
            d = matmul_BM32_BN128_BK16_GM16_nW4_nS0_EU0_kP1_mfma16(a, b, c, bias, current_P, current_locks, M, N, K, num_sms, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1), bias_stride)

    if 'BM32_BN128_BK32_GM8_nW1_nS0_EU0_kP2_mfma16' not in failed_configs:
        print(f"BM32_BN128_BK32_GM8_nW1_nS0_EU0_kP2_mfma16")
        rotating_num = tensors['rotating_num']
        locks = torch.zeros((200, num_sms), device = "cuda", dtype = torch.int32)
        P = torch.zeros((200, num_sms,  32*128), device="cuda", dtype=torch.float32)
        for i in range(200):
            a = tensors['input_a'][i % rotating_num]
            b = tensors['input_b'][i % rotating_num]
            c = tensors['output_c'][i % rotating_num]
            bias = tensors['bias'][i % rotating_num] if bias_size > 0 else None
            bias_stride = bias.stride(0) if bias_size > 0 else 0
            current_locks = locks[i]
            current_P = P[i]
            d = matmul_BM32_BN128_BK32_GM8_nW1_nS0_EU0_kP2_mfma16(a, b, c, bias, current_P, current_locks, M, N, K, num_sms, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1), bias_stride)

    if 'BM32_BN128_BK32_GM1_nW4_nS0_EU0_kP2_mfma16' not in failed_configs:
        print(f"BM32_BN128_BK32_GM1_nW4_nS0_EU0_kP2_mfma16")
        rotating_num = tensors['rotating_num']
        locks = torch.zeros((200, num_sms), device = "cuda", dtype = torch.int32)
        P = torch.zeros((200, num_sms,  32*128), device="cuda", dtype=torch.float32)
        for i in range(200):
            a = tensors['input_a'][i % rotating_num]
            b = tensors['input_b'][i % rotating_num]
            c = tensors['output_c'][i % rotating_num]
            bias = tensors['bias'][i % rotating_num] if bias_size > 0 else None
            bias_stride = bias.stride(0) if bias_size > 0 else 0
            current_locks = locks[i]
            current_P = P[i]
            d = matmul_BM32_BN128_BK32_GM1_nW4_nS0_EU0_kP2_mfma16(a, b, c, bias, current_P, current_locks, M, N, K, num_sms, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1), bias_stride)

    if 'BM32_BN128_BK32_GM8_nW8_nS0_EU0_kP2_mfma16' not in failed_configs:
        print(f"BM32_BN128_BK32_GM8_nW8_nS0_EU0_kP2_mfma16")
        rotating_num = tensors['rotating_num']
        locks = torch.zeros((200, num_sms), device = "cuda", dtype = torch.int32)
        P = torch.zeros((200, num_sms,  32*128), device="cuda", dtype=torch.float32)
        for i in range(200):
            a = tensors['input_a'][i % rotating_num]
            b = tensors['input_b'][i % rotating_num]
            c = tensors['output_c'][i % rotating_num]
            bias = tensors['bias'][i % rotating_num] if bias_size > 0 else None
            bias_stride = bias.stride(0) if bias_size > 0 else 0
            current_locks = locks[i]
            current_P = P[i]
            d = matmul_BM32_BN128_BK32_GM8_nW8_nS0_EU0_kP2_mfma16(a, b, c, bias, current_P, current_locks, M, N, K, num_sms, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1), bias_stride)

    if 'BM32_BN128_BK64_GM1_nW2_nS0_EU0_kP2_mfma16' not in failed_configs:
        print(f"BM32_BN128_BK64_GM1_nW2_nS0_EU0_kP2_mfma16")
        rotating_num = tensors['rotating_num']
        locks = torch.zeros((200, num_sms), device = "cuda", dtype = torch.int32)
        P = torch.zeros((200, num_sms,  32*128), device="cuda", dtype=torch.float32)
        for i in range(200):
            a = tensors['input_a'][i % rotating_num]
            b = tensors['input_b'][i % rotating_num]
            c = tensors['output_c'][i % rotating_num]
            bias = tensors['bias'][i % rotating_num] if bias_size > 0 else None
            bias_stride = bias.stride(0) if bias_size > 0 else 0
            current_locks = locks[i]
            current_P = P[i]
            d = matmul_BM32_BN128_BK64_GM1_nW2_nS0_EU0_kP2_mfma16(a, b, c, bias, current_P, current_locks, M, N, K, num_sms, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1), bias_stride)

    if 'BM32_BN128_BK64_GM8_nW4_nS0_EU0_kP2_mfma16' not in failed_configs:
        print(f"BM32_BN128_BK64_GM8_nW4_nS0_EU0_kP2_mfma16")
        rotating_num = tensors['rotating_num']
        locks = torch.zeros((200, num_sms), device = "cuda", dtype = torch.int32)
        P = torch.zeros((200, num_sms,  32*128), device="cuda", dtype=torch.float32)
        for i in range(200):
            a = tensors['input_a'][i % rotating_num]
            b = tensors['input_b'][i % rotating_num]
            c = tensors['output_c'][i % rotating_num]
            bias = tensors['bias'][i % rotating_num] if bias_size > 0 else None
            bias_stride = bias.stride(0) if bias_size > 0 else 0
            current_locks = locks[i]
            current_P = P[i]
            d = matmul_BM32_BN128_BK64_GM8_nW4_nS0_EU0_kP2_mfma16(a, b, c, bias, current_P, current_locks, M, N, K, num_sms, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1), bias_stride)

    if 'BM32_BN128_BK128_GM1_nW1_nS0_EU0_kP2_mfma16' not in failed_configs:
        print(f"BM32_BN128_BK128_GM1_nW1_nS0_EU0_kP2_mfma16")
        rotating_num = tensors['rotating_num']
        locks = torch.zeros((200, num_sms), device = "cuda", dtype = torch.int32)
        P = torch.zeros((200, num_sms,  32*128), device="cuda", dtype=torch.float32)
        for i in range(200):
            a = tensors['input_a'][i % rotating_num]
            b = tensors['input_b'][i % rotating_num]
            c = tensors['output_c'][i % rotating_num]
            bias = tensors['bias'][i % rotating_num] if bias_size > 0 else None
            bias_stride = bias.stride(0) if bias_size > 0 else 0
            current_locks = locks[i]
            current_P = P[i]
            d = matmul_BM32_BN128_BK128_GM1_nW1_nS0_EU0_kP2_mfma16(a, b, c, bias, current_P, current_locks, M, N, K, num_sms, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1), bias_stride)

    if 'BM32_BN128_BK128_GM8_nW2_nS0_EU0_kP2_mfma16' not in failed_configs:
        print(f"BM32_BN128_BK128_GM8_nW2_nS0_EU0_kP2_mfma16")
        rotating_num = tensors['rotating_num']
        locks = torch.zeros((200, num_sms), device = "cuda", dtype = torch.int32)
        P = torch.zeros((200, num_sms,  32*128), device="cuda", dtype=torch.float32)
        for i in range(200):
            a = tensors['input_a'][i % rotating_num]
            b = tensors['input_b'][i % rotating_num]
            c = tensors['output_c'][i % rotating_num]
            bias = tensors['bias'][i % rotating_num] if bias_size > 0 else None
            bias_stride = bias.stride(0) if bias_size > 0 else 0
            current_locks = locks[i]
            current_P = P[i]
            d = matmul_BM32_BN128_BK128_GM8_nW2_nS0_EU0_kP2_mfma16(a, b, c, bias, current_P, current_locks, M, N, K, num_sms, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1), bias_stride)

    if 'BM32_BN128_BK128_GM1_nW8_nS0_EU0_kP2_mfma16' not in failed_configs:
        print(f"BM32_BN128_BK128_GM1_nW8_nS0_EU0_kP2_mfma16")
        rotating_num = tensors['rotating_num']
        locks = torch.zeros((200, num_sms), device = "cuda", dtype = torch.int32)
        P = torch.zeros((200, num_sms,  32*128), device="cuda", dtype=torch.float32)
        for i in range(200):
            a = tensors['input_a'][i % rotating_num]
            b = tensors['input_b'][i % rotating_num]
            c = tensors['output_c'][i % rotating_num]
            bias = tensors['bias'][i % rotating_num] if bias_size > 0 else None
            bias_stride = bias.stride(0) if bias_size > 0 else 0
            current_locks = locks[i]
            current_P = P[i]
            d = matmul_BM32_BN128_BK128_GM1_nW8_nS0_EU0_kP2_mfma16(a, b, c, bias, current_P, current_locks, M, N, K, num_sms, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1), bias_stride)

    if 'BM32_BN256_BK16_GM16_nW1_nS0_EU0_kP1_mfma16' not in failed_configs:
        print(f"BM32_BN256_BK16_GM16_nW1_nS0_EU0_kP1_mfma16")
        rotating_num = tensors['rotating_num']
        locks = torch.zeros((200, num_sms), device = "cuda", dtype = torch.int32)
        P = torch.zeros((200, num_sms,  32*256), device="cuda", dtype=torch.float32)
        for i in range(200):
            a = tensors['input_a'][i % rotating_num]
            b = tensors['input_b'][i % rotating_num]
            c = tensors['output_c'][i % rotating_num]
            bias = tensors['bias'][i % rotating_num] if bias_size > 0 else None
            bias_stride = bias.stride(0) if bias_size > 0 else 0
            current_locks = locks[i]
            current_P = P[i]
            d = matmul_BM32_BN256_BK16_GM16_nW1_nS0_EU0_kP1_mfma16(a, b, c, bias, current_P, current_locks, M, N, K, num_sms, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1), bias_stride)

    if 'BM32_BN256_BK16_GM16_nW4_nS0_EU0_kP1_mfma16' not in failed_configs:
        print(f"BM32_BN256_BK16_GM16_nW4_nS0_EU0_kP1_mfma16")
        rotating_num = tensors['rotating_num']
        locks = torch.zeros((200, num_sms), device = "cuda", dtype = torch.int32)
        P = torch.zeros((200, num_sms,  32*256), device="cuda", dtype=torch.float32)
        for i in range(200):
            a = tensors['input_a'][i % rotating_num]
            b = tensors['input_b'][i % rotating_num]
            c = tensors['output_c'][i % rotating_num]
            bias = tensors['bias'][i % rotating_num] if bias_size > 0 else None
            bias_stride = bias.stride(0) if bias_size > 0 else 0
            current_locks = locks[i]
            current_P = P[i]
            d = matmul_BM32_BN256_BK16_GM16_nW4_nS0_EU0_kP1_mfma16(a, b, c, bias, current_P, current_locks, M, N, K, num_sms, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1), bias_stride)

    if 'BM32_BN256_BK32_GM8_nW1_nS0_EU0_kP2_mfma16' not in failed_configs:
        print(f"BM32_BN256_BK32_GM8_nW1_nS0_EU0_kP2_mfma16")
        rotating_num = tensors['rotating_num']
        locks = torch.zeros((200, num_sms), device = "cuda", dtype = torch.int32)
        P = torch.zeros((200, num_sms,  32*256), device="cuda", dtype=torch.float32)
        for i in range(200):
            a = tensors['input_a'][i % rotating_num]
            b = tensors['input_b'][i % rotating_num]
            c = tensors['output_c'][i % rotating_num]
            bias = tensors['bias'][i % rotating_num] if bias_size > 0 else None
            bias_stride = bias.stride(0) if bias_size > 0 else 0
            current_locks = locks[i]
            current_P = P[i]
            d = matmul_BM32_BN256_BK32_GM8_nW1_nS0_EU0_kP2_mfma16(a, b, c, bias, current_P, current_locks, M, N, K, num_sms, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1), bias_stride)

    if 'BM32_BN256_BK32_GM1_nW4_nS0_EU0_kP2_mfma16' not in failed_configs:
        print(f"BM32_BN256_BK32_GM1_nW4_nS0_EU0_kP2_mfma16")
        rotating_num = tensors['rotating_num']
        locks = torch.zeros((200, num_sms), device = "cuda", dtype = torch.int32)
        P = torch.zeros((200, num_sms,  32*256), device="cuda", dtype=torch.float32)
        for i in range(200):
            a = tensors['input_a'][i % rotating_num]
            b = tensors['input_b'][i % rotating_num]
            c = tensors['output_c'][i % rotating_num]
            bias = tensors['bias'][i % rotating_num] if bias_size > 0 else None
            bias_stride = bias.stride(0) if bias_size > 0 else 0
            current_locks = locks[i]
            current_P = P[i]
            d = matmul_BM32_BN256_BK32_GM1_nW4_nS0_EU0_kP2_mfma16(a, b, c, bias, current_P, current_locks, M, N, K, num_sms, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1), bias_stride)

    if 'BM32_BN256_BK32_GM8_nW8_nS0_EU0_kP2_mfma16' not in failed_configs:
        print(f"BM32_BN256_BK32_GM8_nW8_nS0_EU0_kP2_mfma16")
        rotating_num = tensors['rotating_num']
        locks = torch.zeros((200, num_sms), device = "cuda", dtype = torch.int32)
        P = torch.zeros((200, num_sms,  32*256), device="cuda", dtype=torch.float32)
        for i in range(200):
            a = tensors['input_a'][i % rotating_num]
            b = tensors['input_b'][i % rotating_num]
            c = tensors['output_c'][i % rotating_num]
            bias = tensors['bias'][i % rotating_num] if bias_size > 0 else None
            bias_stride = bias.stride(0) if bias_size > 0 else 0
            current_locks = locks[i]
            current_P = P[i]
            d = matmul_BM32_BN256_BK32_GM8_nW8_nS0_EU0_kP2_mfma16(a, b, c, bias, current_P, current_locks, M, N, K, num_sms, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1), bias_stride)

    if 'BM32_BN256_BK64_GM1_nW2_nS0_EU0_kP2_mfma16' not in failed_configs:
        print(f"BM32_BN256_BK64_GM1_nW2_nS0_EU0_kP2_mfma16")
        rotating_num = tensors['rotating_num']
        locks = torch.zeros((200, num_sms), device = "cuda", dtype = torch.int32)
        P = torch.zeros((200, num_sms,  32*256), device="cuda", dtype=torch.float32)
        for i in range(200):
            a = tensors['input_a'][i % rotating_num]
            b = tensors['input_b'][i % rotating_num]
            c = tensors['output_c'][i % rotating_num]
            bias = tensors['bias'][i % rotating_num] if bias_size > 0 else None
            bias_stride = bias.stride(0) if bias_size > 0 else 0
            current_locks = locks[i]
            current_P = P[i]
            d = matmul_BM32_BN256_BK64_GM1_nW2_nS0_EU0_kP2_mfma16(a, b, c, bias, current_P, current_locks, M, N, K, num_sms, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1), bias_stride)

    if 'BM32_BN256_BK64_GM8_nW4_nS0_EU0_kP2_mfma16' not in failed_configs:
        print(f"BM32_BN256_BK64_GM8_nW4_nS0_EU0_kP2_mfma16")
        rotating_num = tensors['rotating_num']
        locks = torch.zeros((200, num_sms), device = "cuda", dtype = torch.int32)
        P = torch.zeros((200, num_sms,  32*256), device="cuda", dtype=torch.float32)
        for i in range(200):
            a = tensors['input_a'][i % rotating_num]
            b = tensors['input_b'][i % rotating_num]
            c = tensors['output_c'][i % rotating_num]
            bias = tensors['bias'][i % rotating_num] if bias_size > 0 else None
            bias_stride = bias.stride(0) if bias_size > 0 else 0
            current_locks = locks[i]
            current_P = P[i]
            d = matmul_BM32_BN256_BK64_GM8_nW4_nS0_EU0_kP2_mfma16(a, b, c, bias, current_P, current_locks, M, N, K, num_sms, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1), bias_stride)

    if 'BM64_BN16_BK16_GM4_nW1_nS0_EU0_kP1_mfma16' not in failed_configs:
        print(f"BM64_BN16_BK16_GM4_nW1_nS0_EU0_kP1_mfma16")
        rotating_num = tensors['rotating_num']
        locks = torch.zeros((200, num_sms), device = "cuda", dtype = torch.int32)
        P = torch.zeros((200, num_sms,  64*16), device="cuda", dtype=torch.float32)
        for i in range(200):
            a = tensors['input_a'][i % rotating_num]
            b = tensors['input_b'][i % rotating_num]
            c = tensors['output_c'][i % rotating_num]
            bias = tensors['bias'][i % rotating_num] if bias_size > 0 else None
            bias_stride = bias.stride(0) if bias_size > 0 else 0
            current_locks = locks[i]
            current_P = P[i]
            d = matmul_BM64_BN16_BK16_GM4_nW1_nS0_EU0_kP1_mfma16(a, b, c, bias, current_P, current_locks, M, N, K, num_sms, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1), bias_stride)

    if 'BM64_BN16_BK32_GM1_nW4_nS0_EU0_kP2_mfma16' not in failed_configs:
        print(f"BM64_BN16_BK32_GM1_nW4_nS0_EU0_kP2_mfma16")
        rotating_num = tensors['rotating_num']
        locks = torch.zeros((200, num_sms), device = "cuda", dtype = torch.int32)
        P = torch.zeros((200, num_sms,  64*16), device="cuda", dtype=torch.float32)
        for i in range(200):
            a = tensors['input_a'][i % rotating_num]
            b = tensors['input_b'][i % rotating_num]
            c = tensors['output_c'][i % rotating_num]
            bias = tensors['bias'][i % rotating_num] if bias_size > 0 else None
            bias_stride = bias.stride(0) if bias_size > 0 else 0
            current_locks = locks[i]
            current_P = P[i]
            d = matmul_BM64_BN16_BK32_GM1_nW4_nS0_EU0_kP2_mfma16(a, b, c, bias, current_P, current_locks, M, N, K, num_sms, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1), bias_stride)

    if 'BM64_BN16_BK64_GM1_nW4_nS0_EU0_kP2_mfma16' not in failed_configs:
        print(f"BM64_BN16_BK64_GM1_nW4_nS0_EU0_kP2_mfma16")
        rotating_num = tensors['rotating_num']
        locks = torch.zeros((200, num_sms), device = "cuda", dtype = torch.int32)
        P = torch.zeros((200, num_sms,  64*16), device="cuda", dtype=torch.float32)
        for i in range(200):
            a = tensors['input_a'][i % rotating_num]
            b = tensors['input_b'][i % rotating_num]
            c = tensors['output_c'][i % rotating_num]
            bias = tensors['bias'][i % rotating_num] if bias_size > 0 else None
            bias_stride = bias.stride(0) if bias_size > 0 else 0
            current_locks = locks[i]
            current_P = P[i]
            d = matmul_BM64_BN16_BK64_GM1_nW4_nS0_EU0_kP2_mfma16(a, b, c, bias, current_P, current_locks, M, N, K, num_sms, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1), bias_stride)

    if 'BM64_BN16_BK128_GM1_nW4_nS0_EU0_kP2_mfma16' not in failed_configs:
        print(f"BM64_BN16_BK128_GM1_nW4_nS0_EU0_kP2_mfma16")
        rotating_num = tensors['rotating_num']
        locks = torch.zeros((200, num_sms), device = "cuda", dtype = torch.int32)
        P = torch.zeros((200, num_sms,  64*16), device="cuda", dtype=torch.float32)
        for i in range(200):
            a = tensors['input_a'][i % rotating_num]
            b = tensors['input_b'][i % rotating_num]
            c = tensors['output_c'][i % rotating_num]
            bias = tensors['bias'][i % rotating_num] if bias_size > 0 else None
            bias_stride = bias.stride(0) if bias_size > 0 else 0
            current_locks = locks[i]
            current_P = P[i]
            d = matmul_BM64_BN16_BK128_GM1_nW4_nS0_EU0_kP2_mfma16(a, b, c, bias, current_P, current_locks, M, N, K, num_sms, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1), bias_stride)

    if 'BM64_BN16_BK256_GM1_nW4_nS0_EU0_kP2_mfma16' not in failed_configs:
        print(f"BM64_BN16_BK256_GM1_nW4_nS0_EU0_kP2_mfma16")
        rotating_num = tensors['rotating_num']
        locks = torch.zeros((200, num_sms), device = "cuda", dtype = torch.int32)
        P = torch.zeros((200, num_sms,  64*16), device="cuda", dtype=torch.float32)
        for i in range(200):
            a = tensors['input_a'][i % rotating_num]
            b = tensors['input_b'][i % rotating_num]
            c = tensors['output_c'][i % rotating_num]
            bias = tensors['bias'][i % rotating_num] if bias_size > 0 else None
            bias_stride = bias.stride(0) if bias_size > 0 else 0
            current_locks = locks[i]
            current_P = P[i]
            d = matmul_BM64_BN16_BK256_GM1_nW4_nS0_EU0_kP2_mfma16(a, b, c, bias, current_P, current_locks, M, N, K, num_sms, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1), bias_stride)

    if 'BM64_BN32_BK16_GM4_nW2_nS0_EU0_kP1_mfma32' not in failed_configs:
        print(f"BM64_BN32_BK16_GM4_nW2_nS0_EU0_kP1_mfma32")
        rotating_num = tensors['rotating_num']
        locks = torch.zeros((200, num_sms), device = "cuda", dtype = torch.int32)
        P = torch.zeros((200, num_sms,  64*32), device="cuda", dtype=torch.float32)
        for i in range(200):
            a = tensors['input_a'][i % rotating_num]
            b = tensors['input_b'][i % rotating_num]
            c = tensors['output_c'][i % rotating_num]
            bias = tensors['bias'][i % rotating_num] if bias_size > 0 else None
            bias_stride = bias.stride(0) if bias_size > 0 else 0
            current_locks = locks[i]
            current_P = P[i]
            d = matmul_BM64_BN32_BK16_GM4_nW2_nS0_EU0_kP1_mfma32(a, b, c, bias, current_P, current_locks, M, N, K, num_sms, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1), bias_stride)

    if 'BM64_BN32_BK32_GM1_nW1_nS0_EU0_kP2_mfma16' not in failed_configs:
        print(f"BM64_BN32_BK32_GM1_nW1_nS0_EU0_kP2_mfma16")
        rotating_num = tensors['rotating_num']
        locks = torch.zeros((200, num_sms), device = "cuda", dtype = torch.int32)
        P = torch.zeros((200, num_sms,  64*32), device="cuda", dtype=torch.float32)
        for i in range(200):
            a = tensors['input_a'][i % rotating_num]
            b = tensors['input_b'][i % rotating_num]
            c = tensors['output_c'][i % rotating_num]
            bias = tensors['bias'][i % rotating_num] if bias_size > 0 else None
            bias_stride = bias.stride(0) if bias_size > 0 else 0
            current_locks = locks[i]
            current_P = P[i]
            d = matmul_BM64_BN32_BK32_GM1_nW1_nS0_EU0_kP2_mfma16(a, b, c, bias, current_P, current_locks, M, N, K, num_sms, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1), bias_stride)

    if 'BM64_BN32_BK32_GM1_nW4_nS0_EU0_kP2_mfma16' not in failed_configs:
        print(f"BM64_BN32_BK32_GM1_nW4_nS0_EU0_kP2_mfma16")
        rotating_num = tensors['rotating_num']
        locks = torch.zeros((200, num_sms), device = "cuda", dtype = torch.int32)
        P = torch.zeros((200, num_sms,  64*32), device="cuda", dtype=torch.float32)
        for i in range(200):
            a = tensors['input_a'][i % rotating_num]
            b = tensors['input_b'][i % rotating_num]
            c = tensors['output_c'][i % rotating_num]
            bias = tensors['bias'][i % rotating_num] if bias_size > 0 else None
            bias_stride = bias.stride(0) if bias_size > 0 else 0
            current_locks = locks[i]
            current_P = P[i]
            d = matmul_BM64_BN32_BK32_GM1_nW4_nS0_EU0_kP2_mfma16(a, b, c, bias, current_P, current_locks, M, N, K, num_sms, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1), bias_stride)

    if 'BM64_BN32_BK64_GM1_nW1_nS0_EU0_kP2_mfma16' not in failed_configs:
        print(f"BM64_BN32_BK64_GM1_nW1_nS0_EU0_kP2_mfma16")
        rotating_num = tensors['rotating_num']
        locks = torch.zeros((200, num_sms), device = "cuda", dtype = torch.int32)
        P = torch.zeros((200, num_sms,  64*32), device="cuda", dtype=torch.float32)
        for i in range(200):
            a = tensors['input_a'][i % rotating_num]
            b = tensors['input_b'][i % rotating_num]
            c = tensors['output_c'][i % rotating_num]
            bias = tensors['bias'][i % rotating_num] if bias_size > 0 else None
            bias_stride = bias.stride(0) if bias_size > 0 else 0
            current_locks = locks[i]
            current_P = P[i]
            d = matmul_BM64_BN32_BK64_GM1_nW1_nS0_EU0_kP2_mfma16(a, b, c, bias, current_P, current_locks, M, N, K, num_sms, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1), bias_stride)

    if 'BM64_BN32_BK64_GM1_nW4_nS0_EU0_kP2_mfma16' not in failed_configs:
        print(f"BM64_BN32_BK64_GM1_nW4_nS0_EU0_kP2_mfma16")
        rotating_num = tensors['rotating_num']
        locks = torch.zeros((200, num_sms), device = "cuda", dtype = torch.int32)
        P = torch.zeros((200, num_sms,  64*32), device="cuda", dtype=torch.float32)
        for i in range(200):
            a = tensors['input_a'][i % rotating_num]
            b = tensors['input_b'][i % rotating_num]
            c = tensors['output_c'][i % rotating_num]
            bias = tensors['bias'][i % rotating_num] if bias_size > 0 else None
            bias_stride = bias.stride(0) if bias_size > 0 else 0
            current_locks = locks[i]
            current_P = P[i]
            d = matmul_BM64_BN32_BK64_GM1_nW4_nS0_EU0_kP2_mfma16(a, b, c, bias, current_P, current_locks, M, N, K, num_sms, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1), bias_stride)

    if 'BM64_BN32_BK128_GM1_nW1_nS0_EU0_kP2_mfma16' not in failed_configs:
        print(f"BM64_BN32_BK128_GM1_nW1_nS0_EU0_kP2_mfma16")
        rotating_num = tensors['rotating_num']
        locks = torch.zeros((200, num_sms), device = "cuda", dtype = torch.int32)
        P = torch.zeros((200, num_sms,  64*32), device="cuda", dtype=torch.float32)
        for i in range(200):
            a = tensors['input_a'][i % rotating_num]
            b = tensors['input_b'][i % rotating_num]
            c = tensors['output_c'][i % rotating_num]
            bias = tensors['bias'][i % rotating_num] if bias_size > 0 else None
            bias_stride = bias.stride(0) if bias_size > 0 else 0
            current_locks = locks[i]
            current_P = P[i]
            d = matmul_BM64_BN32_BK128_GM1_nW1_nS0_EU0_kP2_mfma16(a, b, c, bias, current_P, current_locks, M, N, K, num_sms, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1), bias_stride)

    if 'BM64_BN32_BK128_GM1_nW4_nS0_EU0_kP2_mfma16' not in failed_configs:
        print(f"BM64_BN32_BK128_GM1_nW4_nS0_EU0_kP2_mfma16")
        rotating_num = tensors['rotating_num']
        locks = torch.zeros((200, num_sms), device = "cuda", dtype = torch.int32)
        P = torch.zeros((200, num_sms,  64*32), device="cuda", dtype=torch.float32)
        for i in range(200):
            a = tensors['input_a'][i % rotating_num]
            b = tensors['input_b'][i % rotating_num]
            c = tensors['output_c'][i % rotating_num]
            bias = tensors['bias'][i % rotating_num] if bias_size > 0 else None
            bias_stride = bias.stride(0) if bias_size > 0 else 0
            current_locks = locks[i]
            current_P = P[i]
            d = matmul_BM64_BN32_BK128_GM1_nW4_nS0_EU0_kP2_mfma16(a, b, c, bias, current_P, current_locks, M, N, K, num_sms, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1), bias_stride)

    if 'BM64_BN32_BK256_GM1_nW1_nS0_EU0_kP2_mfma16' not in failed_configs:
        print(f"BM64_BN32_BK256_GM1_nW1_nS0_EU0_kP2_mfma16")
        rotating_num = tensors['rotating_num']
        locks = torch.zeros((200, num_sms), device = "cuda", dtype = torch.int32)
        P = torch.zeros((200, num_sms,  64*32), device="cuda", dtype=torch.float32)
        for i in range(200):
            a = tensors['input_a'][i % rotating_num]
            b = tensors['input_b'][i % rotating_num]
            c = tensors['output_c'][i % rotating_num]
            bias = tensors['bias'][i % rotating_num] if bias_size > 0 else None
            bias_stride = bias.stride(0) if bias_size > 0 else 0
            current_locks = locks[i]
            current_P = P[i]
            d = matmul_BM64_BN32_BK256_GM1_nW1_nS0_EU0_kP2_mfma16(a, b, c, bias, current_P, current_locks, M, N, K, num_sms, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1), bias_stride)

    if 'BM64_BN32_BK256_GM1_nW4_nS0_EU0_kP2_mfma16' not in failed_configs:
        print(f"BM64_BN32_BK256_GM1_nW4_nS0_EU0_kP2_mfma16")
        rotating_num = tensors['rotating_num']
        locks = torch.zeros((200, num_sms), device = "cuda", dtype = torch.int32)
        P = torch.zeros((200, num_sms,  64*32), device="cuda", dtype=torch.float32)
        for i in range(200):
            a = tensors['input_a'][i % rotating_num]
            b = tensors['input_b'][i % rotating_num]
            c = tensors['output_c'][i % rotating_num]
            bias = tensors['bias'][i % rotating_num] if bias_size > 0 else None
            bias_stride = bias.stride(0) if bias_size > 0 else 0
            current_locks = locks[i]
            current_P = P[i]
            d = matmul_BM64_BN32_BK256_GM1_nW4_nS0_EU0_kP2_mfma16(a, b, c, bias, current_P, current_locks, M, N, K, num_sms, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1), bias_stride)

    if 'BM64_BN64_BK16_GM1_nW1_nS0_EU0_kP1_mfma32' not in failed_configs:
        print(f"BM64_BN64_BK16_GM1_nW1_nS0_EU0_kP1_mfma32")
        rotating_num = tensors['rotating_num']
        locks = torch.zeros((200, num_sms), device = "cuda", dtype = torch.int32)
        P = torch.zeros((200, num_sms,  64*64), device="cuda", dtype=torch.float32)
        for i in range(200):
            a = tensors['input_a'][i % rotating_num]
            b = tensors['input_b'][i % rotating_num]
            c = tensors['output_c'][i % rotating_num]
            bias = tensors['bias'][i % rotating_num] if bias_size > 0 else None
            bias_stride = bias.stride(0) if bias_size > 0 else 0
            current_locks = locks[i]
            current_P = P[i]
            d = matmul_BM64_BN64_BK16_GM1_nW1_nS0_EU0_kP1_mfma32(a, b, c, bias, current_P, current_locks, M, N, K, num_sms, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1), bias_stride)

    if 'BM64_BN64_BK16_GM8_nW4_nS0_EU0_kP1_mfma32' not in failed_configs:
        print(f"BM64_BN64_BK16_GM8_nW4_nS0_EU0_kP1_mfma32")
        rotating_num = tensors['rotating_num']
        locks = torch.zeros((200, num_sms), device = "cuda", dtype = torch.int32)
        P = torch.zeros((200, num_sms,  64*64), device="cuda", dtype=torch.float32)
        for i in range(200):
            a = tensors['input_a'][i % rotating_num]
            b = tensors['input_b'][i % rotating_num]
            c = tensors['output_c'][i % rotating_num]
            bias = tensors['bias'][i % rotating_num] if bias_size > 0 else None
            bias_stride = bias.stride(0) if bias_size > 0 else 0
            current_locks = locks[i]
            current_P = P[i]
            d = matmul_BM64_BN64_BK16_GM8_nW4_nS0_EU0_kP1_mfma32(a, b, c, bias, current_P, current_locks, M, N, K, num_sms, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1), bias_stride)

    if 'BM64_BN64_BK32_GM1_nW2_nS0_EU0_kP2_mfma16' not in failed_configs:
        print(f"BM64_BN64_BK32_GM1_nW2_nS0_EU0_kP2_mfma16")
        rotating_num = tensors['rotating_num']
        locks = torch.zeros((200, num_sms), device = "cuda", dtype = torch.int32)
        P = torch.zeros((200, num_sms,  64*64), device="cuda", dtype=torch.float32)
        for i in range(200):
            a = tensors['input_a'][i % rotating_num]
            b = tensors['input_b'][i % rotating_num]
            c = tensors['output_c'][i % rotating_num]
            bias = tensors['bias'][i % rotating_num] if bias_size > 0 else None
            bias_stride = bias.stride(0) if bias_size > 0 else 0
            current_locks = locks[i]
            current_P = P[i]
            d = matmul_BM64_BN64_BK32_GM1_nW2_nS0_EU0_kP2_mfma16(a, b, c, bias, current_P, current_locks, M, N, K, num_sms, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1), bias_stride)

    if 'BM64_BN64_BK32_GM1_nW8_nS0_EU0_kP2_mfma16' not in failed_configs:
        print(f"BM64_BN64_BK32_GM1_nW8_nS0_EU0_kP2_mfma16")
        rotating_num = tensors['rotating_num']
        locks = torch.zeros((200, num_sms), device = "cuda", dtype = torch.int32)
        P = torch.zeros((200, num_sms,  64*64), device="cuda", dtype=torch.float32)
        for i in range(200):
            a = tensors['input_a'][i % rotating_num]
            b = tensors['input_b'][i % rotating_num]
            c = tensors['output_c'][i % rotating_num]
            bias = tensors['bias'][i % rotating_num] if bias_size > 0 else None
            bias_stride = bias.stride(0) if bias_size > 0 else 0
            current_locks = locks[i]
            current_P = P[i]
            d = matmul_BM64_BN64_BK32_GM1_nW8_nS0_EU0_kP2_mfma16(a, b, c, bias, current_P, current_locks, M, N, K, num_sms, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1), bias_stride)

    if 'BM64_BN64_BK64_GM1_nW2_nS0_EU0_kP2_mfma16' not in failed_configs:
        print(f"BM64_BN64_BK64_GM1_nW2_nS0_EU0_kP2_mfma16")
        rotating_num = tensors['rotating_num']
        locks = torch.zeros((200, num_sms), device = "cuda", dtype = torch.int32)
        P = torch.zeros((200, num_sms,  64*64), device="cuda", dtype=torch.float32)
        for i in range(200):
            a = tensors['input_a'][i % rotating_num]
            b = tensors['input_b'][i % rotating_num]
            c = tensors['output_c'][i % rotating_num]
            bias = tensors['bias'][i % rotating_num] if bias_size > 0 else None
            bias_stride = bias.stride(0) if bias_size > 0 else 0
            current_locks = locks[i]
            current_P = P[i]
            d = matmul_BM64_BN64_BK64_GM1_nW2_nS0_EU0_kP2_mfma16(a, b, c, bias, current_P, current_locks, M, N, K, num_sms, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1), bias_stride)

    if 'BM64_BN64_BK64_GM1_nW8_nS0_EU0_kP2_mfma16' not in failed_configs:
        print(f"BM64_BN64_BK64_GM1_nW8_nS0_EU0_kP2_mfma16")
        rotating_num = tensors['rotating_num']
        locks = torch.zeros((200, num_sms), device = "cuda", dtype = torch.int32)
        P = torch.zeros((200, num_sms,  64*64), device="cuda", dtype=torch.float32)
        for i in range(200):
            a = tensors['input_a'][i % rotating_num]
            b = tensors['input_b'][i % rotating_num]
            c = tensors['output_c'][i % rotating_num]
            bias = tensors['bias'][i % rotating_num] if bias_size > 0 else None
            bias_stride = bias.stride(0) if bias_size > 0 else 0
            current_locks = locks[i]
            current_P = P[i]
            d = matmul_BM64_BN64_BK64_GM1_nW8_nS0_EU0_kP2_mfma16(a, b, c, bias, current_P, current_locks, M, N, K, num_sms, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1), bias_stride)

    if 'BM64_BN64_BK128_GM1_nW2_nS0_EU0_kP2_mfma16' not in failed_configs:
        print(f"BM64_BN64_BK128_GM1_nW2_nS0_EU0_kP2_mfma16")
        rotating_num = tensors['rotating_num']
        locks = torch.zeros((200, num_sms), device = "cuda", dtype = torch.int32)
        P = torch.zeros((200, num_sms,  64*64), device="cuda", dtype=torch.float32)
        for i in range(200):
            a = tensors['input_a'][i % rotating_num]
            b = tensors['input_b'][i % rotating_num]
            c = tensors['output_c'][i % rotating_num]
            bias = tensors['bias'][i % rotating_num] if bias_size > 0 else None
            bias_stride = bias.stride(0) if bias_size > 0 else 0
            current_locks = locks[i]
            current_P = P[i]
            d = matmul_BM64_BN64_BK128_GM1_nW2_nS0_EU0_kP2_mfma16(a, b, c, bias, current_P, current_locks, M, N, K, num_sms, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1), bias_stride)

    if 'BM64_BN64_BK128_GM1_nW8_nS0_EU0_kP2_mfma16' not in failed_configs:
        print(f"BM64_BN64_BK128_GM1_nW8_nS0_EU0_kP2_mfma16")
        rotating_num = tensors['rotating_num']
        locks = torch.zeros((200, num_sms), device = "cuda", dtype = torch.int32)
        P = torch.zeros((200, num_sms,  64*64), device="cuda", dtype=torch.float32)
        for i in range(200):
            a = tensors['input_a'][i % rotating_num]
            b = tensors['input_b'][i % rotating_num]
            c = tensors['output_c'][i % rotating_num]
            bias = tensors['bias'][i % rotating_num] if bias_size > 0 else None
            bias_stride = bias.stride(0) if bias_size > 0 else 0
            current_locks = locks[i]
            current_P = P[i]
            d = matmul_BM64_BN64_BK128_GM1_nW8_nS0_EU0_kP2_mfma16(a, b, c, bias, current_P, current_locks, M, N, K, num_sms, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1), bias_stride)

    if 'BM64_BN64_BK256_GM1_nW2_nS0_EU0_kP2_mfma16' not in failed_configs:
        print(f"BM64_BN64_BK256_GM1_nW2_nS0_EU0_kP2_mfma16")
        rotating_num = tensors['rotating_num']
        locks = torch.zeros((200, num_sms), device = "cuda", dtype = torch.int32)
        P = torch.zeros((200, num_sms,  64*64), device="cuda", dtype=torch.float32)
        for i in range(200):
            a = tensors['input_a'][i % rotating_num]
            b = tensors['input_b'][i % rotating_num]
            c = tensors['output_c'][i % rotating_num]
            bias = tensors['bias'][i % rotating_num] if bias_size > 0 else None
            bias_stride = bias.stride(0) if bias_size > 0 else 0
            current_locks = locks[i]
            current_P = P[i]
            d = matmul_BM64_BN64_BK256_GM1_nW2_nS0_EU0_kP2_mfma16(a, b, c, bias, current_P, current_locks, M, N, K, num_sms, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1), bias_stride)

    if 'BM64_BN64_BK256_GM1_nW8_nS0_EU0_kP2_mfma16' not in failed_configs:
        print(f"BM64_BN64_BK256_GM1_nW8_nS0_EU0_kP2_mfma16")
        rotating_num = tensors['rotating_num']
        locks = torch.zeros((200, num_sms), device = "cuda", dtype = torch.int32)
        P = torch.zeros((200, num_sms,  64*64), device="cuda", dtype=torch.float32)
        for i in range(200):
            a = tensors['input_a'][i % rotating_num]
            b = tensors['input_b'][i % rotating_num]
            c = tensors['output_c'][i % rotating_num]
            bias = tensors['bias'][i % rotating_num] if bias_size > 0 else None
            bias_stride = bias.stride(0) if bias_size > 0 else 0
            current_locks = locks[i]
            current_P = P[i]
            d = matmul_BM64_BN64_BK256_GM1_nW8_nS0_EU0_kP2_mfma16(a, b, c, bias, current_P, current_locks, M, N, K, num_sms, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1), bias_stride)

    if 'BM64_BN128_BK16_GM4_nW2_nS0_EU0_kP1_mfma32' not in failed_configs:
        print(f"BM64_BN128_BK16_GM4_nW2_nS0_EU0_kP1_mfma32")
        rotating_num = tensors['rotating_num']
        locks = torch.zeros((200, num_sms), device = "cuda", dtype = torch.int32)
        P = torch.zeros((200, num_sms,  64*128), device="cuda", dtype=torch.float32)
        for i in range(200):
            a = tensors['input_a'][i % rotating_num]
            b = tensors['input_b'][i % rotating_num]
            c = tensors['output_c'][i % rotating_num]
            bias = tensors['bias'][i % rotating_num] if bias_size > 0 else None
            bias_stride = bias.stride(0) if bias_size > 0 else 0
            current_locks = locks[i]
            current_P = P[i]
            d = matmul_BM64_BN128_BK16_GM4_nW2_nS0_EU0_kP1_mfma32(a, b, c, bias, current_P, current_locks, M, N, K, num_sms, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1), bias_stride)

    if 'BM64_BN128_BK32_GM1_nW1_nS0_EU0_kP2_mfma16' not in failed_configs:
        print(f"BM64_BN128_BK32_GM1_nW1_nS0_EU0_kP2_mfma16")
        rotating_num = tensors['rotating_num']
        locks = torch.zeros((200, num_sms), device = "cuda", dtype = torch.int32)
        P = torch.zeros((200, num_sms,  64*128), device="cuda", dtype=torch.float32)
        for i in range(200):
            a = tensors['input_a'][i % rotating_num]
            b = tensors['input_b'][i % rotating_num]
            c = tensors['output_c'][i % rotating_num]
            bias = tensors['bias'][i % rotating_num] if bias_size > 0 else None
            bias_stride = bias.stride(0) if bias_size > 0 else 0
            current_locks = locks[i]
            current_P = P[i]
            d = matmul_BM64_BN128_BK32_GM1_nW1_nS0_EU0_kP2_mfma16(a, b, c, bias, current_P, current_locks, M, N, K, num_sms, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1), bias_stride)

    if 'BM64_BN128_BK32_GM1_nW4_nS0_EU0_kP2_mfma16' not in failed_configs:
        print(f"BM64_BN128_BK32_GM1_nW4_nS0_EU0_kP2_mfma16")
        rotating_num = tensors['rotating_num']
        locks = torch.zeros((200, num_sms), device = "cuda", dtype = torch.int32)
        P = torch.zeros((200, num_sms,  64*128), device="cuda", dtype=torch.float32)
        for i in range(200):
            a = tensors['input_a'][i % rotating_num]
            b = tensors['input_b'][i % rotating_num]
            c = tensors['output_c'][i % rotating_num]
            bias = tensors['bias'][i % rotating_num] if bias_size > 0 else None
            bias_stride = bias.stride(0) if bias_size > 0 else 0
            current_locks = locks[i]
            current_P = P[i]
            d = matmul_BM64_BN128_BK32_GM1_nW4_nS0_EU0_kP2_mfma16(a, b, c, bias, current_P, current_locks, M, N, K, num_sms, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1), bias_stride)

    if 'BM64_BN128_BK64_GM1_nW1_nS0_EU0_kP2_mfma16' not in failed_configs:
        print(f"BM64_BN128_BK64_GM1_nW1_nS0_EU0_kP2_mfma16")
        rotating_num = tensors['rotating_num']
        locks = torch.zeros((200, num_sms), device = "cuda", dtype = torch.int32)
        P = torch.zeros((200, num_sms,  64*128), device="cuda", dtype=torch.float32)
        for i in range(200):
            a = tensors['input_a'][i % rotating_num]
            b = tensors['input_b'][i % rotating_num]
            c = tensors['output_c'][i % rotating_num]
            bias = tensors['bias'][i % rotating_num] if bias_size > 0 else None
            bias_stride = bias.stride(0) if bias_size > 0 else 0
            current_locks = locks[i]
            current_P = P[i]
            d = matmul_BM64_BN128_BK64_GM1_nW1_nS0_EU0_kP2_mfma16(a, b, c, bias, current_P, current_locks, M, N, K, num_sms, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1), bias_stride)

    if 'BM64_BN128_BK64_GM1_nW4_nS0_EU0_kP2_mfma16' not in failed_configs:
        print(f"BM64_BN128_BK64_GM1_nW4_nS0_EU0_kP2_mfma16")
        rotating_num = tensors['rotating_num']
        locks = torch.zeros((200, num_sms), device = "cuda", dtype = torch.int32)
        P = torch.zeros((200, num_sms,  64*128), device="cuda", dtype=torch.float32)
        for i in range(200):
            a = tensors['input_a'][i % rotating_num]
            b = tensors['input_b'][i % rotating_num]
            c = tensors['output_c'][i % rotating_num]
            bias = tensors['bias'][i % rotating_num] if bias_size > 0 else None
            bias_stride = bias.stride(0) if bias_size > 0 else 0
            current_locks = locks[i]
            current_P = P[i]
            d = matmul_BM64_BN128_BK64_GM1_nW4_nS0_EU0_kP2_mfma16(a, b, c, bias, current_P, current_locks, M, N, K, num_sms, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1), bias_stride)

    if 'BM64_BN128_BK128_GM1_nW1_nS0_EU0_kP2_mfma16' not in failed_configs:
        print(f"BM64_BN128_BK128_GM1_nW1_nS0_EU0_kP2_mfma16")
        rotating_num = tensors['rotating_num']
        locks = torch.zeros((200, num_sms), device = "cuda", dtype = torch.int32)
        P = torch.zeros((200, num_sms,  64*128), device="cuda", dtype=torch.float32)
        for i in range(200):
            a = tensors['input_a'][i % rotating_num]
            b = tensors['input_b'][i % rotating_num]
            c = tensors['output_c'][i % rotating_num]
            bias = tensors['bias'][i % rotating_num] if bias_size > 0 else None
            bias_stride = bias.stride(0) if bias_size > 0 else 0
            current_locks = locks[i]
            current_P = P[i]
            d = matmul_BM64_BN128_BK128_GM1_nW1_nS0_EU0_kP2_mfma16(a, b, c, bias, current_P, current_locks, M, N, K, num_sms, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1), bias_stride)

    if 'BM64_BN128_BK128_GM1_nW4_nS0_EU0_kP2_mfma16' not in failed_configs:
        print(f"BM64_BN128_BK128_GM1_nW4_nS0_EU0_kP2_mfma16")
        rotating_num = tensors['rotating_num']
        locks = torch.zeros((200, num_sms), device = "cuda", dtype = torch.int32)
        P = torch.zeros((200, num_sms,  64*128), device="cuda", dtype=torch.float32)
        for i in range(200):
            a = tensors['input_a'][i % rotating_num]
            b = tensors['input_b'][i % rotating_num]
            c = tensors['output_c'][i % rotating_num]
            bias = tensors['bias'][i % rotating_num] if bias_size > 0 else None
            bias_stride = bias.stride(0) if bias_size > 0 else 0
            current_locks = locks[i]
            current_P = P[i]
            d = matmul_BM64_BN128_BK128_GM1_nW4_nS0_EU0_kP2_mfma16(a, b, c, bias, current_P, current_locks, M, N, K, num_sms, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1), bias_stride)

    if 'BM64_BN256_BK16_GM1_nW1_nS0_EU0_kP1_mfma32' not in failed_configs:
        print(f"BM64_BN256_BK16_GM1_nW1_nS0_EU0_kP1_mfma32")
        rotating_num = tensors['rotating_num']
        locks = torch.zeros((200, num_sms), device = "cuda", dtype = torch.int32)
        P = torch.zeros((200, num_sms,  64*256), device="cuda", dtype=torch.float32)
        for i in range(200):
            a = tensors['input_a'][i % rotating_num]
            b = tensors['input_b'][i % rotating_num]
            c = tensors['output_c'][i % rotating_num]
            bias = tensors['bias'][i % rotating_num] if bias_size > 0 else None
            bias_stride = bias.stride(0) if bias_size > 0 else 0
            current_locks = locks[i]
            current_P = P[i]
            d = matmul_BM64_BN256_BK16_GM1_nW1_nS0_EU0_kP1_mfma32(a, b, c, bias, current_P, current_locks, M, N, K, num_sms, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1), bias_stride)

    if 'BM64_BN256_BK16_GM8_nW4_nS0_EU0_kP1_mfma32' not in failed_configs:
        print(f"BM64_BN256_BK16_GM8_nW4_nS0_EU0_kP1_mfma32")
        rotating_num = tensors['rotating_num']
        locks = torch.zeros((200, num_sms), device = "cuda", dtype = torch.int32)
        P = torch.zeros((200, num_sms,  64*256), device="cuda", dtype=torch.float32)
        for i in range(200):
            a = tensors['input_a'][i % rotating_num]
            b = tensors['input_b'][i % rotating_num]
            c = tensors['output_c'][i % rotating_num]
            bias = tensors['bias'][i % rotating_num] if bias_size > 0 else None
            bias_stride = bias.stride(0) if bias_size > 0 else 0
            current_locks = locks[i]
            current_P = P[i]
            d = matmul_BM64_BN256_BK16_GM8_nW4_nS0_EU0_kP1_mfma32(a, b, c, bias, current_P, current_locks, M, N, K, num_sms, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1), bias_stride)

    if 'BM64_BN256_BK32_GM1_nW2_nS0_EU0_kP2_mfma16' not in failed_configs:
        print(f"BM64_BN256_BK32_GM1_nW2_nS0_EU0_kP2_mfma16")
        rotating_num = tensors['rotating_num']
        locks = torch.zeros((200, num_sms), device = "cuda", dtype = torch.int32)
        P = torch.zeros((200, num_sms,  64*256), device="cuda", dtype=torch.float32)
        for i in range(200):
            a = tensors['input_a'][i % rotating_num]
            b = tensors['input_b'][i % rotating_num]
            c = tensors['output_c'][i % rotating_num]
            bias = tensors['bias'][i % rotating_num] if bias_size > 0 else None
            bias_stride = bias.stride(0) if bias_size > 0 else 0
            current_locks = locks[i]
            current_P = P[i]
            d = matmul_BM64_BN256_BK32_GM1_nW2_nS0_EU0_kP2_mfma16(a, b, c, bias, current_P, current_locks, M, N, K, num_sms, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1), bias_stride)

    if 'BM64_BN256_BK32_GM1_nW8_nS0_EU0_kP2_mfma16' not in failed_configs:
        print(f"BM64_BN256_BK32_GM1_nW8_nS0_EU0_kP2_mfma16")
        rotating_num = tensors['rotating_num']
        locks = torch.zeros((200, num_sms), device = "cuda", dtype = torch.int32)
        P = torch.zeros((200, num_sms,  64*256), device="cuda", dtype=torch.float32)
        for i in range(200):
            a = tensors['input_a'][i % rotating_num]
            b = tensors['input_b'][i % rotating_num]
            c = tensors['output_c'][i % rotating_num]
            bias = tensors['bias'][i % rotating_num] if bias_size > 0 else None
            bias_stride = bias.stride(0) if bias_size > 0 else 0
            current_locks = locks[i]
            current_P = P[i]
            d = matmul_BM64_BN256_BK32_GM1_nW8_nS0_EU0_kP2_mfma16(a, b, c, bias, current_P, current_locks, M, N, K, num_sms, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1), bias_stride)

    if 'BM64_BN256_BK64_GM1_nW2_nS0_EU0_kP2_mfma16' not in failed_configs:
        print(f"BM64_BN256_BK64_GM1_nW2_nS0_EU0_kP2_mfma16")
        rotating_num = tensors['rotating_num']
        locks = torch.zeros((200, num_sms), device = "cuda", dtype = torch.int32)
        P = torch.zeros((200, num_sms,  64*256), device="cuda", dtype=torch.float32)
        for i in range(200):
            a = tensors['input_a'][i % rotating_num]
            b = tensors['input_b'][i % rotating_num]
            c = tensors['output_c'][i % rotating_num]
            bias = tensors['bias'][i % rotating_num] if bias_size > 0 else None
            bias_stride = bias.stride(0) if bias_size > 0 else 0
            current_locks = locks[i]
            current_P = P[i]
            d = matmul_BM64_BN256_BK64_GM1_nW2_nS0_EU0_kP2_mfma16(a, b, c, bias, current_P, current_locks, M, N, K, num_sms, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1), bias_stride)

    if 'BM64_BN256_BK64_GM1_nW8_nS0_EU0_kP2_mfma16' not in failed_configs:
        print(f"BM64_BN256_BK64_GM1_nW8_nS0_EU0_kP2_mfma16")
        rotating_num = tensors['rotating_num']
        locks = torch.zeros((200, num_sms), device = "cuda", dtype = torch.int32)
        P = torch.zeros((200, num_sms,  64*256), device="cuda", dtype=torch.float32)
        for i in range(200):
            a = tensors['input_a'][i % rotating_num]
            b = tensors['input_b'][i % rotating_num]
            c = tensors['output_c'][i % rotating_num]
            bias = tensors['bias'][i % rotating_num] if bias_size > 0 else None
            bias_stride = bias.stride(0) if bias_size > 0 else 0
            current_locks = locks[i]
            current_P = P[i]
            d = matmul_BM64_BN256_BK64_GM1_nW8_nS0_EU0_kP2_mfma16(a, b, c, bias, current_P, current_locks, M, N, K, num_sms, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1), bias_stride)

    if 'BM128_BN16_BK32_GM1_nW2_nS0_EU0_kP2_mfma16' not in failed_configs:
        print(f"BM128_BN16_BK32_GM1_nW2_nS0_EU0_kP2_mfma16")
        rotating_num = tensors['rotating_num']
        locks = torch.zeros((200, num_sms), device = "cuda", dtype = torch.int32)
        P = torch.zeros((200, num_sms,  128*16), device="cuda", dtype=torch.float32)
        for i in range(200):
            a = tensors['input_a'][i % rotating_num]
            b = tensors['input_b'][i % rotating_num]
            c = tensors['output_c'][i % rotating_num]
            bias = tensors['bias'][i % rotating_num] if bias_size > 0 else None
            bias_stride = bias.stride(0) if bias_size > 0 else 0
            current_locks = locks[i]
            current_P = P[i]
            d = matmul_BM128_BN16_BK32_GM1_nW2_nS0_EU0_kP2_mfma16(a, b, c, bias, current_P, current_locks, M, N, K, num_sms, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1), bias_stride)

    if 'BM128_BN16_BK64_GM1_nW8_nS0_EU0_kP2_mfma16' not in failed_configs:
        print(f"BM128_BN16_BK64_GM1_nW8_nS0_EU0_kP2_mfma16")
        rotating_num = tensors['rotating_num']
        locks = torch.zeros((200, num_sms), device = "cuda", dtype = torch.int32)
        P = torch.zeros((200, num_sms,  128*16), device="cuda", dtype=torch.float32)
        for i in range(200):
            a = tensors['input_a'][i % rotating_num]
            b = tensors['input_b'][i % rotating_num]
            c = tensors['output_c'][i % rotating_num]
            bias = tensors['bias'][i % rotating_num] if bias_size > 0 else None
            bias_stride = bias.stride(0) if bias_size > 0 else 0
            current_locks = locks[i]
            current_P = P[i]
            d = matmul_BM128_BN16_BK64_GM1_nW8_nS0_EU0_kP2_mfma16(a, b, c, bias, current_P, current_locks, M, N, K, num_sms, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1), bias_stride)

    if 'BM128_BN32_BK16_GM4_nW1_nS0_EU0_kP2_mfma32' not in failed_configs:
        print(f"BM128_BN32_BK16_GM4_nW1_nS0_EU0_kP2_mfma32")
        rotating_num = tensors['rotating_num']
        locks = torch.zeros((200, num_sms), device = "cuda", dtype = torch.int32)
        P = torch.zeros((200, num_sms,  128*32), device="cuda", dtype=torch.float32)
        for i in range(200):
            a = tensors['input_a'][i % rotating_num]
            b = tensors['input_b'][i % rotating_num]
            c = tensors['output_c'][i % rotating_num]
            bias = tensors['bias'][i % rotating_num] if bias_size > 0 else None
            bias_stride = bias.stride(0) if bias_size > 0 else 0
            current_locks = locks[i]
            current_P = P[i]
            d = matmul_BM128_BN32_BK16_GM4_nW1_nS0_EU0_kP2_mfma32(a, b, c, bias, current_P, current_locks, M, N, K, num_sms, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1), bias_stride)

    if 'BM128_BN32_BK32_GM4_nW1_nS0_EU0_kP2_mfma16' not in failed_configs:
        print(f"BM128_BN32_BK32_GM4_nW1_nS0_EU0_kP2_mfma16")
        rotating_num = tensors['rotating_num']
        locks = torch.zeros((200, num_sms), device = "cuda", dtype = torch.int32)
        P = torch.zeros((200, num_sms,  128*32), device="cuda", dtype=torch.float32)
        for i in range(200):
            a = tensors['input_a'][i % rotating_num]
            b = tensors['input_b'][i % rotating_num]
            c = tensors['output_c'][i % rotating_num]
            bias = tensors['bias'][i % rotating_num] if bias_size > 0 else None
            bias_stride = bias.stride(0) if bias_size > 0 else 0
            current_locks = locks[i]
            current_P = P[i]
            d = matmul_BM128_BN32_BK32_GM4_nW1_nS0_EU0_kP2_mfma16(a, b, c, bias, current_P, current_locks, M, N, K, num_sms, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1), bias_stride)

    if 'BM128_BN32_BK32_GM4_nW8_nS0_EU0_kP2_mfma16' not in failed_configs:
        print(f"BM128_BN32_BK32_GM4_nW8_nS0_EU0_kP2_mfma16")
        rotating_num = tensors['rotating_num']
        locks = torch.zeros((200, num_sms), device = "cuda", dtype = torch.int32)
        P = torch.zeros((200, num_sms,  128*32), device="cuda", dtype=torch.float32)
        for i in range(200):
            a = tensors['input_a'][i % rotating_num]
            b = tensors['input_b'][i % rotating_num]
            c = tensors['output_c'][i % rotating_num]
            bias = tensors['bias'][i % rotating_num] if bias_size > 0 else None
            bias_stride = bias.stride(0) if bias_size > 0 else 0
            current_locks = locks[i]
            current_P = P[i]
            d = matmul_BM128_BN32_BK32_GM4_nW8_nS0_EU0_kP2_mfma16(a, b, c, bias, current_P, current_locks, M, N, K, num_sms, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1), bias_stride)

    if 'BM128_BN32_BK64_GM4_nW4_nS0_EU0_kP2_mfma16' not in failed_configs:
        print(f"BM128_BN32_BK64_GM4_nW4_nS0_EU0_kP2_mfma16")
        rotating_num = tensors['rotating_num']
        locks = torch.zeros((200, num_sms), device = "cuda", dtype = torch.int32)
        P = torch.zeros((200, num_sms,  128*32), device="cuda", dtype=torch.float32)
        for i in range(200):
            a = tensors['input_a'][i % rotating_num]
            b = tensors['input_b'][i % rotating_num]
            c = tensors['output_c'][i % rotating_num]
            bias = tensors['bias'][i % rotating_num] if bias_size > 0 else None
            bias_stride = bias.stride(0) if bias_size > 0 else 0
            current_locks = locks[i]
            current_P = P[i]
            d = matmul_BM128_BN32_BK64_GM4_nW4_nS0_EU0_kP2_mfma16(a, b, c, bias, current_P, current_locks, M, N, K, num_sms, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1), bias_stride)

    if 'BM128_BN32_BK128_GM4_nW2_nS0_EU0_kP2_mfma16' not in failed_configs:
        print(f"BM128_BN32_BK128_GM4_nW2_nS0_EU0_kP2_mfma16")
        rotating_num = tensors['rotating_num']
        locks = torch.zeros((200, num_sms), device = "cuda", dtype = torch.int32)
        P = torch.zeros((200, num_sms,  128*32), device="cuda", dtype=torch.float32)
        for i in range(200):
            a = tensors['input_a'][i % rotating_num]
            b = tensors['input_b'][i % rotating_num]
            c = tensors['output_c'][i % rotating_num]
            bias = tensors['bias'][i % rotating_num] if bias_size > 0 else None
            bias_stride = bias.stride(0) if bias_size > 0 else 0
            current_locks = locks[i]
            current_P = P[i]
            d = matmul_BM128_BN32_BK128_GM4_nW2_nS0_EU0_kP2_mfma16(a, b, c, bias, current_P, current_locks, M, N, K, num_sms, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1), bias_stride)

    if 'BM128_BN64_BK16_GM4_nW1_nS0_EU0_kP2_mfma32' not in failed_configs:
        print(f"BM128_BN64_BK16_GM4_nW1_nS0_EU0_kP2_mfma32")
        rotating_num = tensors['rotating_num']
        locks = torch.zeros((200, num_sms), device = "cuda", dtype = torch.int32)
        P = torch.zeros((200, num_sms,  128*64), device="cuda", dtype=torch.float32)
        for i in range(200):
            a = tensors['input_a'][i % rotating_num]
            b = tensors['input_b'][i % rotating_num]
            c = tensors['output_c'][i % rotating_num]
            bias = tensors['bias'][i % rotating_num] if bias_size > 0 else None
            bias_stride = bias.stride(0) if bias_size > 0 else 0
            current_locks = locks[i]
            current_P = P[i]
            d = matmul_BM128_BN64_BK16_GM4_nW1_nS0_EU0_kP2_mfma32(a, b, c, bias, current_P, current_locks, M, N, K, num_sms, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1), bias_stride)

    if 'BM128_BN64_BK32_GM4_nW1_nS0_EU0_kP2_mfma16' not in failed_configs:
        print(f"BM128_BN64_BK32_GM4_nW1_nS0_EU0_kP2_mfma16")
        rotating_num = tensors['rotating_num']
        locks = torch.zeros((200, num_sms), device = "cuda", dtype = torch.int32)
        P = torch.zeros((200, num_sms,  128*64), device="cuda", dtype=torch.float32)
        for i in range(200):
            a = tensors['input_a'][i % rotating_num]
            b = tensors['input_b'][i % rotating_num]
            c = tensors['output_c'][i % rotating_num]
            bias = tensors['bias'][i % rotating_num] if bias_size > 0 else None
            bias_stride = bias.stride(0) if bias_size > 0 else 0
            current_locks = locks[i]
            current_P = P[i]
            d = matmul_BM128_BN64_BK32_GM4_nW1_nS0_EU0_kP2_mfma16(a, b, c, bias, current_P, current_locks, M, N, K, num_sms, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1), bias_stride)

    if 'BM128_BN64_BK32_GM4_nW8_nS0_EU0_kP2_mfma16' not in failed_configs:
        print(f"BM128_BN64_BK32_GM4_nW8_nS0_EU0_kP2_mfma16")
        rotating_num = tensors['rotating_num']
        locks = torch.zeros((200, num_sms), device = "cuda", dtype = torch.int32)
        P = torch.zeros((200, num_sms,  128*64), device="cuda", dtype=torch.float32)
        for i in range(200):
            a = tensors['input_a'][i % rotating_num]
            b = tensors['input_b'][i % rotating_num]
            c = tensors['output_c'][i % rotating_num]
            bias = tensors['bias'][i % rotating_num] if bias_size > 0 else None
            bias_stride = bias.stride(0) if bias_size > 0 else 0
            current_locks = locks[i]
            current_P = P[i]
            d = matmul_BM128_BN64_BK32_GM4_nW8_nS0_EU0_kP2_mfma16(a, b, c, bias, current_P, current_locks, M, N, K, num_sms, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1), bias_stride)

    if 'BM128_BN64_BK64_GM4_nW4_nS0_EU0_kP2_mfma16' not in failed_configs:
        print(f"BM128_BN64_BK64_GM4_nW4_nS0_EU0_kP2_mfma16")
        rotating_num = tensors['rotating_num']
        locks = torch.zeros((200, num_sms), device = "cuda", dtype = torch.int32)
        P = torch.zeros((200, num_sms,  128*64), device="cuda", dtype=torch.float32)
        for i in range(200):
            a = tensors['input_a'][i % rotating_num]
            b = tensors['input_b'][i % rotating_num]
            c = tensors['output_c'][i % rotating_num]
            bias = tensors['bias'][i % rotating_num] if bias_size > 0 else None
            bias_stride = bias.stride(0) if bias_size > 0 else 0
            current_locks = locks[i]
            current_P = P[i]
            d = matmul_BM128_BN64_BK64_GM4_nW4_nS0_EU0_kP2_mfma16(a, b, c, bias, current_P, current_locks, M, N, K, num_sms, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1), bias_stride)

    if 'BM128_BN64_BK128_GM4_nW2_nS0_EU0_kP2_mfma16' not in failed_configs:
        print(f"BM128_BN64_BK128_GM4_nW2_nS0_EU0_kP2_mfma16")
        rotating_num = tensors['rotating_num']
        locks = torch.zeros((200, num_sms), device = "cuda", dtype = torch.int32)
        P = torch.zeros((200, num_sms,  128*64), device="cuda", dtype=torch.float32)
        for i in range(200):
            a = tensors['input_a'][i % rotating_num]
            b = tensors['input_b'][i % rotating_num]
            c = tensors['output_c'][i % rotating_num]
            bias = tensors['bias'][i % rotating_num] if bias_size > 0 else None
            bias_stride = bias.stride(0) if bias_size > 0 else 0
            current_locks = locks[i]
            current_P = P[i]
            d = matmul_BM128_BN64_BK128_GM4_nW2_nS0_EU0_kP2_mfma16(a, b, c, bias, current_P, current_locks, M, N, K, num_sms, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1), bias_stride)

    if 'BM128_BN128_BK16_GM4_nW1_nS0_EU0_kP2_mfma32' not in failed_configs:
        print(f"BM128_BN128_BK16_GM4_nW1_nS0_EU0_kP2_mfma32")
        rotating_num = tensors['rotating_num']
        locks = torch.zeros((200, num_sms), device = "cuda", dtype = torch.int32)
        P = torch.zeros((200, num_sms,  128*128), device="cuda", dtype=torch.float32)
        for i in range(200):
            a = tensors['input_a'][i % rotating_num]
            b = tensors['input_b'][i % rotating_num]
            c = tensors['output_c'][i % rotating_num]
            bias = tensors['bias'][i % rotating_num] if bias_size > 0 else None
            bias_stride = bias.stride(0) if bias_size > 0 else 0
            current_locks = locks[i]
            current_P = P[i]
            d = matmul_BM128_BN128_BK16_GM4_nW1_nS0_EU0_kP2_mfma32(a, b, c, bias, current_P, current_locks, M, N, K, num_sms, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1), bias_stride)

    if 'BM128_BN128_BK32_GM4_nW1_nS0_EU0_kP2_mfma16' not in failed_configs:
        print(f"BM128_BN128_BK32_GM4_nW1_nS0_EU0_kP2_mfma16")
        rotating_num = tensors['rotating_num']
        locks = torch.zeros((200, num_sms), device = "cuda", dtype = torch.int32)
        P = torch.zeros((200, num_sms,  128*128), device="cuda", dtype=torch.float32)
        for i in range(200):
            a = tensors['input_a'][i % rotating_num]
            b = tensors['input_b'][i % rotating_num]
            c = tensors['output_c'][i % rotating_num]
            bias = tensors['bias'][i % rotating_num] if bias_size > 0 else None
            bias_stride = bias.stride(0) if bias_size > 0 else 0
            current_locks = locks[i]
            current_P = P[i]
            d = matmul_BM128_BN128_BK32_GM4_nW1_nS0_EU0_kP2_mfma16(a, b, c, bias, current_P, current_locks, M, N, K, num_sms, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1), bias_stride)

    if 'BM128_BN128_BK32_GM4_nW8_nS0_EU0_kP2_mfma16' not in failed_configs:
        print(f"BM128_BN128_BK32_GM4_nW8_nS0_EU0_kP2_mfma16")
        rotating_num = tensors['rotating_num']
        locks = torch.zeros((200, num_sms), device = "cuda", dtype = torch.int32)
        P = torch.zeros((200, num_sms,  128*128), device="cuda", dtype=torch.float32)
        for i in range(200):
            a = tensors['input_a'][i % rotating_num]
            b = tensors['input_b'][i % rotating_num]
            c = tensors['output_c'][i % rotating_num]
            bias = tensors['bias'][i % rotating_num] if bias_size > 0 else None
            bias_stride = bias.stride(0) if bias_size > 0 else 0
            current_locks = locks[i]
            current_P = P[i]
            d = matmul_BM128_BN128_BK32_GM4_nW8_nS0_EU0_kP2_mfma16(a, b, c, bias, current_P, current_locks, M, N, K, num_sms, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1), bias_stride)

    if 'BM128_BN128_BK64_GM4_nW4_nS0_EU0_kP2_mfma16' not in failed_configs:
        print(f"BM128_BN128_BK64_GM4_nW4_nS0_EU0_kP2_mfma16")
        rotating_num = tensors['rotating_num']
        locks = torch.zeros((200, num_sms), device = "cuda", dtype = torch.int32)
        P = torch.zeros((200, num_sms,  128*128), device="cuda", dtype=torch.float32)
        for i in range(200):
            a = tensors['input_a'][i % rotating_num]
            b = tensors['input_b'][i % rotating_num]
            c = tensors['output_c'][i % rotating_num]
            bias = tensors['bias'][i % rotating_num] if bias_size > 0 else None
            bias_stride = bias.stride(0) if bias_size > 0 else 0
            current_locks = locks[i]
            current_P = P[i]
            d = matmul_BM128_BN128_BK64_GM4_nW4_nS0_EU0_kP2_mfma16(a, b, c, bias, current_P, current_locks, M, N, K, num_sms, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1), bias_stride)

    if 'BM128_BN128_BK128_GM1_nW4_nS0_EU0_kP2_mfma16' not in failed_configs:
        print(f"BM128_BN128_BK128_GM1_nW4_nS0_EU0_kP2_mfma16")
        rotating_num = tensors['rotating_num']
        locks = torch.zeros((200, num_sms), device = "cuda", dtype = torch.int32)
        P = torch.zeros((200, num_sms,  128*128), device="cuda", dtype=torch.float32)
        for i in range(200):
            a = tensors['input_a'][i % rotating_num]
            b = tensors['input_b'][i % rotating_num]
            c = tensors['output_c'][i % rotating_num]
            bias = tensors['bias'][i % rotating_num] if bias_size > 0 else None
            bias_stride = bias.stride(0) if bias_size > 0 else 0
            current_locks = locks[i]
            current_P = P[i]
            d = matmul_BM128_BN128_BK128_GM1_nW4_nS0_EU0_kP2_mfma16(a, b, c, bias, current_P, current_locks, M, N, K, num_sms, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1), bias_stride)

    if 'BM128_BN256_BK16_GM1_nW4_nS0_EU0_kP1_mfma32' not in failed_configs:
        print(f"BM128_BN256_BK16_GM1_nW4_nS0_EU0_kP1_mfma32")
        rotating_num = tensors['rotating_num']
        locks = torch.zeros((200, num_sms), device = "cuda", dtype = torch.int32)
        P = torch.zeros((200, num_sms,  128*256), device="cuda", dtype=torch.float32)
        for i in range(200):
            a = tensors['input_a'][i % rotating_num]
            b = tensors['input_b'][i % rotating_num]
            c = tensors['output_c'][i % rotating_num]
            bias = tensors['bias'][i % rotating_num] if bias_size > 0 else None
            bias_stride = bias.stride(0) if bias_size > 0 else 0
            current_locks = locks[i]
            current_P = P[i]
            d = matmul_BM128_BN256_BK16_GM1_nW4_nS0_EU0_kP1_mfma32(a, b, c, bias, current_P, current_locks, M, N, K, num_sms, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1), bias_stride)

    if 'BM128_BN256_BK32_GM4_nW2_nS0_EU0_kP2_mfma16' not in failed_configs:
        print(f"BM128_BN256_BK32_GM4_nW2_nS0_EU0_kP2_mfma16")
        rotating_num = tensors['rotating_num']
        locks = torch.zeros((200, num_sms), device = "cuda", dtype = torch.int32)
        P = torch.zeros((200, num_sms,  128*256), device="cuda", dtype=torch.float32)
        for i in range(200):
            a = tensors['input_a'][i % rotating_num]
            b = tensors['input_b'][i % rotating_num]
            c = tensors['output_c'][i % rotating_num]
            bias = tensors['bias'][i % rotating_num] if bias_size > 0 else None
            bias_stride = bias.stride(0) if bias_size > 0 else 0
            current_locks = locks[i]
            current_P = P[i]
            d = matmul_BM128_BN256_BK32_GM4_nW2_nS0_EU0_kP2_mfma16(a, b, c, bias, current_P, current_locks, M, N, K, num_sms, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1), bias_stride)

    if 'BM128_BN256_BK64_GM4_nW1_nS0_EU0_kP2_mfma16' not in failed_configs:
        print(f"BM128_BN256_BK64_GM4_nW1_nS0_EU0_kP2_mfma16")
        rotating_num = tensors['rotating_num']
        locks = torch.zeros((200, num_sms), device = "cuda", dtype = torch.int32)
        P = torch.zeros((200, num_sms,  128*256), device="cuda", dtype=torch.float32)
        for i in range(200):
            a = tensors['input_a'][i % rotating_num]
            b = tensors['input_b'][i % rotating_num]
            c = tensors['output_c'][i % rotating_num]
            bias = tensors['bias'][i % rotating_num] if bias_size > 0 else None
            bias_stride = bias.stride(0) if bias_size > 0 else 0
            current_locks = locks[i]
            current_P = P[i]
            d = matmul_BM128_BN256_BK64_GM4_nW1_nS0_EU0_kP2_mfma16(a, b, c, bias, current_P, current_locks, M, N, K, num_sms, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1), bias_stride)

    if 'BM128_BN256_BK64_GM4_nW8_nS0_EU0_kP2_mfma16' not in failed_configs:
        print(f"BM128_BN256_BK64_GM4_nW8_nS0_EU0_kP2_mfma16")
        rotating_num = tensors['rotating_num']
        locks = torch.zeros((200, num_sms), device = "cuda", dtype = torch.int32)
        P = torch.zeros((200, num_sms,  128*256), device="cuda", dtype=torch.float32)
        for i in range(200):
            a = tensors['input_a'][i % rotating_num]
            b = tensors['input_b'][i % rotating_num]
            c = tensors['output_c'][i % rotating_num]
            bias = tensors['bias'][i % rotating_num] if bias_size > 0 else None
            bias_stride = bias.stride(0) if bias_size > 0 else 0
            current_locks = locks[i]
            current_P = P[i]
            d = matmul_BM128_BN256_BK64_GM4_nW8_nS0_EU0_kP2_mfma16(a, b, c, bias, current_P, current_locks, M, N, K, num_sms, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1), bias_stride)

    if 'BM256_BN32_BK16_GM1_nW1_nS0_EU0_kP1_mfma32' not in failed_configs:
        print(f"BM256_BN32_BK16_GM1_nW1_nS0_EU0_kP1_mfma32")
        rotating_num = tensors['rotating_num']
        locks = torch.zeros((200, num_sms), device = "cuda", dtype = torch.int32)
        P = torch.zeros((200, num_sms,  256*32), device="cuda", dtype=torch.float32)
        for i in range(200):
            a = tensors['input_a'][i % rotating_num]
            b = tensors['input_b'][i % rotating_num]
            c = tensors['output_c'][i % rotating_num]
            bias = tensors['bias'][i % rotating_num] if bias_size > 0 else None
            bias_stride = bias.stride(0) if bias_size > 0 else 0
            current_locks = locks[i]
            current_P = P[i]
            d = matmul_BM256_BN32_BK16_GM1_nW1_nS0_EU0_kP1_mfma32(a, b, c, bias, current_P, current_locks, M, N, K, num_sms, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1), bias_stride)

    if 'BM256_BN32_BK32_GM1_nW8_nS0_EU0_kP2_mfma16' not in failed_configs:
        print(f"BM256_BN32_BK32_GM1_nW8_nS0_EU0_kP2_mfma16")
        rotating_num = tensors['rotating_num']
        locks = torch.zeros((200, num_sms), device = "cuda", dtype = torch.int32)
        P = torch.zeros((200, num_sms,  256*32), device="cuda", dtype=torch.float32)
        for i in range(200):
            a = tensors['input_a'][i % rotating_num]
            b = tensors['input_b'][i % rotating_num]
            c = tensors['output_c'][i % rotating_num]
            bias = tensors['bias'][i % rotating_num] if bias_size > 0 else None
            bias_stride = bias.stride(0) if bias_size > 0 else 0
            current_locks = locks[i]
            current_P = P[i]
            d = matmul_BM256_BN32_BK32_GM1_nW8_nS0_EU0_kP2_mfma16(a, b, c, bias, current_P, current_locks, M, N, K, num_sms, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1), bias_stride)

    if 'BM256_BN64_BK16_GM1_nW2_nS0_EU0_kP2_mfma32' not in failed_configs:
        print(f"BM256_BN64_BK16_GM1_nW2_nS0_EU0_kP2_mfma32")
        rotating_num = tensors['rotating_num']
        locks = torch.zeros((200, num_sms), device = "cuda", dtype = torch.int32)
        P = torch.zeros((200, num_sms,  256*64), device="cuda", dtype=torch.float32)
        for i in range(200):
            a = tensors['input_a'][i % rotating_num]
            b = tensors['input_b'][i % rotating_num]
            c = tensors['output_c'][i % rotating_num]
            bias = tensors['bias'][i % rotating_num] if bias_size > 0 else None
            bias_stride = bias.stride(0) if bias_size > 0 else 0
            current_locks = locks[i]
            current_P = P[i]
            d = matmul_BM256_BN64_BK16_GM1_nW2_nS0_EU0_kP2_mfma32(a, b, c, bias, current_P, current_locks, M, N, K, num_sms, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1), bias_stride)

    if 'BM256_BN64_BK64_GM1_nW1_nS0_EU0_kP2_mfma16' not in failed_configs:
        print(f"BM256_BN64_BK64_GM1_nW1_nS0_EU0_kP2_mfma16")
        rotating_num = tensors['rotating_num']
        locks = torch.zeros((200, num_sms), device = "cuda", dtype = torch.int32)
        P = torch.zeros((200, num_sms,  256*64), device="cuda", dtype=torch.float32)
        for i in range(200):
            a = tensors['input_a'][i % rotating_num]
            b = tensors['input_b'][i % rotating_num]
            c = tensors['output_c'][i % rotating_num]
            bias = tensors['bias'][i % rotating_num] if bias_size > 0 else None
            bias_stride = bias.stride(0) if bias_size > 0 else 0
            current_locks = locks[i]
            current_P = P[i]
            d = matmul_BM256_BN64_BK64_GM1_nW1_nS0_EU0_kP2_mfma16(a, b, c, bias, current_P, current_locks, M, N, K, num_sms, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1), bias_stride)

    if 'BM256_BN128_BK16_GM1_nW8_nS0_EU0_kP1_mfma16' not in failed_configs:
        print(f"BM256_BN128_BK16_GM1_nW8_nS0_EU0_kP1_mfma16")
        rotating_num = tensors['rotating_num']
        locks = torch.zeros((200, num_sms), device = "cuda", dtype = torch.int32)
        P = torch.zeros((200, num_sms,  256*128), device="cuda", dtype=torch.float32)
        for i in range(200):
            a = tensors['input_a'][i % rotating_num]
            b = tensors['input_b'][i % rotating_num]
            c = tensors['output_c'][i % rotating_num]
            bias = tensors['bias'][i % rotating_num] if bias_size > 0 else None
            bias_stride = bias.stride(0) if bias_size > 0 else 0
            current_locks = locks[i]
            current_P = P[i]
            d = matmul_BM256_BN128_BK16_GM1_nW8_nS0_EU0_kP1_mfma16(a, b, c, bias, current_P, current_locks, M, N, K, num_sms, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1), bias_stride)

    if 'BM256_BN128_BK64_GM1_nW2_nS0_EU0_kP2_mfma16' not in failed_configs:
        print(f"BM256_BN128_BK64_GM1_nW2_nS0_EU0_kP2_mfma16")
        rotating_num = tensors['rotating_num']
        locks = torch.zeros((200, num_sms), device = "cuda", dtype = torch.int32)
        P = torch.zeros((200, num_sms,  256*128), device="cuda", dtype=torch.float32)
        for i in range(200):
            a = tensors['input_a'][i % rotating_num]
            b = tensors['input_b'][i % rotating_num]
            c = tensors['output_c'][i % rotating_num]
            bias = tensors['bias'][i % rotating_num] if bias_size > 0 else None
            bias_stride = bias.stride(0) if bias_size > 0 else 0
            current_locks = locks[i]
            current_P = P[i]
            d = matmul_BM256_BN128_BK64_GM1_nW2_nS0_EU0_kP2_mfma16(a, b, c, bias, current_P, current_locks, M, N, K, num_sms, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1), bias_stride)

    if 'BM256_BN256_BK32_GM1_nW1_nS0_EU0_kP2_mfma16' not in failed_configs:
        print(f"BM256_BN256_BK32_GM1_nW1_nS0_EU0_kP2_mfma16")
        rotating_num = tensors['rotating_num']
        locks = torch.zeros((200, num_sms), device = "cuda", dtype = torch.int32)
        P = torch.zeros((200, num_sms,  256*256), device="cuda", dtype=torch.float32)
        for i in range(200):
            a = tensors['input_a'][i % rotating_num]
            b = tensors['input_b'][i % rotating_num]
            c = tensors['output_c'][i % rotating_num]
            bias = tensors['bias'][i % rotating_num] if bias_size > 0 else None
            bias_stride = bias.stride(0) if bias_size > 0 else 0
            current_locks = locks[i]
            current_P = P[i]
            d = matmul_BM256_BN256_BK32_GM1_nW1_nS0_EU0_kP2_mfma16(a, b, c, bias, current_P, current_locks, M, N, K, num_sms, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1), bias_stride)

    if 'BM256_BN256_BK64_GM1_nW4_nS0_EU0_kP2_mfma16' not in failed_configs:
        print(f"BM256_BN256_BK64_GM1_nW4_nS0_EU0_kP2_mfma16")
        rotating_num = tensors['rotating_num']
        locks = torch.zeros((200, num_sms), device = "cuda", dtype = torch.int32)
        P = torch.zeros((200, num_sms,  256*256), device="cuda", dtype=torch.float32)
        for i in range(200):
            a = tensors['input_a'][i % rotating_num]
            b = tensors['input_b'][i % rotating_num]
            c = tensors['output_c'][i % rotating_num]
            bias = tensors['bias'][i % rotating_num] if bias_size > 0 else None
            bias_stride = bias.stride(0) if bias_size > 0 else 0
            current_locks = locks[i]
            current_P = P[i]
            d = matmul_BM256_BN256_BK64_GM1_nW4_nS0_EU0_kP2_mfma16(a, b, c, bias, current_P, current_locks, M, N, K, num_sms, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1), bias_stride)
    return d

def main():
    parser = argparse.ArgumentParser(
        prog="tune a specific gemm size",
        allow_abbrev=False,)
    parser.add_argument("-n", type=int, default=1, help='number of threads')
    parser.add_argument("-rotating_tensor", type=int, default=0, help='size of rotating buffer (MB), default: 0')
    args = parser.parse_args()
    numThreads = args.n
    rotating_buffer_size = args.rotating_tensor
    num_sms = 304
    test_gemm(512, 512, 512, num_sms, rotating_buffer_size, 0)

if __name__ == '__main__':
   sys.exit(main())