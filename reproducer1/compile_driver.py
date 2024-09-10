import torch
import triton
import triton.language as tl
import argparse
import sys
import multiprocessing
from tune_streamk import gen_rotating_tensors
from myKernels import *


def matmul_BM64_BN64_BK64_GM1_nW4_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM64_BN64_BK64_GM1_nW4_nS2_EU0_kP1_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 1,
        num_warps = 4,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 1,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM64_BN64_BK64_GM1_nW4_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM64_BN64_BK64_GM1_nW4_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM64_BN64_BK64_GM1_nW4_nS2_EU0_kP1_mfma16: ', e, flush=True)
        return False


def matmul_BM64_BN64_BK64_GM1_nW4_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM64_BN64_BK64_GM1_nW4_nS2_EU0_kP2_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 1,
        num_warps = 4,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 2,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM64_BN64_BK64_GM1_nW4_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM64_BN64_BK64_GM1_nW4_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM64_BN64_BK64_GM1_nW4_nS2_EU0_kP2_mfma16: ', e, flush=True)
        return False


def matmul_BM64_BN64_BK64_GM1_nW4_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM64_BN64_BK64_GM1_nW4_nS2_EU0_kP1_mfma32.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 1,
        num_warps = 4,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 1,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM64_BN64_BK64_GM1_nW4_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM64_BN64_BK64_GM1_nW4_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM64_BN64_BK64_GM1_nW4_nS2_EU0_kP1_mfma32: ', e, flush=True)
        return False


def matmul_BM64_BN64_BK64_GM1_nW4_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM64_BN64_BK64_GM1_nW4_nS2_EU0_kP2_mfma32.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 1,
        num_warps = 4,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 2,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM64_BN64_BK64_GM1_nW4_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM64_BN64_BK64_GM1_nW4_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM64_BN64_BK64_GM1_nW4_nS2_EU0_kP2_mfma32: ', e, flush=True)
        return False


def matmul_BM64_BN64_BK64_GM4_nW4_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM64_BN64_BK64_GM4_nW4_nS2_EU0_kP1_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 4,
        num_warps = 4,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 1,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM64_BN64_BK64_GM4_nW4_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM64_BN64_BK64_GM4_nW4_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM64_BN64_BK64_GM4_nW4_nS2_EU0_kP1_mfma16: ', e, flush=True)
        return False


def matmul_BM64_BN64_BK64_GM4_nW4_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM64_BN64_BK64_GM4_nW4_nS2_EU0_kP2_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 4,
        num_warps = 4,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 2,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM64_BN64_BK64_GM4_nW4_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM64_BN64_BK64_GM4_nW4_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM64_BN64_BK64_GM4_nW4_nS2_EU0_kP2_mfma16: ', e, flush=True)
        return False


def matmul_BM64_BN64_BK64_GM4_nW4_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM64_BN64_BK64_GM4_nW4_nS2_EU0_kP1_mfma32.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 4,
        num_warps = 4,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 1,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM64_BN64_BK64_GM4_nW4_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM64_BN64_BK64_GM4_nW4_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM64_BN64_BK64_GM4_nW4_nS2_EU0_kP1_mfma32: ', e, flush=True)
        return False


def matmul_BM64_BN64_BK64_GM4_nW4_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM64_BN64_BK64_GM4_nW4_nS2_EU0_kP2_mfma32.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 4,
        num_warps = 4,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 2,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM64_BN64_BK64_GM4_nW4_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM64_BN64_BK64_GM4_nW4_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM64_BN64_BK64_GM4_nW4_nS2_EU0_kP2_mfma32: ', e, flush=True)
        return False


def matmul_BM64_BN64_BK64_GM8_nW4_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM64_BN64_BK64_GM8_nW4_nS2_EU0_kP1_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 8,
        num_warps = 4,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 1,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM64_BN64_BK64_GM8_nW4_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM64_BN64_BK64_GM8_nW4_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM64_BN64_BK64_GM8_nW4_nS2_EU0_kP1_mfma16: ', e, flush=True)
        return False


def matmul_BM64_BN64_BK64_GM8_nW4_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM64_BN64_BK64_GM8_nW4_nS2_EU0_kP2_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 8,
        num_warps = 4,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 2,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM64_BN64_BK64_GM8_nW4_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM64_BN64_BK64_GM8_nW4_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM64_BN64_BK64_GM8_nW4_nS2_EU0_kP2_mfma16: ', e, flush=True)
        return False


def matmul_BM64_BN64_BK64_GM8_nW4_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM64_BN64_BK64_GM8_nW4_nS2_EU0_kP1_mfma32.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 8,
        num_warps = 4,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 1,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM64_BN64_BK64_GM8_nW4_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM64_BN64_BK64_GM8_nW4_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM64_BN64_BK64_GM8_nW4_nS2_EU0_kP1_mfma32: ', e, flush=True)
        return False


def matmul_BM64_BN64_BK64_GM8_nW4_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM64_BN64_BK64_GM8_nW4_nS2_EU0_kP2_mfma32.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 8,
        num_warps = 4,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 2,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM64_BN64_BK64_GM8_nW4_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM64_BN64_BK64_GM8_nW4_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM64_BN64_BK64_GM8_nW4_nS2_EU0_kP2_mfma32: ', e, flush=True)
        return False


def matmul_BM64_BN64_BK64_GM16_nW4_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM64_BN64_BK64_GM16_nW4_nS2_EU0_kP1_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 16,
        num_warps = 4,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 1,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM64_BN64_BK64_GM16_nW4_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM64_BN64_BK64_GM16_nW4_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM64_BN64_BK64_GM16_nW4_nS2_EU0_kP1_mfma16: ', e, flush=True)
        return False


def matmul_BM64_BN64_BK64_GM16_nW4_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM64_BN64_BK64_GM16_nW4_nS2_EU0_kP2_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 16,
        num_warps = 4,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 2,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM64_BN64_BK64_GM16_nW4_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM64_BN64_BK64_GM16_nW4_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM64_BN64_BK64_GM16_nW4_nS2_EU0_kP2_mfma16: ', e, flush=True)
        return False


def matmul_BM64_BN64_BK64_GM16_nW4_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM64_BN64_BK64_GM16_nW4_nS2_EU0_kP1_mfma32.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 16,
        num_warps = 4,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 1,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM64_BN64_BK64_GM16_nW4_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM64_BN64_BK64_GM16_nW4_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM64_BN64_BK64_GM16_nW4_nS2_EU0_kP1_mfma32: ', e, flush=True)
        return False


def matmul_BM64_BN64_BK64_GM16_nW4_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM64_BN64_BK64_GM16_nW4_nS2_EU0_kP2_mfma32.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 16,
        num_warps = 4,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 2,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM64_BN64_BK64_GM16_nW4_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM64_BN64_BK64_GM16_nW4_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM64_BN64_BK64_GM16_nW4_nS2_EU0_kP2_mfma32: ', e, flush=True)
        return False


def matmul_BM64_BN64_BK64_GM32_nW4_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM64_BN64_BK64_GM32_nW4_nS2_EU0_kP1_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 32,
        num_warps = 4,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 1,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM64_BN64_BK64_GM32_nW4_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM64_BN64_BK64_GM32_nW4_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM64_BN64_BK64_GM32_nW4_nS2_EU0_kP1_mfma16: ', e, flush=True)
        return False


def matmul_BM64_BN64_BK64_GM32_nW4_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM64_BN64_BK64_GM32_nW4_nS2_EU0_kP2_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 32,
        num_warps = 4,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 2,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM64_BN64_BK64_GM32_nW4_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM64_BN64_BK64_GM32_nW4_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM64_BN64_BK64_GM32_nW4_nS2_EU0_kP2_mfma16: ', e, flush=True)
        return False


def matmul_BM64_BN64_BK64_GM32_nW4_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM64_BN64_BK64_GM32_nW4_nS2_EU0_kP1_mfma32.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 32,
        num_warps = 4,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 1,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM64_BN64_BK64_GM32_nW4_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM64_BN64_BK64_GM32_nW4_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM64_BN64_BK64_GM32_nW4_nS2_EU0_kP1_mfma32: ', e, flush=True)
        return False


def matmul_BM64_BN64_BK64_GM32_nW4_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM64_BN64_BK64_GM32_nW4_nS2_EU0_kP2_mfma32.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 32,
        num_warps = 4,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 2,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM64_BN64_BK64_GM32_nW4_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM64_BN64_BK64_GM32_nW4_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM64_BN64_BK64_GM32_nW4_nS2_EU0_kP2_mfma32: ', e, flush=True)
        return False


def matmul_BM64_BN64_BK64_GM1_nW8_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM64_BN64_BK64_GM1_nW8_nS2_EU0_kP1_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 1,
        num_warps = 8,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 1,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM64_BN64_BK64_GM1_nW8_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM64_BN64_BK64_GM1_nW8_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM64_BN64_BK64_GM1_nW8_nS2_EU0_kP1_mfma16: ', e, flush=True)
        return False


def matmul_BM64_BN64_BK64_GM1_nW8_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM64_BN64_BK64_GM1_nW8_nS2_EU0_kP2_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 1,
        num_warps = 8,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 2,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM64_BN64_BK64_GM1_nW8_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM64_BN64_BK64_GM1_nW8_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM64_BN64_BK64_GM1_nW8_nS2_EU0_kP2_mfma16: ', e, flush=True)
        return False


def matmul_BM64_BN64_BK64_GM1_nW8_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM64_BN64_BK64_GM1_nW8_nS2_EU0_kP1_mfma32.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 1,
        num_warps = 8,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 1,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM64_BN64_BK64_GM1_nW8_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM64_BN64_BK64_GM1_nW8_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM64_BN64_BK64_GM1_nW8_nS2_EU0_kP1_mfma32: ', e, flush=True)
        return False


def matmul_BM64_BN64_BK64_GM1_nW8_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM64_BN64_BK64_GM1_nW8_nS2_EU0_kP2_mfma32.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 1,
        num_warps = 8,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 2,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM64_BN64_BK64_GM1_nW8_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM64_BN64_BK64_GM1_nW8_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM64_BN64_BK64_GM1_nW8_nS2_EU0_kP2_mfma32: ', e, flush=True)
        return False


def matmul_BM64_BN64_BK64_GM4_nW8_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM64_BN64_BK64_GM4_nW8_nS2_EU0_kP1_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 4,
        num_warps = 8,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 1,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM64_BN64_BK64_GM4_nW8_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM64_BN64_BK64_GM4_nW8_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM64_BN64_BK64_GM4_nW8_nS2_EU0_kP1_mfma16: ', e, flush=True)
        return False


def matmul_BM64_BN64_BK64_GM4_nW8_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM64_BN64_BK64_GM4_nW8_nS2_EU0_kP2_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 4,
        num_warps = 8,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 2,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM64_BN64_BK64_GM4_nW8_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM64_BN64_BK64_GM4_nW8_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM64_BN64_BK64_GM4_nW8_nS2_EU0_kP2_mfma16: ', e, flush=True)
        return False


def matmul_BM64_BN64_BK64_GM4_nW8_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM64_BN64_BK64_GM4_nW8_nS2_EU0_kP1_mfma32.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 4,
        num_warps = 8,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 1,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM64_BN64_BK64_GM4_nW8_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM64_BN64_BK64_GM4_nW8_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM64_BN64_BK64_GM4_nW8_nS2_EU0_kP1_mfma32: ', e, flush=True)
        return False


def matmul_BM64_BN64_BK64_GM4_nW8_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM64_BN64_BK64_GM4_nW8_nS2_EU0_kP2_mfma32.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 4,
        num_warps = 8,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 2,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM64_BN64_BK64_GM4_nW8_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM64_BN64_BK64_GM4_nW8_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM64_BN64_BK64_GM4_nW8_nS2_EU0_kP2_mfma32: ', e, flush=True)
        return False


def matmul_BM64_BN64_BK64_GM8_nW8_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM64_BN64_BK64_GM8_nW8_nS2_EU0_kP1_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 8,
        num_warps = 8,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 1,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM64_BN64_BK64_GM8_nW8_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM64_BN64_BK64_GM8_nW8_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM64_BN64_BK64_GM8_nW8_nS2_EU0_kP1_mfma16: ', e, flush=True)
        return False


def matmul_BM64_BN64_BK64_GM8_nW8_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM64_BN64_BK64_GM8_nW8_nS2_EU0_kP2_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 8,
        num_warps = 8,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 2,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM64_BN64_BK64_GM8_nW8_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM64_BN64_BK64_GM8_nW8_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM64_BN64_BK64_GM8_nW8_nS2_EU0_kP2_mfma16: ', e, flush=True)
        return False


def matmul_BM64_BN64_BK64_GM8_nW8_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM64_BN64_BK64_GM8_nW8_nS2_EU0_kP1_mfma32.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 8,
        num_warps = 8,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 1,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM64_BN64_BK64_GM8_nW8_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM64_BN64_BK64_GM8_nW8_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM64_BN64_BK64_GM8_nW8_nS2_EU0_kP1_mfma32: ', e, flush=True)
        return False


def matmul_BM64_BN64_BK64_GM8_nW8_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM64_BN64_BK64_GM8_nW8_nS2_EU0_kP2_mfma32.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 8,
        num_warps = 8,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 2,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM64_BN64_BK64_GM8_nW8_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM64_BN64_BK64_GM8_nW8_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM64_BN64_BK64_GM8_nW8_nS2_EU0_kP2_mfma32: ', e, flush=True)
        return False


def matmul_BM64_BN64_BK64_GM16_nW8_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM64_BN64_BK64_GM16_nW8_nS2_EU0_kP1_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 16,
        num_warps = 8,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 1,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM64_BN64_BK64_GM16_nW8_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM64_BN64_BK64_GM16_nW8_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM64_BN64_BK64_GM16_nW8_nS2_EU0_kP1_mfma16: ', e, flush=True)
        return False


def matmul_BM64_BN64_BK64_GM16_nW8_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM64_BN64_BK64_GM16_nW8_nS2_EU0_kP2_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 16,
        num_warps = 8,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 2,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM64_BN64_BK64_GM16_nW8_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM64_BN64_BK64_GM16_nW8_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM64_BN64_BK64_GM16_nW8_nS2_EU0_kP2_mfma16: ', e, flush=True)
        return False


def matmul_BM64_BN64_BK64_GM16_nW8_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM64_BN64_BK64_GM16_nW8_nS2_EU0_kP1_mfma32.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 16,
        num_warps = 8,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 1,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM64_BN64_BK64_GM16_nW8_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM64_BN64_BK64_GM16_nW8_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM64_BN64_BK64_GM16_nW8_nS2_EU0_kP1_mfma32: ', e, flush=True)
        return False


def matmul_BM64_BN64_BK64_GM16_nW8_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM64_BN64_BK64_GM16_nW8_nS2_EU0_kP2_mfma32.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 16,
        num_warps = 8,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 2,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM64_BN64_BK64_GM16_nW8_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM64_BN64_BK64_GM16_nW8_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM64_BN64_BK64_GM16_nW8_nS2_EU0_kP2_mfma32: ', e, flush=True)
        return False


def matmul_BM64_BN64_BK64_GM32_nW8_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM64_BN64_BK64_GM32_nW8_nS2_EU0_kP1_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 32,
        num_warps = 8,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 1,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM64_BN64_BK64_GM32_nW8_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM64_BN64_BK64_GM32_nW8_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM64_BN64_BK64_GM32_nW8_nS2_EU0_kP1_mfma16: ', e, flush=True)
        return False


def matmul_BM64_BN64_BK64_GM32_nW8_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM64_BN64_BK64_GM32_nW8_nS2_EU0_kP2_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 32,
        num_warps = 8,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 2,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM64_BN64_BK64_GM32_nW8_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM64_BN64_BK64_GM32_nW8_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM64_BN64_BK64_GM32_nW8_nS2_EU0_kP2_mfma16: ', e, flush=True)
        return False


def matmul_BM64_BN64_BK64_GM32_nW8_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM64_BN64_BK64_GM32_nW8_nS2_EU0_kP1_mfma32.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 32,
        num_warps = 8,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 1,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM64_BN64_BK64_GM32_nW8_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM64_BN64_BK64_GM32_nW8_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM64_BN64_BK64_GM32_nW8_nS2_EU0_kP1_mfma32: ', e, flush=True)
        return False


def matmul_BM64_BN64_BK64_GM32_nW8_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM64_BN64_BK64_GM32_nW8_nS2_EU0_kP2_mfma32.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 32,
        num_warps = 8,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 2,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM64_BN64_BK64_GM32_nW8_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM64_BN64_BK64_GM32_nW8_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM64_BN64_BK64_GM32_nW8_nS2_EU0_kP2_mfma32: ', e, flush=True)
        return False


def matmul_BM64_BN64_BK128_GM1_nW4_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM64_BN64_BK128_GM1_nW4_nS2_EU0_kP1_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 128,
        GROUP_SIZE_M = 1,
        num_warps = 4,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 1,
        BIAS=False,
        EVEN_K=False,
        grid=(1,),
    )
    return None

def try_compile_config_BM64_BN64_BK128_GM1_nW4_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM64_BN64_BK128_GM1_nW4_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM64_BN64_BK128_GM1_nW4_nS2_EU0_kP1_mfma16: ', e, flush=True)
        return False


def matmul_BM64_BN64_BK128_GM1_nW4_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM64_BN64_BK128_GM1_nW4_nS2_EU0_kP2_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 128,
        GROUP_SIZE_M = 1,
        num_warps = 4,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 2,
        BIAS=False,
        EVEN_K=False,
        grid=(1,),
    )
    return None

def try_compile_config_BM64_BN64_BK128_GM1_nW4_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM64_BN64_BK128_GM1_nW4_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM64_BN64_BK128_GM1_nW4_nS2_EU0_kP2_mfma16: ', e, flush=True)
        return False


def matmul_BM64_BN64_BK128_GM1_nW4_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM64_BN64_BK128_GM1_nW4_nS2_EU0_kP1_mfma32.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 128,
        GROUP_SIZE_M = 1,
        num_warps = 4,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 1,
        BIAS=False,
        EVEN_K=False,
        grid=(1,),
    )
    return None

def try_compile_config_BM64_BN64_BK128_GM1_nW4_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM64_BN64_BK128_GM1_nW4_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM64_BN64_BK128_GM1_nW4_nS2_EU0_kP1_mfma32: ', e, flush=True)
        return False


def matmul_BM64_BN64_BK128_GM1_nW4_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM64_BN64_BK128_GM1_nW4_nS2_EU0_kP2_mfma32.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 128,
        GROUP_SIZE_M = 1,
        num_warps = 4,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 2,
        BIAS=False,
        EVEN_K=False,
        grid=(1,),
    )
    return None

def try_compile_config_BM64_BN64_BK128_GM1_nW4_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM64_BN64_BK128_GM1_nW4_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM64_BN64_BK128_GM1_nW4_nS2_EU0_kP2_mfma32: ', e, flush=True)
        return False


def matmul_BM64_BN64_BK128_GM4_nW4_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM64_BN64_BK128_GM4_nW4_nS2_EU0_kP1_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 128,
        GROUP_SIZE_M = 4,
        num_warps = 4,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 1,
        BIAS=False,
        EVEN_K=False,
        grid=(1,),
    )
    return None

def try_compile_config_BM64_BN64_BK128_GM4_nW4_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM64_BN64_BK128_GM4_nW4_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM64_BN64_BK128_GM4_nW4_nS2_EU0_kP1_mfma16: ', e, flush=True)
        return False


def matmul_BM64_BN64_BK128_GM4_nW4_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM64_BN64_BK128_GM4_nW4_nS2_EU0_kP2_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 128,
        GROUP_SIZE_M = 4,
        num_warps = 4,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 2,
        BIAS=False,
        EVEN_K=False,
        grid=(1,),
    )
    return None

def try_compile_config_BM64_BN64_BK128_GM4_nW4_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM64_BN64_BK128_GM4_nW4_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM64_BN64_BK128_GM4_nW4_nS2_EU0_kP2_mfma16: ', e, flush=True)
        return False


def matmul_BM64_BN64_BK128_GM4_nW4_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM64_BN64_BK128_GM4_nW4_nS2_EU0_kP1_mfma32.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 128,
        GROUP_SIZE_M = 4,
        num_warps = 4,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 1,
        BIAS=False,
        EVEN_K=False,
        grid=(1,),
    )
    return None

def try_compile_config_BM64_BN64_BK128_GM4_nW4_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM64_BN64_BK128_GM4_nW4_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM64_BN64_BK128_GM4_nW4_nS2_EU0_kP1_mfma32: ', e, flush=True)
        return False


def matmul_BM64_BN64_BK128_GM4_nW4_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM64_BN64_BK128_GM4_nW4_nS2_EU0_kP2_mfma32.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 128,
        GROUP_SIZE_M = 4,
        num_warps = 4,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 2,
        BIAS=False,
        EVEN_K=False,
        grid=(1,),
    )
    return None

def try_compile_config_BM64_BN64_BK128_GM4_nW4_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM64_BN64_BK128_GM4_nW4_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM64_BN64_BK128_GM4_nW4_nS2_EU0_kP2_mfma32: ', e, flush=True)
        return False


def matmul_BM64_BN64_BK128_GM8_nW4_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM64_BN64_BK128_GM8_nW4_nS2_EU0_kP1_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 128,
        GROUP_SIZE_M = 8,
        num_warps = 4,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 1,
        BIAS=False,
        EVEN_K=False,
        grid=(1,),
    )
    return None

def try_compile_config_BM64_BN64_BK128_GM8_nW4_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM64_BN64_BK128_GM8_nW4_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM64_BN64_BK128_GM8_nW4_nS2_EU0_kP1_mfma16: ', e, flush=True)
        return False


def matmul_BM64_BN64_BK128_GM8_nW4_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM64_BN64_BK128_GM8_nW4_nS2_EU0_kP2_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 128,
        GROUP_SIZE_M = 8,
        num_warps = 4,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 2,
        BIAS=False,
        EVEN_K=False,
        grid=(1,),
    )
    return None

def try_compile_config_BM64_BN64_BK128_GM8_nW4_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM64_BN64_BK128_GM8_nW4_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM64_BN64_BK128_GM8_nW4_nS2_EU0_kP2_mfma16: ', e, flush=True)
        return False


def matmul_BM64_BN64_BK128_GM8_nW4_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM64_BN64_BK128_GM8_nW4_nS2_EU0_kP1_mfma32.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 128,
        GROUP_SIZE_M = 8,
        num_warps = 4,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 1,
        BIAS=False,
        EVEN_K=False,
        grid=(1,),
    )
    return None

def try_compile_config_BM64_BN64_BK128_GM8_nW4_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM64_BN64_BK128_GM8_nW4_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM64_BN64_BK128_GM8_nW4_nS2_EU0_kP1_mfma32: ', e, flush=True)
        return False


def matmul_BM64_BN64_BK128_GM8_nW4_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM64_BN64_BK128_GM8_nW4_nS2_EU0_kP2_mfma32.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 128,
        GROUP_SIZE_M = 8,
        num_warps = 4,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 2,
        BIAS=False,
        EVEN_K=False,
        grid=(1,),
    )
    return None

def try_compile_config_BM64_BN64_BK128_GM8_nW4_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM64_BN64_BK128_GM8_nW4_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM64_BN64_BK128_GM8_nW4_nS2_EU0_kP2_mfma32: ', e, flush=True)
        return False


def matmul_BM64_BN64_BK128_GM16_nW4_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM64_BN64_BK128_GM16_nW4_nS2_EU0_kP1_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 128,
        GROUP_SIZE_M = 16,
        num_warps = 4,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 1,
        BIAS=False,
        EVEN_K=False,
        grid=(1,),
    )
    return None

def try_compile_config_BM64_BN64_BK128_GM16_nW4_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM64_BN64_BK128_GM16_nW4_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM64_BN64_BK128_GM16_nW4_nS2_EU0_kP1_mfma16: ', e, flush=True)
        return False


def matmul_BM64_BN64_BK128_GM16_nW4_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM64_BN64_BK128_GM16_nW4_nS2_EU0_kP2_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 128,
        GROUP_SIZE_M = 16,
        num_warps = 4,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 2,
        BIAS=False,
        EVEN_K=False,
        grid=(1,),
    )
    return None

def try_compile_config_BM64_BN64_BK128_GM16_nW4_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM64_BN64_BK128_GM16_nW4_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM64_BN64_BK128_GM16_nW4_nS2_EU0_kP2_mfma16: ', e, flush=True)
        return False


def matmul_BM64_BN64_BK128_GM16_nW4_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM64_BN64_BK128_GM16_nW4_nS2_EU0_kP1_mfma32.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 128,
        GROUP_SIZE_M = 16,
        num_warps = 4,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 1,
        BIAS=False,
        EVEN_K=False,
        grid=(1,),
    )
    return None

def try_compile_config_BM64_BN64_BK128_GM16_nW4_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM64_BN64_BK128_GM16_nW4_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM64_BN64_BK128_GM16_nW4_nS2_EU0_kP1_mfma32: ', e, flush=True)
        return False


def matmul_BM64_BN64_BK128_GM16_nW4_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM64_BN64_BK128_GM16_nW4_nS2_EU0_kP2_mfma32.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 128,
        GROUP_SIZE_M = 16,
        num_warps = 4,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 2,
        BIAS=False,
        EVEN_K=False,
        grid=(1,),
    )
    return None

def try_compile_config_BM64_BN64_BK128_GM16_nW4_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM64_BN64_BK128_GM16_nW4_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM64_BN64_BK128_GM16_nW4_nS2_EU0_kP2_mfma32: ', e, flush=True)
        return False


def matmul_BM64_BN64_BK128_GM32_nW4_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM64_BN64_BK128_GM32_nW4_nS2_EU0_kP1_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 128,
        GROUP_SIZE_M = 32,
        num_warps = 4,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 1,
        BIAS=False,
        EVEN_K=False,
        grid=(1,),
    )
    return None

def try_compile_config_BM64_BN64_BK128_GM32_nW4_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM64_BN64_BK128_GM32_nW4_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM64_BN64_BK128_GM32_nW4_nS2_EU0_kP1_mfma16: ', e, flush=True)
        return False


def matmul_BM64_BN64_BK128_GM32_nW4_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM64_BN64_BK128_GM32_nW4_nS2_EU0_kP2_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 128,
        GROUP_SIZE_M = 32,
        num_warps = 4,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 2,
        BIAS=False,
        EVEN_K=False,
        grid=(1,),
    )
    return None

def try_compile_config_BM64_BN64_BK128_GM32_nW4_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM64_BN64_BK128_GM32_nW4_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM64_BN64_BK128_GM32_nW4_nS2_EU0_kP2_mfma16: ', e, flush=True)
        return False


def matmul_BM64_BN64_BK128_GM32_nW4_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM64_BN64_BK128_GM32_nW4_nS2_EU0_kP1_mfma32.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 128,
        GROUP_SIZE_M = 32,
        num_warps = 4,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 1,
        BIAS=False,
        EVEN_K=False,
        grid=(1,),
    )
    return None

def try_compile_config_BM64_BN64_BK128_GM32_nW4_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM64_BN64_BK128_GM32_nW4_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM64_BN64_BK128_GM32_nW4_nS2_EU0_kP1_mfma32: ', e, flush=True)
        return False


def matmul_BM64_BN64_BK128_GM32_nW4_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM64_BN64_BK128_GM32_nW4_nS2_EU0_kP2_mfma32.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 128,
        GROUP_SIZE_M = 32,
        num_warps = 4,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 2,
        BIAS=False,
        EVEN_K=False,
        grid=(1,),
    )
    return None

def try_compile_config_BM64_BN64_BK128_GM32_nW4_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM64_BN64_BK128_GM32_nW4_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM64_BN64_BK128_GM32_nW4_nS2_EU0_kP2_mfma32: ', e, flush=True)
        return False


def matmul_BM64_BN64_BK128_GM1_nW8_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM64_BN64_BK128_GM1_nW8_nS2_EU0_kP1_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 128,
        GROUP_SIZE_M = 1,
        num_warps = 8,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 1,
        BIAS=False,
        EVEN_K=False,
        grid=(1,),
    )
    return None

def try_compile_config_BM64_BN64_BK128_GM1_nW8_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM64_BN64_BK128_GM1_nW8_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM64_BN64_BK128_GM1_nW8_nS2_EU0_kP1_mfma16: ', e, flush=True)
        return False


def matmul_BM64_BN64_BK128_GM1_nW8_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM64_BN64_BK128_GM1_nW8_nS2_EU0_kP2_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 128,
        GROUP_SIZE_M = 1,
        num_warps = 8,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 2,
        BIAS=False,
        EVEN_K=False,
        grid=(1,),
    )
    return None

def try_compile_config_BM64_BN64_BK128_GM1_nW8_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM64_BN64_BK128_GM1_nW8_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM64_BN64_BK128_GM1_nW8_nS2_EU0_kP2_mfma16: ', e, flush=True)
        return False


def matmul_BM64_BN64_BK128_GM1_nW8_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM64_BN64_BK128_GM1_nW8_nS2_EU0_kP1_mfma32.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 128,
        GROUP_SIZE_M = 1,
        num_warps = 8,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 1,
        BIAS=False,
        EVEN_K=False,
        grid=(1,),
    )
    return None

def try_compile_config_BM64_BN64_BK128_GM1_nW8_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM64_BN64_BK128_GM1_nW8_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM64_BN64_BK128_GM1_nW8_nS2_EU0_kP1_mfma32: ', e, flush=True)
        return False


def matmul_BM64_BN64_BK128_GM1_nW8_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM64_BN64_BK128_GM1_nW8_nS2_EU0_kP2_mfma32.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 128,
        GROUP_SIZE_M = 1,
        num_warps = 8,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 2,
        BIAS=False,
        EVEN_K=False,
        grid=(1,),
    )
    return None

def try_compile_config_BM64_BN64_BK128_GM1_nW8_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM64_BN64_BK128_GM1_nW8_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM64_BN64_BK128_GM1_nW8_nS2_EU0_kP2_mfma32: ', e, flush=True)
        return False


def matmul_BM64_BN64_BK128_GM4_nW8_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM64_BN64_BK128_GM4_nW8_nS2_EU0_kP1_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 128,
        GROUP_SIZE_M = 4,
        num_warps = 8,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 1,
        BIAS=False,
        EVEN_K=False,
        grid=(1,),
    )
    return None

def try_compile_config_BM64_BN64_BK128_GM4_nW8_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM64_BN64_BK128_GM4_nW8_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM64_BN64_BK128_GM4_nW8_nS2_EU0_kP1_mfma16: ', e, flush=True)
        return False


def matmul_BM64_BN64_BK128_GM4_nW8_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM64_BN64_BK128_GM4_nW8_nS2_EU0_kP2_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 128,
        GROUP_SIZE_M = 4,
        num_warps = 8,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 2,
        BIAS=False,
        EVEN_K=False,
        grid=(1,),
    )
    return None

def try_compile_config_BM64_BN64_BK128_GM4_nW8_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM64_BN64_BK128_GM4_nW8_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM64_BN64_BK128_GM4_nW8_nS2_EU0_kP2_mfma16: ', e, flush=True)
        return False


def matmul_BM64_BN64_BK128_GM4_nW8_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM64_BN64_BK128_GM4_nW8_nS2_EU0_kP1_mfma32.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 128,
        GROUP_SIZE_M = 4,
        num_warps = 8,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 1,
        BIAS=False,
        EVEN_K=False,
        grid=(1,),
    )
    return None

def try_compile_config_BM64_BN64_BK128_GM4_nW8_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM64_BN64_BK128_GM4_nW8_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM64_BN64_BK128_GM4_nW8_nS2_EU0_kP1_mfma32: ', e, flush=True)
        return False


def matmul_BM64_BN64_BK128_GM4_nW8_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM64_BN64_BK128_GM4_nW8_nS2_EU0_kP2_mfma32.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 128,
        GROUP_SIZE_M = 4,
        num_warps = 8,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 2,
        BIAS=False,
        EVEN_K=False,
        grid=(1,),
    )
    return None

def try_compile_config_BM64_BN64_BK128_GM4_nW8_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM64_BN64_BK128_GM4_nW8_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM64_BN64_BK128_GM4_nW8_nS2_EU0_kP2_mfma32: ', e, flush=True)
        return False


def matmul_BM64_BN64_BK128_GM8_nW8_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM64_BN64_BK128_GM8_nW8_nS2_EU0_kP1_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 128,
        GROUP_SIZE_M = 8,
        num_warps = 8,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 1,
        BIAS=False,
        EVEN_K=False,
        grid=(1,),
    )
    return None

def try_compile_config_BM64_BN64_BK128_GM8_nW8_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM64_BN64_BK128_GM8_nW8_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM64_BN64_BK128_GM8_nW8_nS2_EU0_kP1_mfma16: ', e, flush=True)
        return False


def matmul_BM64_BN64_BK128_GM8_nW8_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM64_BN64_BK128_GM8_nW8_nS2_EU0_kP2_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 128,
        GROUP_SIZE_M = 8,
        num_warps = 8,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 2,
        BIAS=False,
        EVEN_K=False,
        grid=(1,),
    )
    return None

def try_compile_config_BM64_BN64_BK128_GM8_nW8_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM64_BN64_BK128_GM8_nW8_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM64_BN64_BK128_GM8_nW8_nS2_EU0_kP2_mfma16: ', e, flush=True)
        return False


def matmul_BM64_BN64_BK128_GM8_nW8_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM64_BN64_BK128_GM8_nW8_nS2_EU0_kP1_mfma32.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 128,
        GROUP_SIZE_M = 8,
        num_warps = 8,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 1,
        BIAS=False,
        EVEN_K=False,
        grid=(1,),
    )
    return None

def try_compile_config_BM64_BN64_BK128_GM8_nW8_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM64_BN64_BK128_GM8_nW8_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM64_BN64_BK128_GM8_nW8_nS2_EU0_kP1_mfma32: ', e, flush=True)
        return False


def matmul_BM64_BN64_BK128_GM8_nW8_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM64_BN64_BK128_GM8_nW8_nS2_EU0_kP2_mfma32.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 128,
        GROUP_SIZE_M = 8,
        num_warps = 8,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 2,
        BIAS=False,
        EVEN_K=False,
        grid=(1,),
    )
    return None

def try_compile_config_BM64_BN64_BK128_GM8_nW8_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM64_BN64_BK128_GM8_nW8_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM64_BN64_BK128_GM8_nW8_nS2_EU0_kP2_mfma32: ', e, flush=True)
        return False


def matmul_BM64_BN64_BK128_GM16_nW8_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM64_BN64_BK128_GM16_nW8_nS2_EU0_kP1_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 128,
        GROUP_SIZE_M = 16,
        num_warps = 8,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 1,
        BIAS=False,
        EVEN_K=False,
        grid=(1,),
    )
    return None

def try_compile_config_BM64_BN64_BK128_GM16_nW8_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM64_BN64_BK128_GM16_nW8_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM64_BN64_BK128_GM16_nW8_nS2_EU0_kP1_mfma16: ', e, flush=True)
        return False


def matmul_BM64_BN64_BK128_GM16_nW8_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM64_BN64_BK128_GM16_nW8_nS2_EU0_kP2_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 128,
        GROUP_SIZE_M = 16,
        num_warps = 8,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 2,
        BIAS=False,
        EVEN_K=False,
        grid=(1,),
    )
    return None

def try_compile_config_BM64_BN64_BK128_GM16_nW8_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM64_BN64_BK128_GM16_nW8_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM64_BN64_BK128_GM16_nW8_nS2_EU0_kP2_mfma16: ', e, flush=True)
        return False


def matmul_BM64_BN64_BK128_GM16_nW8_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM64_BN64_BK128_GM16_nW8_nS2_EU0_kP1_mfma32.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 128,
        GROUP_SIZE_M = 16,
        num_warps = 8,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 1,
        BIAS=False,
        EVEN_K=False,
        grid=(1,),
    )
    return None

def try_compile_config_BM64_BN64_BK128_GM16_nW8_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM64_BN64_BK128_GM16_nW8_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM64_BN64_BK128_GM16_nW8_nS2_EU0_kP1_mfma32: ', e, flush=True)
        return False


def matmul_BM64_BN64_BK128_GM16_nW8_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM64_BN64_BK128_GM16_nW8_nS2_EU0_kP2_mfma32.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 128,
        GROUP_SIZE_M = 16,
        num_warps = 8,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 2,
        BIAS=False,
        EVEN_K=False,
        grid=(1,),
    )
    return None

def try_compile_config_BM64_BN64_BK128_GM16_nW8_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM64_BN64_BK128_GM16_nW8_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM64_BN64_BK128_GM16_nW8_nS2_EU0_kP2_mfma32: ', e, flush=True)
        return False


def matmul_BM64_BN64_BK128_GM32_nW8_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM64_BN64_BK128_GM32_nW8_nS2_EU0_kP1_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 128,
        GROUP_SIZE_M = 32,
        num_warps = 8,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 1,
        BIAS=False,
        EVEN_K=False,
        grid=(1,),
    )
    return None

def try_compile_config_BM64_BN64_BK128_GM32_nW8_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM64_BN64_BK128_GM32_nW8_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM64_BN64_BK128_GM32_nW8_nS2_EU0_kP1_mfma16: ', e, flush=True)
        return False


def matmul_BM64_BN64_BK128_GM32_nW8_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM64_BN64_BK128_GM32_nW8_nS2_EU0_kP2_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 128,
        GROUP_SIZE_M = 32,
        num_warps = 8,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 2,
        BIAS=False,
        EVEN_K=False,
        grid=(1,),
    )
    return None

def try_compile_config_BM64_BN64_BK128_GM32_nW8_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM64_BN64_BK128_GM32_nW8_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM64_BN64_BK128_GM32_nW8_nS2_EU0_kP2_mfma16: ', e, flush=True)
        return False


def matmul_BM64_BN64_BK128_GM32_nW8_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM64_BN64_BK128_GM32_nW8_nS2_EU0_kP1_mfma32.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 128,
        GROUP_SIZE_M = 32,
        num_warps = 8,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 1,
        BIAS=False,
        EVEN_K=False,
        grid=(1,),
    )
    return None

def try_compile_config_BM64_BN64_BK128_GM32_nW8_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM64_BN64_BK128_GM32_nW8_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM64_BN64_BK128_GM32_nW8_nS2_EU0_kP1_mfma32: ', e, flush=True)
        return False


def matmul_BM64_BN64_BK128_GM32_nW8_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM64_BN64_BK128_GM32_nW8_nS2_EU0_kP2_mfma32.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 128,
        GROUP_SIZE_M = 32,
        num_warps = 8,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 2,
        BIAS=False,
        EVEN_K=False,
        grid=(1,),
    )
    return None

def try_compile_config_BM64_BN64_BK128_GM32_nW8_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM64_BN64_BK128_GM32_nW8_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM64_BN64_BK128_GM32_nW8_nS2_EU0_kP2_mfma32: ', e, flush=True)
        return False


def matmul_BM64_BN64_BK256_GM1_nW4_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM64_BN64_BK256_GM1_nW4_nS2_EU0_kP1_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 256,
        GROUP_SIZE_M = 1,
        num_warps = 4,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 1,
        BIAS=False,
        EVEN_K=False,
        grid=(1,),
    )
    return None

def try_compile_config_BM64_BN64_BK256_GM1_nW4_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM64_BN64_BK256_GM1_nW4_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM64_BN64_BK256_GM1_nW4_nS2_EU0_kP1_mfma16: ', e, flush=True)
        return False


def matmul_BM64_BN64_BK256_GM1_nW4_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM64_BN64_BK256_GM1_nW4_nS2_EU0_kP2_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 256,
        GROUP_SIZE_M = 1,
        num_warps = 4,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 2,
        BIAS=False,
        EVEN_K=False,
        grid=(1,),
    )
    return None

def try_compile_config_BM64_BN64_BK256_GM1_nW4_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM64_BN64_BK256_GM1_nW4_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM64_BN64_BK256_GM1_nW4_nS2_EU0_kP2_mfma16: ', e, flush=True)
        return False


def matmul_BM64_BN64_BK256_GM1_nW4_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM64_BN64_BK256_GM1_nW4_nS2_EU0_kP1_mfma32.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 256,
        GROUP_SIZE_M = 1,
        num_warps = 4,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 1,
        BIAS=False,
        EVEN_K=False,
        grid=(1,),
    )
    return None

def try_compile_config_BM64_BN64_BK256_GM1_nW4_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM64_BN64_BK256_GM1_nW4_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM64_BN64_BK256_GM1_nW4_nS2_EU0_kP1_mfma32: ', e, flush=True)
        return False


def matmul_BM64_BN64_BK256_GM1_nW4_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM64_BN64_BK256_GM1_nW4_nS2_EU0_kP2_mfma32.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 256,
        GROUP_SIZE_M = 1,
        num_warps = 4,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 2,
        BIAS=False,
        EVEN_K=False,
        grid=(1,),
    )
    return None

def try_compile_config_BM64_BN64_BK256_GM1_nW4_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM64_BN64_BK256_GM1_nW4_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM64_BN64_BK256_GM1_nW4_nS2_EU0_kP2_mfma32: ', e, flush=True)
        return False


def matmul_BM64_BN64_BK256_GM4_nW4_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM64_BN64_BK256_GM4_nW4_nS2_EU0_kP1_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 256,
        GROUP_SIZE_M = 4,
        num_warps = 4,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 1,
        BIAS=False,
        EVEN_K=False,
        grid=(1,),
    )
    return None

def try_compile_config_BM64_BN64_BK256_GM4_nW4_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM64_BN64_BK256_GM4_nW4_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM64_BN64_BK256_GM4_nW4_nS2_EU0_kP1_mfma16: ', e, flush=True)
        return False


def matmul_BM64_BN64_BK256_GM4_nW4_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM64_BN64_BK256_GM4_nW4_nS2_EU0_kP2_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 256,
        GROUP_SIZE_M = 4,
        num_warps = 4,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 2,
        BIAS=False,
        EVEN_K=False,
        grid=(1,),
    )
    return None

def try_compile_config_BM64_BN64_BK256_GM4_nW4_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM64_BN64_BK256_GM4_nW4_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM64_BN64_BK256_GM4_nW4_nS2_EU0_kP2_mfma16: ', e, flush=True)
        return False


def matmul_BM64_BN64_BK256_GM4_nW4_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM64_BN64_BK256_GM4_nW4_nS2_EU0_kP1_mfma32.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 256,
        GROUP_SIZE_M = 4,
        num_warps = 4,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 1,
        BIAS=False,
        EVEN_K=False,
        grid=(1,),
    )
    return None

def try_compile_config_BM64_BN64_BK256_GM4_nW4_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM64_BN64_BK256_GM4_nW4_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM64_BN64_BK256_GM4_nW4_nS2_EU0_kP1_mfma32: ', e, flush=True)
        return False


def matmul_BM64_BN64_BK256_GM4_nW4_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM64_BN64_BK256_GM4_nW4_nS2_EU0_kP2_mfma32.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 256,
        GROUP_SIZE_M = 4,
        num_warps = 4,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 2,
        BIAS=False,
        EVEN_K=False,
        grid=(1,),
    )
    return None

def try_compile_config_BM64_BN64_BK256_GM4_nW4_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM64_BN64_BK256_GM4_nW4_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM64_BN64_BK256_GM4_nW4_nS2_EU0_kP2_mfma32: ', e, flush=True)
        return False


def matmul_BM64_BN64_BK256_GM8_nW4_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM64_BN64_BK256_GM8_nW4_nS2_EU0_kP1_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 256,
        GROUP_SIZE_M = 8,
        num_warps = 4,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 1,
        BIAS=False,
        EVEN_K=False,
        grid=(1,),
    )
    return None

def try_compile_config_BM64_BN64_BK256_GM8_nW4_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM64_BN64_BK256_GM8_nW4_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM64_BN64_BK256_GM8_nW4_nS2_EU0_kP1_mfma16: ', e, flush=True)
        return False


def matmul_BM64_BN64_BK256_GM8_nW4_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM64_BN64_BK256_GM8_nW4_nS2_EU0_kP2_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 256,
        GROUP_SIZE_M = 8,
        num_warps = 4,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 2,
        BIAS=False,
        EVEN_K=False,
        grid=(1,),
    )
    return None

def try_compile_config_BM64_BN64_BK256_GM8_nW4_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM64_BN64_BK256_GM8_nW4_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM64_BN64_BK256_GM8_nW4_nS2_EU0_kP2_mfma16: ', e, flush=True)
        return False


def matmul_BM64_BN64_BK256_GM8_nW4_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM64_BN64_BK256_GM8_nW4_nS2_EU0_kP1_mfma32.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 256,
        GROUP_SIZE_M = 8,
        num_warps = 4,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 1,
        BIAS=False,
        EVEN_K=False,
        grid=(1,),
    )
    return None

def try_compile_config_BM64_BN64_BK256_GM8_nW4_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM64_BN64_BK256_GM8_nW4_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM64_BN64_BK256_GM8_nW4_nS2_EU0_kP1_mfma32: ', e, flush=True)
        return False


def matmul_BM64_BN64_BK256_GM8_nW4_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM64_BN64_BK256_GM8_nW4_nS2_EU0_kP2_mfma32.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 256,
        GROUP_SIZE_M = 8,
        num_warps = 4,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 2,
        BIAS=False,
        EVEN_K=False,
        grid=(1,),
    )
    return None

def try_compile_config_BM64_BN64_BK256_GM8_nW4_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM64_BN64_BK256_GM8_nW4_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM64_BN64_BK256_GM8_nW4_nS2_EU0_kP2_mfma32: ', e, flush=True)
        return False


def matmul_BM64_BN64_BK256_GM16_nW4_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM64_BN64_BK256_GM16_nW4_nS2_EU0_kP1_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 256,
        GROUP_SIZE_M = 16,
        num_warps = 4,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 1,
        BIAS=False,
        EVEN_K=False,
        grid=(1,),
    )
    return None

def try_compile_config_BM64_BN64_BK256_GM16_nW4_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM64_BN64_BK256_GM16_nW4_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM64_BN64_BK256_GM16_nW4_nS2_EU0_kP1_mfma16: ', e, flush=True)
        return False


def matmul_BM64_BN64_BK256_GM16_nW4_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM64_BN64_BK256_GM16_nW4_nS2_EU0_kP2_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 256,
        GROUP_SIZE_M = 16,
        num_warps = 4,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 2,
        BIAS=False,
        EVEN_K=False,
        grid=(1,),
    )
    return None

def try_compile_config_BM64_BN64_BK256_GM16_nW4_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM64_BN64_BK256_GM16_nW4_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM64_BN64_BK256_GM16_nW4_nS2_EU0_kP2_mfma16: ', e, flush=True)
        return False


def matmul_BM64_BN64_BK256_GM16_nW4_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM64_BN64_BK256_GM16_nW4_nS2_EU0_kP1_mfma32.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 256,
        GROUP_SIZE_M = 16,
        num_warps = 4,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 1,
        BIAS=False,
        EVEN_K=False,
        grid=(1,),
    )
    return None

def try_compile_config_BM64_BN64_BK256_GM16_nW4_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM64_BN64_BK256_GM16_nW4_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM64_BN64_BK256_GM16_nW4_nS2_EU0_kP1_mfma32: ', e, flush=True)
        return False


def matmul_BM64_BN64_BK256_GM16_nW4_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM64_BN64_BK256_GM16_nW4_nS2_EU0_kP2_mfma32.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 256,
        GROUP_SIZE_M = 16,
        num_warps = 4,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 2,
        BIAS=False,
        EVEN_K=False,
        grid=(1,),
    )
    return None

def try_compile_config_BM64_BN64_BK256_GM16_nW4_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM64_BN64_BK256_GM16_nW4_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM64_BN64_BK256_GM16_nW4_nS2_EU0_kP2_mfma32: ', e, flush=True)
        return False


def matmul_BM64_BN64_BK256_GM32_nW4_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM64_BN64_BK256_GM32_nW4_nS2_EU0_kP1_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 256,
        GROUP_SIZE_M = 32,
        num_warps = 4,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 1,
        BIAS=False,
        EVEN_K=False,
        grid=(1,),
    )
    return None

def try_compile_config_BM64_BN64_BK256_GM32_nW4_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM64_BN64_BK256_GM32_nW4_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM64_BN64_BK256_GM32_nW4_nS2_EU0_kP1_mfma16: ', e, flush=True)
        return False


def matmul_BM64_BN64_BK256_GM32_nW4_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM64_BN64_BK256_GM32_nW4_nS2_EU0_kP2_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 256,
        GROUP_SIZE_M = 32,
        num_warps = 4,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 2,
        BIAS=False,
        EVEN_K=False,
        grid=(1,),
    )
    return None

def try_compile_config_BM64_BN64_BK256_GM32_nW4_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM64_BN64_BK256_GM32_nW4_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM64_BN64_BK256_GM32_nW4_nS2_EU0_kP2_mfma16: ', e, flush=True)
        return False


def matmul_BM64_BN64_BK256_GM32_nW4_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM64_BN64_BK256_GM32_nW4_nS2_EU0_kP1_mfma32.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 256,
        GROUP_SIZE_M = 32,
        num_warps = 4,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 1,
        BIAS=False,
        EVEN_K=False,
        grid=(1,),
    )
    return None

def try_compile_config_BM64_BN64_BK256_GM32_nW4_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM64_BN64_BK256_GM32_nW4_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM64_BN64_BK256_GM32_nW4_nS2_EU0_kP1_mfma32: ', e, flush=True)
        return False


def matmul_BM64_BN64_BK256_GM32_nW4_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM64_BN64_BK256_GM32_nW4_nS2_EU0_kP2_mfma32.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 256,
        GROUP_SIZE_M = 32,
        num_warps = 4,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 2,
        BIAS=False,
        EVEN_K=False,
        grid=(1,),
    )
    return None

def try_compile_config_BM64_BN64_BK256_GM32_nW4_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM64_BN64_BK256_GM32_nW4_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM64_BN64_BK256_GM32_nW4_nS2_EU0_kP2_mfma32: ', e, flush=True)
        return False


def matmul_BM64_BN64_BK256_GM1_nW8_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM64_BN64_BK256_GM1_nW8_nS2_EU0_kP1_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 256,
        GROUP_SIZE_M = 1,
        num_warps = 8,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 1,
        BIAS=False,
        EVEN_K=False,
        grid=(1,),
    )
    return None

def try_compile_config_BM64_BN64_BK256_GM1_nW8_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM64_BN64_BK256_GM1_nW8_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM64_BN64_BK256_GM1_nW8_nS2_EU0_kP1_mfma16: ', e, flush=True)
        return False


def matmul_BM64_BN64_BK256_GM1_nW8_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM64_BN64_BK256_GM1_nW8_nS2_EU0_kP2_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 256,
        GROUP_SIZE_M = 1,
        num_warps = 8,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 2,
        BIAS=False,
        EVEN_K=False,
        grid=(1,),
    )
    return None

def try_compile_config_BM64_BN64_BK256_GM1_nW8_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM64_BN64_BK256_GM1_nW8_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM64_BN64_BK256_GM1_nW8_nS2_EU0_kP2_mfma16: ', e, flush=True)
        return False


def matmul_BM64_BN64_BK256_GM1_nW8_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM64_BN64_BK256_GM1_nW8_nS2_EU0_kP1_mfma32.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 256,
        GROUP_SIZE_M = 1,
        num_warps = 8,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 1,
        BIAS=False,
        EVEN_K=False,
        grid=(1,),
    )
    return None

def try_compile_config_BM64_BN64_BK256_GM1_nW8_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM64_BN64_BK256_GM1_nW8_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM64_BN64_BK256_GM1_nW8_nS2_EU0_kP1_mfma32: ', e, flush=True)
        return False


def matmul_BM64_BN64_BK256_GM1_nW8_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM64_BN64_BK256_GM1_nW8_nS2_EU0_kP2_mfma32.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 256,
        GROUP_SIZE_M = 1,
        num_warps = 8,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 2,
        BIAS=False,
        EVEN_K=False,
        grid=(1,),
    )
    return None

def try_compile_config_BM64_BN64_BK256_GM1_nW8_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM64_BN64_BK256_GM1_nW8_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM64_BN64_BK256_GM1_nW8_nS2_EU0_kP2_mfma32: ', e, flush=True)
        return False


def matmul_BM64_BN64_BK256_GM4_nW8_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM64_BN64_BK256_GM4_nW8_nS2_EU0_kP1_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 256,
        GROUP_SIZE_M = 4,
        num_warps = 8,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 1,
        BIAS=False,
        EVEN_K=False,
        grid=(1,),
    )
    return None

def try_compile_config_BM64_BN64_BK256_GM4_nW8_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM64_BN64_BK256_GM4_nW8_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM64_BN64_BK256_GM4_nW8_nS2_EU0_kP1_mfma16: ', e, flush=True)
        return False


def matmul_BM64_BN64_BK256_GM4_nW8_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM64_BN64_BK256_GM4_nW8_nS2_EU0_kP2_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 256,
        GROUP_SIZE_M = 4,
        num_warps = 8,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 2,
        BIAS=False,
        EVEN_K=False,
        grid=(1,),
    )
    return None

def try_compile_config_BM64_BN64_BK256_GM4_nW8_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM64_BN64_BK256_GM4_nW8_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM64_BN64_BK256_GM4_nW8_nS2_EU0_kP2_mfma16: ', e, flush=True)
        return False


def matmul_BM64_BN64_BK256_GM4_nW8_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM64_BN64_BK256_GM4_nW8_nS2_EU0_kP1_mfma32.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 256,
        GROUP_SIZE_M = 4,
        num_warps = 8,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 1,
        BIAS=False,
        EVEN_K=False,
        grid=(1,),
    )
    return None

def try_compile_config_BM64_BN64_BK256_GM4_nW8_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM64_BN64_BK256_GM4_nW8_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM64_BN64_BK256_GM4_nW8_nS2_EU0_kP1_mfma32: ', e, flush=True)
        return False


def matmul_BM64_BN64_BK256_GM4_nW8_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM64_BN64_BK256_GM4_nW8_nS2_EU0_kP2_mfma32.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 256,
        GROUP_SIZE_M = 4,
        num_warps = 8,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 2,
        BIAS=False,
        EVEN_K=False,
        grid=(1,),
    )
    return None

def try_compile_config_BM64_BN64_BK256_GM4_nW8_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM64_BN64_BK256_GM4_nW8_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM64_BN64_BK256_GM4_nW8_nS2_EU0_kP2_mfma32: ', e, flush=True)
        return False


def matmul_BM64_BN64_BK256_GM8_nW8_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM64_BN64_BK256_GM8_nW8_nS2_EU0_kP1_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 256,
        GROUP_SIZE_M = 8,
        num_warps = 8,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 1,
        BIAS=False,
        EVEN_K=False,
        grid=(1,),
    )
    return None

def try_compile_config_BM64_BN64_BK256_GM8_nW8_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM64_BN64_BK256_GM8_nW8_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM64_BN64_BK256_GM8_nW8_nS2_EU0_kP1_mfma16: ', e, flush=True)
        return False


def matmul_BM64_BN64_BK256_GM8_nW8_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM64_BN64_BK256_GM8_nW8_nS2_EU0_kP2_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 256,
        GROUP_SIZE_M = 8,
        num_warps = 8,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 2,
        BIAS=False,
        EVEN_K=False,
        grid=(1,),
    )
    return None

def try_compile_config_BM64_BN64_BK256_GM8_nW8_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM64_BN64_BK256_GM8_nW8_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM64_BN64_BK256_GM8_nW8_nS2_EU0_kP2_mfma16: ', e, flush=True)
        return False


def matmul_BM64_BN64_BK256_GM8_nW8_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM64_BN64_BK256_GM8_nW8_nS2_EU0_kP1_mfma32.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 256,
        GROUP_SIZE_M = 8,
        num_warps = 8,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 1,
        BIAS=False,
        EVEN_K=False,
        grid=(1,),
    )
    return None

def try_compile_config_BM64_BN64_BK256_GM8_nW8_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM64_BN64_BK256_GM8_nW8_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM64_BN64_BK256_GM8_nW8_nS2_EU0_kP1_mfma32: ', e, flush=True)
        return False


def matmul_BM64_BN64_BK256_GM8_nW8_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM64_BN64_BK256_GM8_nW8_nS2_EU0_kP2_mfma32.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 256,
        GROUP_SIZE_M = 8,
        num_warps = 8,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 2,
        BIAS=False,
        EVEN_K=False,
        grid=(1,),
    )
    return None

def try_compile_config_BM64_BN64_BK256_GM8_nW8_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM64_BN64_BK256_GM8_nW8_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM64_BN64_BK256_GM8_nW8_nS2_EU0_kP2_mfma32: ', e, flush=True)
        return False


def matmul_BM64_BN64_BK256_GM16_nW8_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM64_BN64_BK256_GM16_nW8_nS2_EU0_kP1_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 256,
        GROUP_SIZE_M = 16,
        num_warps = 8,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 1,
        BIAS=False,
        EVEN_K=False,
        grid=(1,),
    )
    return None

def try_compile_config_BM64_BN64_BK256_GM16_nW8_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM64_BN64_BK256_GM16_nW8_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM64_BN64_BK256_GM16_nW8_nS2_EU0_kP1_mfma16: ', e, flush=True)
        return False


def matmul_BM64_BN64_BK256_GM16_nW8_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM64_BN64_BK256_GM16_nW8_nS2_EU0_kP2_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 256,
        GROUP_SIZE_M = 16,
        num_warps = 8,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 2,
        BIAS=False,
        EVEN_K=False,
        grid=(1,),
    )
    return None

def try_compile_config_BM64_BN64_BK256_GM16_nW8_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM64_BN64_BK256_GM16_nW8_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM64_BN64_BK256_GM16_nW8_nS2_EU0_kP2_mfma16: ', e, flush=True)
        return False


def matmul_BM64_BN64_BK256_GM16_nW8_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM64_BN64_BK256_GM16_nW8_nS2_EU0_kP1_mfma32.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 256,
        GROUP_SIZE_M = 16,
        num_warps = 8,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 1,
        BIAS=False,
        EVEN_K=False,
        grid=(1,),
    )
    return None

def try_compile_config_BM64_BN64_BK256_GM16_nW8_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM64_BN64_BK256_GM16_nW8_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM64_BN64_BK256_GM16_nW8_nS2_EU0_kP1_mfma32: ', e, flush=True)
        return False


def matmul_BM64_BN64_BK256_GM16_nW8_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM64_BN64_BK256_GM16_nW8_nS2_EU0_kP2_mfma32.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 256,
        GROUP_SIZE_M = 16,
        num_warps = 8,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 2,
        BIAS=False,
        EVEN_K=False,
        grid=(1,),
    )
    return None

def try_compile_config_BM64_BN64_BK256_GM16_nW8_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM64_BN64_BK256_GM16_nW8_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM64_BN64_BK256_GM16_nW8_nS2_EU0_kP2_mfma32: ', e, flush=True)
        return False


def matmul_BM64_BN64_BK256_GM32_nW8_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM64_BN64_BK256_GM32_nW8_nS2_EU0_kP1_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 256,
        GROUP_SIZE_M = 32,
        num_warps = 8,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 1,
        BIAS=False,
        EVEN_K=False,
        grid=(1,),
    )
    return None

def try_compile_config_BM64_BN64_BK256_GM32_nW8_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM64_BN64_BK256_GM32_nW8_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM64_BN64_BK256_GM32_nW8_nS2_EU0_kP1_mfma16: ', e, flush=True)
        return False


def matmul_BM64_BN64_BK256_GM32_nW8_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM64_BN64_BK256_GM32_nW8_nS2_EU0_kP2_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 256,
        GROUP_SIZE_M = 32,
        num_warps = 8,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 2,
        BIAS=False,
        EVEN_K=False,
        grid=(1,),
    )
    return None

def try_compile_config_BM64_BN64_BK256_GM32_nW8_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM64_BN64_BK256_GM32_nW8_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM64_BN64_BK256_GM32_nW8_nS2_EU0_kP2_mfma16: ', e, flush=True)
        return False


def matmul_BM64_BN64_BK256_GM32_nW8_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM64_BN64_BK256_GM32_nW8_nS2_EU0_kP1_mfma32.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 256,
        GROUP_SIZE_M = 32,
        num_warps = 8,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 1,
        BIAS=False,
        EVEN_K=False,
        grid=(1,),
    )
    return None

def try_compile_config_BM64_BN64_BK256_GM32_nW8_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM64_BN64_BK256_GM32_nW8_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM64_BN64_BK256_GM32_nW8_nS2_EU0_kP1_mfma32: ', e, flush=True)
        return False


def matmul_BM64_BN64_BK256_GM32_nW8_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM64_BN64_BK256_GM32_nW8_nS2_EU0_kP2_mfma32.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 256,
        GROUP_SIZE_M = 32,
        num_warps = 8,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 2,
        BIAS=False,
        EVEN_K=False,
        grid=(1,),
    )
    return None

def try_compile_config_BM64_BN64_BK256_GM32_nW8_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM64_BN64_BK256_GM32_nW8_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM64_BN64_BK256_GM32_nW8_nS2_EU0_kP2_mfma32: ', e, flush=True)
        return False


def matmul_BM64_BN128_BK64_GM1_nW4_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM64_BN128_BK64_GM1_nW4_nS2_EU0_kP1_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 128,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 1,
        num_warps = 4,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 1,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM64_BN128_BK64_GM1_nW4_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM64_BN128_BK64_GM1_nW4_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM64_BN128_BK64_GM1_nW4_nS2_EU0_kP1_mfma16: ', e, flush=True)
        return False


def matmul_BM64_BN128_BK64_GM1_nW4_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM64_BN128_BK64_GM1_nW4_nS2_EU0_kP2_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 128,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 1,
        num_warps = 4,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 2,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM64_BN128_BK64_GM1_nW4_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM64_BN128_BK64_GM1_nW4_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM64_BN128_BK64_GM1_nW4_nS2_EU0_kP2_mfma16: ', e, flush=True)
        return False


def matmul_BM64_BN128_BK64_GM1_nW4_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM64_BN128_BK64_GM1_nW4_nS2_EU0_kP1_mfma32.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 128,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 1,
        num_warps = 4,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 1,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM64_BN128_BK64_GM1_nW4_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM64_BN128_BK64_GM1_nW4_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM64_BN128_BK64_GM1_nW4_nS2_EU0_kP1_mfma32: ', e, flush=True)
        return False


def matmul_BM64_BN128_BK64_GM1_nW4_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM64_BN128_BK64_GM1_nW4_nS2_EU0_kP2_mfma32.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 128,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 1,
        num_warps = 4,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 2,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM64_BN128_BK64_GM1_nW4_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM64_BN128_BK64_GM1_nW4_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM64_BN128_BK64_GM1_nW4_nS2_EU0_kP2_mfma32: ', e, flush=True)
        return False


def matmul_BM64_BN128_BK64_GM4_nW4_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM64_BN128_BK64_GM4_nW4_nS2_EU0_kP1_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 128,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 4,
        num_warps = 4,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 1,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM64_BN128_BK64_GM4_nW4_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM64_BN128_BK64_GM4_nW4_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM64_BN128_BK64_GM4_nW4_nS2_EU0_kP1_mfma16: ', e, flush=True)
        return False


def matmul_BM64_BN128_BK64_GM4_nW4_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM64_BN128_BK64_GM4_nW4_nS2_EU0_kP2_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 128,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 4,
        num_warps = 4,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 2,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM64_BN128_BK64_GM4_nW4_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM64_BN128_BK64_GM4_nW4_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM64_BN128_BK64_GM4_nW4_nS2_EU0_kP2_mfma16: ', e, flush=True)
        return False


def matmul_BM64_BN128_BK64_GM4_nW4_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM64_BN128_BK64_GM4_nW4_nS2_EU0_kP1_mfma32.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 128,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 4,
        num_warps = 4,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 1,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM64_BN128_BK64_GM4_nW4_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM64_BN128_BK64_GM4_nW4_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM64_BN128_BK64_GM4_nW4_nS2_EU0_kP1_mfma32: ', e, flush=True)
        return False


def matmul_BM64_BN128_BK64_GM4_nW4_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM64_BN128_BK64_GM4_nW4_nS2_EU0_kP2_mfma32.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 128,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 4,
        num_warps = 4,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 2,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM64_BN128_BK64_GM4_nW4_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM64_BN128_BK64_GM4_nW4_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM64_BN128_BK64_GM4_nW4_nS2_EU0_kP2_mfma32: ', e, flush=True)
        return False


def matmul_BM64_BN128_BK64_GM8_nW4_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM64_BN128_BK64_GM8_nW4_nS2_EU0_kP1_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 128,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 8,
        num_warps = 4,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 1,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM64_BN128_BK64_GM8_nW4_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM64_BN128_BK64_GM8_nW4_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM64_BN128_BK64_GM8_nW4_nS2_EU0_kP1_mfma16: ', e, flush=True)
        return False


def matmul_BM64_BN128_BK64_GM8_nW4_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM64_BN128_BK64_GM8_nW4_nS2_EU0_kP2_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 128,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 8,
        num_warps = 4,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 2,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM64_BN128_BK64_GM8_nW4_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM64_BN128_BK64_GM8_nW4_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM64_BN128_BK64_GM8_nW4_nS2_EU0_kP2_mfma16: ', e, flush=True)
        return False


def matmul_BM64_BN128_BK64_GM8_nW4_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM64_BN128_BK64_GM8_nW4_nS2_EU0_kP1_mfma32.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 128,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 8,
        num_warps = 4,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 1,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM64_BN128_BK64_GM8_nW4_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM64_BN128_BK64_GM8_nW4_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM64_BN128_BK64_GM8_nW4_nS2_EU0_kP1_mfma32: ', e, flush=True)
        return False


def matmul_BM64_BN128_BK64_GM8_nW4_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM64_BN128_BK64_GM8_nW4_nS2_EU0_kP2_mfma32.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 128,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 8,
        num_warps = 4,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 2,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM64_BN128_BK64_GM8_nW4_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM64_BN128_BK64_GM8_nW4_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM64_BN128_BK64_GM8_nW4_nS2_EU0_kP2_mfma32: ', e, flush=True)
        return False


def matmul_BM64_BN128_BK64_GM16_nW4_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM64_BN128_BK64_GM16_nW4_nS2_EU0_kP1_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 128,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 16,
        num_warps = 4,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 1,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM64_BN128_BK64_GM16_nW4_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM64_BN128_BK64_GM16_nW4_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM64_BN128_BK64_GM16_nW4_nS2_EU0_kP1_mfma16: ', e, flush=True)
        return False


def matmul_BM64_BN128_BK64_GM16_nW4_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM64_BN128_BK64_GM16_nW4_nS2_EU0_kP2_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 128,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 16,
        num_warps = 4,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 2,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM64_BN128_BK64_GM16_nW4_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM64_BN128_BK64_GM16_nW4_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM64_BN128_BK64_GM16_nW4_nS2_EU0_kP2_mfma16: ', e, flush=True)
        return False


def matmul_BM64_BN128_BK64_GM16_nW4_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM64_BN128_BK64_GM16_nW4_nS2_EU0_kP1_mfma32.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 128,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 16,
        num_warps = 4,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 1,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM64_BN128_BK64_GM16_nW4_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM64_BN128_BK64_GM16_nW4_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM64_BN128_BK64_GM16_nW4_nS2_EU0_kP1_mfma32: ', e, flush=True)
        return False


def matmul_BM64_BN128_BK64_GM16_nW4_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM64_BN128_BK64_GM16_nW4_nS2_EU0_kP2_mfma32.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 128,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 16,
        num_warps = 4,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 2,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM64_BN128_BK64_GM16_nW4_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM64_BN128_BK64_GM16_nW4_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM64_BN128_BK64_GM16_nW4_nS2_EU0_kP2_mfma32: ', e, flush=True)
        return False


def matmul_BM64_BN128_BK64_GM32_nW4_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM64_BN128_BK64_GM32_nW4_nS2_EU0_kP1_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 128,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 32,
        num_warps = 4,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 1,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM64_BN128_BK64_GM32_nW4_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM64_BN128_BK64_GM32_nW4_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM64_BN128_BK64_GM32_nW4_nS2_EU0_kP1_mfma16: ', e, flush=True)
        return False


def matmul_BM64_BN128_BK64_GM32_nW4_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM64_BN128_BK64_GM32_nW4_nS2_EU0_kP2_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 128,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 32,
        num_warps = 4,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 2,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM64_BN128_BK64_GM32_nW4_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM64_BN128_BK64_GM32_nW4_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM64_BN128_BK64_GM32_nW4_nS2_EU0_kP2_mfma16: ', e, flush=True)
        return False


def matmul_BM64_BN128_BK64_GM32_nW4_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM64_BN128_BK64_GM32_nW4_nS2_EU0_kP1_mfma32.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 128,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 32,
        num_warps = 4,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 1,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM64_BN128_BK64_GM32_nW4_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM64_BN128_BK64_GM32_nW4_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM64_BN128_BK64_GM32_nW4_nS2_EU0_kP1_mfma32: ', e, flush=True)
        return False


def matmul_BM64_BN128_BK64_GM32_nW4_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM64_BN128_BK64_GM32_nW4_nS2_EU0_kP2_mfma32.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 128,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 32,
        num_warps = 4,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 2,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM64_BN128_BK64_GM32_nW4_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM64_BN128_BK64_GM32_nW4_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM64_BN128_BK64_GM32_nW4_nS2_EU0_kP2_mfma32: ', e, flush=True)
        return False


def matmul_BM64_BN128_BK64_GM1_nW8_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM64_BN128_BK64_GM1_nW8_nS2_EU0_kP1_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 128,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 1,
        num_warps = 8,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 1,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM64_BN128_BK64_GM1_nW8_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM64_BN128_BK64_GM1_nW8_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM64_BN128_BK64_GM1_nW8_nS2_EU0_kP1_mfma16: ', e, flush=True)
        return False


def matmul_BM64_BN128_BK64_GM1_nW8_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM64_BN128_BK64_GM1_nW8_nS2_EU0_kP2_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 128,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 1,
        num_warps = 8,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 2,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM64_BN128_BK64_GM1_nW8_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM64_BN128_BK64_GM1_nW8_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM64_BN128_BK64_GM1_nW8_nS2_EU0_kP2_mfma16: ', e, flush=True)
        return False


def matmul_BM64_BN128_BK64_GM1_nW8_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM64_BN128_BK64_GM1_nW8_nS2_EU0_kP1_mfma32.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 128,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 1,
        num_warps = 8,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 1,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM64_BN128_BK64_GM1_nW8_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM64_BN128_BK64_GM1_nW8_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM64_BN128_BK64_GM1_nW8_nS2_EU0_kP1_mfma32: ', e, flush=True)
        return False


def matmul_BM64_BN128_BK64_GM1_nW8_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM64_BN128_BK64_GM1_nW8_nS2_EU0_kP2_mfma32.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 128,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 1,
        num_warps = 8,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 2,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM64_BN128_BK64_GM1_nW8_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM64_BN128_BK64_GM1_nW8_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM64_BN128_BK64_GM1_nW8_nS2_EU0_kP2_mfma32: ', e, flush=True)
        return False


def matmul_BM64_BN128_BK64_GM4_nW8_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM64_BN128_BK64_GM4_nW8_nS2_EU0_kP1_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 128,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 4,
        num_warps = 8,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 1,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM64_BN128_BK64_GM4_nW8_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM64_BN128_BK64_GM4_nW8_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM64_BN128_BK64_GM4_nW8_nS2_EU0_kP1_mfma16: ', e, flush=True)
        return False


def matmul_BM64_BN128_BK64_GM4_nW8_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM64_BN128_BK64_GM4_nW8_nS2_EU0_kP2_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 128,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 4,
        num_warps = 8,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 2,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM64_BN128_BK64_GM4_nW8_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM64_BN128_BK64_GM4_nW8_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM64_BN128_BK64_GM4_nW8_nS2_EU0_kP2_mfma16: ', e, flush=True)
        return False


def matmul_BM64_BN128_BK64_GM4_nW8_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM64_BN128_BK64_GM4_nW8_nS2_EU0_kP1_mfma32.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 128,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 4,
        num_warps = 8,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 1,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM64_BN128_BK64_GM4_nW8_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM64_BN128_BK64_GM4_nW8_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM64_BN128_BK64_GM4_nW8_nS2_EU0_kP1_mfma32: ', e, flush=True)
        return False


def matmul_BM64_BN128_BK64_GM4_nW8_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM64_BN128_BK64_GM4_nW8_nS2_EU0_kP2_mfma32.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 128,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 4,
        num_warps = 8,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 2,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM64_BN128_BK64_GM4_nW8_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM64_BN128_BK64_GM4_nW8_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM64_BN128_BK64_GM4_nW8_nS2_EU0_kP2_mfma32: ', e, flush=True)
        return False


def matmul_BM64_BN128_BK64_GM8_nW8_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM64_BN128_BK64_GM8_nW8_nS2_EU0_kP1_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 128,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 8,
        num_warps = 8,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 1,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM64_BN128_BK64_GM8_nW8_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM64_BN128_BK64_GM8_nW8_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM64_BN128_BK64_GM8_nW8_nS2_EU0_kP1_mfma16: ', e, flush=True)
        return False


def matmul_BM64_BN128_BK64_GM8_nW8_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM64_BN128_BK64_GM8_nW8_nS2_EU0_kP2_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 128,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 8,
        num_warps = 8,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 2,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM64_BN128_BK64_GM8_nW8_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM64_BN128_BK64_GM8_nW8_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM64_BN128_BK64_GM8_nW8_nS2_EU0_kP2_mfma16: ', e, flush=True)
        return False


def matmul_BM64_BN128_BK64_GM8_nW8_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM64_BN128_BK64_GM8_nW8_nS2_EU0_kP1_mfma32.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 128,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 8,
        num_warps = 8,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 1,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM64_BN128_BK64_GM8_nW8_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM64_BN128_BK64_GM8_nW8_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM64_BN128_BK64_GM8_nW8_nS2_EU0_kP1_mfma32: ', e, flush=True)
        return False


def matmul_BM64_BN128_BK64_GM8_nW8_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM64_BN128_BK64_GM8_nW8_nS2_EU0_kP2_mfma32.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 128,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 8,
        num_warps = 8,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 2,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM64_BN128_BK64_GM8_nW8_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM64_BN128_BK64_GM8_nW8_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM64_BN128_BK64_GM8_nW8_nS2_EU0_kP2_mfma32: ', e, flush=True)
        return False


def matmul_BM64_BN128_BK64_GM16_nW8_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM64_BN128_BK64_GM16_nW8_nS2_EU0_kP1_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 128,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 16,
        num_warps = 8,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 1,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM64_BN128_BK64_GM16_nW8_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM64_BN128_BK64_GM16_nW8_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM64_BN128_BK64_GM16_nW8_nS2_EU0_kP1_mfma16: ', e, flush=True)
        return False


def matmul_BM64_BN128_BK64_GM16_nW8_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM64_BN128_BK64_GM16_nW8_nS2_EU0_kP2_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 128,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 16,
        num_warps = 8,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 2,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM64_BN128_BK64_GM16_nW8_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM64_BN128_BK64_GM16_nW8_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM64_BN128_BK64_GM16_nW8_nS2_EU0_kP2_mfma16: ', e, flush=True)
        return False


def matmul_BM64_BN128_BK64_GM16_nW8_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM64_BN128_BK64_GM16_nW8_nS2_EU0_kP1_mfma32.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 128,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 16,
        num_warps = 8,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 1,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM64_BN128_BK64_GM16_nW8_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM64_BN128_BK64_GM16_nW8_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM64_BN128_BK64_GM16_nW8_nS2_EU0_kP1_mfma32: ', e, flush=True)
        return False


def matmul_BM64_BN128_BK64_GM16_nW8_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM64_BN128_BK64_GM16_nW8_nS2_EU0_kP2_mfma32.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 128,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 16,
        num_warps = 8,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 2,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM64_BN128_BK64_GM16_nW8_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM64_BN128_BK64_GM16_nW8_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM64_BN128_BK64_GM16_nW8_nS2_EU0_kP2_mfma32: ', e, flush=True)
        return False


def matmul_BM64_BN128_BK64_GM32_nW8_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM64_BN128_BK64_GM32_nW8_nS2_EU0_kP1_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 128,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 32,
        num_warps = 8,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 1,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM64_BN128_BK64_GM32_nW8_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM64_BN128_BK64_GM32_nW8_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM64_BN128_BK64_GM32_nW8_nS2_EU0_kP1_mfma16: ', e, flush=True)
        return False


def matmul_BM64_BN128_BK64_GM32_nW8_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM64_BN128_BK64_GM32_nW8_nS2_EU0_kP2_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 128,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 32,
        num_warps = 8,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 2,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM64_BN128_BK64_GM32_nW8_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM64_BN128_BK64_GM32_nW8_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM64_BN128_BK64_GM32_nW8_nS2_EU0_kP2_mfma16: ', e, flush=True)
        return False


def matmul_BM64_BN128_BK64_GM32_nW8_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM64_BN128_BK64_GM32_nW8_nS2_EU0_kP1_mfma32.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 128,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 32,
        num_warps = 8,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 1,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM64_BN128_BK64_GM32_nW8_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM64_BN128_BK64_GM32_nW8_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM64_BN128_BK64_GM32_nW8_nS2_EU0_kP1_mfma32: ', e, flush=True)
        return False


def matmul_BM64_BN128_BK64_GM32_nW8_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM64_BN128_BK64_GM32_nW8_nS2_EU0_kP2_mfma32.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 128,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 32,
        num_warps = 8,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 2,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM64_BN128_BK64_GM32_nW8_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM64_BN128_BK64_GM32_nW8_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM64_BN128_BK64_GM32_nW8_nS2_EU0_kP2_mfma32: ', e, flush=True)
        return False


def matmul_BM64_BN128_BK128_GM1_nW4_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM64_BN128_BK128_GM1_nW4_nS2_EU0_kP1_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 128,
        BLOCK_SIZE_K = 128,
        GROUP_SIZE_M = 1,
        num_warps = 4,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 1,
        BIAS=False,
        EVEN_K=False,
        grid=(1,),
    )
    return None

def try_compile_config_BM64_BN128_BK128_GM1_nW4_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM64_BN128_BK128_GM1_nW4_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM64_BN128_BK128_GM1_nW4_nS2_EU0_kP1_mfma16: ', e, flush=True)
        return False


def matmul_BM64_BN128_BK128_GM1_nW4_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM64_BN128_BK128_GM1_nW4_nS2_EU0_kP2_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 128,
        BLOCK_SIZE_K = 128,
        GROUP_SIZE_M = 1,
        num_warps = 4,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 2,
        BIAS=False,
        EVEN_K=False,
        grid=(1,),
    )
    return None

def try_compile_config_BM64_BN128_BK128_GM1_nW4_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM64_BN128_BK128_GM1_nW4_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM64_BN128_BK128_GM1_nW4_nS2_EU0_kP2_mfma16: ', e, flush=True)
        return False


def matmul_BM64_BN128_BK128_GM1_nW4_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM64_BN128_BK128_GM1_nW4_nS2_EU0_kP1_mfma32.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 128,
        BLOCK_SIZE_K = 128,
        GROUP_SIZE_M = 1,
        num_warps = 4,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 1,
        BIAS=False,
        EVEN_K=False,
        grid=(1,),
    )
    return None

def try_compile_config_BM64_BN128_BK128_GM1_nW4_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM64_BN128_BK128_GM1_nW4_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM64_BN128_BK128_GM1_nW4_nS2_EU0_kP1_mfma32: ', e, flush=True)
        return False


def matmul_BM64_BN128_BK128_GM1_nW4_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM64_BN128_BK128_GM1_nW4_nS2_EU0_kP2_mfma32.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 128,
        BLOCK_SIZE_K = 128,
        GROUP_SIZE_M = 1,
        num_warps = 4,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 2,
        BIAS=False,
        EVEN_K=False,
        grid=(1,),
    )
    return None

def try_compile_config_BM64_BN128_BK128_GM1_nW4_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM64_BN128_BK128_GM1_nW4_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM64_BN128_BK128_GM1_nW4_nS2_EU0_kP2_mfma32: ', e, flush=True)
        return False


def matmul_BM64_BN128_BK128_GM4_nW4_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM64_BN128_BK128_GM4_nW4_nS2_EU0_kP1_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 128,
        BLOCK_SIZE_K = 128,
        GROUP_SIZE_M = 4,
        num_warps = 4,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 1,
        BIAS=False,
        EVEN_K=False,
        grid=(1,),
    )
    return None

def try_compile_config_BM64_BN128_BK128_GM4_nW4_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM64_BN128_BK128_GM4_nW4_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM64_BN128_BK128_GM4_nW4_nS2_EU0_kP1_mfma16: ', e, flush=True)
        return False


def matmul_BM64_BN128_BK128_GM4_nW4_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM64_BN128_BK128_GM4_nW4_nS2_EU0_kP2_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 128,
        BLOCK_SIZE_K = 128,
        GROUP_SIZE_M = 4,
        num_warps = 4,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 2,
        BIAS=False,
        EVEN_K=False,
        grid=(1,),
    )
    return None

def try_compile_config_BM64_BN128_BK128_GM4_nW4_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM64_BN128_BK128_GM4_nW4_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM64_BN128_BK128_GM4_nW4_nS2_EU0_kP2_mfma16: ', e, flush=True)
        return False


def matmul_BM64_BN128_BK128_GM4_nW4_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM64_BN128_BK128_GM4_nW4_nS2_EU0_kP1_mfma32.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 128,
        BLOCK_SIZE_K = 128,
        GROUP_SIZE_M = 4,
        num_warps = 4,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 1,
        BIAS=False,
        EVEN_K=False,
        grid=(1,),
    )
    return None

def try_compile_config_BM64_BN128_BK128_GM4_nW4_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM64_BN128_BK128_GM4_nW4_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM64_BN128_BK128_GM4_nW4_nS2_EU0_kP1_mfma32: ', e, flush=True)
        return False


def matmul_BM64_BN128_BK128_GM4_nW4_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM64_BN128_BK128_GM4_nW4_nS2_EU0_kP2_mfma32.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 128,
        BLOCK_SIZE_K = 128,
        GROUP_SIZE_M = 4,
        num_warps = 4,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 2,
        BIAS=False,
        EVEN_K=False,
        grid=(1,),
    )
    return None

def try_compile_config_BM64_BN128_BK128_GM4_nW4_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM64_BN128_BK128_GM4_nW4_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM64_BN128_BK128_GM4_nW4_nS2_EU0_kP2_mfma32: ', e, flush=True)
        return False


def matmul_BM64_BN128_BK128_GM8_nW4_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM64_BN128_BK128_GM8_nW4_nS2_EU0_kP1_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 128,
        BLOCK_SIZE_K = 128,
        GROUP_SIZE_M = 8,
        num_warps = 4,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 1,
        BIAS=False,
        EVEN_K=False,
        grid=(1,),
    )
    return None

def try_compile_config_BM64_BN128_BK128_GM8_nW4_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM64_BN128_BK128_GM8_nW4_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM64_BN128_BK128_GM8_nW4_nS2_EU0_kP1_mfma16: ', e, flush=True)
        return False


def matmul_BM64_BN128_BK128_GM8_nW4_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM64_BN128_BK128_GM8_nW4_nS2_EU0_kP2_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 128,
        BLOCK_SIZE_K = 128,
        GROUP_SIZE_M = 8,
        num_warps = 4,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 2,
        BIAS=False,
        EVEN_K=False,
        grid=(1,),
    )
    return None

def try_compile_config_BM64_BN128_BK128_GM8_nW4_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM64_BN128_BK128_GM8_nW4_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM64_BN128_BK128_GM8_nW4_nS2_EU0_kP2_mfma16: ', e, flush=True)
        return False


def matmul_BM64_BN128_BK128_GM8_nW4_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM64_BN128_BK128_GM8_nW4_nS2_EU0_kP1_mfma32.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 128,
        BLOCK_SIZE_K = 128,
        GROUP_SIZE_M = 8,
        num_warps = 4,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 1,
        BIAS=False,
        EVEN_K=False,
        grid=(1,),
    )
    return None

def try_compile_config_BM64_BN128_BK128_GM8_nW4_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM64_BN128_BK128_GM8_nW4_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM64_BN128_BK128_GM8_nW4_nS2_EU0_kP1_mfma32: ', e, flush=True)
        return False


def matmul_BM64_BN128_BK128_GM8_nW4_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM64_BN128_BK128_GM8_nW4_nS2_EU0_kP2_mfma32.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 128,
        BLOCK_SIZE_K = 128,
        GROUP_SIZE_M = 8,
        num_warps = 4,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 2,
        BIAS=False,
        EVEN_K=False,
        grid=(1,),
    )
    return None

def try_compile_config_BM64_BN128_BK128_GM8_nW4_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM64_BN128_BK128_GM8_nW4_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM64_BN128_BK128_GM8_nW4_nS2_EU0_kP2_mfma32: ', e, flush=True)
        return False


def matmul_BM64_BN128_BK128_GM16_nW4_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM64_BN128_BK128_GM16_nW4_nS2_EU0_kP1_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 128,
        BLOCK_SIZE_K = 128,
        GROUP_SIZE_M = 16,
        num_warps = 4,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 1,
        BIAS=False,
        EVEN_K=False,
        grid=(1,),
    )
    return None

def try_compile_config_BM64_BN128_BK128_GM16_nW4_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM64_BN128_BK128_GM16_nW4_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM64_BN128_BK128_GM16_nW4_nS2_EU0_kP1_mfma16: ', e, flush=True)
        return False


def matmul_BM64_BN128_BK128_GM16_nW4_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM64_BN128_BK128_GM16_nW4_nS2_EU0_kP2_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 128,
        BLOCK_SIZE_K = 128,
        GROUP_SIZE_M = 16,
        num_warps = 4,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 2,
        BIAS=False,
        EVEN_K=False,
        grid=(1,),
    )
    return None

def try_compile_config_BM64_BN128_BK128_GM16_nW4_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM64_BN128_BK128_GM16_nW4_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM64_BN128_BK128_GM16_nW4_nS2_EU0_kP2_mfma16: ', e, flush=True)
        return False


def matmul_BM64_BN128_BK128_GM16_nW4_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM64_BN128_BK128_GM16_nW4_nS2_EU0_kP1_mfma32.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 128,
        BLOCK_SIZE_K = 128,
        GROUP_SIZE_M = 16,
        num_warps = 4,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 1,
        BIAS=False,
        EVEN_K=False,
        grid=(1,),
    )
    return None

def try_compile_config_BM64_BN128_BK128_GM16_nW4_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM64_BN128_BK128_GM16_nW4_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM64_BN128_BK128_GM16_nW4_nS2_EU0_kP1_mfma32: ', e, flush=True)
        return False


def matmul_BM64_BN128_BK128_GM16_nW4_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM64_BN128_BK128_GM16_nW4_nS2_EU0_kP2_mfma32.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 128,
        BLOCK_SIZE_K = 128,
        GROUP_SIZE_M = 16,
        num_warps = 4,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 2,
        BIAS=False,
        EVEN_K=False,
        grid=(1,),
    )
    return None

def try_compile_config_BM64_BN128_BK128_GM16_nW4_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM64_BN128_BK128_GM16_nW4_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM64_BN128_BK128_GM16_nW4_nS2_EU0_kP2_mfma32: ', e, flush=True)
        return False


def matmul_BM64_BN128_BK128_GM32_nW4_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM64_BN128_BK128_GM32_nW4_nS2_EU0_kP1_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 128,
        BLOCK_SIZE_K = 128,
        GROUP_SIZE_M = 32,
        num_warps = 4,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 1,
        BIAS=False,
        EVEN_K=False,
        grid=(1,),
    )
    return None

def try_compile_config_BM64_BN128_BK128_GM32_nW4_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM64_BN128_BK128_GM32_nW4_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM64_BN128_BK128_GM32_nW4_nS2_EU0_kP1_mfma16: ', e, flush=True)
        return False


def matmul_BM64_BN128_BK128_GM32_nW4_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM64_BN128_BK128_GM32_nW4_nS2_EU0_kP2_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 128,
        BLOCK_SIZE_K = 128,
        GROUP_SIZE_M = 32,
        num_warps = 4,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 2,
        BIAS=False,
        EVEN_K=False,
        grid=(1,),
    )
    return None

def try_compile_config_BM64_BN128_BK128_GM32_nW4_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM64_BN128_BK128_GM32_nW4_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM64_BN128_BK128_GM32_nW4_nS2_EU0_kP2_mfma16: ', e, flush=True)
        return False


def matmul_BM64_BN128_BK128_GM32_nW4_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM64_BN128_BK128_GM32_nW4_nS2_EU0_kP1_mfma32.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 128,
        BLOCK_SIZE_K = 128,
        GROUP_SIZE_M = 32,
        num_warps = 4,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 1,
        BIAS=False,
        EVEN_K=False,
        grid=(1,),
    )
    return None

def try_compile_config_BM64_BN128_BK128_GM32_nW4_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM64_BN128_BK128_GM32_nW4_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM64_BN128_BK128_GM32_nW4_nS2_EU0_kP1_mfma32: ', e, flush=True)
        return False


def matmul_BM64_BN128_BK128_GM32_nW4_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM64_BN128_BK128_GM32_nW4_nS2_EU0_kP2_mfma32.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 128,
        BLOCK_SIZE_K = 128,
        GROUP_SIZE_M = 32,
        num_warps = 4,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 2,
        BIAS=False,
        EVEN_K=False,
        grid=(1,),
    )
    return None

def try_compile_config_BM64_BN128_BK128_GM32_nW4_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM64_BN128_BK128_GM32_nW4_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM64_BN128_BK128_GM32_nW4_nS2_EU0_kP2_mfma32: ', e, flush=True)
        return False


def matmul_BM64_BN128_BK128_GM1_nW8_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM64_BN128_BK128_GM1_nW8_nS2_EU0_kP1_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 128,
        BLOCK_SIZE_K = 128,
        GROUP_SIZE_M = 1,
        num_warps = 8,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 1,
        BIAS=False,
        EVEN_K=False,
        grid=(1,),
    )
    return None

def try_compile_config_BM64_BN128_BK128_GM1_nW8_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM64_BN128_BK128_GM1_nW8_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM64_BN128_BK128_GM1_nW8_nS2_EU0_kP1_mfma16: ', e, flush=True)
        return False


def matmul_BM64_BN128_BK128_GM1_nW8_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM64_BN128_BK128_GM1_nW8_nS2_EU0_kP2_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 128,
        BLOCK_SIZE_K = 128,
        GROUP_SIZE_M = 1,
        num_warps = 8,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 2,
        BIAS=False,
        EVEN_K=False,
        grid=(1,),
    )
    return None

def try_compile_config_BM64_BN128_BK128_GM1_nW8_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM64_BN128_BK128_GM1_nW8_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM64_BN128_BK128_GM1_nW8_nS2_EU0_kP2_mfma16: ', e, flush=True)
        return False


def matmul_BM64_BN128_BK128_GM1_nW8_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM64_BN128_BK128_GM1_nW8_nS2_EU0_kP1_mfma32.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 128,
        BLOCK_SIZE_K = 128,
        GROUP_SIZE_M = 1,
        num_warps = 8,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 1,
        BIAS=False,
        EVEN_K=False,
        grid=(1,),
    )
    return None

def try_compile_config_BM64_BN128_BK128_GM1_nW8_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM64_BN128_BK128_GM1_nW8_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM64_BN128_BK128_GM1_nW8_nS2_EU0_kP1_mfma32: ', e, flush=True)
        return False


def matmul_BM64_BN128_BK128_GM1_nW8_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM64_BN128_BK128_GM1_nW8_nS2_EU0_kP2_mfma32.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 128,
        BLOCK_SIZE_K = 128,
        GROUP_SIZE_M = 1,
        num_warps = 8,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 2,
        BIAS=False,
        EVEN_K=False,
        grid=(1,),
    )
    return None

def try_compile_config_BM64_BN128_BK128_GM1_nW8_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM64_BN128_BK128_GM1_nW8_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM64_BN128_BK128_GM1_nW8_nS2_EU0_kP2_mfma32: ', e, flush=True)
        return False


def matmul_BM64_BN128_BK128_GM4_nW8_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM64_BN128_BK128_GM4_nW8_nS2_EU0_kP1_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 128,
        BLOCK_SIZE_K = 128,
        GROUP_SIZE_M = 4,
        num_warps = 8,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 1,
        BIAS=False,
        EVEN_K=False,
        grid=(1,),
    )
    return None

def try_compile_config_BM64_BN128_BK128_GM4_nW8_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM64_BN128_BK128_GM4_nW8_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM64_BN128_BK128_GM4_nW8_nS2_EU0_kP1_mfma16: ', e, flush=True)
        return False


def matmul_BM64_BN128_BK128_GM4_nW8_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM64_BN128_BK128_GM4_nW8_nS2_EU0_kP2_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 128,
        BLOCK_SIZE_K = 128,
        GROUP_SIZE_M = 4,
        num_warps = 8,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 2,
        BIAS=False,
        EVEN_K=False,
        grid=(1,),
    )
    return None

def try_compile_config_BM64_BN128_BK128_GM4_nW8_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM64_BN128_BK128_GM4_nW8_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM64_BN128_BK128_GM4_nW8_nS2_EU0_kP2_mfma16: ', e, flush=True)
        return False


def matmul_BM64_BN128_BK128_GM4_nW8_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM64_BN128_BK128_GM4_nW8_nS2_EU0_kP1_mfma32.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 128,
        BLOCK_SIZE_K = 128,
        GROUP_SIZE_M = 4,
        num_warps = 8,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 1,
        BIAS=False,
        EVEN_K=False,
        grid=(1,),
    )
    return None

def try_compile_config_BM64_BN128_BK128_GM4_nW8_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM64_BN128_BK128_GM4_nW8_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM64_BN128_BK128_GM4_nW8_nS2_EU0_kP1_mfma32: ', e, flush=True)
        return False


def matmul_BM64_BN128_BK128_GM4_nW8_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM64_BN128_BK128_GM4_nW8_nS2_EU0_kP2_mfma32.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 128,
        BLOCK_SIZE_K = 128,
        GROUP_SIZE_M = 4,
        num_warps = 8,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 2,
        BIAS=False,
        EVEN_K=False,
        grid=(1,),
    )
    return None

def try_compile_config_BM64_BN128_BK128_GM4_nW8_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM64_BN128_BK128_GM4_nW8_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM64_BN128_BK128_GM4_nW8_nS2_EU0_kP2_mfma32: ', e, flush=True)
        return False


def matmul_BM64_BN128_BK128_GM8_nW8_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM64_BN128_BK128_GM8_nW8_nS2_EU0_kP1_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 128,
        BLOCK_SIZE_K = 128,
        GROUP_SIZE_M = 8,
        num_warps = 8,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 1,
        BIAS=False,
        EVEN_K=False,
        grid=(1,),
    )
    return None

def try_compile_config_BM64_BN128_BK128_GM8_nW8_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM64_BN128_BK128_GM8_nW8_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM64_BN128_BK128_GM8_nW8_nS2_EU0_kP1_mfma16: ', e, flush=True)
        return False


def matmul_BM64_BN128_BK128_GM8_nW8_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM64_BN128_BK128_GM8_nW8_nS2_EU0_kP2_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 128,
        BLOCK_SIZE_K = 128,
        GROUP_SIZE_M = 8,
        num_warps = 8,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 2,
        BIAS=False,
        EVEN_K=False,
        grid=(1,),
    )
    return None

def try_compile_config_BM64_BN128_BK128_GM8_nW8_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM64_BN128_BK128_GM8_nW8_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM64_BN128_BK128_GM8_nW8_nS2_EU0_kP2_mfma16: ', e, flush=True)
        return False


def matmul_BM64_BN128_BK128_GM8_nW8_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM64_BN128_BK128_GM8_nW8_nS2_EU0_kP1_mfma32.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 128,
        BLOCK_SIZE_K = 128,
        GROUP_SIZE_M = 8,
        num_warps = 8,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 1,
        BIAS=False,
        EVEN_K=False,
        grid=(1,),
    )
    return None

def try_compile_config_BM64_BN128_BK128_GM8_nW8_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM64_BN128_BK128_GM8_nW8_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM64_BN128_BK128_GM8_nW8_nS2_EU0_kP1_mfma32: ', e, flush=True)
        return False


def matmul_BM64_BN128_BK128_GM8_nW8_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM64_BN128_BK128_GM8_nW8_nS2_EU0_kP2_mfma32.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 128,
        BLOCK_SIZE_K = 128,
        GROUP_SIZE_M = 8,
        num_warps = 8,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 2,
        BIAS=False,
        EVEN_K=False,
        grid=(1,),
    )
    return None

def try_compile_config_BM64_BN128_BK128_GM8_nW8_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM64_BN128_BK128_GM8_nW8_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM64_BN128_BK128_GM8_nW8_nS2_EU0_kP2_mfma32: ', e, flush=True)
        return False


def matmul_BM64_BN128_BK128_GM16_nW8_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM64_BN128_BK128_GM16_nW8_nS2_EU0_kP1_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 128,
        BLOCK_SIZE_K = 128,
        GROUP_SIZE_M = 16,
        num_warps = 8,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 1,
        BIAS=False,
        EVEN_K=False,
        grid=(1,),
    )
    return None

def try_compile_config_BM64_BN128_BK128_GM16_nW8_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM64_BN128_BK128_GM16_nW8_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM64_BN128_BK128_GM16_nW8_nS2_EU0_kP1_mfma16: ', e, flush=True)
        return False


def matmul_BM64_BN128_BK128_GM16_nW8_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM64_BN128_BK128_GM16_nW8_nS2_EU0_kP2_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 128,
        BLOCK_SIZE_K = 128,
        GROUP_SIZE_M = 16,
        num_warps = 8,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 2,
        BIAS=False,
        EVEN_K=False,
        grid=(1,),
    )
    return None

def try_compile_config_BM64_BN128_BK128_GM16_nW8_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM64_BN128_BK128_GM16_nW8_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM64_BN128_BK128_GM16_nW8_nS2_EU0_kP2_mfma16: ', e, flush=True)
        return False


def matmul_BM64_BN128_BK128_GM16_nW8_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM64_BN128_BK128_GM16_nW8_nS2_EU0_kP1_mfma32.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 128,
        BLOCK_SIZE_K = 128,
        GROUP_SIZE_M = 16,
        num_warps = 8,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 1,
        BIAS=False,
        EVEN_K=False,
        grid=(1,),
    )
    return None

def try_compile_config_BM64_BN128_BK128_GM16_nW8_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM64_BN128_BK128_GM16_nW8_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM64_BN128_BK128_GM16_nW8_nS2_EU0_kP1_mfma32: ', e, flush=True)
        return False


def matmul_BM64_BN128_BK128_GM16_nW8_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM64_BN128_BK128_GM16_nW8_nS2_EU0_kP2_mfma32.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 128,
        BLOCK_SIZE_K = 128,
        GROUP_SIZE_M = 16,
        num_warps = 8,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 2,
        BIAS=False,
        EVEN_K=False,
        grid=(1,),
    )
    return None

def try_compile_config_BM64_BN128_BK128_GM16_nW8_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM64_BN128_BK128_GM16_nW8_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM64_BN128_BK128_GM16_nW8_nS2_EU0_kP2_mfma32: ', e, flush=True)
        return False


def matmul_BM64_BN128_BK128_GM32_nW8_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM64_BN128_BK128_GM32_nW8_nS2_EU0_kP1_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 128,
        BLOCK_SIZE_K = 128,
        GROUP_SIZE_M = 32,
        num_warps = 8,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 1,
        BIAS=False,
        EVEN_K=False,
        grid=(1,),
    )
    return None

def try_compile_config_BM64_BN128_BK128_GM32_nW8_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM64_BN128_BK128_GM32_nW8_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM64_BN128_BK128_GM32_nW8_nS2_EU0_kP1_mfma16: ', e, flush=True)
        return False


def matmul_BM64_BN128_BK128_GM32_nW8_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM64_BN128_BK128_GM32_nW8_nS2_EU0_kP2_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 128,
        BLOCK_SIZE_K = 128,
        GROUP_SIZE_M = 32,
        num_warps = 8,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 2,
        BIAS=False,
        EVEN_K=False,
        grid=(1,),
    )
    return None

def try_compile_config_BM64_BN128_BK128_GM32_nW8_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM64_BN128_BK128_GM32_nW8_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM64_BN128_BK128_GM32_nW8_nS2_EU0_kP2_mfma16: ', e, flush=True)
        return False


def matmul_BM64_BN128_BK128_GM32_nW8_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM64_BN128_BK128_GM32_nW8_nS2_EU0_kP1_mfma32.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 128,
        BLOCK_SIZE_K = 128,
        GROUP_SIZE_M = 32,
        num_warps = 8,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 1,
        BIAS=False,
        EVEN_K=False,
        grid=(1,),
    )
    return None

def try_compile_config_BM64_BN128_BK128_GM32_nW8_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM64_BN128_BK128_GM32_nW8_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM64_BN128_BK128_GM32_nW8_nS2_EU0_kP1_mfma32: ', e, flush=True)
        return False


def matmul_BM64_BN128_BK128_GM32_nW8_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM64_BN128_BK128_GM32_nW8_nS2_EU0_kP2_mfma32.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 128,
        BLOCK_SIZE_K = 128,
        GROUP_SIZE_M = 32,
        num_warps = 8,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 2,
        BIAS=False,
        EVEN_K=False,
        grid=(1,),
    )
    return None

def try_compile_config_BM64_BN128_BK128_GM32_nW8_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM64_BN128_BK128_GM32_nW8_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM64_BN128_BK128_GM32_nW8_nS2_EU0_kP2_mfma32: ', e, flush=True)
        return False


def matmul_BM64_BN256_BK64_GM1_nW4_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM64_BN256_BK64_GM1_nW4_nS2_EU0_kP1_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 256,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 1,
        num_warps = 4,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 1,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM64_BN256_BK64_GM1_nW4_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM64_BN256_BK64_GM1_nW4_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM64_BN256_BK64_GM1_nW4_nS2_EU0_kP1_mfma16: ', e, flush=True)
        return False


def matmul_BM64_BN256_BK64_GM1_nW4_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM64_BN256_BK64_GM1_nW4_nS2_EU0_kP2_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 256,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 1,
        num_warps = 4,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 2,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM64_BN256_BK64_GM1_nW4_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM64_BN256_BK64_GM1_nW4_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM64_BN256_BK64_GM1_nW4_nS2_EU0_kP2_mfma16: ', e, flush=True)
        return False


def matmul_BM64_BN256_BK64_GM1_nW4_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM64_BN256_BK64_GM1_nW4_nS2_EU0_kP1_mfma32.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 256,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 1,
        num_warps = 4,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 1,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM64_BN256_BK64_GM1_nW4_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM64_BN256_BK64_GM1_nW4_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM64_BN256_BK64_GM1_nW4_nS2_EU0_kP1_mfma32: ', e, flush=True)
        return False


def matmul_BM64_BN256_BK64_GM1_nW4_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM64_BN256_BK64_GM1_nW4_nS2_EU0_kP2_mfma32.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 256,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 1,
        num_warps = 4,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 2,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM64_BN256_BK64_GM1_nW4_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM64_BN256_BK64_GM1_nW4_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM64_BN256_BK64_GM1_nW4_nS2_EU0_kP2_mfma32: ', e, flush=True)
        return False


def matmul_BM64_BN256_BK64_GM4_nW4_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM64_BN256_BK64_GM4_nW4_nS2_EU0_kP1_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 256,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 4,
        num_warps = 4,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 1,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM64_BN256_BK64_GM4_nW4_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM64_BN256_BK64_GM4_nW4_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM64_BN256_BK64_GM4_nW4_nS2_EU0_kP1_mfma16: ', e, flush=True)
        return False


def matmul_BM64_BN256_BK64_GM4_nW4_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM64_BN256_BK64_GM4_nW4_nS2_EU0_kP2_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 256,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 4,
        num_warps = 4,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 2,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM64_BN256_BK64_GM4_nW4_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM64_BN256_BK64_GM4_nW4_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM64_BN256_BK64_GM4_nW4_nS2_EU0_kP2_mfma16: ', e, flush=True)
        return False


def matmul_BM64_BN256_BK64_GM4_nW4_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM64_BN256_BK64_GM4_nW4_nS2_EU0_kP1_mfma32.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 256,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 4,
        num_warps = 4,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 1,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM64_BN256_BK64_GM4_nW4_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM64_BN256_BK64_GM4_nW4_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM64_BN256_BK64_GM4_nW4_nS2_EU0_kP1_mfma32: ', e, flush=True)
        return False


def matmul_BM64_BN256_BK64_GM4_nW4_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM64_BN256_BK64_GM4_nW4_nS2_EU0_kP2_mfma32.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 256,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 4,
        num_warps = 4,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 2,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM64_BN256_BK64_GM4_nW4_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM64_BN256_BK64_GM4_nW4_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM64_BN256_BK64_GM4_nW4_nS2_EU0_kP2_mfma32: ', e, flush=True)
        return False


def matmul_BM64_BN256_BK64_GM8_nW4_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM64_BN256_BK64_GM8_nW4_nS2_EU0_kP1_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 256,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 8,
        num_warps = 4,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 1,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM64_BN256_BK64_GM8_nW4_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM64_BN256_BK64_GM8_nW4_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM64_BN256_BK64_GM8_nW4_nS2_EU0_kP1_mfma16: ', e, flush=True)
        return False


def matmul_BM64_BN256_BK64_GM8_nW4_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM64_BN256_BK64_GM8_nW4_nS2_EU0_kP2_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 256,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 8,
        num_warps = 4,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 2,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM64_BN256_BK64_GM8_nW4_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM64_BN256_BK64_GM8_nW4_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM64_BN256_BK64_GM8_nW4_nS2_EU0_kP2_mfma16: ', e, flush=True)
        return False


def matmul_BM64_BN256_BK64_GM8_nW4_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM64_BN256_BK64_GM8_nW4_nS2_EU0_kP1_mfma32.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 256,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 8,
        num_warps = 4,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 1,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM64_BN256_BK64_GM8_nW4_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM64_BN256_BK64_GM8_nW4_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM64_BN256_BK64_GM8_nW4_nS2_EU0_kP1_mfma32: ', e, flush=True)
        return False


def matmul_BM64_BN256_BK64_GM8_nW4_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM64_BN256_BK64_GM8_nW4_nS2_EU0_kP2_mfma32.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 256,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 8,
        num_warps = 4,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 2,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM64_BN256_BK64_GM8_nW4_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM64_BN256_BK64_GM8_nW4_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM64_BN256_BK64_GM8_nW4_nS2_EU0_kP2_mfma32: ', e, flush=True)
        return False


def matmul_BM64_BN256_BK64_GM16_nW4_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM64_BN256_BK64_GM16_nW4_nS2_EU0_kP1_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 256,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 16,
        num_warps = 4,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 1,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM64_BN256_BK64_GM16_nW4_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM64_BN256_BK64_GM16_nW4_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM64_BN256_BK64_GM16_nW4_nS2_EU0_kP1_mfma16: ', e, flush=True)
        return False


def matmul_BM64_BN256_BK64_GM16_nW4_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM64_BN256_BK64_GM16_nW4_nS2_EU0_kP2_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 256,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 16,
        num_warps = 4,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 2,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM64_BN256_BK64_GM16_nW4_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM64_BN256_BK64_GM16_nW4_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM64_BN256_BK64_GM16_nW4_nS2_EU0_kP2_mfma16: ', e, flush=True)
        return False


def matmul_BM64_BN256_BK64_GM16_nW4_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM64_BN256_BK64_GM16_nW4_nS2_EU0_kP1_mfma32.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 256,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 16,
        num_warps = 4,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 1,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM64_BN256_BK64_GM16_nW4_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM64_BN256_BK64_GM16_nW4_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM64_BN256_BK64_GM16_nW4_nS2_EU0_kP1_mfma32: ', e, flush=True)
        return False


def matmul_BM64_BN256_BK64_GM16_nW4_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM64_BN256_BK64_GM16_nW4_nS2_EU0_kP2_mfma32.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 256,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 16,
        num_warps = 4,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 2,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM64_BN256_BK64_GM16_nW4_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM64_BN256_BK64_GM16_nW4_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM64_BN256_BK64_GM16_nW4_nS2_EU0_kP2_mfma32: ', e, flush=True)
        return False


def matmul_BM64_BN256_BK64_GM32_nW4_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM64_BN256_BK64_GM32_nW4_nS2_EU0_kP1_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 256,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 32,
        num_warps = 4,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 1,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM64_BN256_BK64_GM32_nW4_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM64_BN256_BK64_GM32_nW4_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM64_BN256_BK64_GM32_nW4_nS2_EU0_kP1_mfma16: ', e, flush=True)
        return False


def matmul_BM64_BN256_BK64_GM32_nW4_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM64_BN256_BK64_GM32_nW4_nS2_EU0_kP2_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 256,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 32,
        num_warps = 4,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 2,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM64_BN256_BK64_GM32_nW4_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM64_BN256_BK64_GM32_nW4_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM64_BN256_BK64_GM32_nW4_nS2_EU0_kP2_mfma16: ', e, flush=True)
        return False


def matmul_BM64_BN256_BK64_GM32_nW4_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM64_BN256_BK64_GM32_nW4_nS2_EU0_kP1_mfma32.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 256,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 32,
        num_warps = 4,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 1,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM64_BN256_BK64_GM32_nW4_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM64_BN256_BK64_GM32_nW4_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM64_BN256_BK64_GM32_nW4_nS2_EU0_kP1_mfma32: ', e, flush=True)
        return False


def matmul_BM64_BN256_BK64_GM32_nW4_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM64_BN256_BK64_GM32_nW4_nS2_EU0_kP2_mfma32.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 256,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 32,
        num_warps = 4,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 2,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM64_BN256_BK64_GM32_nW4_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM64_BN256_BK64_GM32_nW4_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM64_BN256_BK64_GM32_nW4_nS2_EU0_kP2_mfma32: ', e, flush=True)
        return False


def matmul_BM64_BN256_BK64_GM1_nW8_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM64_BN256_BK64_GM1_nW8_nS2_EU0_kP1_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 256,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 1,
        num_warps = 8,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 1,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM64_BN256_BK64_GM1_nW8_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM64_BN256_BK64_GM1_nW8_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM64_BN256_BK64_GM1_nW8_nS2_EU0_kP1_mfma16: ', e, flush=True)
        return False


def matmul_BM64_BN256_BK64_GM1_nW8_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM64_BN256_BK64_GM1_nW8_nS2_EU0_kP2_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 256,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 1,
        num_warps = 8,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 2,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM64_BN256_BK64_GM1_nW8_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM64_BN256_BK64_GM1_nW8_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM64_BN256_BK64_GM1_nW8_nS2_EU0_kP2_mfma16: ', e, flush=True)
        return False


def matmul_BM64_BN256_BK64_GM1_nW8_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM64_BN256_BK64_GM1_nW8_nS2_EU0_kP1_mfma32.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 256,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 1,
        num_warps = 8,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 1,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM64_BN256_BK64_GM1_nW8_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM64_BN256_BK64_GM1_nW8_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM64_BN256_BK64_GM1_nW8_nS2_EU0_kP1_mfma32: ', e, flush=True)
        return False


def matmul_BM64_BN256_BK64_GM1_nW8_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM64_BN256_BK64_GM1_nW8_nS2_EU0_kP2_mfma32.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 256,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 1,
        num_warps = 8,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 2,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM64_BN256_BK64_GM1_nW8_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM64_BN256_BK64_GM1_nW8_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM64_BN256_BK64_GM1_nW8_nS2_EU0_kP2_mfma32: ', e, flush=True)
        return False


def matmul_BM64_BN256_BK64_GM4_nW8_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM64_BN256_BK64_GM4_nW8_nS2_EU0_kP1_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 256,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 4,
        num_warps = 8,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 1,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM64_BN256_BK64_GM4_nW8_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM64_BN256_BK64_GM4_nW8_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM64_BN256_BK64_GM4_nW8_nS2_EU0_kP1_mfma16: ', e, flush=True)
        return False


def matmul_BM64_BN256_BK64_GM4_nW8_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM64_BN256_BK64_GM4_nW8_nS2_EU0_kP2_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 256,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 4,
        num_warps = 8,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 2,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM64_BN256_BK64_GM4_nW8_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM64_BN256_BK64_GM4_nW8_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM64_BN256_BK64_GM4_nW8_nS2_EU0_kP2_mfma16: ', e, flush=True)
        return False


def matmul_BM64_BN256_BK64_GM4_nW8_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM64_BN256_BK64_GM4_nW8_nS2_EU0_kP1_mfma32.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 256,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 4,
        num_warps = 8,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 1,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM64_BN256_BK64_GM4_nW8_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM64_BN256_BK64_GM4_nW8_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM64_BN256_BK64_GM4_nW8_nS2_EU0_kP1_mfma32: ', e, flush=True)
        return False


def matmul_BM64_BN256_BK64_GM4_nW8_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM64_BN256_BK64_GM4_nW8_nS2_EU0_kP2_mfma32.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 256,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 4,
        num_warps = 8,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 2,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM64_BN256_BK64_GM4_nW8_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM64_BN256_BK64_GM4_nW8_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM64_BN256_BK64_GM4_nW8_nS2_EU0_kP2_mfma32: ', e, flush=True)
        return False


def matmul_BM64_BN256_BK64_GM8_nW8_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM64_BN256_BK64_GM8_nW8_nS2_EU0_kP1_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 256,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 8,
        num_warps = 8,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 1,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM64_BN256_BK64_GM8_nW8_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM64_BN256_BK64_GM8_nW8_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM64_BN256_BK64_GM8_nW8_nS2_EU0_kP1_mfma16: ', e, flush=True)
        return False


def matmul_BM64_BN256_BK64_GM8_nW8_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM64_BN256_BK64_GM8_nW8_nS2_EU0_kP2_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 256,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 8,
        num_warps = 8,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 2,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM64_BN256_BK64_GM8_nW8_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM64_BN256_BK64_GM8_nW8_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM64_BN256_BK64_GM8_nW8_nS2_EU0_kP2_mfma16: ', e, flush=True)
        return False


def matmul_BM64_BN256_BK64_GM8_nW8_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM64_BN256_BK64_GM8_nW8_nS2_EU0_kP1_mfma32.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 256,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 8,
        num_warps = 8,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 1,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM64_BN256_BK64_GM8_nW8_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM64_BN256_BK64_GM8_nW8_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM64_BN256_BK64_GM8_nW8_nS2_EU0_kP1_mfma32: ', e, flush=True)
        return False


def matmul_BM64_BN256_BK64_GM8_nW8_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM64_BN256_BK64_GM8_nW8_nS2_EU0_kP2_mfma32.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 256,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 8,
        num_warps = 8,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 2,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM64_BN256_BK64_GM8_nW8_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM64_BN256_BK64_GM8_nW8_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM64_BN256_BK64_GM8_nW8_nS2_EU0_kP2_mfma32: ', e, flush=True)
        return False


def matmul_BM64_BN256_BK64_GM16_nW8_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM64_BN256_BK64_GM16_nW8_nS2_EU0_kP1_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 256,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 16,
        num_warps = 8,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 1,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM64_BN256_BK64_GM16_nW8_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM64_BN256_BK64_GM16_nW8_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM64_BN256_BK64_GM16_nW8_nS2_EU0_kP1_mfma16: ', e, flush=True)
        return False


def matmul_BM64_BN256_BK64_GM16_nW8_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM64_BN256_BK64_GM16_nW8_nS2_EU0_kP2_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 256,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 16,
        num_warps = 8,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 2,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM64_BN256_BK64_GM16_nW8_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM64_BN256_BK64_GM16_nW8_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM64_BN256_BK64_GM16_nW8_nS2_EU0_kP2_mfma16: ', e, flush=True)
        return False


def matmul_BM64_BN256_BK64_GM16_nW8_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM64_BN256_BK64_GM16_nW8_nS2_EU0_kP1_mfma32.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 256,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 16,
        num_warps = 8,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 1,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM64_BN256_BK64_GM16_nW8_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM64_BN256_BK64_GM16_nW8_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM64_BN256_BK64_GM16_nW8_nS2_EU0_kP1_mfma32: ', e, flush=True)
        return False


def matmul_BM64_BN256_BK64_GM16_nW8_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM64_BN256_BK64_GM16_nW8_nS2_EU0_kP2_mfma32.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 256,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 16,
        num_warps = 8,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 2,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM64_BN256_BK64_GM16_nW8_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM64_BN256_BK64_GM16_nW8_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM64_BN256_BK64_GM16_nW8_nS2_EU0_kP2_mfma32: ', e, flush=True)
        return False


def matmul_BM64_BN256_BK64_GM32_nW8_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM64_BN256_BK64_GM32_nW8_nS2_EU0_kP1_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 256,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 32,
        num_warps = 8,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 1,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM64_BN256_BK64_GM32_nW8_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM64_BN256_BK64_GM32_nW8_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM64_BN256_BK64_GM32_nW8_nS2_EU0_kP1_mfma16: ', e, flush=True)
        return False


def matmul_BM64_BN256_BK64_GM32_nW8_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM64_BN256_BK64_GM32_nW8_nS2_EU0_kP2_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 256,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 32,
        num_warps = 8,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 2,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM64_BN256_BK64_GM32_nW8_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM64_BN256_BK64_GM32_nW8_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM64_BN256_BK64_GM32_nW8_nS2_EU0_kP2_mfma16: ', e, flush=True)
        return False


def matmul_BM64_BN256_BK64_GM32_nW8_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM64_BN256_BK64_GM32_nW8_nS2_EU0_kP1_mfma32.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 256,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 32,
        num_warps = 8,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 1,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM64_BN256_BK64_GM32_nW8_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM64_BN256_BK64_GM32_nW8_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM64_BN256_BK64_GM32_nW8_nS2_EU0_kP1_mfma32: ', e, flush=True)
        return False


def matmul_BM64_BN256_BK64_GM32_nW8_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM64_BN256_BK64_GM32_nW8_nS2_EU0_kP2_mfma32.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 256,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 32,
        num_warps = 8,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 2,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM64_BN256_BK64_GM32_nW8_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM64_BN256_BK64_GM32_nW8_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM64_BN256_BK64_GM32_nW8_nS2_EU0_kP2_mfma32: ', e, flush=True)
        return False


def matmul_BM128_BN64_BK64_GM1_nW4_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM128_BN64_BK64_GM1_nW4_nS2_EU0_kP1_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 128,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 1,
        num_warps = 4,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 1,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM128_BN64_BK64_GM1_nW4_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM128_BN64_BK64_GM1_nW4_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM128_BN64_BK64_GM1_nW4_nS2_EU0_kP1_mfma16: ', e, flush=True)
        return False


def matmul_BM128_BN64_BK64_GM1_nW4_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM128_BN64_BK64_GM1_nW4_nS2_EU0_kP2_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 128,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 1,
        num_warps = 4,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 2,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM128_BN64_BK64_GM1_nW4_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM128_BN64_BK64_GM1_nW4_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM128_BN64_BK64_GM1_nW4_nS2_EU0_kP2_mfma16: ', e, flush=True)
        return False


def matmul_BM128_BN64_BK64_GM1_nW4_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM128_BN64_BK64_GM1_nW4_nS2_EU0_kP1_mfma32.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 128,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 1,
        num_warps = 4,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 1,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM128_BN64_BK64_GM1_nW4_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM128_BN64_BK64_GM1_nW4_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM128_BN64_BK64_GM1_nW4_nS2_EU0_kP1_mfma32: ', e, flush=True)
        return False


def matmul_BM128_BN64_BK64_GM1_nW4_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM128_BN64_BK64_GM1_nW4_nS2_EU0_kP2_mfma32.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 128,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 1,
        num_warps = 4,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 2,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM128_BN64_BK64_GM1_nW4_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM128_BN64_BK64_GM1_nW4_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM128_BN64_BK64_GM1_nW4_nS2_EU0_kP2_mfma32: ', e, flush=True)
        return False


def matmul_BM128_BN64_BK64_GM4_nW4_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM128_BN64_BK64_GM4_nW4_nS2_EU0_kP1_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 128,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 4,
        num_warps = 4,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 1,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM128_BN64_BK64_GM4_nW4_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM128_BN64_BK64_GM4_nW4_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM128_BN64_BK64_GM4_nW4_nS2_EU0_kP1_mfma16: ', e, flush=True)
        return False


def matmul_BM128_BN64_BK64_GM4_nW4_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM128_BN64_BK64_GM4_nW4_nS2_EU0_kP2_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 128,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 4,
        num_warps = 4,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 2,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM128_BN64_BK64_GM4_nW4_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM128_BN64_BK64_GM4_nW4_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM128_BN64_BK64_GM4_nW4_nS2_EU0_kP2_mfma16: ', e, flush=True)
        return False


def matmul_BM128_BN64_BK64_GM4_nW4_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM128_BN64_BK64_GM4_nW4_nS2_EU0_kP1_mfma32.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 128,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 4,
        num_warps = 4,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 1,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM128_BN64_BK64_GM4_nW4_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM128_BN64_BK64_GM4_nW4_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM128_BN64_BK64_GM4_nW4_nS2_EU0_kP1_mfma32: ', e, flush=True)
        return False


def matmul_BM128_BN64_BK64_GM4_nW4_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM128_BN64_BK64_GM4_nW4_nS2_EU0_kP2_mfma32.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 128,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 4,
        num_warps = 4,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 2,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM128_BN64_BK64_GM4_nW4_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM128_BN64_BK64_GM4_nW4_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM128_BN64_BK64_GM4_nW4_nS2_EU0_kP2_mfma32: ', e, flush=True)
        return False


def matmul_BM128_BN64_BK64_GM8_nW4_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM128_BN64_BK64_GM8_nW4_nS2_EU0_kP1_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 128,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 8,
        num_warps = 4,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 1,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM128_BN64_BK64_GM8_nW4_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM128_BN64_BK64_GM8_nW4_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM128_BN64_BK64_GM8_nW4_nS2_EU0_kP1_mfma16: ', e, flush=True)
        return False


def matmul_BM128_BN64_BK64_GM8_nW4_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM128_BN64_BK64_GM8_nW4_nS2_EU0_kP2_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 128,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 8,
        num_warps = 4,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 2,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM128_BN64_BK64_GM8_nW4_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM128_BN64_BK64_GM8_nW4_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM128_BN64_BK64_GM8_nW4_nS2_EU0_kP2_mfma16: ', e, flush=True)
        return False


def matmul_BM128_BN64_BK64_GM8_nW4_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM128_BN64_BK64_GM8_nW4_nS2_EU0_kP1_mfma32.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 128,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 8,
        num_warps = 4,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 1,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM128_BN64_BK64_GM8_nW4_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM128_BN64_BK64_GM8_nW4_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM128_BN64_BK64_GM8_nW4_nS2_EU0_kP1_mfma32: ', e, flush=True)
        return False


def matmul_BM128_BN64_BK64_GM8_nW4_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM128_BN64_BK64_GM8_nW4_nS2_EU0_kP2_mfma32.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 128,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 8,
        num_warps = 4,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 2,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM128_BN64_BK64_GM8_nW4_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM128_BN64_BK64_GM8_nW4_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM128_BN64_BK64_GM8_nW4_nS2_EU0_kP2_mfma32: ', e, flush=True)
        return False


def matmul_BM128_BN64_BK64_GM16_nW4_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM128_BN64_BK64_GM16_nW4_nS2_EU0_kP1_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 128,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 16,
        num_warps = 4,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 1,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM128_BN64_BK64_GM16_nW4_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM128_BN64_BK64_GM16_nW4_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM128_BN64_BK64_GM16_nW4_nS2_EU0_kP1_mfma16: ', e, flush=True)
        return False


def matmul_BM128_BN64_BK64_GM16_nW4_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM128_BN64_BK64_GM16_nW4_nS2_EU0_kP2_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 128,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 16,
        num_warps = 4,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 2,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM128_BN64_BK64_GM16_nW4_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM128_BN64_BK64_GM16_nW4_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM128_BN64_BK64_GM16_nW4_nS2_EU0_kP2_mfma16: ', e, flush=True)
        return False


def matmul_BM128_BN64_BK64_GM16_nW4_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM128_BN64_BK64_GM16_nW4_nS2_EU0_kP1_mfma32.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 128,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 16,
        num_warps = 4,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 1,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM128_BN64_BK64_GM16_nW4_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM128_BN64_BK64_GM16_nW4_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM128_BN64_BK64_GM16_nW4_nS2_EU0_kP1_mfma32: ', e, flush=True)
        return False


def matmul_BM128_BN64_BK64_GM16_nW4_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM128_BN64_BK64_GM16_nW4_nS2_EU0_kP2_mfma32.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 128,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 16,
        num_warps = 4,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 2,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM128_BN64_BK64_GM16_nW4_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM128_BN64_BK64_GM16_nW4_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM128_BN64_BK64_GM16_nW4_nS2_EU0_kP2_mfma32: ', e, flush=True)
        return False


def matmul_BM128_BN64_BK64_GM32_nW4_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM128_BN64_BK64_GM32_nW4_nS2_EU0_kP1_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 128,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 32,
        num_warps = 4,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 1,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM128_BN64_BK64_GM32_nW4_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM128_BN64_BK64_GM32_nW4_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM128_BN64_BK64_GM32_nW4_nS2_EU0_kP1_mfma16: ', e, flush=True)
        return False


def matmul_BM128_BN64_BK64_GM32_nW4_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM128_BN64_BK64_GM32_nW4_nS2_EU0_kP2_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 128,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 32,
        num_warps = 4,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 2,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM128_BN64_BK64_GM32_nW4_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM128_BN64_BK64_GM32_nW4_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM128_BN64_BK64_GM32_nW4_nS2_EU0_kP2_mfma16: ', e, flush=True)
        return False


def matmul_BM128_BN64_BK64_GM32_nW4_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM128_BN64_BK64_GM32_nW4_nS2_EU0_kP1_mfma32.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 128,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 32,
        num_warps = 4,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 1,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM128_BN64_BK64_GM32_nW4_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM128_BN64_BK64_GM32_nW4_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM128_BN64_BK64_GM32_nW4_nS2_EU0_kP1_mfma32: ', e, flush=True)
        return False


def matmul_BM128_BN64_BK64_GM32_nW4_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM128_BN64_BK64_GM32_nW4_nS2_EU0_kP2_mfma32.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 128,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 32,
        num_warps = 4,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 2,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM128_BN64_BK64_GM32_nW4_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM128_BN64_BK64_GM32_nW4_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM128_BN64_BK64_GM32_nW4_nS2_EU0_kP2_mfma32: ', e, flush=True)
        return False


def matmul_BM128_BN64_BK64_GM1_nW8_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM128_BN64_BK64_GM1_nW8_nS2_EU0_kP1_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 128,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 1,
        num_warps = 8,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 1,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM128_BN64_BK64_GM1_nW8_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM128_BN64_BK64_GM1_nW8_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM128_BN64_BK64_GM1_nW8_nS2_EU0_kP1_mfma16: ', e, flush=True)
        return False


def matmul_BM128_BN64_BK64_GM1_nW8_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM128_BN64_BK64_GM1_nW8_nS2_EU0_kP2_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 128,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 1,
        num_warps = 8,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 2,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM128_BN64_BK64_GM1_nW8_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM128_BN64_BK64_GM1_nW8_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM128_BN64_BK64_GM1_nW8_nS2_EU0_kP2_mfma16: ', e, flush=True)
        return False


def matmul_BM128_BN64_BK64_GM1_nW8_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM128_BN64_BK64_GM1_nW8_nS2_EU0_kP1_mfma32.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 128,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 1,
        num_warps = 8,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 1,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM128_BN64_BK64_GM1_nW8_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM128_BN64_BK64_GM1_nW8_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM128_BN64_BK64_GM1_nW8_nS2_EU0_kP1_mfma32: ', e, flush=True)
        return False


def matmul_BM128_BN64_BK64_GM1_nW8_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM128_BN64_BK64_GM1_nW8_nS2_EU0_kP2_mfma32.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 128,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 1,
        num_warps = 8,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 2,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM128_BN64_BK64_GM1_nW8_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM128_BN64_BK64_GM1_nW8_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM128_BN64_BK64_GM1_nW8_nS2_EU0_kP2_mfma32: ', e, flush=True)
        return False


def matmul_BM128_BN64_BK64_GM4_nW8_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM128_BN64_BK64_GM4_nW8_nS2_EU0_kP1_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 128,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 4,
        num_warps = 8,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 1,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM128_BN64_BK64_GM4_nW8_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM128_BN64_BK64_GM4_nW8_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM128_BN64_BK64_GM4_nW8_nS2_EU0_kP1_mfma16: ', e, flush=True)
        return False


def matmul_BM128_BN64_BK64_GM4_nW8_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM128_BN64_BK64_GM4_nW8_nS2_EU0_kP2_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 128,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 4,
        num_warps = 8,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 2,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM128_BN64_BK64_GM4_nW8_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM128_BN64_BK64_GM4_nW8_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM128_BN64_BK64_GM4_nW8_nS2_EU0_kP2_mfma16: ', e, flush=True)
        return False


def matmul_BM128_BN64_BK64_GM4_nW8_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM128_BN64_BK64_GM4_nW8_nS2_EU0_kP1_mfma32.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 128,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 4,
        num_warps = 8,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 1,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM128_BN64_BK64_GM4_nW8_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM128_BN64_BK64_GM4_nW8_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM128_BN64_BK64_GM4_nW8_nS2_EU0_kP1_mfma32: ', e, flush=True)
        return False


def matmul_BM128_BN64_BK64_GM4_nW8_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM128_BN64_BK64_GM4_nW8_nS2_EU0_kP2_mfma32.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 128,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 4,
        num_warps = 8,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 2,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM128_BN64_BK64_GM4_nW8_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM128_BN64_BK64_GM4_nW8_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM128_BN64_BK64_GM4_nW8_nS2_EU0_kP2_mfma32: ', e, flush=True)
        return False


def matmul_BM128_BN64_BK64_GM8_nW8_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM128_BN64_BK64_GM8_nW8_nS2_EU0_kP1_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 128,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 8,
        num_warps = 8,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 1,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM128_BN64_BK64_GM8_nW8_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM128_BN64_BK64_GM8_nW8_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM128_BN64_BK64_GM8_nW8_nS2_EU0_kP1_mfma16: ', e, flush=True)
        return False


def matmul_BM128_BN64_BK64_GM8_nW8_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM128_BN64_BK64_GM8_nW8_nS2_EU0_kP2_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 128,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 8,
        num_warps = 8,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 2,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM128_BN64_BK64_GM8_nW8_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM128_BN64_BK64_GM8_nW8_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM128_BN64_BK64_GM8_nW8_nS2_EU0_kP2_mfma16: ', e, flush=True)
        return False


def matmul_BM128_BN64_BK64_GM8_nW8_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM128_BN64_BK64_GM8_nW8_nS2_EU0_kP1_mfma32.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 128,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 8,
        num_warps = 8,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 1,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM128_BN64_BK64_GM8_nW8_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM128_BN64_BK64_GM8_nW8_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM128_BN64_BK64_GM8_nW8_nS2_EU0_kP1_mfma32: ', e, flush=True)
        return False


def matmul_BM128_BN64_BK64_GM8_nW8_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM128_BN64_BK64_GM8_nW8_nS2_EU0_kP2_mfma32.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 128,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 8,
        num_warps = 8,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 2,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM128_BN64_BK64_GM8_nW8_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM128_BN64_BK64_GM8_nW8_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM128_BN64_BK64_GM8_nW8_nS2_EU0_kP2_mfma32: ', e, flush=True)
        return False


def matmul_BM128_BN64_BK64_GM16_nW8_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM128_BN64_BK64_GM16_nW8_nS2_EU0_kP1_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 128,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 16,
        num_warps = 8,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 1,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM128_BN64_BK64_GM16_nW8_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM128_BN64_BK64_GM16_nW8_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM128_BN64_BK64_GM16_nW8_nS2_EU0_kP1_mfma16: ', e, flush=True)
        return False


def matmul_BM128_BN64_BK64_GM16_nW8_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM128_BN64_BK64_GM16_nW8_nS2_EU0_kP2_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 128,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 16,
        num_warps = 8,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 2,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM128_BN64_BK64_GM16_nW8_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM128_BN64_BK64_GM16_nW8_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM128_BN64_BK64_GM16_nW8_nS2_EU0_kP2_mfma16: ', e, flush=True)
        return False


def matmul_BM128_BN64_BK64_GM16_nW8_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM128_BN64_BK64_GM16_nW8_nS2_EU0_kP1_mfma32.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 128,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 16,
        num_warps = 8,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 1,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM128_BN64_BK64_GM16_nW8_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM128_BN64_BK64_GM16_nW8_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM128_BN64_BK64_GM16_nW8_nS2_EU0_kP1_mfma32: ', e, flush=True)
        return False


def matmul_BM128_BN64_BK64_GM16_nW8_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM128_BN64_BK64_GM16_nW8_nS2_EU0_kP2_mfma32.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 128,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 16,
        num_warps = 8,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 2,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM128_BN64_BK64_GM16_nW8_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM128_BN64_BK64_GM16_nW8_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM128_BN64_BK64_GM16_nW8_nS2_EU0_kP2_mfma32: ', e, flush=True)
        return False


def matmul_BM128_BN64_BK64_GM32_nW8_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM128_BN64_BK64_GM32_nW8_nS2_EU0_kP1_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 128,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 32,
        num_warps = 8,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 1,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM128_BN64_BK64_GM32_nW8_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM128_BN64_BK64_GM32_nW8_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM128_BN64_BK64_GM32_nW8_nS2_EU0_kP1_mfma16: ', e, flush=True)
        return False


def matmul_BM128_BN64_BK64_GM32_nW8_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM128_BN64_BK64_GM32_nW8_nS2_EU0_kP2_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 128,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 32,
        num_warps = 8,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 2,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM128_BN64_BK64_GM32_nW8_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM128_BN64_BK64_GM32_nW8_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM128_BN64_BK64_GM32_nW8_nS2_EU0_kP2_mfma16: ', e, flush=True)
        return False


def matmul_BM128_BN64_BK64_GM32_nW8_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM128_BN64_BK64_GM32_nW8_nS2_EU0_kP1_mfma32.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 128,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 32,
        num_warps = 8,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 1,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM128_BN64_BK64_GM32_nW8_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM128_BN64_BK64_GM32_nW8_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM128_BN64_BK64_GM32_nW8_nS2_EU0_kP1_mfma32: ', e, flush=True)
        return False


def matmul_BM128_BN64_BK64_GM32_nW8_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM128_BN64_BK64_GM32_nW8_nS2_EU0_kP2_mfma32.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 128,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 32,
        num_warps = 8,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 2,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM128_BN64_BK64_GM32_nW8_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM128_BN64_BK64_GM32_nW8_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM128_BN64_BK64_GM32_nW8_nS2_EU0_kP2_mfma32: ', e, flush=True)
        return False


def matmul_BM128_BN64_BK128_GM1_nW4_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM128_BN64_BK128_GM1_nW4_nS2_EU0_kP1_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 128,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 128,
        GROUP_SIZE_M = 1,
        num_warps = 4,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 1,
        BIAS=False,
        EVEN_K=False,
        grid=(1,),
    )
    return None

def try_compile_config_BM128_BN64_BK128_GM1_nW4_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM128_BN64_BK128_GM1_nW4_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM128_BN64_BK128_GM1_nW4_nS2_EU0_kP1_mfma16: ', e, flush=True)
        return False


def matmul_BM128_BN64_BK128_GM1_nW4_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM128_BN64_BK128_GM1_nW4_nS2_EU0_kP2_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 128,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 128,
        GROUP_SIZE_M = 1,
        num_warps = 4,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 2,
        BIAS=False,
        EVEN_K=False,
        grid=(1,),
    )
    return None

def try_compile_config_BM128_BN64_BK128_GM1_nW4_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM128_BN64_BK128_GM1_nW4_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM128_BN64_BK128_GM1_nW4_nS2_EU0_kP2_mfma16: ', e, flush=True)
        return False


def matmul_BM128_BN64_BK128_GM1_nW4_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM128_BN64_BK128_GM1_nW4_nS2_EU0_kP1_mfma32.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 128,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 128,
        GROUP_SIZE_M = 1,
        num_warps = 4,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 1,
        BIAS=False,
        EVEN_K=False,
        grid=(1,),
    )
    return None

def try_compile_config_BM128_BN64_BK128_GM1_nW4_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM128_BN64_BK128_GM1_nW4_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM128_BN64_BK128_GM1_nW4_nS2_EU0_kP1_mfma32: ', e, flush=True)
        return False


def matmul_BM128_BN64_BK128_GM1_nW4_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM128_BN64_BK128_GM1_nW4_nS2_EU0_kP2_mfma32.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 128,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 128,
        GROUP_SIZE_M = 1,
        num_warps = 4,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 2,
        BIAS=False,
        EVEN_K=False,
        grid=(1,),
    )
    return None

def try_compile_config_BM128_BN64_BK128_GM1_nW4_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM128_BN64_BK128_GM1_nW4_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM128_BN64_BK128_GM1_nW4_nS2_EU0_kP2_mfma32: ', e, flush=True)
        return False


def matmul_BM128_BN64_BK128_GM4_nW4_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM128_BN64_BK128_GM4_nW4_nS2_EU0_kP1_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 128,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 128,
        GROUP_SIZE_M = 4,
        num_warps = 4,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 1,
        BIAS=False,
        EVEN_K=False,
        grid=(1,),
    )
    return None

def try_compile_config_BM128_BN64_BK128_GM4_nW4_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM128_BN64_BK128_GM4_nW4_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM128_BN64_BK128_GM4_nW4_nS2_EU0_kP1_mfma16: ', e, flush=True)
        return False


def matmul_BM128_BN64_BK128_GM4_nW4_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM128_BN64_BK128_GM4_nW4_nS2_EU0_kP2_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 128,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 128,
        GROUP_SIZE_M = 4,
        num_warps = 4,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 2,
        BIAS=False,
        EVEN_K=False,
        grid=(1,),
    )
    return None

def try_compile_config_BM128_BN64_BK128_GM4_nW4_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM128_BN64_BK128_GM4_nW4_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM128_BN64_BK128_GM4_nW4_nS2_EU0_kP2_mfma16: ', e, flush=True)
        return False


def matmul_BM128_BN64_BK128_GM4_nW4_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM128_BN64_BK128_GM4_nW4_nS2_EU0_kP1_mfma32.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 128,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 128,
        GROUP_SIZE_M = 4,
        num_warps = 4,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 1,
        BIAS=False,
        EVEN_K=False,
        grid=(1,),
    )
    return None

def try_compile_config_BM128_BN64_BK128_GM4_nW4_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM128_BN64_BK128_GM4_nW4_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM128_BN64_BK128_GM4_nW4_nS2_EU0_kP1_mfma32: ', e, flush=True)
        return False


def matmul_BM128_BN64_BK128_GM4_nW4_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM128_BN64_BK128_GM4_nW4_nS2_EU0_kP2_mfma32.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 128,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 128,
        GROUP_SIZE_M = 4,
        num_warps = 4,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 2,
        BIAS=False,
        EVEN_K=False,
        grid=(1,),
    )
    return None

def try_compile_config_BM128_BN64_BK128_GM4_nW4_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM128_BN64_BK128_GM4_nW4_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM128_BN64_BK128_GM4_nW4_nS2_EU0_kP2_mfma32: ', e, flush=True)
        return False


def matmul_BM128_BN64_BK128_GM8_nW4_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM128_BN64_BK128_GM8_nW4_nS2_EU0_kP1_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 128,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 128,
        GROUP_SIZE_M = 8,
        num_warps = 4,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 1,
        BIAS=False,
        EVEN_K=False,
        grid=(1,),
    )
    return None

def try_compile_config_BM128_BN64_BK128_GM8_nW4_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM128_BN64_BK128_GM8_nW4_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM128_BN64_BK128_GM8_nW4_nS2_EU0_kP1_mfma16: ', e, flush=True)
        return False


def matmul_BM128_BN64_BK128_GM8_nW4_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM128_BN64_BK128_GM8_nW4_nS2_EU0_kP2_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 128,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 128,
        GROUP_SIZE_M = 8,
        num_warps = 4,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 2,
        BIAS=False,
        EVEN_K=False,
        grid=(1,),
    )
    return None

def try_compile_config_BM128_BN64_BK128_GM8_nW4_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM128_BN64_BK128_GM8_nW4_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM128_BN64_BK128_GM8_nW4_nS2_EU0_kP2_mfma16: ', e, flush=True)
        return False


def matmul_BM128_BN64_BK128_GM8_nW4_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM128_BN64_BK128_GM8_nW4_nS2_EU0_kP1_mfma32.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 128,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 128,
        GROUP_SIZE_M = 8,
        num_warps = 4,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 1,
        BIAS=False,
        EVEN_K=False,
        grid=(1,),
    )
    return None

def try_compile_config_BM128_BN64_BK128_GM8_nW4_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM128_BN64_BK128_GM8_nW4_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM128_BN64_BK128_GM8_nW4_nS2_EU0_kP1_mfma32: ', e, flush=True)
        return False


def matmul_BM128_BN64_BK128_GM8_nW4_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM128_BN64_BK128_GM8_nW4_nS2_EU0_kP2_mfma32.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 128,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 128,
        GROUP_SIZE_M = 8,
        num_warps = 4,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 2,
        BIAS=False,
        EVEN_K=False,
        grid=(1,),
    )
    return None

def try_compile_config_BM128_BN64_BK128_GM8_nW4_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM128_BN64_BK128_GM8_nW4_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM128_BN64_BK128_GM8_nW4_nS2_EU0_kP2_mfma32: ', e, flush=True)
        return False


def matmul_BM128_BN64_BK128_GM16_nW4_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM128_BN64_BK128_GM16_nW4_nS2_EU0_kP1_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 128,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 128,
        GROUP_SIZE_M = 16,
        num_warps = 4,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 1,
        BIAS=False,
        EVEN_K=False,
        grid=(1,),
    )
    return None

def try_compile_config_BM128_BN64_BK128_GM16_nW4_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM128_BN64_BK128_GM16_nW4_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM128_BN64_BK128_GM16_nW4_nS2_EU0_kP1_mfma16: ', e, flush=True)
        return False


def matmul_BM128_BN64_BK128_GM16_nW4_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM128_BN64_BK128_GM16_nW4_nS2_EU0_kP2_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 128,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 128,
        GROUP_SIZE_M = 16,
        num_warps = 4,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 2,
        BIAS=False,
        EVEN_K=False,
        grid=(1,),
    )
    return None

def try_compile_config_BM128_BN64_BK128_GM16_nW4_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM128_BN64_BK128_GM16_nW4_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM128_BN64_BK128_GM16_nW4_nS2_EU0_kP2_mfma16: ', e, flush=True)
        return False


def matmul_BM128_BN64_BK128_GM16_nW4_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM128_BN64_BK128_GM16_nW4_nS2_EU0_kP1_mfma32.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 128,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 128,
        GROUP_SIZE_M = 16,
        num_warps = 4,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 1,
        BIAS=False,
        EVEN_K=False,
        grid=(1,),
    )
    return None

def try_compile_config_BM128_BN64_BK128_GM16_nW4_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM128_BN64_BK128_GM16_nW4_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM128_BN64_BK128_GM16_nW4_nS2_EU0_kP1_mfma32: ', e, flush=True)
        return False


def matmul_BM128_BN64_BK128_GM16_nW4_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM128_BN64_BK128_GM16_nW4_nS2_EU0_kP2_mfma32.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 128,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 128,
        GROUP_SIZE_M = 16,
        num_warps = 4,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 2,
        BIAS=False,
        EVEN_K=False,
        grid=(1,),
    )
    return None

def try_compile_config_BM128_BN64_BK128_GM16_nW4_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM128_BN64_BK128_GM16_nW4_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM128_BN64_BK128_GM16_nW4_nS2_EU0_kP2_mfma32: ', e, flush=True)
        return False


def matmul_BM128_BN64_BK128_GM32_nW4_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM128_BN64_BK128_GM32_nW4_nS2_EU0_kP1_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 128,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 128,
        GROUP_SIZE_M = 32,
        num_warps = 4,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 1,
        BIAS=False,
        EVEN_K=False,
        grid=(1,),
    )
    return None

def try_compile_config_BM128_BN64_BK128_GM32_nW4_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM128_BN64_BK128_GM32_nW4_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM128_BN64_BK128_GM32_nW4_nS2_EU0_kP1_mfma16: ', e, flush=True)
        return False


def matmul_BM128_BN64_BK128_GM32_nW4_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM128_BN64_BK128_GM32_nW4_nS2_EU0_kP2_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 128,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 128,
        GROUP_SIZE_M = 32,
        num_warps = 4,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 2,
        BIAS=False,
        EVEN_K=False,
        grid=(1,),
    )
    return None

def try_compile_config_BM128_BN64_BK128_GM32_nW4_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM128_BN64_BK128_GM32_nW4_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM128_BN64_BK128_GM32_nW4_nS2_EU0_kP2_mfma16: ', e, flush=True)
        return False


def matmul_BM128_BN64_BK128_GM32_nW4_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM128_BN64_BK128_GM32_nW4_nS2_EU0_kP1_mfma32.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 128,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 128,
        GROUP_SIZE_M = 32,
        num_warps = 4,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 1,
        BIAS=False,
        EVEN_K=False,
        grid=(1,),
    )
    return None

def try_compile_config_BM128_BN64_BK128_GM32_nW4_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM128_BN64_BK128_GM32_nW4_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM128_BN64_BK128_GM32_nW4_nS2_EU0_kP1_mfma32: ', e, flush=True)
        return False


def matmul_BM128_BN64_BK128_GM32_nW4_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM128_BN64_BK128_GM32_nW4_nS2_EU0_kP2_mfma32.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 128,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 128,
        GROUP_SIZE_M = 32,
        num_warps = 4,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 2,
        BIAS=False,
        EVEN_K=False,
        grid=(1,),
    )
    return None

def try_compile_config_BM128_BN64_BK128_GM32_nW4_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM128_BN64_BK128_GM32_nW4_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM128_BN64_BK128_GM32_nW4_nS2_EU0_kP2_mfma32: ', e, flush=True)
        return False


def matmul_BM128_BN64_BK128_GM1_nW8_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM128_BN64_BK128_GM1_nW8_nS2_EU0_kP1_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 128,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 128,
        GROUP_SIZE_M = 1,
        num_warps = 8,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 1,
        BIAS=False,
        EVEN_K=False,
        grid=(1,),
    )
    return None

def try_compile_config_BM128_BN64_BK128_GM1_nW8_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM128_BN64_BK128_GM1_nW8_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM128_BN64_BK128_GM1_nW8_nS2_EU0_kP1_mfma16: ', e, flush=True)
        return False


def matmul_BM128_BN64_BK128_GM1_nW8_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM128_BN64_BK128_GM1_nW8_nS2_EU0_kP2_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 128,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 128,
        GROUP_SIZE_M = 1,
        num_warps = 8,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 2,
        BIAS=False,
        EVEN_K=False,
        grid=(1,),
    )
    return None

def try_compile_config_BM128_BN64_BK128_GM1_nW8_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM128_BN64_BK128_GM1_nW8_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM128_BN64_BK128_GM1_nW8_nS2_EU0_kP2_mfma16: ', e, flush=True)
        return False


def matmul_BM128_BN64_BK128_GM1_nW8_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM128_BN64_BK128_GM1_nW8_nS2_EU0_kP1_mfma32.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 128,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 128,
        GROUP_SIZE_M = 1,
        num_warps = 8,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 1,
        BIAS=False,
        EVEN_K=False,
        grid=(1,),
    )
    return None

def try_compile_config_BM128_BN64_BK128_GM1_nW8_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM128_BN64_BK128_GM1_nW8_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM128_BN64_BK128_GM1_nW8_nS2_EU0_kP1_mfma32: ', e, flush=True)
        return False


def matmul_BM128_BN64_BK128_GM1_nW8_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM128_BN64_BK128_GM1_nW8_nS2_EU0_kP2_mfma32.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 128,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 128,
        GROUP_SIZE_M = 1,
        num_warps = 8,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 2,
        BIAS=False,
        EVEN_K=False,
        grid=(1,),
    )
    return None

def try_compile_config_BM128_BN64_BK128_GM1_nW8_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM128_BN64_BK128_GM1_nW8_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM128_BN64_BK128_GM1_nW8_nS2_EU0_kP2_mfma32: ', e, flush=True)
        return False


def matmul_BM128_BN64_BK128_GM4_nW8_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM128_BN64_BK128_GM4_nW8_nS2_EU0_kP1_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 128,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 128,
        GROUP_SIZE_M = 4,
        num_warps = 8,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 1,
        BIAS=False,
        EVEN_K=False,
        grid=(1,),
    )
    return None

def try_compile_config_BM128_BN64_BK128_GM4_nW8_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM128_BN64_BK128_GM4_nW8_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM128_BN64_BK128_GM4_nW8_nS2_EU0_kP1_mfma16: ', e, flush=True)
        return False


def matmul_BM128_BN64_BK128_GM4_nW8_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM128_BN64_BK128_GM4_nW8_nS2_EU0_kP2_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 128,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 128,
        GROUP_SIZE_M = 4,
        num_warps = 8,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 2,
        BIAS=False,
        EVEN_K=False,
        grid=(1,),
    )
    return None

def try_compile_config_BM128_BN64_BK128_GM4_nW8_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM128_BN64_BK128_GM4_nW8_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM128_BN64_BK128_GM4_nW8_nS2_EU0_kP2_mfma16: ', e, flush=True)
        return False


def matmul_BM128_BN64_BK128_GM4_nW8_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM128_BN64_BK128_GM4_nW8_nS2_EU0_kP1_mfma32.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 128,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 128,
        GROUP_SIZE_M = 4,
        num_warps = 8,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 1,
        BIAS=False,
        EVEN_K=False,
        grid=(1,),
    )
    return None

def try_compile_config_BM128_BN64_BK128_GM4_nW8_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM128_BN64_BK128_GM4_nW8_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM128_BN64_BK128_GM4_nW8_nS2_EU0_kP1_mfma32: ', e, flush=True)
        return False


def matmul_BM128_BN64_BK128_GM4_nW8_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM128_BN64_BK128_GM4_nW8_nS2_EU0_kP2_mfma32.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 128,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 128,
        GROUP_SIZE_M = 4,
        num_warps = 8,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 2,
        BIAS=False,
        EVEN_K=False,
        grid=(1,),
    )
    return None

def try_compile_config_BM128_BN64_BK128_GM4_nW8_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM128_BN64_BK128_GM4_nW8_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM128_BN64_BK128_GM4_nW8_nS2_EU0_kP2_mfma32: ', e, flush=True)
        return False


def matmul_BM128_BN64_BK128_GM8_nW8_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM128_BN64_BK128_GM8_nW8_nS2_EU0_kP1_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 128,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 128,
        GROUP_SIZE_M = 8,
        num_warps = 8,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 1,
        BIAS=False,
        EVEN_K=False,
        grid=(1,),
    )
    return None

def try_compile_config_BM128_BN64_BK128_GM8_nW8_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM128_BN64_BK128_GM8_nW8_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM128_BN64_BK128_GM8_nW8_nS2_EU0_kP1_mfma16: ', e, flush=True)
        return False


def matmul_BM128_BN64_BK128_GM8_nW8_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM128_BN64_BK128_GM8_nW8_nS2_EU0_kP2_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 128,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 128,
        GROUP_SIZE_M = 8,
        num_warps = 8,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 2,
        BIAS=False,
        EVEN_K=False,
        grid=(1,),
    )
    return None

def try_compile_config_BM128_BN64_BK128_GM8_nW8_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM128_BN64_BK128_GM8_nW8_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM128_BN64_BK128_GM8_nW8_nS2_EU0_kP2_mfma16: ', e, flush=True)
        return False


def matmul_BM128_BN64_BK128_GM8_nW8_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM128_BN64_BK128_GM8_nW8_nS2_EU0_kP1_mfma32.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 128,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 128,
        GROUP_SIZE_M = 8,
        num_warps = 8,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 1,
        BIAS=False,
        EVEN_K=False,
        grid=(1,),
    )
    return None

def try_compile_config_BM128_BN64_BK128_GM8_nW8_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM128_BN64_BK128_GM8_nW8_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM128_BN64_BK128_GM8_nW8_nS2_EU0_kP1_mfma32: ', e, flush=True)
        return False


def matmul_BM128_BN64_BK128_GM8_nW8_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM128_BN64_BK128_GM8_nW8_nS2_EU0_kP2_mfma32.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 128,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 128,
        GROUP_SIZE_M = 8,
        num_warps = 8,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 2,
        BIAS=False,
        EVEN_K=False,
        grid=(1,),
    )
    return None

def try_compile_config_BM128_BN64_BK128_GM8_nW8_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM128_BN64_BK128_GM8_nW8_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM128_BN64_BK128_GM8_nW8_nS2_EU0_kP2_mfma32: ', e, flush=True)
        return False


def matmul_BM128_BN64_BK128_GM16_nW8_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM128_BN64_BK128_GM16_nW8_nS2_EU0_kP1_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 128,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 128,
        GROUP_SIZE_M = 16,
        num_warps = 8,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 1,
        BIAS=False,
        EVEN_K=False,
        grid=(1,),
    )
    return None

def try_compile_config_BM128_BN64_BK128_GM16_nW8_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM128_BN64_BK128_GM16_nW8_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM128_BN64_BK128_GM16_nW8_nS2_EU0_kP1_mfma16: ', e, flush=True)
        return False


def matmul_BM128_BN64_BK128_GM16_nW8_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM128_BN64_BK128_GM16_nW8_nS2_EU0_kP2_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 128,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 128,
        GROUP_SIZE_M = 16,
        num_warps = 8,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 2,
        BIAS=False,
        EVEN_K=False,
        grid=(1,),
    )
    return None

def try_compile_config_BM128_BN64_BK128_GM16_nW8_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM128_BN64_BK128_GM16_nW8_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM128_BN64_BK128_GM16_nW8_nS2_EU0_kP2_mfma16: ', e, flush=True)
        return False


def matmul_BM128_BN64_BK128_GM16_nW8_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM128_BN64_BK128_GM16_nW8_nS2_EU0_kP1_mfma32.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 128,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 128,
        GROUP_SIZE_M = 16,
        num_warps = 8,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 1,
        BIAS=False,
        EVEN_K=False,
        grid=(1,),
    )
    return None

def try_compile_config_BM128_BN64_BK128_GM16_nW8_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM128_BN64_BK128_GM16_nW8_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM128_BN64_BK128_GM16_nW8_nS2_EU0_kP1_mfma32: ', e, flush=True)
        return False


def matmul_BM128_BN64_BK128_GM16_nW8_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM128_BN64_BK128_GM16_nW8_nS2_EU0_kP2_mfma32.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 128,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 128,
        GROUP_SIZE_M = 16,
        num_warps = 8,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 2,
        BIAS=False,
        EVEN_K=False,
        grid=(1,),
    )
    return None

def try_compile_config_BM128_BN64_BK128_GM16_nW8_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM128_BN64_BK128_GM16_nW8_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM128_BN64_BK128_GM16_nW8_nS2_EU0_kP2_mfma32: ', e, flush=True)
        return False


def matmul_BM128_BN64_BK128_GM32_nW8_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM128_BN64_BK128_GM32_nW8_nS2_EU0_kP1_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 128,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 128,
        GROUP_SIZE_M = 32,
        num_warps = 8,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 1,
        BIAS=False,
        EVEN_K=False,
        grid=(1,),
    )
    return None

def try_compile_config_BM128_BN64_BK128_GM32_nW8_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM128_BN64_BK128_GM32_nW8_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM128_BN64_BK128_GM32_nW8_nS2_EU0_kP1_mfma16: ', e, flush=True)
        return False


def matmul_BM128_BN64_BK128_GM32_nW8_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM128_BN64_BK128_GM32_nW8_nS2_EU0_kP2_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 128,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 128,
        GROUP_SIZE_M = 32,
        num_warps = 8,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 2,
        BIAS=False,
        EVEN_K=False,
        grid=(1,),
    )
    return None

def try_compile_config_BM128_BN64_BK128_GM32_nW8_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM128_BN64_BK128_GM32_nW8_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM128_BN64_BK128_GM32_nW8_nS2_EU0_kP2_mfma16: ', e, flush=True)
        return False


def matmul_BM128_BN64_BK128_GM32_nW8_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM128_BN64_BK128_GM32_nW8_nS2_EU0_kP1_mfma32.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 128,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 128,
        GROUP_SIZE_M = 32,
        num_warps = 8,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 1,
        BIAS=False,
        EVEN_K=False,
        grid=(1,),
    )
    return None

def try_compile_config_BM128_BN64_BK128_GM32_nW8_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM128_BN64_BK128_GM32_nW8_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM128_BN64_BK128_GM32_nW8_nS2_EU0_kP1_mfma32: ', e, flush=True)
        return False


def matmul_BM128_BN64_BK128_GM32_nW8_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM128_BN64_BK128_GM32_nW8_nS2_EU0_kP2_mfma32.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 128,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 128,
        GROUP_SIZE_M = 32,
        num_warps = 8,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 2,
        BIAS=False,
        EVEN_K=False,
        grid=(1,),
    )
    return None

def try_compile_config_BM128_BN64_BK128_GM32_nW8_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM128_BN64_BK128_GM32_nW8_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM128_BN64_BK128_GM32_nW8_nS2_EU0_kP2_mfma32: ', e, flush=True)
        return False


def matmul_BM128_BN128_BK64_GM1_nW4_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM128_BN128_BK64_GM1_nW4_nS2_EU0_kP1_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 128,
        BLOCK_SIZE_N = 128,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 1,
        num_warps = 4,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 1,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM128_BN128_BK64_GM1_nW4_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM128_BN128_BK64_GM1_nW4_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM128_BN128_BK64_GM1_nW4_nS2_EU0_kP1_mfma16: ', e, flush=True)
        return False


def matmul_BM128_BN128_BK64_GM1_nW4_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM128_BN128_BK64_GM1_nW4_nS2_EU0_kP2_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 128,
        BLOCK_SIZE_N = 128,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 1,
        num_warps = 4,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 2,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM128_BN128_BK64_GM1_nW4_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM128_BN128_BK64_GM1_nW4_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM128_BN128_BK64_GM1_nW4_nS2_EU0_kP2_mfma16: ', e, flush=True)
        return False


def matmul_BM128_BN128_BK64_GM1_nW4_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM128_BN128_BK64_GM1_nW4_nS2_EU0_kP1_mfma32.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 128,
        BLOCK_SIZE_N = 128,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 1,
        num_warps = 4,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 1,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM128_BN128_BK64_GM1_nW4_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM128_BN128_BK64_GM1_nW4_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM128_BN128_BK64_GM1_nW4_nS2_EU0_kP1_mfma32: ', e, flush=True)
        return False


def matmul_BM128_BN128_BK64_GM1_nW4_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM128_BN128_BK64_GM1_nW4_nS2_EU0_kP2_mfma32.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 128,
        BLOCK_SIZE_N = 128,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 1,
        num_warps = 4,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 2,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM128_BN128_BK64_GM1_nW4_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM128_BN128_BK64_GM1_nW4_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM128_BN128_BK64_GM1_nW4_nS2_EU0_kP2_mfma32: ', e, flush=True)
        return False


def matmul_BM128_BN128_BK64_GM4_nW4_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM128_BN128_BK64_GM4_nW4_nS2_EU0_kP1_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 128,
        BLOCK_SIZE_N = 128,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 4,
        num_warps = 4,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 1,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM128_BN128_BK64_GM4_nW4_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM128_BN128_BK64_GM4_nW4_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM128_BN128_BK64_GM4_nW4_nS2_EU0_kP1_mfma16: ', e, flush=True)
        return False


def matmul_BM128_BN128_BK64_GM4_nW4_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM128_BN128_BK64_GM4_nW4_nS2_EU0_kP2_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 128,
        BLOCK_SIZE_N = 128,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 4,
        num_warps = 4,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 2,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM128_BN128_BK64_GM4_nW4_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM128_BN128_BK64_GM4_nW4_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM128_BN128_BK64_GM4_nW4_nS2_EU0_kP2_mfma16: ', e, flush=True)
        return False


def matmul_BM128_BN128_BK64_GM4_nW4_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM128_BN128_BK64_GM4_nW4_nS2_EU0_kP1_mfma32.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 128,
        BLOCK_SIZE_N = 128,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 4,
        num_warps = 4,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 1,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM128_BN128_BK64_GM4_nW4_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM128_BN128_BK64_GM4_nW4_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM128_BN128_BK64_GM4_nW4_nS2_EU0_kP1_mfma32: ', e, flush=True)
        return False


def matmul_BM128_BN128_BK64_GM4_nW4_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM128_BN128_BK64_GM4_nW4_nS2_EU0_kP2_mfma32.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 128,
        BLOCK_SIZE_N = 128,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 4,
        num_warps = 4,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 2,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM128_BN128_BK64_GM4_nW4_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM128_BN128_BK64_GM4_nW4_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM128_BN128_BK64_GM4_nW4_nS2_EU0_kP2_mfma32: ', e, flush=True)
        return False


def matmul_BM128_BN128_BK64_GM8_nW4_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM128_BN128_BK64_GM8_nW4_nS2_EU0_kP1_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 128,
        BLOCK_SIZE_N = 128,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 8,
        num_warps = 4,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 1,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM128_BN128_BK64_GM8_nW4_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM128_BN128_BK64_GM8_nW4_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM128_BN128_BK64_GM8_nW4_nS2_EU0_kP1_mfma16: ', e, flush=True)
        return False


def matmul_BM128_BN128_BK64_GM8_nW4_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM128_BN128_BK64_GM8_nW4_nS2_EU0_kP2_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 128,
        BLOCK_SIZE_N = 128,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 8,
        num_warps = 4,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 2,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM128_BN128_BK64_GM8_nW4_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM128_BN128_BK64_GM8_nW4_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM128_BN128_BK64_GM8_nW4_nS2_EU0_kP2_mfma16: ', e, flush=True)
        return False


def matmul_BM128_BN128_BK64_GM8_nW4_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM128_BN128_BK64_GM8_nW4_nS2_EU0_kP1_mfma32.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 128,
        BLOCK_SIZE_N = 128,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 8,
        num_warps = 4,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 1,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM128_BN128_BK64_GM8_nW4_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM128_BN128_BK64_GM8_nW4_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM128_BN128_BK64_GM8_nW4_nS2_EU0_kP1_mfma32: ', e, flush=True)
        return False


def matmul_BM128_BN128_BK64_GM8_nW4_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM128_BN128_BK64_GM8_nW4_nS2_EU0_kP2_mfma32.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 128,
        BLOCK_SIZE_N = 128,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 8,
        num_warps = 4,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 2,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM128_BN128_BK64_GM8_nW4_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM128_BN128_BK64_GM8_nW4_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM128_BN128_BK64_GM8_nW4_nS2_EU0_kP2_mfma32: ', e, flush=True)
        return False


def matmul_BM128_BN128_BK64_GM16_nW4_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM128_BN128_BK64_GM16_nW4_nS2_EU0_kP1_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 128,
        BLOCK_SIZE_N = 128,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 16,
        num_warps = 4,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 1,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM128_BN128_BK64_GM16_nW4_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM128_BN128_BK64_GM16_nW4_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM128_BN128_BK64_GM16_nW4_nS2_EU0_kP1_mfma16: ', e, flush=True)
        return False


def matmul_BM128_BN128_BK64_GM16_nW4_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM128_BN128_BK64_GM16_nW4_nS2_EU0_kP2_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 128,
        BLOCK_SIZE_N = 128,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 16,
        num_warps = 4,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 2,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM128_BN128_BK64_GM16_nW4_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM128_BN128_BK64_GM16_nW4_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM128_BN128_BK64_GM16_nW4_nS2_EU0_kP2_mfma16: ', e, flush=True)
        return False


def matmul_BM128_BN128_BK64_GM16_nW4_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM128_BN128_BK64_GM16_nW4_nS2_EU0_kP1_mfma32.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 128,
        BLOCK_SIZE_N = 128,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 16,
        num_warps = 4,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 1,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM128_BN128_BK64_GM16_nW4_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM128_BN128_BK64_GM16_nW4_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM128_BN128_BK64_GM16_nW4_nS2_EU0_kP1_mfma32: ', e, flush=True)
        return False


def matmul_BM128_BN128_BK64_GM16_nW4_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM128_BN128_BK64_GM16_nW4_nS2_EU0_kP2_mfma32.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 128,
        BLOCK_SIZE_N = 128,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 16,
        num_warps = 4,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 2,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM128_BN128_BK64_GM16_nW4_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM128_BN128_BK64_GM16_nW4_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM128_BN128_BK64_GM16_nW4_nS2_EU0_kP2_mfma32: ', e, flush=True)
        return False


def matmul_BM128_BN128_BK64_GM32_nW4_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM128_BN128_BK64_GM32_nW4_nS2_EU0_kP1_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 128,
        BLOCK_SIZE_N = 128,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 32,
        num_warps = 4,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 1,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM128_BN128_BK64_GM32_nW4_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM128_BN128_BK64_GM32_nW4_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM128_BN128_BK64_GM32_nW4_nS2_EU0_kP1_mfma16: ', e, flush=True)
        return False


def matmul_BM128_BN128_BK64_GM32_nW4_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM128_BN128_BK64_GM32_nW4_nS2_EU0_kP2_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 128,
        BLOCK_SIZE_N = 128,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 32,
        num_warps = 4,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 2,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM128_BN128_BK64_GM32_nW4_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM128_BN128_BK64_GM32_nW4_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM128_BN128_BK64_GM32_nW4_nS2_EU0_kP2_mfma16: ', e, flush=True)
        return False


def matmul_BM128_BN128_BK64_GM32_nW4_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM128_BN128_BK64_GM32_nW4_nS2_EU0_kP1_mfma32.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 128,
        BLOCK_SIZE_N = 128,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 32,
        num_warps = 4,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 1,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM128_BN128_BK64_GM32_nW4_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM128_BN128_BK64_GM32_nW4_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM128_BN128_BK64_GM32_nW4_nS2_EU0_kP1_mfma32: ', e, flush=True)
        return False


def matmul_BM128_BN128_BK64_GM32_nW4_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM128_BN128_BK64_GM32_nW4_nS2_EU0_kP2_mfma32.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 128,
        BLOCK_SIZE_N = 128,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 32,
        num_warps = 4,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 2,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM128_BN128_BK64_GM32_nW4_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM128_BN128_BK64_GM32_nW4_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM128_BN128_BK64_GM32_nW4_nS2_EU0_kP2_mfma32: ', e, flush=True)
        return False


def matmul_BM128_BN128_BK64_GM1_nW8_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM128_BN128_BK64_GM1_nW8_nS2_EU0_kP1_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 128,
        BLOCK_SIZE_N = 128,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 1,
        num_warps = 8,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 1,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM128_BN128_BK64_GM1_nW8_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM128_BN128_BK64_GM1_nW8_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM128_BN128_BK64_GM1_nW8_nS2_EU0_kP1_mfma16: ', e, flush=True)
        return False


def matmul_BM128_BN128_BK64_GM1_nW8_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM128_BN128_BK64_GM1_nW8_nS2_EU0_kP2_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 128,
        BLOCK_SIZE_N = 128,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 1,
        num_warps = 8,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 2,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM128_BN128_BK64_GM1_nW8_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM128_BN128_BK64_GM1_nW8_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM128_BN128_BK64_GM1_nW8_nS2_EU0_kP2_mfma16: ', e, flush=True)
        return False


def matmul_BM128_BN128_BK64_GM1_nW8_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM128_BN128_BK64_GM1_nW8_nS2_EU0_kP1_mfma32.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 128,
        BLOCK_SIZE_N = 128,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 1,
        num_warps = 8,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 1,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM128_BN128_BK64_GM1_nW8_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM128_BN128_BK64_GM1_nW8_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM128_BN128_BK64_GM1_nW8_nS2_EU0_kP1_mfma32: ', e, flush=True)
        return False


def matmul_BM128_BN128_BK64_GM1_nW8_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM128_BN128_BK64_GM1_nW8_nS2_EU0_kP2_mfma32.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 128,
        BLOCK_SIZE_N = 128,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 1,
        num_warps = 8,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 2,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM128_BN128_BK64_GM1_nW8_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM128_BN128_BK64_GM1_nW8_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM128_BN128_BK64_GM1_nW8_nS2_EU0_kP2_mfma32: ', e, flush=True)
        return False


def matmul_BM128_BN128_BK64_GM4_nW8_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM128_BN128_BK64_GM4_nW8_nS2_EU0_kP1_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 128,
        BLOCK_SIZE_N = 128,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 4,
        num_warps = 8,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 1,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM128_BN128_BK64_GM4_nW8_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM128_BN128_BK64_GM4_nW8_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM128_BN128_BK64_GM4_nW8_nS2_EU0_kP1_mfma16: ', e, flush=True)
        return False


def matmul_BM128_BN128_BK64_GM4_nW8_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM128_BN128_BK64_GM4_nW8_nS2_EU0_kP2_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 128,
        BLOCK_SIZE_N = 128,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 4,
        num_warps = 8,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 2,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM128_BN128_BK64_GM4_nW8_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM128_BN128_BK64_GM4_nW8_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM128_BN128_BK64_GM4_nW8_nS2_EU0_kP2_mfma16: ', e, flush=True)
        return False


def matmul_BM128_BN128_BK64_GM4_nW8_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM128_BN128_BK64_GM4_nW8_nS2_EU0_kP1_mfma32.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 128,
        BLOCK_SIZE_N = 128,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 4,
        num_warps = 8,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 1,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM128_BN128_BK64_GM4_nW8_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM128_BN128_BK64_GM4_nW8_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM128_BN128_BK64_GM4_nW8_nS2_EU0_kP1_mfma32: ', e, flush=True)
        return False


def matmul_BM128_BN128_BK64_GM4_nW8_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM128_BN128_BK64_GM4_nW8_nS2_EU0_kP2_mfma32.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 128,
        BLOCK_SIZE_N = 128,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 4,
        num_warps = 8,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 2,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM128_BN128_BK64_GM4_nW8_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM128_BN128_BK64_GM4_nW8_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM128_BN128_BK64_GM4_nW8_nS2_EU0_kP2_mfma32: ', e, flush=True)
        return False


def matmul_BM128_BN128_BK64_GM8_nW8_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM128_BN128_BK64_GM8_nW8_nS2_EU0_kP1_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 128,
        BLOCK_SIZE_N = 128,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 8,
        num_warps = 8,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 1,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM128_BN128_BK64_GM8_nW8_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM128_BN128_BK64_GM8_nW8_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM128_BN128_BK64_GM8_nW8_nS2_EU0_kP1_mfma16: ', e, flush=True)
        return False


def matmul_BM128_BN128_BK64_GM8_nW8_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM128_BN128_BK64_GM8_nW8_nS2_EU0_kP2_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 128,
        BLOCK_SIZE_N = 128,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 8,
        num_warps = 8,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 2,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM128_BN128_BK64_GM8_nW8_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM128_BN128_BK64_GM8_nW8_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM128_BN128_BK64_GM8_nW8_nS2_EU0_kP2_mfma16: ', e, flush=True)
        return False


def matmul_BM128_BN128_BK64_GM8_nW8_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM128_BN128_BK64_GM8_nW8_nS2_EU0_kP1_mfma32.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 128,
        BLOCK_SIZE_N = 128,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 8,
        num_warps = 8,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 1,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM128_BN128_BK64_GM8_nW8_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM128_BN128_BK64_GM8_nW8_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM128_BN128_BK64_GM8_nW8_nS2_EU0_kP1_mfma32: ', e, flush=True)
        return False


def matmul_BM128_BN128_BK64_GM8_nW8_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM128_BN128_BK64_GM8_nW8_nS2_EU0_kP2_mfma32.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 128,
        BLOCK_SIZE_N = 128,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 8,
        num_warps = 8,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 2,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM128_BN128_BK64_GM8_nW8_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM128_BN128_BK64_GM8_nW8_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM128_BN128_BK64_GM8_nW8_nS2_EU0_kP2_mfma32: ', e, flush=True)
        return False


def matmul_BM128_BN128_BK64_GM16_nW8_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM128_BN128_BK64_GM16_nW8_nS2_EU0_kP1_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 128,
        BLOCK_SIZE_N = 128,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 16,
        num_warps = 8,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 1,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM128_BN128_BK64_GM16_nW8_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM128_BN128_BK64_GM16_nW8_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM128_BN128_BK64_GM16_nW8_nS2_EU0_kP1_mfma16: ', e, flush=True)
        return False


def matmul_BM128_BN128_BK64_GM16_nW8_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM128_BN128_BK64_GM16_nW8_nS2_EU0_kP2_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 128,
        BLOCK_SIZE_N = 128,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 16,
        num_warps = 8,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 2,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM128_BN128_BK64_GM16_nW8_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM128_BN128_BK64_GM16_nW8_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM128_BN128_BK64_GM16_nW8_nS2_EU0_kP2_mfma16: ', e, flush=True)
        return False


def matmul_BM128_BN128_BK64_GM16_nW8_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM128_BN128_BK64_GM16_nW8_nS2_EU0_kP1_mfma32.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 128,
        BLOCK_SIZE_N = 128,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 16,
        num_warps = 8,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 1,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM128_BN128_BK64_GM16_nW8_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM128_BN128_BK64_GM16_nW8_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM128_BN128_BK64_GM16_nW8_nS2_EU0_kP1_mfma32: ', e, flush=True)
        return False


def matmul_BM128_BN128_BK64_GM16_nW8_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM128_BN128_BK64_GM16_nW8_nS2_EU0_kP2_mfma32.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 128,
        BLOCK_SIZE_N = 128,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 16,
        num_warps = 8,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 2,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM128_BN128_BK64_GM16_nW8_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM128_BN128_BK64_GM16_nW8_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM128_BN128_BK64_GM16_nW8_nS2_EU0_kP2_mfma32: ', e, flush=True)
        return False


def matmul_BM128_BN128_BK64_GM32_nW8_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM128_BN128_BK64_GM32_nW8_nS2_EU0_kP1_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 128,
        BLOCK_SIZE_N = 128,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 32,
        num_warps = 8,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 1,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM128_BN128_BK64_GM32_nW8_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM128_BN128_BK64_GM32_nW8_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM128_BN128_BK64_GM32_nW8_nS2_EU0_kP1_mfma16: ', e, flush=True)
        return False


def matmul_BM128_BN128_BK64_GM32_nW8_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM128_BN128_BK64_GM32_nW8_nS2_EU0_kP2_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 128,
        BLOCK_SIZE_N = 128,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 32,
        num_warps = 8,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 2,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM128_BN128_BK64_GM32_nW8_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM128_BN128_BK64_GM32_nW8_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM128_BN128_BK64_GM32_nW8_nS2_EU0_kP2_mfma16: ', e, flush=True)
        return False


def matmul_BM128_BN128_BK64_GM32_nW8_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM128_BN128_BK64_GM32_nW8_nS2_EU0_kP1_mfma32.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 128,
        BLOCK_SIZE_N = 128,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 32,
        num_warps = 8,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 1,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM128_BN128_BK64_GM32_nW8_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM128_BN128_BK64_GM32_nW8_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM128_BN128_BK64_GM32_nW8_nS2_EU0_kP1_mfma32: ', e, flush=True)
        return False


def matmul_BM128_BN128_BK64_GM32_nW8_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM128_BN128_BK64_GM32_nW8_nS2_EU0_kP2_mfma32.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 128,
        BLOCK_SIZE_N = 128,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 32,
        num_warps = 8,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 2,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM128_BN128_BK64_GM32_nW8_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM128_BN128_BK64_GM32_nW8_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM128_BN128_BK64_GM32_nW8_nS2_EU0_kP2_mfma32: ', e, flush=True)
        return False


def matmul_BM128_BN128_BK128_GM1_nW4_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM128_BN128_BK128_GM1_nW4_nS2_EU0_kP1_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 128,
        BLOCK_SIZE_N = 128,
        BLOCK_SIZE_K = 128,
        GROUP_SIZE_M = 1,
        num_warps = 4,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 1,
        BIAS=False,
        EVEN_K=False,
        grid=(1,),
    )
    return None

def try_compile_config_BM128_BN128_BK128_GM1_nW4_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM128_BN128_BK128_GM1_nW4_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM128_BN128_BK128_GM1_nW4_nS2_EU0_kP1_mfma16: ', e, flush=True)
        return False


def matmul_BM128_BN128_BK128_GM1_nW4_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM128_BN128_BK128_GM1_nW4_nS2_EU0_kP2_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 128,
        BLOCK_SIZE_N = 128,
        BLOCK_SIZE_K = 128,
        GROUP_SIZE_M = 1,
        num_warps = 4,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 2,
        BIAS=False,
        EVEN_K=False,
        grid=(1,),
    )
    return None

def try_compile_config_BM128_BN128_BK128_GM1_nW4_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM128_BN128_BK128_GM1_nW4_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM128_BN128_BK128_GM1_nW4_nS2_EU0_kP2_mfma16: ', e, flush=True)
        return False


def matmul_BM128_BN128_BK128_GM1_nW4_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM128_BN128_BK128_GM1_nW4_nS2_EU0_kP1_mfma32.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 128,
        BLOCK_SIZE_N = 128,
        BLOCK_SIZE_K = 128,
        GROUP_SIZE_M = 1,
        num_warps = 4,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 1,
        BIAS=False,
        EVEN_K=False,
        grid=(1,),
    )
    return None

def try_compile_config_BM128_BN128_BK128_GM1_nW4_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM128_BN128_BK128_GM1_nW4_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM128_BN128_BK128_GM1_nW4_nS2_EU0_kP1_mfma32: ', e, flush=True)
        return False


def matmul_BM128_BN128_BK128_GM4_nW4_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM128_BN128_BK128_GM4_nW4_nS2_EU0_kP1_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 128,
        BLOCK_SIZE_N = 128,
        BLOCK_SIZE_K = 128,
        GROUP_SIZE_M = 4,
        num_warps = 4,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 1,
        BIAS=False,
        EVEN_K=False,
        grid=(1,),
    )
    return None

def try_compile_config_BM128_BN128_BK128_GM4_nW4_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM128_BN128_BK128_GM4_nW4_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM128_BN128_BK128_GM4_nW4_nS2_EU0_kP1_mfma16: ', e, flush=True)
        return False


def matmul_BM128_BN128_BK128_GM4_nW4_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM128_BN128_BK128_GM4_nW4_nS2_EU0_kP2_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 128,
        BLOCK_SIZE_N = 128,
        BLOCK_SIZE_K = 128,
        GROUP_SIZE_M = 4,
        num_warps = 4,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 2,
        BIAS=False,
        EVEN_K=False,
        grid=(1,),
    )
    return None

def try_compile_config_BM128_BN128_BK128_GM4_nW4_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM128_BN128_BK128_GM4_nW4_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM128_BN128_BK128_GM4_nW4_nS2_EU0_kP2_mfma16: ', e, flush=True)
        return False


def matmul_BM128_BN128_BK128_GM4_nW4_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM128_BN128_BK128_GM4_nW4_nS2_EU0_kP1_mfma32.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 128,
        BLOCK_SIZE_N = 128,
        BLOCK_SIZE_K = 128,
        GROUP_SIZE_M = 4,
        num_warps = 4,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 1,
        BIAS=False,
        EVEN_K=False,
        grid=(1,),
    )
    return None

def try_compile_config_BM128_BN128_BK128_GM4_nW4_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM128_BN128_BK128_GM4_nW4_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM128_BN128_BK128_GM4_nW4_nS2_EU0_kP1_mfma32: ', e, flush=True)
        return False


def matmul_BM128_BN128_BK128_GM8_nW4_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM128_BN128_BK128_GM8_nW4_nS2_EU0_kP1_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 128,
        BLOCK_SIZE_N = 128,
        BLOCK_SIZE_K = 128,
        GROUP_SIZE_M = 8,
        num_warps = 4,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 1,
        BIAS=False,
        EVEN_K=False,
        grid=(1,),
    )
    return None

def try_compile_config_BM128_BN128_BK128_GM8_nW4_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM128_BN128_BK128_GM8_nW4_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM128_BN128_BK128_GM8_nW4_nS2_EU0_kP1_mfma16: ', e, flush=True)
        return False


def matmul_BM128_BN128_BK128_GM8_nW4_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM128_BN128_BK128_GM8_nW4_nS2_EU0_kP2_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 128,
        BLOCK_SIZE_N = 128,
        BLOCK_SIZE_K = 128,
        GROUP_SIZE_M = 8,
        num_warps = 4,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 2,
        BIAS=False,
        EVEN_K=False,
        grid=(1,),
    )
    return None

def try_compile_config_BM128_BN128_BK128_GM8_nW4_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM128_BN128_BK128_GM8_nW4_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM128_BN128_BK128_GM8_nW4_nS2_EU0_kP2_mfma16: ', e, flush=True)
        return False


def matmul_BM128_BN128_BK128_GM8_nW4_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM128_BN128_BK128_GM8_nW4_nS2_EU0_kP1_mfma32.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 128,
        BLOCK_SIZE_N = 128,
        BLOCK_SIZE_K = 128,
        GROUP_SIZE_M = 8,
        num_warps = 4,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 1,
        BIAS=False,
        EVEN_K=False,
        grid=(1,),
    )
    return None

def try_compile_config_BM128_BN128_BK128_GM8_nW4_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM128_BN128_BK128_GM8_nW4_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM128_BN128_BK128_GM8_nW4_nS2_EU0_kP1_mfma32: ', e, flush=True)
        return False


def matmul_BM128_BN128_BK128_GM16_nW4_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM128_BN128_BK128_GM16_nW4_nS2_EU0_kP1_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 128,
        BLOCK_SIZE_N = 128,
        BLOCK_SIZE_K = 128,
        GROUP_SIZE_M = 16,
        num_warps = 4,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 1,
        BIAS=False,
        EVEN_K=False,
        grid=(1,),
    )
    return None

def try_compile_config_BM128_BN128_BK128_GM16_nW4_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM128_BN128_BK128_GM16_nW4_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM128_BN128_BK128_GM16_nW4_nS2_EU0_kP1_mfma16: ', e, flush=True)
        return False


def matmul_BM128_BN128_BK128_GM16_nW4_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM128_BN128_BK128_GM16_nW4_nS2_EU0_kP2_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 128,
        BLOCK_SIZE_N = 128,
        BLOCK_SIZE_K = 128,
        GROUP_SIZE_M = 16,
        num_warps = 4,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 2,
        BIAS=False,
        EVEN_K=False,
        grid=(1,),
    )
    return None

def try_compile_config_BM128_BN128_BK128_GM16_nW4_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM128_BN128_BK128_GM16_nW4_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM128_BN128_BK128_GM16_nW4_nS2_EU0_kP2_mfma16: ', e, flush=True)
        return False


def matmul_BM128_BN128_BK128_GM16_nW4_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM128_BN128_BK128_GM16_nW4_nS2_EU0_kP1_mfma32.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 128,
        BLOCK_SIZE_N = 128,
        BLOCK_SIZE_K = 128,
        GROUP_SIZE_M = 16,
        num_warps = 4,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 1,
        BIAS=False,
        EVEN_K=False,
        grid=(1,),
    )
    return None

def try_compile_config_BM128_BN128_BK128_GM16_nW4_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM128_BN128_BK128_GM16_nW4_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM128_BN128_BK128_GM16_nW4_nS2_EU0_kP1_mfma32: ', e, flush=True)
        return False


def matmul_BM128_BN128_BK128_GM32_nW4_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM128_BN128_BK128_GM32_nW4_nS2_EU0_kP1_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 128,
        BLOCK_SIZE_N = 128,
        BLOCK_SIZE_K = 128,
        GROUP_SIZE_M = 32,
        num_warps = 4,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 1,
        BIAS=False,
        EVEN_K=False,
        grid=(1,),
    )
    return None

def try_compile_config_BM128_BN128_BK128_GM32_nW4_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM128_BN128_BK128_GM32_nW4_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM128_BN128_BK128_GM32_nW4_nS2_EU0_kP1_mfma16: ', e, flush=True)
        return False


def matmul_BM128_BN128_BK128_GM32_nW4_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM128_BN128_BK128_GM32_nW4_nS2_EU0_kP2_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 128,
        BLOCK_SIZE_N = 128,
        BLOCK_SIZE_K = 128,
        GROUP_SIZE_M = 32,
        num_warps = 4,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 2,
        BIAS=False,
        EVEN_K=False,
        grid=(1,),
    )
    return None

def try_compile_config_BM128_BN128_BK128_GM32_nW4_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM128_BN128_BK128_GM32_nW4_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM128_BN128_BK128_GM32_nW4_nS2_EU0_kP2_mfma16: ', e, flush=True)
        return False


def matmul_BM128_BN128_BK128_GM32_nW4_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM128_BN128_BK128_GM32_nW4_nS2_EU0_kP1_mfma32.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 128,
        BLOCK_SIZE_N = 128,
        BLOCK_SIZE_K = 128,
        GROUP_SIZE_M = 32,
        num_warps = 4,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 1,
        BIAS=False,
        EVEN_K=False,
        grid=(1,),
    )
    return None

def try_compile_config_BM128_BN128_BK128_GM32_nW4_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM128_BN128_BK128_GM32_nW4_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM128_BN128_BK128_GM32_nW4_nS2_EU0_kP1_mfma32: ', e, flush=True)
        return False


def matmul_BM128_BN128_BK128_GM1_nW8_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM128_BN128_BK128_GM1_nW8_nS2_EU0_kP1_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 128,
        BLOCK_SIZE_N = 128,
        BLOCK_SIZE_K = 128,
        GROUP_SIZE_M = 1,
        num_warps = 8,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 1,
        BIAS=False,
        EVEN_K=False,
        grid=(1,),
    )
    return None

def try_compile_config_BM128_BN128_BK128_GM1_nW8_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM128_BN128_BK128_GM1_nW8_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM128_BN128_BK128_GM1_nW8_nS2_EU0_kP1_mfma16: ', e, flush=True)
        return False


def matmul_BM128_BN128_BK128_GM1_nW8_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM128_BN128_BK128_GM1_nW8_nS2_EU0_kP2_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 128,
        BLOCK_SIZE_N = 128,
        BLOCK_SIZE_K = 128,
        GROUP_SIZE_M = 1,
        num_warps = 8,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 2,
        BIAS=False,
        EVEN_K=False,
        grid=(1,),
    )
    return None

def try_compile_config_BM128_BN128_BK128_GM1_nW8_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM128_BN128_BK128_GM1_nW8_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM128_BN128_BK128_GM1_nW8_nS2_EU0_kP2_mfma16: ', e, flush=True)
        return False


def matmul_BM128_BN128_BK128_GM1_nW8_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM128_BN128_BK128_GM1_nW8_nS2_EU0_kP1_mfma32.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 128,
        BLOCK_SIZE_N = 128,
        BLOCK_SIZE_K = 128,
        GROUP_SIZE_M = 1,
        num_warps = 8,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 1,
        BIAS=False,
        EVEN_K=False,
        grid=(1,),
    )
    return None

def try_compile_config_BM128_BN128_BK128_GM1_nW8_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM128_BN128_BK128_GM1_nW8_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM128_BN128_BK128_GM1_nW8_nS2_EU0_kP1_mfma32: ', e, flush=True)
        return False


def matmul_BM128_BN128_BK128_GM4_nW8_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM128_BN128_BK128_GM4_nW8_nS2_EU0_kP1_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 128,
        BLOCK_SIZE_N = 128,
        BLOCK_SIZE_K = 128,
        GROUP_SIZE_M = 4,
        num_warps = 8,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 1,
        BIAS=False,
        EVEN_K=False,
        grid=(1,),
    )
    return None

def try_compile_config_BM128_BN128_BK128_GM4_nW8_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM128_BN128_BK128_GM4_nW8_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM128_BN128_BK128_GM4_nW8_nS2_EU0_kP1_mfma16: ', e, flush=True)
        return False


def matmul_BM128_BN128_BK128_GM4_nW8_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM128_BN128_BK128_GM4_nW8_nS2_EU0_kP2_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 128,
        BLOCK_SIZE_N = 128,
        BLOCK_SIZE_K = 128,
        GROUP_SIZE_M = 4,
        num_warps = 8,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 2,
        BIAS=False,
        EVEN_K=False,
        grid=(1,),
    )
    return None

def try_compile_config_BM128_BN128_BK128_GM4_nW8_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM128_BN128_BK128_GM4_nW8_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM128_BN128_BK128_GM4_nW8_nS2_EU0_kP2_mfma16: ', e, flush=True)
        return False


def matmul_BM128_BN128_BK128_GM4_nW8_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM128_BN128_BK128_GM4_nW8_nS2_EU0_kP1_mfma32.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 128,
        BLOCK_SIZE_N = 128,
        BLOCK_SIZE_K = 128,
        GROUP_SIZE_M = 4,
        num_warps = 8,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 1,
        BIAS=False,
        EVEN_K=False,
        grid=(1,),
    )
    return None

def try_compile_config_BM128_BN128_BK128_GM4_nW8_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM128_BN128_BK128_GM4_nW8_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM128_BN128_BK128_GM4_nW8_nS2_EU0_kP1_mfma32: ', e, flush=True)
        return False


def matmul_BM128_BN128_BK128_GM8_nW8_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM128_BN128_BK128_GM8_nW8_nS2_EU0_kP1_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 128,
        BLOCK_SIZE_N = 128,
        BLOCK_SIZE_K = 128,
        GROUP_SIZE_M = 8,
        num_warps = 8,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 1,
        BIAS=False,
        EVEN_K=False,
        grid=(1,),
    )
    return None

def try_compile_config_BM128_BN128_BK128_GM8_nW8_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM128_BN128_BK128_GM8_nW8_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM128_BN128_BK128_GM8_nW8_nS2_EU0_kP1_mfma16: ', e, flush=True)
        return False


def matmul_BM128_BN128_BK128_GM8_nW8_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM128_BN128_BK128_GM8_nW8_nS2_EU0_kP2_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 128,
        BLOCK_SIZE_N = 128,
        BLOCK_SIZE_K = 128,
        GROUP_SIZE_M = 8,
        num_warps = 8,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 2,
        BIAS=False,
        EVEN_K=False,
        grid=(1,),
    )
    return None

def try_compile_config_BM128_BN128_BK128_GM8_nW8_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM128_BN128_BK128_GM8_nW8_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM128_BN128_BK128_GM8_nW8_nS2_EU0_kP2_mfma16: ', e, flush=True)
        return False


def matmul_BM128_BN128_BK128_GM8_nW8_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM128_BN128_BK128_GM8_nW8_nS2_EU0_kP1_mfma32.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 128,
        BLOCK_SIZE_N = 128,
        BLOCK_SIZE_K = 128,
        GROUP_SIZE_M = 8,
        num_warps = 8,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 1,
        BIAS=False,
        EVEN_K=False,
        grid=(1,),
    )
    return None

def try_compile_config_BM128_BN128_BK128_GM8_nW8_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM128_BN128_BK128_GM8_nW8_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM128_BN128_BK128_GM8_nW8_nS2_EU0_kP1_mfma32: ', e, flush=True)
        return False


def matmul_BM128_BN128_BK128_GM16_nW8_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM128_BN128_BK128_GM16_nW8_nS2_EU0_kP1_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 128,
        BLOCK_SIZE_N = 128,
        BLOCK_SIZE_K = 128,
        GROUP_SIZE_M = 16,
        num_warps = 8,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 1,
        BIAS=False,
        EVEN_K=False,
        grid=(1,),
    )
    return None

def try_compile_config_BM128_BN128_BK128_GM16_nW8_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM128_BN128_BK128_GM16_nW8_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM128_BN128_BK128_GM16_nW8_nS2_EU0_kP1_mfma16: ', e, flush=True)
        return False


def matmul_BM128_BN128_BK128_GM16_nW8_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM128_BN128_BK128_GM16_nW8_nS2_EU0_kP2_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 128,
        BLOCK_SIZE_N = 128,
        BLOCK_SIZE_K = 128,
        GROUP_SIZE_M = 16,
        num_warps = 8,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 2,
        BIAS=False,
        EVEN_K=False,
        grid=(1,),
    )
    return None

def try_compile_config_BM128_BN128_BK128_GM16_nW8_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM128_BN128_BK128_GM16_nW8_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM128_BN128_BK128_GM16_nW8_nS2_EU0_kP2_mfma16: ', e, flush=True)
        return False


def matmul_BM128_BN128_BK128_GM16_nW8_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM128_BN128_BK128_GM16_nW8_nS2_EU0_kP1_mfma32.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 128,
        BLOCK_SIZE_N = 128,
        BLOCK_SIZE_K = 128,
        GROUP_SIZE_M = 16,
        num_warps = 8,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 1,
        BIAS=False,
        EVEN_K=False,
        grid=(1,),
    )
    return None

def try_compile_config_BM128_BN128_BK128_GM16_nW8_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM128_BN128_BK128_GM16_nW8_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM128_BN128_BK128_GM16_nW8_nS2_EU0_kP1_mfma32: ', e, flush=True)
        return False


def matmul_BM128_BN128_BK128_GM32_nW8_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM128_BN128_BK128_GM32_nW8_nS2_EU0_kP1_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 128,
        BLOCK_SIZE_N = 128,
        BLOCK_SIZE_K = 128,
        GROUP_SIZE_M = 32,
        num_warps = 8,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 1,
        BIAS=False,
        EVEN_K=False,
        grid=(1,),
    )
    return None

def try_compile_config_BM128_BN128_BK128_GM32_nW8_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM128_BN128_BK128_GM32_nW8_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM128_BN128_BK128_GM32_nW8_nS2_EU0_kP1_mfma16: ', e, flush=True)
        return False


def matmul_BM128_BN128_BK128_GM32_nW8_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM128_BN128_BK128_GM32_nW8_nS2_EU0_kP2_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 128,
        BLOCK_SIZE_N = 128,
        BLOCK_SIZE_K = 128,
        GROUP_SIZE_M = 32,
        num_warps = 8,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 2,
        BIAS=False,
        EVEN_K=False,
        grid=(1,),
    )
    return None

def try_compile_config_BM128_BN128_BK128_GM32_nW8_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM128_BN128_BK128_GM32_nW8_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM128_BN128_BK128_GM32_nW8_nS2_EU0_kP2_mfma16: ', e, flush=True)
        return False


def matmul_BM128_BN128_BK128_GM32_nW8_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM128_BN128_BK128_GM32_nW8_nS2_EU0_kP1_mfma32.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 128,
        BLOCK_SIZE_N = 128,
        BLOCK_SIZE_K = 128,
        GROUP_SIZE_M = 32,
        num_warps = 8,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 1,
        BIAS=False,
        EVEN_K=False,
        grid=(1,),
    )
    return None

def try_compile_config_BM128_BN128_BK128_GM32_nW8_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM128_BN128_BK128_GM32_nW8_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM128_BN128_BK128_GM32_nW8_nS2_EU0_kP1_mfma32: ', e, flush=True)
        return False


def matmul_BM128_BN256_BK64_GM1_nW4_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM128_BN256_BK64_GM1_nW4_nS2_EU0_kP1_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 128,
        BLOCK_SIZE_N = 256,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 1,
        num_warps = 4,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 1,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM128_BN256_BK64_GM1_nW4_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM128_BN256_BK64_GM1_nW4_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM128_BN256_BK64_GM1_nW4_nS2_EU0_kP1_mfma16: ', e, flush=True)
        return False


def matmul_BM128_BN256_BK64_GM1_nW4_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM128_BN256_BK64_GM1_nW4_nS2_EU0_kP2_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 128,
        BLOCK_SIZE_N = 256,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 1,
        num_warps = 4,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 2,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM128_BN256_BK64_GM1_nW4_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM128_BN256_BK64_GM1_nW4_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM128_BN256_BK64_GM1_nW4_nS2_EU0_kP2_mfma16: ', e, flush=True)
        return False


def matmul_BM128_BN256_BK64_GM1_nW4_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM128_BN256_BK64_GM1_nW4_nS2_EU0_kP1_mfma32.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 128,
        BLOCK_SIZE_N = 256,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 1,
        num_warps = 4,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 1,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM128_BN256_BK64_GM1_nW4_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM128_BN256_BK64_GM1_nW4_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM128_BN256_BK64_GM1_nW4_nS2_EU0_kP1_mfma32: ', e, flush=True)
        return False


def matmul_BM128_BN256_BK64_GM1_nW4_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM128_BN256_BK64_GM1_nW4_nS2_EU0_kP2_mfma32.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 128,
        BLOCK_SIZE_N = 256,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 1,
        num_warps = 4,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 2,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM128_BN256_BK64_GM1_nW4_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM128_BN256_BK64_GM1_nW4_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM128_BN256_BK64_GM1_nW4_nS2_EU0_kP2_mfma32: ', e, flush=True)
        return False


def matmul_BM128_BN256_BK64_GM4_nW4_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM128_BN256_BK64_GM4_nW4_nS2_EU0_kP1_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 128,
        BLOCK_SIZE_N = 256,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 4,
        num_warps = 4,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 1,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM128_BN256_BK64_GM4_nW4_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM128_BN256_BK64_GM4_nW4_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM128_BN256_BK64_GM4_nW4_nS2_EU0_kP1_mfma16: ', e, flush=True)
        return False


def matmul_BM128_BN256_BK64_GM4_nW4_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM128_BN256_BK64_GM4_nW4_nS2_EU0_kP2_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 128,
        BLOCK_SIZE_N = 256,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 4,
        num_warps = 4,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 2,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM128_BN256_BK64_GM4_nW4_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM128_BN256_BK64_GM4_nW4_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM128_BN256_BK64_GM4_nW4_nS2_EU0_kP2_mfma16: ', e, flush=True)
        return False


def matmul_BM128_BN256_BK64_GM4_nW4_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM128_BN256_BK64_GM4_nW4_nS2_EU0_kP1_mfma32.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 128,
        BLOCK_SIZE_N = 256,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 4,
        num_warps = 4,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 1,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM128_BN256_BK64_GM4_nW4_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM128_BN256_BK64_GM4_nW4_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM128_BN256_BK64_GM4_nW4_nS2_EU0_kP1_mfma32: ', e, flush=True)
        return False


def matmul_BM128_BN256_BK64_GM4_nW4_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM128_BN256_BK64_GM4_nW4_nS2_EU0_kP2_mfma32.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 128,
        BLOCK_SIZE_N = 256,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 4,
        num_warps = 4,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 2,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM128_BN256_BK64_GM4_nW4_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM128_BN256_BK64_GM4_nW4_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM128_BN256_BK64_GM4_nW4_nS2_EU0_kP2_mfma32: ', e, flush=True)
        return False


def matmul_BM128_BN256_BK64_GM8_nW4_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM128_BN256_BK64_GM8_nW4_nS2_EU0_kP1_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 128,
        BLOCK_SIZE_N = 256,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 8,
        num_warps = 4,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 1,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM128_BN256_BK64_GM8_nW4_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM128_BN256_BK64_GM8_nW4_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM128_BN256_BK64_GM8_nW4_nS2_EU0_kP1_mfma16: ', e, flush=True)
        return False


def matmul_BM128_BN256_BK64_GM8_nW4_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM128_BN256_BK64_GM8_nW4_nS2_EU0_kP2_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 128,
        BLOCK_SIZE_N = 256,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 8,
        num_warps = 4,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 2,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM128_BN256_BK64_GM8_nW4_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM128_BN256_BK64_GM8_nW4_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM128_BN256_BK64_GM8_nW4_nS2_EU0_kP2_mfma16: ', e, flush=True)
        return False


def matmul_BM128_BN256_BK64_GM8_nW4_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM128_BN256_BK64_GM8_nW4_nS2_EU0_kP1_mfma32.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 128,
        BLOCK_SIZE_N = 256,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 8,
        num_warps = 4,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 1,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM128_BN256_BK64_GM8_nW4_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM128_BN256_BK64_GM8_nW4_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM128_BN256_BK64_GM8_nW4_nS2_EU0_kP1_mfma32: ', e, flush=True)
        return False


def matmul_BM128_BN256_BK64_GM8_nW4_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM128_BN256_BK64_GM8_nW4_nS2_EU0_kP2_mfma32.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 128,
        BLOCK_SIZE_N = 256,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 8,
        num_warps = 4,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 2,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM128_BN256_BK64_GM8_nW4_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM128_BN256_BK64_GM8_nW4_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM128_BN256_BK64_GM8_nW4_nS2_EU0_kP2_mfma32: ', e, flush=True)
        return False


def matmul_BM128_BN256_BK64_GM16_nW4_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM128_BN256_BK64_GM16_nW4_nS2_EU0_kP1_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 128,
        BLOCK_SIZE_N = 256,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 16,
        num_warps = 4,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 1,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM128_BN256_BK64_GM16_nW4_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM128_BN256_BK64_GM16_nW4_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM128_BN256_BK64_GM16_nW4_nS2_EU0_kP1_mfma16: ', e, flush=True)
        return False


def matmul_BM128_BN256_BK64_GM16_nW4_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM128_BN256_BK64_GM16_nW4_nS2_EU0_kP2_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 128,
        BLOCK_SIZE_N = 256,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 16,
        num_warps = 4,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 2,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM128_BN256_BK64_GM16_nW4_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM128_BN256_BK64_GM16_nW4_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM128_BN256_BK64_GM16_nW4_nS2_EU0_kP2_mfma16: ', e, flush=True)
        return False


def matmul_BM128_BN256_BK64_GM16_nW4_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM128_BN256_BK64_GM16_nW4_nS2_EU0_kP1_mfma32.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 128,
        BLOCK_SIZE_N = 256,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 16,
        num_warps = 4,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 1,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM128_BN256_BK64_GM16_nW4_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM128_BN256_BK64_GM16_nW4_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM128_BN256_BK64_GM16_nW4_nS2_EU0_kP1_mfma32: ', e, flush=True)
        return False


def matmul_BM128_BN256_BK64_GM16_nW4_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM128_BN256_BK64_GM16_nW4_nS2_EU0_kP2_mfma32.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 128,
        BLOCK_SIZE_N = 256,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 16,
        num_warps = 4,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 2,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM128_BN256_BK64_GM16_nW4_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM128_BN256_BK64_GM16_nW4_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM128_BN256_BK64_GM16_nW4_nS2_EU0_kP2_mfma32: ', e, flush=True)
        return False


def matmul_BM128_BN256_BK64_GM32_nW4_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM128_BN256_BK64_GM32_nW4_nS2_EU0_kP1_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 128,
        BLOCK_SIZE_N = 256,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 32,
        num_warps = 4,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 1,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM128_BN256_BK64_GM32_nW4_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM128_BN256_BK64_GM32_nW4_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM128_BN256_BK64_GM32_nW4_nS2_EU0_kP1_mfma16: ', e, flush=True)
        return False


def matmul_BM128_BN256_BK64_GM32_nW4_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM128_BN256_BK64_GM32_nW4_nS2_EU0_kP2_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 128,
        BLOCK_SIZE_N = 256,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 32,
        num_warps = 4,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 2,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM128_BN256_BK64_GM32_nW4_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM128_BN256_BK64_GM32_nW4_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM128_BN256_BK64_GM32_nW4_nS2_EU0_kP2_mfma16: ', e, flush=True)
        return False


def matmul_BM128_BN256_BK64_GM32_nW4_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM128_BN256_BK64_GM32_nW4_nS2_EU0_kP1_mfma32.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 128,
        BLOCK_SIZE_N = 256,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 32,
        num_warps = 4,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 1,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM128_BN256_BK64_GM32_nW4_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM128_BN256_BK64_GM32_nW4_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM128_BN256_BK64_GM32_nW4_nS2_EU0_kP1_mfma32: ', e, flush=True)
        return False


def matmul_BM128_BN256_BK64_GM32_nW4_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM128_BN256_BK64_GM32_nW4_nS2_EU0_kP2_mfma32.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 128,
        BLOCK_SIZE_N = 256,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 32,
        num_warps = 4,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 2,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM128_BN256_BK64_GM32_nW4_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM128_BN256_BK64_GM32_nW4_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM128_BN256_BK64_GM32_nW4_nS2_EU0_kP2_mfma32: ', e, flush=True)
        return False


def matmul_BM128_BN256_BK64_GM1_nW8_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM128_BN256_BK64_GM1_nW8_nS2_EU0_kP1_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 128,
        BLOCK_SIZE_N = 256,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 1,
        num_warps = 8,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 1,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM128_BN256_BK64_GM1_nW8_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM128_BN256_BK64_GM1_nW8_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM128_BN256_BK64_GM1_nW8_nS2_EU0_kP1_mfma16: ', e, flush=True)
        return False


def matmul_BM128_BN256_BK64_GM1_nW8_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM128_BN256_BK64_GM1_nW8_nS2_EU0_kP2_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 128,
        BLOCK_SIZE_N = 256,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 1,
        num_warps = 8,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 2,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM128_BN256_BK64_GM1_nW8_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM128_BN256_BK64_GM1_nW8_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM128_BN256_BK64_GM1_nW8_nS2_EU0_kP2_mfma16: ', e, flush=True)
        return False


def matmul_BM128_BN256_BK64_GM1_nW8_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM128_BN256_BK64_GM1_nW8_nS2_EU0_kP1_mfma32.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 128,
        BLOCK_SIZE_N = 256,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 1,
        num_warps = 8,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 1,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM128_BN256_BK64_GM1_nW8_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM128_BN256_BK64_GM1_nW8_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM128_BN256_BK64_GM1_nW8_nS2_EU0_kP1_mfma32: ', e, flush=True)
        return False


def matmul_BM128_BN256_BK64_GM1_nW8_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM128_BN256_BK64_GM1_nW8_nS2_EU0_kP2_mfma32.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 128,
        BLOCK_SIZE_N = 256,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 1,
        num_warps = 8,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 2,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM128_BN256_BK64_GM1_nW8_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM128_BN256_BK64_GM1_nW8_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM128_BN256_BK64_GM1_nW8_nS2_EU0_kP2_mfma32: ', e, flush=True)
        return False


def matmul_BM128_BN256_BK64_GM4_nW8_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM128_BN256_BK64_GM4_nW8_nS2_EU0_kP1_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 128,
        BLOCK_SIZE_N = 256,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 4,
        num_warps = 8,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 1,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM128_BN256_BK64_GM4_nW8_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM128_BN256_BK64_GM4_nW8_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM128_BN256_BK64_GM4_nW8_nS2_EU0_kP1_mfma16: ', e, flush=True)
        return False


def matmul_BM128_BN256_BK64_GM4_nW8_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM128_BN256_BK64_GM4_nW8_nS2_EU0_kP2_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 128,
        BLOCK_SIZE_N = 256,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 4,
        num_warps = 8,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 2,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM128_BN256_BK64_GM4_nW8_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM128_BN256_BK64_GM4_nW8_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM128_BN256_BK64_GM4_nW8_nS2_EU0_kP2_mfma16: ', e, flush=True)
        return False


def matmul_BM128_BN256_BK64_GM4_nW8_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM128_BN256_BK64_GM4_nW8_nS2_EU0_kP1_mfma32.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 128,
        BLOCK_SIZE_N = 256,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 4,
        num_warps = 8,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 1,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM128_BN256_BK64_GM4_nW8_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM128_BN256_BK64_GM4_nW8_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM128_BN256_BK64_GM4_nW8_nS2_EU0_kP1_mfma32: ', e, flush=True)
        return False


def matmul_BM128_BN256_BK64_GM4_nW8_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM128_BN256_BK64_GM4_nW8_nS2_EU0_kP2_mfma32.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 128,
        BLOCK_SIZE_N = 256,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 4,
        num_warps = 8,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 2,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM128_BN256_BK64_GM4_nW8_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM128_BN256_BK64_GM4_nW8_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM128_BN256_BK64_GM4_nW8_nS2_EU0_kP2_mfma32: ', e, flush=True)
        return False


def matmul_BM128_BN256_BK64_GM8_nW8_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM128_BN256_BK64_GM8_nW8_nS2_EU0_kP1_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 128,
        BLOCK_SIZE_N = 256,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 8,
        num_warps = 8,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 1,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM128_BN256_BK64_GM8_nW8_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM128_BN256_BK64_GM8_nW8_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM128_BN256_BK64_GM8_nW8_nS2_EU0_kP1_mfma16: ', e, flush=True)
        return False


def matmul_BM128_BN256_BK64_GM8_nW8_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM128_BN256_BK64_GM8_nW8_nS2_EU0_kP2_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 128,
        BLOCK_SIZE_N = 256,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 8,
        num_warps = 8,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 2,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM128_BN256_BK64_GM8_nW8_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM128_BN256_BK64_GM8_nW8_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM128_BN256_BK64_GM8_nW8_nS2_EU0_kP2_mfma16: ', e, flush=True)
        return False


def matmul_BM128_BN256_BK64_GM8_nW8_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM128_BN256_BK64_GM8_nW8_nS2_EU0_kP1_mfma32.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 128,
        BLOCK_SIZE_N = 256,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 8,
        num_warps = 8,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 1,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM128_BN256_BK64_GM8_nW8_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM128_BN256_BK64_GM8_nW8_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM128_BN256_BK64_GM8_nW8_nS2_EU0_kP1_mfma32: ', e, flush=True)
        return False


def matmul_BM128_BN256_BK64_GM8_nW8_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM128_BN256_BK64_GM8_nW8_nS2_EU0_kP2_mfma32.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 128,
        BLOCK_SIZE_N = 256,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 8,
        num_warps = 8,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 2,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM128_BN256_BK64_GM8_nW8_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM128_BN256_BK64_GM8_nW8_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM128_BN256_BK64_GM8_nW8_nS2_EU0_kP2_mfma32: ', e, flush=True)
        return False


def matmul_BM128_BN256_BK64_GM16_nW8_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM128_BN256_BK64_GM16_nW8_nS2_EU0_kP1_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 128,
        BLOCK_SIZE_N = 256,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 16,
        num_warps = 8,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 1,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM128_BN256_BK64_GM16_nW8_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM128_BN256_BK64_GM16_nW8_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM128_BN256_BK64_GM16_nW8_nS2_EU0_kP1_mfma16: ', e, flush=True)
        return False


def matmul_BM128_BN256_BK64_GM16_nW8_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM128_BN256_BK64_GM16_nW8_nS2_EU0_kP2_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 128,
        BLOCK_SIZE_N = 256,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 16,
        num_warps = 8,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 2,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM128_BN256_BK64_GM16_nW8_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM128_BN256_BK64_GM16_nW8_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM128_BN256_BK64_GM16_nW8_nS2_EU0_kP2_mfma16: ', e, flush=True)
        return False


def matmul_BM128_BN256_BK64_GM16_nW8_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM128_BN256_BK64_GM16_nW8_nS2_EU0_kP1_mfma32.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 128,
        BLOCK_SIZE_N = 256,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 16,
        num_warps = 8,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 1,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM128_BN256_BK64_GM16_nW8_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM128_BN256_BK64_GM16_nW8_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM128_BN256_BK64_GM16_nW8_nS2_EU0_kP1_mfma32: ', e, flush=True)
        return False


def matmul_BM128_BN256_BK64_GM16_nW8_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM128_BN256_BK64_GM16_nW8_nS2_EU0_kP2_mfma32.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 128,
        BLOCK_SIZE_N = 256,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 16,
        num_warps = 8,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 2,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM128_BN256_BK64_GM16_nW8_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM128_BN256_BK64_GM16_nW8_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM128_BN256_BK64_GM16_nW8_nS2_EU0_kP2_mfma32: ', e, flush=True)
        return False


def matmul_BM128_BN256_BK64_GM32_nW8_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM128_BN256_BK64_GM32_nW8_nS2_EU0_kP1_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 128,
        BLOCK_SIZE_N = 256,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 32,
        num_warps = 8,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 1,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM128_BN256_BK64_GM32_nW8_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM128_BN256_BK64_GM32_nW8_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM128_BN256_BK64_GM32_nW8_nS2_EU0_kP1_mfma16: ', e, flush=True)
        return False


def matmul_BM128_BN256_BK64_GM32_nW8_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM128_BN256_BK64_GM32_nW8_nS2_EU0_kP2_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 128,
        BLOCK_SIZE_N = 256,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 32,
        num_warps = 8,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 2,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM128_BN256_BK64_GM32_nW8_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM128_BN256_BK64_GM32_nW8_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM128_BN256_BK64_GM32_nW8_nS2_EU0_kP2_mfma16: ', e, flush=True)
        return False


def matmul_BM128_BN256_BK64_GM32_nW8_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM128_BN256_BK64_GM32_nW8_nS2_EU0_kP1_mfma32.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 128,
        BLOCK_SIZE_N = 256,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 32,
        num_warps = 8,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 1,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM128_BN256_BK64_GM32_nW8_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM128_BN256_BK64_GM32_nW8_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM128_BN256_BK64_GM32_nW8_nS2_EU0_kP1_mfma32: ', e, flush=True)
        return False


def matmul_BM128_BN256_BK64_GM32_nW8_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM128_BN256_BK64_GM32_nW8_nS2_EU0_kP2_mfma32.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 128,
        BLOCK_SIZE_N = 256,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 32,
        num_warps = 8,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 2,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM128_BN256_BK64_GM32_nW8_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM128_BN256_BK64_GM32_nW8_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM128_BN256_BK64_GM32_nW8_nS2_EU0_kP2_mfma32: ', e, flush=True)
        return False


def matmul_BM256_BN64_BK64_GM1_nW4_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM256_BN64_BK64_GM1_nW4_nS2_EU0_kP1_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 256,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 1,
        num_warps = 4,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 1,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM256_BN64_BK64_GM1_nW4_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM256_BN64_BK64_GM1_nW4_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM256_BN64_BK64_GM1_nW4_nS2_EU0_kP1_mfma16: ', e, flush=True)
        return False


def matmul_BM256_BN64_BK64_GM1_nW4_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM256_BN64_BK64_GM1_nW4_nS2_EU0_kP2_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 256,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 1,
        num_warps = 4,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 2,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM256_BN64_BK64_GM1_nW4_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM256_BN64_BK64_GM1_nW4_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM256_BN64_BK64_GM1_nW4_nS2_EU0_kP2_mfma16: ', e, flush=True)
        return False


def matmul_BM256_BN64_BK64_GM1_nW4_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM256_BN64_BK64_GM1_nW4_nS2_EU0_kP1_mfma32.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 256,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 1,
        num_warps = 4,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 1,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM256_BN64_BK64_GM1_nW4_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM256_BN64_BK64_GM1_nW4_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM256_BN64_BK64_GM1_nW4_nS2_EU0_kP1_mfma32: ', e, flush=True)
        return False


def matmul_BM256_BN64_BK64_GM1_nW4_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM256_BN64_BK64_GM1_nW4_nS2_EU0_kP2_mfma32.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 256,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 1,
        num_warps = 4,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 2,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM256_BN64_BK64_GM1_nW4_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM256_BN64_BK64_GM1_nW4_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM256_BN64_BK64_GM1_nW4_nS2_EU0_kP2_mfma32: ', e, flush=True)
        return False


def matmul_BM256_BN64_BK64_GM4_nW4_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM256_BN64_BK64_GM4_nW4_nS2_EU0_kP1_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 256,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 4,
        num_warps = 4,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 1,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM256_BN64_BK64_GM4_nW4_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM256_BN64_BK64_GM4_nW4_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM256_BN64_BK64_GM4_nW4_nS2_EU0_kP1_mfma16: ', e, flush=True)
        return False


def matmul_BM256_BN64_BK64_GM4_nW4_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM256_BN64_BK64_GM4_nW4_nS2_EU0_kP2_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 256,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 4,
        num_warps = 4,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 2,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM256_BN64_BK64_GM4_nW4_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM256_BN64_BK64_GM4_nW4_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM256_BN64_BK64_GM4_nW4_nS2_EU0_kP2_mfma16: ', e, flush=True)
        return False


def matmul_BM256_BN64_BK64_GM4_nW4_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM256_BN64_BK64_GM4_nW4_nS2_EU0_kP1_mfma32.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 256,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 4,
        num_warps = 4,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 1,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM256_BN64_BK64_GM4_nW4_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM256_BN64_BK64_GM4_nW4_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM256_BN64_BK64_GM4_nW4_nS2_EU0_kP1_mfma32: ', e, flush=True)
        return False


def matmul_BM256_BN64_BK64_GM4_nW4_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM256_BN64_BK64_GM4_nW4_nS2_EU0_kP2_mfma32.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 256,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 4,
        num_warps = 4,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 2,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM256_BN64_BK64_GM4_nW4_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM256_BN64_BK64_GM4_nW4_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM256_BN64_BK64_GM4_nW4_nS2_EU0_kP2_mfma32: ', e, flush=True)
        return False


def matmul_BM256_BN64_BK64_GM8_nW4_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM256_BN64_BK64_GM8_nW4_nS2_EU0_kP1_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 256,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 8,
        num_warps = 4,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 1,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM256_BN64_BK64_GM8_nW4_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM256_BN64_BK64_GM8_nW4_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM256_BN64_BK64_GM8_nW4_nS2_EU0_kP1_mfma16: ', e, flush=True)
        return False


def matmul_BM256_BN64_BK64_GM8_nW4_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM256_BN64_BK64_GM8_nW4_nS2_EU0_kP2_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 256,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 8,
        num_warps = 4,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 2,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM256_BN64_BK64_GM8_nW4_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM256_BN64_BK64_GM8_nW4_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM256_BN64_BK64_GM8_nW4_nS2_EU0_kP2_mfma16: ', e, flush=True)
        return False


def matmul_BM256_BN64_BK64_GM8_nW4_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM256_BN64_BK64_GM8_nW4_nS2_EU0_kP1_mfma32.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 256,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 8,
        num_warps = 4,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 1,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM256_BN64_BK64_GM8_nW4_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM256_BN64_BK64_GM8_nW4_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM256_BN64_BK64_GM8_nW4_nS2_EU0_kP1_mfma32: ', e, flush=True)
        return False


def matmul_BM256_BN64_BK64_GM8_nW4_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM256_BN64_BK64_GM8_nW4_nS2_EU0_kP2_mfma32.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 256,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 8,
        num_warps = 4,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 2,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM256_BN64_BK64_GM8_nW4_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM256_BN64_BK64_GM8_nW4_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM256_BN64_BK64_GM8_nW4_nS2_EU0_kP2_mfma32: ', e, flush=True)
        return False


def matmul_BM256_BN64_BK64_GM16_nW4_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM256_BN64_BK64_GM16_nW4_nS2_EU0_kP1_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 256,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 16,
        num_warps = 4,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 1,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM256_BN64_BK64_GM16_nW4_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM256_BN64_BK64_GM16_nW4_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM256_BN64_BK64_GM16_nW4_nS2_EU0_kP1_mfma16: ', e, flush=True)
        return False


def matmul_BM256_BN64_BK64_GM16_nW4_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM256_BN64_BK64_GM16_nW4_nS2_EU0_kP2_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 256,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 16,
        num_warps = 4,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 2,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM256_BN64_BK64_GM16_nW4_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM256_BN64_BK64_GM16_nW4_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM256_BN64_BK64_GM16_nW4_nS2_EU0_kP2_mfma16: ', e, flush=True)
        return False


def matmul_BM256_BN64_BK64_GM16_nW4_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM256_BN64_BK64_GM16_nW4_nS2_EU0_kP1_mfma32.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 256,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 16,
        num_warps = 4,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 1,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM256_BN64_BK64_GM16_nW4_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM256_BN64_BK64_GM16_nW4_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM256_BN64_BK64_GM16_nW4_nS2_EU0_kP1_mfma32: ', e, flush=True)
        return False


def matmul_BM256_BN64_BK64_GM16_nW4_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM256_BN64_BK64_GM16_nW4_nS2_EU0_kP2_mfma32.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 256,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 16,
        num_warps = 4,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 2,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM256_BN64_BK64_GM16_nW4_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM256_BN64_BK64_GM16_nW4_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM256_BN64_BK64_GM16_nW4_nS2_EU0_kP2_mfma32: ', e, flush=True)
        return False


def matmul_BM256_BN64_BK64_GM1_nW8_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM256_BN64_BK64_GM1_nW8_nS2_EU0_kP1_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 256,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 1,
        num_warps = 8,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 1,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM256_BN64_BK64_GM1_nW8_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM256_BN64_BK64_GM1_nW8_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM256_BN64_BK64_GM1_nW8_nS2_EU0_kP1_mfma16: ', e, flush=True)
        return False


def matmul_BM256_BN64_BK64_GM1_nW8_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM256_BN64_BK64_GM1_nW8_nS2_EU0_kP2_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 256,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 1,
        num_warps = 8,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 2,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM256_BN64_BK64_GM1_nW8_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM256_BN64_BK64_GM1_nW8_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM256_BN64_BK64_GM1_nW8_nS2_EU0_kP2_mfma16: ', e, flush=True)
        return False


def matmul_BM256_BN64_BK64_GM1_nW8_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM256_BN64_BK64_GM1_nW8_nS2_EU0_kP1_mfma32.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 256,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 1,
        num_warps = 8,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 1,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM256_BN64_BK64_GM1_nW8_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM256_BN64_BK64_GM1_nW8_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM256_BN64_BK64_GM1_nW8_nS2_EU0_kP1_mfma32: ', e, flush=True)
        return False


def matmul_BM256_BN64_BK64_GM1_nW8_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM256_BN64_BK64_GM1_nW8_nS2_EU0_kP2_mfma32.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 256,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 1,
        num_warps = 8,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 2,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM256_BN64_BK64_GM1_nW8_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM256_BN64_BK64_GM1_nW8_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM256_BN64_BK64_GM1_nW8_nS2_EU0_kP2_mfma32: ', e, flush=True)
        return False


def matmul_BM256_BN64_BK64_GM4_nW8_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM256_BN64_BK64_GM4_nW8_nS2_EU0_kP1_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 256,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 4,
        num_warps = 8,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 1,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM256_BN64_BK64_GM4_nW8_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM256_BN64_BK64_GM4_nW8_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM256_BN64_BK64_GM4_nW8_nS2_EU0_kP1_mfma16: ', e, flush=True)
        return False


def matmul_BM256_BN64_BK64_GM4_nW8_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM256_BN64_BK64_GM4_nW8_nS2_EU0_kP2_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 256,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 4,
        num_warps = 8,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 2,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM256_BN64_BK64_GM4_nW8_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM256_BN64_BK64_GM4_nW8_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM256_BN64_BK64_GM4_nW8_nS2_EU0_kP2_mfma16: ', e, flush=True)
        return False


def matmul_BM256_BN64_BK64_GM4_nW8_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM256_BN64_BK64_GM4_nW8_nS2_EU0_kP1_mfma32.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 256,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 4,
        num_warps = 8,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 1,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM256_BN64_BK64_GM4_nW8_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM256_BN64_BK64_GM4_nW8_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM256_BN64_BK64_GM4_nW8_nS2_EU0_kP1_mfma32: ', e, flush=True)
        return False


def matmul_BM256_BN64_BK64_GM4_nW8_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM256_BN64_BK64_GM4_nW8_nS2_EU0_kP2_mfma32.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 256,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 4,
        num_warps = 8,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 2,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM256_BN64_BK64_GM4_nW8_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM256_BN64_BK64_GM4_nW8_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM256_BN64_BK64_GM4_nW8_nS2_EU0_kP2_mfma32: ', e, flush=True)
        return False


def matmul_BM256_BN64_BK64_GM8_nW8_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM256_BN64_BK64_GM8_nW8_nS2_EU0_kP1_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 256,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 8,
        num_warps = 8,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 1,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM256_BN64_BK64_GM8_nW8_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM256_BN64_BK64_GM8_nW8_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM256_BN64_BK64_GM8_nW8_nS2_EU0_kP1_mfma16: ', e, flush=True)
        return False


def matmul_BM256_BN64_BK64_GM8_nW8_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM256_BN64_BK64_GM8_nW8_nS2_EU0_kP2_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 256,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 8,
        num_warps = 8,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 2,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM256_BN64_BK64_GM8_nW8_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM256_BN64_BK64_GM8_nW8_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM256_BN64_BK64_GM8_nW8_nS2_EU0_kP2_mfma16: ', e, flush=True)
        return False


def matmul_BM256_BN64_BK64_GM8_nW8_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM256_BN64_BK64_GM8_nW8_nS2_EU0_kP1_mfma32.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 256,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 8,
        num_warps = 8,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 1,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM256_BN64_BK64_GM8_nW8_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM256_BN64_BK64_GM8_nW8_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM256_BN64_BK64_GM8_nW8_nS2_EU0_kP1_mfma32: ', e, flush=True)
        return False


def matmul_BM256_BN64_BK64_GM8_nW8_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM256_BN64_BK64_GM8_nW8_nS2_EU0_kP2_mfma32.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 256,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 8,
        num_warps = 8,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 2,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM256_BN64_BK64_GM8_nW8_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM256_BN64_BK64_GM8_nW8_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM256_BN64_BK64_GM8_nW8_nS2_EU0_kP2_mfma32: ', e, flush=True)
        return False


def matmul_BM256_BN64_BK64_GM16_nW8_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM256_BN64_BK64_GM16_nW8_nS2_EU0_kP1_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 256,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 16,
        num_warps = 8,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 1,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM256_BN64_BK64_GM16_nW8_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM256_BN64_BK64_GM16_nW8_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM256_BN64_BK64_GM16_nW8_nS2_EU0_kP1_mfma16: ', e, flush=True)
        return False


def matmul_BM256_BN64_BK64_GM16_nW8_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM256_BN64_BK64_GM16_nW8_nS2_EU0_kP2_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 256,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 16,
        num_warps = 8,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 2,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM256_BN64_BK64_GM16_nW8_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM256_BN64_BK64_GM16_nW8_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM256_BN64_BK64_GM16_nW8_nS2_EU0_kP2_mfma16: ', e, flush=True)
        return False


def matmul_BM256_BN64_BK64_GM16_nW8_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM256_BN64_BK64_GM16_nW8_nS2_EU0_kP1_mfma32.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 256,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 16,
        num_warps = 8,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 1,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM256_BN64_BK64_GM16_nW8_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM256_BN64_BK64_GM16_nW8_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM256_BN64_BK64_GM16_nW8_nS2_EU0_kP1_mfma32: ', e, flush=True)
        return False


def matmul_BM256_BN64_BK64_GM16_nW8_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM256_BN64_BK64_GM16_nW8_nS2_EU0_kP2_mfma32.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 256,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 16,
        num_warps = 8,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 2,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM256_BN64_BK64_GM16_nW8_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM256_BN64_BK64_GM16_nW8_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM256_BN64_BK64_GM16_nW8_nS2_EU0_kP2_mfma32: ', e, flush=True)
        return False


def matmul_BM256_BN128_BK64_GM1_nW4_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM256_BN128_BK64_GM1_nW4_nS2_EU0_kP1_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 256,
        BLOCK_SIZE_N = 128,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 1,
        num_warps = 4,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 1,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM256_BN128_BK64_GM1_nW4_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM256_BN128_BK64_GM1_nW4_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM256_BN128_BK64_GM1_nW4_nS2_EU0_kP1_mfma16: ', e, flush=True)
        return False


def matmul_BM256_BN128_BK64_GM1_nW4_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM256_BN128_BK64_GM1_nW4_nS2_EU0_kP2_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 256,
        BLOCK_SIZE_N = 128,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 1,
        num_warps = 4,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 2,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM256_BN128_BK64_GM1_nW4_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM256_BN128_BK64_GM1_nW4_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM256_BN128_BK64_GM1_nW4_nS2_EU0_kP2_mfma16: ', e, flush=True)
        return False


def matmul_BM256_BN128_BK64_GM1_nW4_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM256_BN128_BK64_GM1_nW4_nS2_EU0_kP1_mfma32.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 256,
        BLOCK_SIZE_N = 128,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 1,
        num_warps = 4,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 1,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM256_BN128_BK64_GM1_nW4_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM256_BN128_BK64_GM1_nW4_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM256_BN128_BK64_GM1_nW4_nS2_EU0_kP1_mfma32: ', e, flush=True)
        return False


def matmul_BM256_BN128_BK64_GM1_nW4_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM256_BN128_BK64_GM1_nW4_nS2_EU0_kP2_mfma32.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 256,
        BLOCK_SIZE_N = 128,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 1,
        num_warps = 4,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 2,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM256_BN128_BK64_GM1_nW4_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM256_BN128_BK64_GM1_nW4_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM256_BN128_BK64_GM1_nW4_nS2_EU0_kP2_mfma32: ', e, flush=True)
        return False


def matmul_BM256_BN128_BK64_GM4_nW4_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM256_BN128_BK64_GM4_nW4_nS2_EU0_kP1_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 256,
        BLOCK_SIZE_N = 128,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 4,
        num_warps = 4,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 1,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM256_BN128_BK64_GM4_nW4_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM256_BN128_BK64_GM4_nW4_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM256_BN128_BK64_GM4_nW4_nS2_EU0_kP1_mfma16: ', e, flush=True)
        return False


def matmul_BM256_BN128_BK64_GM4_nW4_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM256_BN128_BK64_GM4_nW4_nS2_EU0_kP2_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 256,
        BLOCK_SIZE_N = 128,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 4,
        num_warps = 4,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 2,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM256_BN128_BK64_GM4_nW4_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM256_BN128_BK64_GM4_nW4_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM256_BN128_BK64_GM4_nW4_nS2_EU0_kP2_mfma16: ', e, flush=True)
        return False


def matmul_BM256_BN128_BK64_GM4_nW4_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM256_BN128_BK64_GM4_nW4_nS2_EU0_kP1_mfma32.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 256,
        BLOCK_SIZE_N = 128,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 4,
        num_warps = 4,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 1,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM256_BN128_BK64_GM4_nW4_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM256_BN128_BK64_GM4_nW4_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM256_BN128_BK64_GM4_nW4_nS2_EU0_kP1_mfma32: ', e, flush=True)
        return False


def matmul_BM256_BN128_BK64_GM4_nW4_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM256_BN128_BK64_GM4_nW4_nS2_EU0_kP2_mfma32.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 256,
        BLOCK_SIZE_N = 128,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 4,
        num_warps = 4,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 2,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM256_BN128_BK64_GM4_nW4_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM256_BN128_BK64_GM4_nW4_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM256_BN128_BK64_GM4_nW4_nS2_EU0_kP2_mfma32: ', e, flush=True)
        return False


def matmul_BM256_BN128_BK64_GM8_nW4_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM256_BN128_BK64_GM8_nW4_nS2_EU0_kP1_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 256,
        BLOCK_SIZE_N = 128,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 8,
        num_warps = 4,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 1,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM256_BN128_BK64_GM8_nW4_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM256_BN128_BK64_GM8_nW4_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM256_BN128_BK64_GM8_nW4_nS2_EU0_kP1_mfma16: ', e, flush=True)
        return False


def matmul_BM256_BN128_BK64_GM8_nW4_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM256_BN128_BK64_GM8_nW4_nS2_EU0_kP2_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 256,
        BLOCK_SIZE_N = 128,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 8,
        num_warps = 4,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 2,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM256_BN128_BK64_GM8_nW4_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM256_BN128_BK64_GM8_nW4_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM256_BN128_BK64_GM8_nW4_nS2_EU0_kP2_mfma16: ', e, flush=True)
        return False


def matmul_BM256_BN128_BK64_GM8_nW4_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM256_BN128_BK64_GM8_nW4_nS2_EU0_kP1_mfma32.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 256,
        BLOCK_SIZE_N = 128,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 8,
        num_warps = 4,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 1,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM256_BN128_BK64_GM8_nW4_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM256_BN128_BK64_GM8_nW4_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM256_BN128_BK64_GM8_nW4_nS2_EU0_kP1_mfma32: ', e, flush=True)
        return False


def matmul_BM256_BN128_BK64_GM8_nW4_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM256_BN128_BK64_GM8_nW4_nS2_EU0_kP2_mfma32.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 256,
        BLOCK_SIZE_N = 128,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 8,
        num_warps = 4,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 2,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM256_BN128_BK64_GM8_nW4_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM256_BN128_BK64_GM8_nW4_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM256_BN128_BK64_GM8_nW4_nS2_EU0_kP2_mfma32: ', e, flush=True)
        return False


def matmul_BM256_BN128_BK64_GM16_nW4_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM256_BN128_BK64_GM16_nW4_nS2_EU0_kP1_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 256,
        BLOCK_SIZE_N = 128,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 16,
        num_warps = 4,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 1,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM256_BN128_BK64_GM16_nW4_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM256_BN128_BK64_GM16_nW4_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM256_BN128_BK64_GM16_nW4_nS2_EU0_kP1_mfma16: ', e, flush=True)
        return False


def matmul_BM256_BN128_BK64_GM16_nW4_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM256_BN128_BK64_GM16_nW4_nS2_EU0_kP2_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 256,
        BLOCK_SIZE_N = 128,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 16,
        num_warps = 4,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 2,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM256_BN128_BK64_GM16_nW4_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM256_BN128_BK64_GM16_nW4_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM256_BN128_BK64_GM16_nW4_nS2_EU0_kP2_mfma16: ', e, flush=True)
        return False


def matmul_BM256_BN128_BK64_GM16_nW4_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM256_BN128_BK64_GM16_nW4_nS2_EU0_kP1_mfma32.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 256,
        BLOCK_SIZE_N = 128,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 16,
        num_warps = 4,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 1,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM256_BN128_BK64_GM16_nW4_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM256_BN128_BK64_GM16_nW4_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM256_BN128_BK64_GM16_nW4_nS2_EU0_kP1_mfma32: ', e, flush=True)
        return False


def matmul_BM256_BN128_BK64_GM16_nW4_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM256_BN128_BK64_GM16_nW4_nS2_EU0_kP2_mfma32.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 256,
        BLOCK_SIZE_N = 128,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 16,
        num_warps = 4,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 2,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM256_BN128_BK64_GM16_nW4_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM256_BN128_BK64_GM16_nW4_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM256_BN128_BK64_GM16_nW4_nS2_EU0_kP2_mfma32: ', e, flush=True)
        return False


def matmul_BM256_BN128_BK64_GM1_nW8_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM256_BN128_BK64_GM1_nW8_nS2_EU0_kP1_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 256,
        BLOCK_SIZE_N = 128,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 1,
        num_warps = 8,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 1,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM256_BN128_BK64_GM1_nW8_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM256_BN128_BK64_GM1_nW8_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM256_BN128_BK64_GM1_nW8_nS2_EU0_kP1_mfma16: ', e, flush=True)
        return False


def matmul_BM256_BN128_BK64_GM1_nW8_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM256_BN128_BK64_GM1_nW8_nS2_EU0_kP2_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 256,
        BLOCK_SIZE_N = 128,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 1,
        num_warps = 8,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 2,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM256_BN128_BK64_GM1_nW8_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM256_BN128_BK64_GM1_nW8_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM256_BN128_BK64_GM1_nW8_nS2_EU0_kP2_mfma16: ', e, flush=True)
        return False


def matmul_BM256_BN128_BK64_GM1_nW8_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM256_BN128_BK64_GM1_nW8_nS2_EU0_kP1_mfma32.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 256,
        BLOCK_SIZE_N = 128,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 1,
        num_warps = 8,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 1,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM256_BN128_BK64_GM1_nW8_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM256_BN128_BK64_GM1_nW8_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM256_BN128_BK64_GM1_nW8_nS2_EU0_kP1_mfma32: ', e, flush=True)
        return False


def matmul_BM256_BN128_BK64_GM1_nW8_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM256_BN128_BK64_GM1_nW8_nS2_EU0_kP2_mfma32.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 256,
        BLOCK_SIZE_N = 128,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 1,
        num_warps = 8,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 2,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM256_BN128_BK64_GM1_nW8_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM256_BN128_BK64_GM1_nW8_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM256_BN128_BK64_GM1_nW8_nS2_EU0_kP2_mfma32: ', e, flush=True)
        return False


def matmul_BM256_BN128_BK64_GM4_nW8_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM256_BN128_BK64_GM4_nW8_nS2_EU0_kP1_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 256,
        BLOCK_SIZE_N = 128,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 4,
        num_warps = 8,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 1,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM256_BN128_BK64_GM4_nW8_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM256_BN128_BK64_GM4_nW8_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM256_BN128_BK64_GM4_nW8_nS2_EU0_kP1_mfma16: ', e, flush=True)
        return False


def matmul_BM256_BN128_BK64_GM4_nW8_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM256_BN128_BK64_GM4_nW8_nS2_EU0_kP2_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 256,
        BLOCK_SIZE_N = 128,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 4,
        num_warps = 8,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 2,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM256_BN128_BK64_GM4_nW8_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM256_BN128_BK64_GM4_nW8_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM256_BN128_BK64_GM4_nW8_nS2_EU0_kP2_mfma16: ', e, flush=True)
        return False


def matmul_BM256_BN128_BK64_GM4_nW8_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM256_BN128_BK64_GM4_nW8_nS2_EU0_kP1_mfma32.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 256,
        BLOCK_SIZE_N = 128,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 4,
        num_warps = 8,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 1,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM256_BN128_BK64_GM4_nW8_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM256_BN128_BK64_GM4_nW8_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM256_BN128_BK64_GM4_nW8_nS2_EU0_kP1_mfma32: ', e, flush=True)
        return False


def matmul_BM256_BN128_BK64_GM4_nW8_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM256_BN128_BK64_GM4_nW8_nS2_EU0_kP2_mfma32.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 256,
        BLOCK_SIZE_N = 128,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 4,
        num_warps = 8,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 2,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM256_BN128_BK64_GM4_nW8_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM256_BN128_BK64_GM4_nW8_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM256_BN128_BK64_GM4_nW8_nS2_EU0_kP2_mfma32: ', e, flush=True)
        return False


def matmul_BM256_BN128_BK64_GM8_nW8_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM256_BN128_BK64_GM8_nW8_nS2_EU0_kP1_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 256,
        BLOCK_SIZE_N = 128,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 8,
        num_warps = 8,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 1,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM256_BN128_BK64_GM8_nW8_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM256_BN128_BK64_GM8_nW8_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM256_BN128_BK64_GM8_nW8_nS2_EU0_kP1_mfma16: ', e, flush=True)
        return False


def matmul_BM256_BN128_BK64_GM8_nW8_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM256_BN128_BK64_GM8_nW8_nS2_EU0_kP2_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 256,
        BLOCK_SIZE_N = 128,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 8,
        num_warps = 8,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 2,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM256_BN128_BK64_GM8_nW8_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM256_BN128_BK64_GM8_nW8_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM256_BN128_BK64_GM8_nW8_nS2_EU0_kP2_mfma16: ', e, flush=True)
        return False


def matmul_BM256_BN128_BK64_GM8_nW8_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM256_BN128_BK64_GM8_nW8_nS2_EU0_kP1_mfma32.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 256,
        BLOCK_SIZE_N = 128,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 8,
        num_warps = 8,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 1,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM256_BN128_BK64_GM8_nW8_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM256_BN128_BK64_GM8_nW8_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM256_BN128_BK64_GM8_nW8_nS2_EU0_kP1_mfma32: ', e, flush=True)
        return False


def matmul_BM256_BN128_BK64_GM8_nW8_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM256_BN128_BK64_GM8_nW8_nS2_EU0_kP2_mfma32.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 256,
        BLOCK_SIZE_N = 128,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 8,
        num_warps = 8,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 2,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM256_BN128_BK64_GM8_nW8_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM256_BN128_BK64_GM8_nW8_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM256_BN128_BK64_GM8_nW8_nS2_EU0_kP2_mfma32: ', e, flush=True)
        return False


def matmul_BM256_BN128_BK64_GM16_nW8_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM256_BN128_BK64_GM16_nW8_nS2_EU0_kP1_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 256,
        BLOCK_SIZE_N = 128,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 16,
        num_warps = 8,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 1,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM256_BN128_BK64_GM16_nW8_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM256_BN128_BK64_GM16_nW8_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM256_BN128_BK64_GM16_nW8_nS2_EU0_kP1_mfma16: ', e, flush=True)
        return False


def matmul_BM256_BN128_BK64_GM16_nW8_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM256_BN128_BK64_GM16_nW8_nS2_EU0_kP2_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 256,
        BLOCK_SIZE_N = 128,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 16,
        num_warps = 8,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 2,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM256_BN128_BK64_GM16_nW8_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM256_BN128_BK64_GM16_nW8_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM256_BN128_BK64_GM16_nW8_nS2_EU0_kP2_mfma16: ', e, flush=True)
        return False


def matmul_BM256_BN128_BK64_GM16_nW8_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM256_BN128_BK64_GM16_nW8_nS2_EU0_kP1_mfma32.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 256,
        BLOCK_SIZE_N = 128,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 16,
        num_warps = 8,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 1,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM256_BN128_BK64_GM16_nW8_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM256_BN128_BK64_GM16_nW8_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM256_BN128_BK64_GM16_nW8_nS2_EU0_kP1_mfma32: ', e, flush=True)
        return False


def matmul_BM256_BN128_BK64_GM16_nW8_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM256_BN128_BK64_GM16_nW8_nS2_EU0_kP2_mfma32.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 256,
        BLOCK_SIZE_N = 128,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 16,
        num_warps = 8,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 2,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM256_BN128_BK64_GM16_nW8_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM256_BN128_BK64_GM16_nW8_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM256_BN128_BK64_GM16_nW8_nS2_EU0_kP2_mfma32: ', e, flush=True)
        return False


def matmul_BM256_BN256_BK64_GM1_nW4_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM256_BN256_BK64_GM1_nW4_nS2_EU0_kP1_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 256,
        BLOCK_SIZE_N = 256,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 1,
        num_warps = 4,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 1,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM256_BN256_BK64_GM1_nW4_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM256_BN256_BK64_GM1_nW4_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM256_BN256_BK64_GM1_nW4_nS2_EU0_kP1_mfma16: ', e, flush=True)
        return False


def matmul_BM256_BN256_BK64_GM1_nW4_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM256_BN256_BK64_GM1_nW4_nS2_EU0_kP2_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 256,
        BLOCK_SIZE_N = 256,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 1,
        num_warps = 4,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 2,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM256_BN256_BK64_GM1_nW4_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM256_BN256_BK64_GM1_nW4_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM256_BN256_BK64_GM1_nW4_nS2_EU0_kP2_mfma16: ', e, flush=True)
        return False


def matmul_BM256_BN256_BK64_GM1_nW4_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM256_BN256_BK64_GM1_nW4_nS2_EU0_kP1_mfma32.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 256,
        BLOCK_SIZE_N = 256,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 1,
        num_warps = 4,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 1,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM256_BN256_BK64_GM1_nW4_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM256_BN256_BK64_GM1_nW4_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM256_BN256_BK64_GM1_nW4_nS2_EU0_kP1_mfma32: ', e, flush=True)
        return False


def matmul_BM256_BN256_BK64_GM1_nW4_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM256_BN256_BK64_GM1_nW4_nS2_EU0_kP2_mfma32.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 256,
        BLOCK_SIZE_N = 256,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 1,
        num_warps = 4,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 2,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM256_BN256_BK64_GM1_nW4_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM256_BN256_BK64_GM1_nW4_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM256_BN256_BK64_GM1_nW4_nS2_EU0_kP2_mfma32: ', e, flush=True)
        return False


def matmul_BM256_BN256_BK64_GM4_nW4_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM256_BN256_BK64_GM4_nW4_nS2_EU0_kP1_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 256,
        BLOCK_SIZE_N = 256,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 4,
        num_warps = 4,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 1,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM256_BN256_BK64_GM4_nW4_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM256_BN256_BK64_GM4_nW4_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM256_BN256_BK64_GM4_nW4_nS2_EU0_kP1_mfma16: ', e, flush=True)
        return False


def matmul_BM256_BN256_BK64_GM4_nW4_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM256_BN256_BK64_GM4_nW4_nS2_EU0_kP2_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 256,
        BLOCK_SIZE_N = 256,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 4,
        num_warps = 4,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 2,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM256_BN256_BK64_GM4_nW4_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM256_BN256_BK64_GM4_nW4_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM256_BN256_BK64_GM4_nW4_nS2_EU0_kP2_mfma16: ', e, flush=True)
        return False


def matmul_BM256_BN256_BK64_GM4_nW4_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM256_BN256_BK64_GM4_nW4_nS2_EU0_kP1_mfma32.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 256,
        BLOCK_SIZE_N = 256,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 4,
        num_warps = 4,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 1,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM256_BN256_BK64_GM4_nW4_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM256_BN256_BK64_GM4_nW4_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM256_BN256_BK64_GM4_nW4_nS2_EU0_kP1_mfma32: ', e, flush=True)
        return False


def matmul_BM256_BN256_BK64_GM4_nW4_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM256_BN256_BK64_GM4_nW4_nS2_EU0_kP2_mfma32.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 256,
        BLOCK_SIZE_N = 256,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 4,
        num_warps = 4,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 2,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM256_BN256_BK64_GM4_nW4_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM256_BN256_BK64_GM4_nW4_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM256_BN256_BK64_GM4_nW4_nS2_EU0_kP2_mfma32: ', e, flush=True)
        return False


def matmul_BM256_BN256_BK64_GM8_nW4_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM256_BN256_BK64_GM8_nW4_nS2_EU0_kP1_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 256,
        BLOCK_SIZE_N = 256,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 8,
        num_warps = 4,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 1,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM256_BN256_BK64_GM8_nW4_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM256_BN256_BK64_GM8_nW4_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM256_BN256_BK64_GM8_nW4_nS2_EU0_kP1_mfma16: ', e, flush=True)
        return False


def matmul_BM256_BN256_BK64_GM8_nW4_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM256_BN256_BK64_GM8_nW4_nS2_EU0_kP2_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 256,
        BLOCK_SIZE_N = 256,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 8,
        num_warps = 4,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 2,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM256_BN256_BK64_GM8_nW4_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM256_BN256_BK64_GM8_nW4_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM256_BN256_BK64_GM8_nW4_nS2_EU0_kP2_mfma16: ', e, flush=True)
        return False


def matmul_BM256_BN256_BK64_GM8_nW4_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM256_BN256_BK64_GM8_nW4_nS2_EU0_kP1_mfma32.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 256,
        BLOCK_SIZE_N = 256,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 8,
        num_warps = 4,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 1,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM256_BN256_BK64_GM8_nW4_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM256_BN256_BK64_GM8_nW4_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM256_BN256_BK64_GM8_nW4_nS2_EU0_kP1_mfma32: ', e, flush=True)
        return False


def matmul_BM256_BN256_BK64_GM8_nW4_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM256_BN256_BK64_GM8_nW4_nS2_EU0_kP2_mfma32.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 256,
        BLOCK_SIZE_N = 256,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 8,
        num_warps = 4,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 2,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM256_BN256_BK64_GM8_nW4_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM256_BN256_BK64_GM8_nW4_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM256_BN256_BK64_GM8_nW4_nS2_EU0_kP2_mfma32: ', e, flush=True)
        return False


def matmul_BM256_BN256_BK64_GM16_nW4_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM256_BN256_BK64_GM16_nW4_nS2_EU0_kP1_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 256,
        BLOCK_SIZE_N = 256,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 16,
        num_warps = 4,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 1,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM256_BN256_BK64_GM16_nW4_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM256_BN256_BK64_GM16_nW4_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM256_BN256_BK64_GM16_nW4_nS2_EU0_kP1_mfma16: ', e, flush=True)
        return False


def matmul_BM256_BN256_BK64_GM16_nW4_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM256_BN256_BK64_GM16_nW4_nS2_EU0_kP2_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 256,
        BLOCK_SIZE_N = 256,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 16,
        num_warps = 4,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 2,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM256_BN256_BK64_GM16_nW4_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM256_BN256_BK64_GM16_nW4_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM256_BN256_BK64_GM16_nW4_nS2_EU0_kP2_mfma16: ', e, flush=True)
        return False


def matmul_BM256_BN256_BK64_GM16_nW4_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM256_BN256_BK64_GM16_nW4_nS2_EU0_kP1_mfma32.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 256,
        BLOCK_SIZE_N = 256,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 16,
        num_warps = 4,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 1,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM256_BN256_BK64_GM16_nW4_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM256_BN256_BK64_GM16_nW4_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM256_BN256_BK64_GM16_nW4_nS2_EU0_kP1_mfma32: ', e, flush=True)
        return False


def matmul_BM256_BN256_BK64_GM16_nW4_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM256_BN256_BK64_GM16_nW4_nS2_EU0_kP2_mfma32.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 256,
        BLOCK_SIZE_N = 256,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 16,
        num_warps = 4,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 2,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM256_BN256_BK64_GM16_nW4_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM256_BN256_BK64_GM16_nW4_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM256_BN256_BK64_GM16_nW4_nS2_EU0_kP2_mfma32: ', e, flush=True)
        return False


def matmul_BM256_BN256_BK64_GM1_nW8_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM256_BN256_BK64_GM1_nW8_nS2_EU0_kP1_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 256,
        BLOCK_SIZE_N = 256,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 1,
        num_warps = 8,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 1,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM256_BN256_BK64_GM1_nW8_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM256_BN256_BK64_GM1_nW8_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM256_BN256_BK64_GM1_nW8_nS2_EU0_kP1_mfma16: ', e, flush=True)
        return False


def matmul_BM256_BN256_BK64_GM1_nW8_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM256_BN256_BK64_GM1_nW8_nS2_EU0_kP2_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 256,
        BLOCK_SIZE_N = 256,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 1,
        num_warps = 8,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 2,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM256_BN256_BK64_GM1_nW8_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM256_BN256_BK64_GM1_nW8_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM256_BN256_BK64_GM1_nW8_nS2_EU0_kP2_mfma16: ', e, flush=True)
        return False


def matmul_BM256_BN256_BK64_GM1_nW8_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM256_BN256_BK64_GM1_nW8_nS2_EU0_kP1_mfma32.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 256,
        BLOCK_SIZE_N = 256,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 1,
        num_warps = 8,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 1,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM256_BN256_BK64_GM1_nW8_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM256_BN256_BK64_GM1_nW8_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM256_BN256_BK64_GM1_nW8_nS2_EU0_kP1_mfma32: ', e, flush=True)
        return False


def matmul_BM256_BN256_BK64_GM1_nW8_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM256_BN256_BK64_GM1_nW8_nS2_EU0_kP2_mfma32.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 256,
        BLOCK_SIZE_N = 256,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 1,
        num_warps = 8,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 2,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM256_BN256_BK64_GM1_nW8_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM256_BN256_BK64_GM1_nW8_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM256_BN256_BK64_GM1_nW8_nS2_EU0_kP2_mfma32: ', e, flush=True)
        return False


def matmul_BM256_BN256_BK64_GM4_nW8_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM256_BN256_BK64_GM4_nW8_nS2_EU0_kP1_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 256,
        BLOCK_SIZE_N = 256,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 4,
        num_warps = 8,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 1,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM256_BN256_BK64_GM4_nW8_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM256_BN256_BK64_GM4_nW8_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM256_BN256_BK64_GM4_nW8_nS2_EU0_kP1_mfma16: ', e, flush=True)
        return False


def matmul_BM256_BN256_BK64_GM4_nW8_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM256_BN256_BK64_GM4_nW8_nS2_EU0_kP2_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 256,
        BLOCK_SIZE_N = 256,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 4,
        num_warps = 8,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 2,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM256_BN256_BK64_GM4_nW8_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM256_BN256_BK64_GM4_nW8_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM256_BN256_BK64_GM4_nW8_nS2_EU0_kP2_mfma16: ', e, flush=True)
        return False


def matmul_BM256_BN256_BK64_GM4_nW8_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM256_BN256_BK64_GM4_nW8_nS2_EU0_kP1_mfma32.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 256,
        BLOCK_SIZE_N = 256,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 4,
        num_warps = 8,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 1,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM256_BN256_BK64_GM4_nW8_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM256_BN256_BK64_GM4_nW8_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM256_BN256_BK64_GM4_nW8_nS2_EU0_kP1_mfma32: ', e, flush=True)
        return False


def matmul_BM256_BN256_BK64_GM4_nW8_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM256_BN256_BK64_GM4_nW8_nS2_EU0_kP2_mfma32.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 256,
        BLOCK_SIZE_N = 256,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 4,
        num_warps = 8,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 2,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM256_BN256_BK64_GM4_nW8_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM256_BN256_BK64_GM4_nW8_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM256_BN256_BK64_GM4_nW8_nS2_EU0_kP2_mfma32: ', e, flush=True)
        return False


def matmul_BM256_BN256_BK64_GM8_nW8_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM256_BN256_BK64_GM8_nW8_nS2_EU0_kP1_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 256,
        BLOCK_SIZE_N = 256,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 8,
        num_warps = 8,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 1,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM256_BN256_BK64_GM8_nW8_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM256_BN256_BK64_GM8_nW8_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM256_BN256_BK64_GM8_nW8_nS2_EU0_kP1_mfma16: ', e, flush=True)
        return False


def matmul_BM256_BN256_BK64_GM8_nW8_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM256_BN256_BK64_GM8_nW8_nS2_EU0_kP2_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 256,
        BLOCK_SIZE_N = 256,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 8,
        num_warps = 8,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 2,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM256_BN256_BK64_GM8_nW8_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM256_BN256_BK64_GM8_nW8_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM256_BN256_BK64_GM8_nW8_nS2_EU0_kP2_mfma16: ', e, flush=True)
        return False


def matmul_BM256_BN256_BK64_GM8_nW8_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM256_BN256_BK64_GM8_nW8_nS2_EU0_kP1_mfma32.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 256,
        BLOCK_SIZE_N = 256,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 8,
        num_warps = 8,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 1,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM256_BN256_BK64_GM8_nW8_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM256_BN256_BK64_GM8_nW8_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM256_BN256_BK64_GM8_nW8_nS2_EU0_kP1_mfma32: ', e, flush=True)
        return False


def matmul_BM256_BN256_BK64_GM8_nW8_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM256_BN256_BK64_GM8_nW8_nS2_EU0_kP2_mfma32.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 256,
        BLOCK_SIZE_N = 256,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 8,
        num_warps = 8,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 2,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM256_BN256_BK64_GM8_nW8_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM256_BN256_BK64_GM8_nW8_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM256_BN256_BK64_GM8_nW8_nS2_EU0_kP2_mfma32: ', e, flush=True)
        return False


def matmul_BM256_BN256_BK64_GM16_nW8_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM256_BN256_BK64_GM16_nW8_nS2_EU0_kP1_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 256,
        BLOCK_SIZE_N = 256,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 16,
        num_warps = 8,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 1,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM256_BN256_BK64_GM16_nW8_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM256_BN256_BK64_GM16_nW8_nS2_EU0_kP1_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM256_BN256_BK64_GM16_nW8_nS2_EU0_kP1_mfma16: ', e, flush=True)
        return False


def matmul_BM256_BN256_BK64_GM16_nW8_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM256_BN256_BK64_GM16_nW8_nS2_EU0_kP2_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 256,
        BLOCK_SIZE_N = 256,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 16,
        num_warps = 8,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 16,
        kpack = 2,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM256_BN256_BK64_GM16_nW8_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM256_BN256_BK64_GM16_nW8_nS2_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM256_BN256_BK64_GM16_nW8_nS2_EU0_kP2_mfma16: ', e, flush=True)
        return False


def matmul_BM256_BN256_BK64_GM16_nW8_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM256_BN256_BK64_GM16_nW8_nS2_EU0_kP1_mfma32.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 256,
        BLOCK_SIZE_N = 256,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 16,
        num_warps = 8,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 1,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM256_BN256_BK64_GM16_nW8_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM256_BN256_BK64_GM16_nW8_nS2_EU0_kP1_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM256_BN256_BK64_GM16_nW8_nS2_EU0_kP1_mfma32: ', e, flush=True)
        return False


def matmul_BM256_BN256_BK64_GM16_nW8_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM256_BN256_BK64_GM16_nW8_nS2_EU0_kP2_mfma32.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
        M, N, K, num_sms,
        am, ak, bk, bn, cm, cn, biasn,
        BLOCK_SIZE_M = 256,
        BLOCK_SIZE_N = 256,
        BLOCK_SIZE_K = 64,
        GROUP_SIZE_M = 16,
        num_warps = 8,
        num_stages = 2,
        waves_per_eu = 0,
        matrix_instr_nonkdim = 32,
        kpack = 2,
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM256_BN256_BK64_GM16_nW8_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        matmul_BM256_BN256_BK64_GM16_nW8_nS2_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM256_BN256_BK64_GM16_nW8_nS2_EU0_kP2_mfma32: ', e, flush=True)
        return False

def compile_kernels(M, N, K, num_sms, rotating_buffer_size, bias_size, num_threads):
    thread_pool = multiprocessing.Pool(processes=num_threads)

    assert bias_size == M or bias_size == 0

    stride_bias = 1 if bias_size > 0 else 0
    stride_am, stride_ak = M, 1
    stride_bk, stride_bn = 1, N
    stride_cm, stride_cn = N, 1
    task_args = (M, N, K, num_sms,
                 stride_am, stride_ak,
                 stride_bk, stride_bn,
                 stride_cm, stride_cn, stride_bias)

    results = []
    config_names = []

    results += [thread_pool.apply_async(try_compile_config_BM64_BN64_BK64_GM1_nW4_nS2_EU0_kP1_mfma16, args=task_args)]
    config_names += ['BM64_BN64_BK64_GM1_nW4_nS2_EU0_kP1_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM64_BN64_BK64_GM1_nW4_nS2_EU0_kP2_mfma16, args=task_args)]
    config_names += ['BM64_BN64_BK64_GM1_nW4_nS2_EU0_kP2_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM64_BN64_BK64_GM1_nW4_nS2_EU0_kP1_mfma32, args=task_args)]
    config_names += ['BM64_BN64_BK64_GM1_nW4_nS2_EU0_kP1_mfma32']
    results += [thread_pool.apply_async(try_compile_config_BM64_BN64_BK64_GM1_nW4_nS2_EU0_kP2_mfma32, args=task_args)]
    config_names += ['BM64_BN64_BK64_GM1_nW4_nS2_EU0_kP2_mfma32']
    results += [thread_pool.apply_async(try_compile_config_BM64_BN64_BK64_GM4_nW4_nS2_EU0_kP1_mfma16, args=task_args)]
    config_names += ['BM64_BN64_BK64_GM4_nW4_nS2_EU0_kP1_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM64_BN64_BK64_GM4_nW4_nS2_EU0_kP2_mfma16, args=task_args)]
    config_names += ['BM64_BN64_BK64_GM4_nW4_nS2_EU0_kP2_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM64_BN64_BK64_GM4_nW4_nS2_EU0_kP1_mfma32, args=task_args)]
    config_names += ['BM64_BN64_BK64_GM4_nW4_nS2_EU0_kP1_mfma32']
    results += [thread_pool.apply_async(try_compile_config_BM64_BN64_BK64_GM4_nW4_nS2_EU0_kP2_mfma32, args=task_args)]
    config_names += ['BM64_BN64_BK64_GM4_nW4_nS2_EU0_kP2_mfma32']
    results += [thread_pool.apply_async(try_compile_config_BM64_BN64_BK64_GM8_nW4_nS2_EU0_kP1_mfma16, args=task_args)]
    config_names += ['BM64_BN64_BK64_GM8_nW4_nS2_EU0_kP1_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM64_BN64_BK64_GM8_nW4_nS2_EU0_kP2_mfma16, args=task_args)]
    config_names += ['BM64_BN64_BK64_GM8_nW4_nS2_EU0_kP2_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM64_BN64_BK64_GM8_nW4_nS2_EU0_kP1_mfma32, args=task_args)]
    config_names += ['BM64_BN64_BK64_GM8_nW4_nS2_EU0_kP1_mfma32']
    results += [thread_pool.apply_async(try_compile_config_BM64_BN64_BK64_GM8_nW4_nS2_EU0_kP2_mfma32, args=task_args)]
    config_names += ['BM64_BN64_BK64_GM8_nW4_nS2_EU0_kP2_mfma32']
    results += [thread_pool.apply_async(try_compile_config_BM64_BN64_BK64_GM16_nW4_nS2_EU0_kP1_mfma16, args=task_args)]
    config_names += ['BM64_BN64_BK64_GM16_nW4_nS2_EU0_kP1_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM64_BN64_BK64_GM16_nW4_nS2_EU0_kP2_mfma16, args=task_args)]
    config_names += ['BM64_BN64_BK64_GM16_nW4_nS2_EU0_kP2_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM64_BN64_BK64_GM16_nW4_nS2_EU0_kP1_mfma32, args=task_args)]
    config_names += ['BM64_BN64_BK64_GM16_nW4_nS2_EU0_kP1_mfma32']
    results += [thread_pool.apply_async(try_compile_config_BM64_BN64_BK64_GM16_nW4_nS2_EU0_kP2_mfma32, args=task_args)]
    config_names += ['BM64_BN64_BK64_GM16_nW4_nS2_EU0_kP2_mfma32']
    results += [thread_pool.apply_async(try_compile_config_BM64_BN64_BK64_GM32_nW4_nS2_EU0_kP1_mfma16, args=task_args)]
    config_names += ['BM64_BN64_BK64_GM32_nW4_nS2_EU0_kP1_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM64_BN64_BK64_GM32_nW4_nS2_EU0_kP2_mfma16, args=task_args)]
    config_names += ['BM64_BN64_BK64_GM32_nW4_nS2_EU0_kP2_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM64_BN64_BK64_GM32_nW4_nS2_EU0_kP1_mfma32, args=task_args)]
    config_names += ['BM64_BN64_BK64_GM32_nW4_nS2_EU0_kP1_mfma32']
    results += [thread_pool.apply_async(try_compile_config_BM64_BN64_BK64_GM32_nW4_nS2_EU0_kP2_mfma32, args=task_args)]
    config_names += ['BM64_BN64_BK64_GM32_nW4_nS2_EU0_kP2_mfma32']
    results += [thread_pool.apply_async(try_compile_config_BM64_BN64_BK64_GM1_nW8_nS2_EU0_kP1_mfma16, args=task_args)]
    config_names += ['BM64_BN64_BK64_GM1_nW8_nS2_EU0_kP1_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM64_BN64_BK64_GM1_nW8_nS2_EU0_kP2_mfma16, args=task_args)]
    config_names += ['BM64_BN64_BK64_GM1_nW8_nS2_EU0_kP2_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM64_BN64_BK64_GM1_nW8_nS2_EU0_kP1_mfma32, args=task_args)]
    config_names += ['BM64_BN64_BK64_GM1_nW8_nS2_EU0_kP1_mfma32']
    results += [thread_pool.apply_async(try_compile_config_BM64_BN64_BK64_GM1_nW8_nS2_EU0_kP2_mfma32, args=task_args)]
    config_names += ['BM64_BN64_BK64_GM1_nW8_nS2_EU0_kP2_mfma32']
    results += [thread_pool.apply_async(try_compile_config_BM64_BN64_BK64_GM4_nW8_nS2_EU0_kP1_mfma16, args=task_args)]
    config_names += ['BM64_BN64_BK64_GM4_nW8_nS2_EU0_kP1_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM64_BN64_BK64_GM4_nW8_nS2_EU0_kP2_mfma16, args=task_args)]
    config_names += ['BM64_BN64_BK64_GM4_nW8_nS2_EU0_kP2_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM64_BN64_BK64_GM4_nW8_nS2_EU0_kP1_mfma32, args=task_args)]
    config_names += ['BM64_BN64_BK64_GM4_nW8_nS2_EU0_kP1_mfma32']
    results += [thread_pool.apply_async(try_compile_config_BM64_BN64_BK64_GM4_nW8_nS2_EU0_kP2_mfma32, args=task_args)]
    config_names += ['BM64_BN64_BK64_GM4_nW8_nS2_EU0_kP2_mfma32']
    results += [thread_pool.apply_async(try_compile_config_BM64_BN64_BK64_GM8_nW8_nS2_EU0_kP1_mfma16, args=task_args)]
    config_names += ['BM64_BN64_BK64_GM8_nW8_nS2_EU0_kP1_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM64_BN64_BK64_GM8_nW8_nS2_EU0_kP2_mfma16, args=task_args)]
    config_names += ['BM64_BN64_BK64_GM8_nW8_nS2_EU0_kP2_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM64_BN64_BK64_GM8_nW8_nS2_EU0_kP1_mfma32, args=task_args)]
    config_names += ['BM64_BN64_BK64_GM8_nW8_nS2_EU0_kP1_mfma32']
    results += [thread_pool.apply_async(try_compile_config_BM64_BN64_BK64_GM8_nW8_nS2_EU0_kP2_mfma32, args=task_args)]
    config_names += ['BM64_BN64_BK64_GM8_nW8_nS2_EU0_kP2_mfma32']
    results += [thread_pool.apply_async(try_compile_config_BM64_BN64_BK64_GM16_nW8_nS2_EU0_kP1_mfma16, args=task_args)]
    config_names += ['BM64_BN64_BK64_GM16_nW8_nS2_EU0_kP1_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM64_BN64_BK64_GM16_nW8_nS2_EU0_kP2_mfma16, args=task_args)]
    config_names += ['BM64_BN64_BK64_GM16_nW8_nS2_EU0_kP2_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM64_BN64_BK64_GM16_nW8_nS2_EU0_kP1_mfma32, args=task_args)]
    config_names += ['BM64_BN64_BK64_GM16_nW8_nS2_EU0_kP1_mfma32']
    results += [thread_pool.apply_async(try_compile_config_BM64_BN64_BK64_GM16_nW8_nS2_EU0_kP2_mfma32, args=task_args)]
    config_names += ['BM64_BN64_BK64_GM16_nW8_nS2_EU0_kP2_mfma32']
    results += [thread_pool.apply_async(try_compile_config_BM64_BN64_BK64_GM32_nW8_nS2_EU0_kP1_mfma16, args=task_args)]
    config_names += ['BM64_BN64_BK64_GM32_nW8_nS2_EU0_kP1_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM64_BN64_BK64_GM32_nW8_nS2_EU0_kP2_mfma16, args=task_args)]
    config_names += ['BM64_BN64_BK64_GM32_nW8_nS2_EU0_kP2_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM64_BN64_BK64_GM32_nW8_nS2_EU0_kP1_mfma32, args=task_args)]
    config_names += ['BM64_BN64_BK64_GM32_nW8_nS2_EU0_kP1_mfma32']
    results += [thread_pool.apply_async(try_compile_config_BM64_BN64_BK64_GM32_nW8_nS2_EU0_kP2_mfma32, args=task_args)]
    config_names += ['BM64_BN64_BK64_GM32_nW8_nS2_EU0_kP2_mfma32']
    results += [thread_pool.apply_async(try_compile_config_BM64_BN64_BK128_GM1_nW4_nS2_EU0_kP1_mfma16, args=task_args)]
    config_names += ['BM64_BN64_BK128_GM1_nW4_nS2_EU0_kP1_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM64_BN64_BK128_GM1_nW4_nS2_EU0_kP2_mfma16, args=task_args)]
    config_names += ['BM64_BN64_BK128_GM1_nW4_nS2_EU0_kP2_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM64_BN64_BK128_GM1_nW4_nS2_EU0_kP1_mfma32, args=task_args)]
    config_names += ['BM64_BN64_BK128_GM1_nW4_nS2_EU0_kP1_mfma32']
    results += [thread_pool.apply_async(try_compile_config_BM64_BN64_BK128_GM1_nW4_nS2_EU0_kP2_mfma32, args=task_args)]
    config_names += ['BM64_BN64_BK128_GM1_nW4_nS2_EU0_kP2_mfma32']
    results += [thread_pool.apply_async(try_compile_config_BM64_BN64_BK128_GM4_nW4_nS2_EU0_kP1_mfma16, args=task_args)]
    config_names += ['BM64_BN64_BK128_GM4_nW4_nS2_EU0_kP1_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM64_BN64_BK128_GM4_nW4_nS2_EU0_kP2_mfma16, args=task_args)]
    config_names += ['BM64_BN64_BK128_GM4_nW4_nS2_EU0_kP2_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM64_BN64_BK128_GM4_nW4_nS2_EU0_kP1_mfma32, args=task_args)]
    config_names += ['BM64_BN64_BK128_GM4_nW4_nS2_EU0_kP1_mfma32']
    results += [thread_pool.apply_async(try_compile_config_BM64_BN64_BK128_GM4_nW4_nS2_EU0_kP2_mfma32, args=task_args)]
    config_names += ['BM64_BN64_BK128_GM4_nW4_nS2_EU0_kP2_mfma32']
    results += [thread_pool.apply_async(try_compile_config_BM64_BN64_BK128_GM8_nW4_nS2_EU0_kP1_mfma16, args=task_args)]
    config_names += ['BM64_BN64_BK128_GM8_nW4_nS2_EU0_kP1_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM64_BN64_BK128_GM8_nW4_nS2_EU0_kP2_mfma16, args=task_args)]
    config_names += ['BM64_BN64_BK128_GM8_nW4_nS2_EU0_kP2_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM64_BN64_BK128_GM8_nW4_nS2_EU0_kP1_mfma32, args=task_args)]
    config_names += ['BM64_BN64_BK128_GM8_nW4_nS2_EU0_kP1_mfma32']
    results += [thread_pool.apply_async(try_compile_config_BM64_BN64_BK128_GM8_nW4_nS2_EU0_kP2_mfma32, args=task_args)]
    config_names += ['BM64_BN64_BK128_GM8_nW4_nS2_EU0_kP2_mfma32']
    results += [thread_pool.apply_async(try_compile_config_BM64_BN64_BK128_GM16_nW4_nS2_EU0_kP1_mfma16, args=task_args)]
    config_names += ['BM64_BN64_BK128_GM16_nW4_nS2_EU0_kP1_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM64_BN64_BK128_GM16_nW4_nS2_EU0_kP2_mfma16, args=task_args)]
    config_names += ['BM64_BN64_BK128_GM16_nW4_nS2_EU0_kP2_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM64_BN64_BK128_GM16_nW4_nS2_EU0_kP1_mfma32, args=task_args)]
    config_names += ['BM64_BN64_BK128_GM16_nW4_nS2_EU0_kP1_mfma32']
    results += [thread_pool.apply_async(try_compile_config_BM64_BN64_BK128_GM16_nW4_nS2_EU0_kP2_mfma32, args=task_args)]
    config_names += ['BM64_BN64_BK128_GM16_nW4_nS2_EU0_kP2_mfma32']
    results += [thread_pool.apply_async(try_compile_config_BM64_BN64_BK128_GM32_nW4_nS2_EU0_kP1_mfma16, args=task_args)]
    config_names += ['BM64_BN64_BK128_GM32_nW4_nS2_EU0_kP1_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM64_BN64_BK128_GM32_nW4_nS2_EU0_kP2_mfma16, args=task_args)]
    config_names += ['BM64_BN64_BK128_GM32_nW4_nS2_EU0_kP2_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM64_BN64_BK128_GM32_nW4_nS2_EU0_kP1_mfma32, args=task_args)]
    config_names += ['BM64_BN64_BK128_GM32_nW4_nS2_EU0_kP1_mfma32']
    results += [thread_pool.apply_async(try_compile_config_BM64_BN64_BK128_GM32_nW4_nS2_EU0_kP2_mfma32, args=task_args)]
    config_names += ['BM64_BN64_BK128_GM32_nW4_nS2_EU0_kP2_mfma32']
    results += [thread_pool.apply_async(try_compile_config_BM64_BN64_BK128_GM1_nW8_nS2_EU0_kP1_mfma16, args=task_args)]
    config_names += ['BM64_BN64_BK128_GM1_nW8_nS2_EU0_kP1_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM64_BN64_BK128_GM1_nW8_nS2_EU0_kP2_mfma16, args=task_args)]
    config_names += ['BM64_BN64_BK128_GM1_nW8_nS2_EU0_kP2_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM64_BN64_BK128_GM1_nW8_nS2_EU0_kP1_mfma32, args=task_args)]
    config_names += ['BM64_BN64_BK128_GM1_nW8_nS2_EU0_kP1_mfma32']
    results += [thread_pool.apply_async(try_compile_config_BM64_BN64_BK128_GM1_nW8_nS2_EU0_kP2_mfma32, args=task_args)]
    config_names += ['BM64_BN64_BK128_GM1_nW8_nS2_EU0_kP2_mfma32']
    results += [thread_pool.apply_async(try_compile_config_BM64_BN64_BK128_GM4_nW8_nS2_EU0_kP1_mfma16, args=task_args)]
    config_names += ['BM64_BN64_BK128_GM4_nW8_nS2_EU0_kP1_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM64_BN64_BK128_GM4_nW8_nS2_EU0_kP2_mfma16, args=task_args)]
    config_names += ['BM64_BN64_BK128_GM4_nW8_nS2_EU0_kP2_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM64_BN64_BK128_GM4_nW8_nS2_EU0_kP1_mfma32, args=task_args)]
    config_names += ['BM64_BN64_BK128_GM4_nW8_nS2_EU0_kP1_mfma32']
    results += [thread_pool.apply_async(try_compile_config_BM64_BN64_BK128_GM4_nW8_nS2_EU0_kP2_mfma32, args=task_args)]
    config_names += ['BM64_BN64_BK128_GM4_nW8_nS2_EU0_kP2_mfma32']
    results += [thread_pool.apply_async(try_compile_config_BM64_BN64_BK128_GM8_nW8_nS2_EU0_kP1_mfma16, args=task_args)]
    config_names += ['BM64_BN64_BK128_GM8_nW8_nS2_EU0_kP1_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM64_BN64_BK128_GM8_nW8_nS2_EU0_kP2_mfma16, args=task_args)]
    config_names += ['BM64_BN64_BK128_GM8_nW8_nS2_EU0_kP2_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM64_BN64_BK128_GM8_nW8_nS2_EU0_kP1_mfma32, args=task_args)]
    config_names += ['BM64_BN64_BK128_GM8_nW8_nS2_EU0_kP1_mfma32']
    results += [thread_pool.apply_async(try_compile_config_BM64_BN64_BK128_GM8_nW8_nS2_EU0_kP2_mfma32, args=task_args)]
    config_names += ['BM64_BN64_BK128_GM8_nW8_nS2_EU0_kP2_mfma32']
    results += [thread_pool.apply_async(try_compile_config_BM64_BN64_BK128_GM16_nW8_nS2_EU0_kP1_mfma16, args=task_args)]
    config_names += ['BM64_BN64_BK128_GM16_nW8_nS2_EU0_kP1_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM64_BN64_BK128_GM16_nW8_nS2_EU0_kP2_mfma16, args=task_args)]
    config_names += ['BM64_BN64_BK128_GM16_nW8_nS2_EU0_kP2_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM64_BN64_BK128_GM16_nW8_nS2_EU0_kP1_mfma32, args=task_args)]
    config_names += ['BM64_BN64_BK128_GM16_nW8_nS2_EU0_kP1_mfma32']
    results += [thread_pool.apply_async(try_compile_config_BM64_BN64_BK128_GM16_nW8_nS2_EU0_kP2_mfma32, args=task_args)]
    config_names += ['BM64_BN64_BK128_GM16_nW8_nS2_EU0_kP2_mfma32']
    results += [thread_pool.apply_async(try_compile_config_BM64_BN64_BK128_GM32_nW8_nS2_EU0_kP1_mfma16, args=task_args)]
    config_names += ['BM64_BN64_BK128_GM32_nW8_nS2_EU0_kP1_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM64_BN64_BK128_GM32_nW8_nS2_EU0_kP2_mfma16, args=task_args)]
    config_names += ['BM64_BN64_BK128_GM32_nW8_nS2_EU0_kP2_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM64_BN64_BK128_GM32_nW8_nS2_EU0_kP1_mfma32, args=task_args)]
    config_names += ['BM64_BN64_BK128_GM32_nW8_nS2_EU0_kP1_mfma32']
    results += [thread_pool.apply_async(try_compile_config_BM64_BN64_BK128_GM32_nW8_nS2_EU0_kP2_mfma32, args=task_args)]
    config_names += ['BM64_BN64_BK128_GM32_nW8_nS2_EU0_kP2_mfma32']
    results += [thread_pool.apply_async(try_compile_config_BM64_BN64_BK256_GM1_nW4_nS2_EU0_kP1_mfma16, args=task_args)]
    config_names += ['BM64_BN64_BK256_GM1_nW4_nS2_EU0_kP1_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM64_BN64_BK256_GM1_nW4_nS2_EU0_kP2_mfma16, args=task_args)]
    config_names += ['BM64_BN64_BK256_GM1_nW4_nS2_EU0_kP2_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM64_BN64_BK256_GM1_nW4_nS2_EU0_kP1_mfma32, args=task_args)]
    config_names += ['BM64_BN64_BK256_GM1_nW4_nS2_EU0_kP1_mfma32']
    results += [thread_pool.apply_async(try_compile_config_BM64_BN64_BK256_GM1_nW4_nS2_EU0_kP2_mfma32, args=task_args)]
    config_names += ['BM64_BN64_BK256_GM1_nW4_nS2_EU0_kP2_mfma32']
    results += [thread_pool.apply_async(try_compile_config_BM64_BN64_BK256_GM4_nW4_nS2_EU0_kP1_mfma16, args=task_args)]
    config_names += ['BM64_BN64_BK256_GM4_nW4_nS2_EU0_kP1_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM64_BN64_BK256_GM4_nW4_nS2_EU0_kP2_mfma16, args=task_args)]
    config_names += ['BM64_BN64_BK256_GM4_nW4_nS2_EU0_kP2_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM64_BN64_BK256_GM4_nW4_nS2_EU0_kP1_mfma32, args=task_args)]
    config_names += ['BM64_BN64_BK256_GM4_nW4_nS2_EU0_kP1_mfma32']
    results += [thread_pool.apply_async(try_compile_config_BM64_BN64_BK256_GM4_nW4_nS2_EU0_kP2_mfma32, args=task_args)]
    config_names += ['BM64_BN64_BK256_GM4_nW4_nS2_EU0_kP2_mfma32']
    results += [thread_pool.apply_async(try_compile_config_BM64_BN64_BK256_GM8_nW4_nS2_EU0_kP1_mfma16, args=task_args)]
    config_names += ['BM64_BN64_BK256_GM8_nW4_nS2_EU0_kP1_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM64_BN64_BK256_GM8_nW4_nS2_EU0_kP2_mfma16, args=task_args)]
    config_names += ['BM64_BN64_BK256_GM8_nW4_nS2_EU0_kP2_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM64_BN64_BK256_GM8_nW4_nS2_EU0_kP1_mfma32, args=task_args)]
    config_names += ['BM64_BN64_BK256_GM8_nW4_nS2_EU0_kP1_mfma32']
    results += [thread_pool.apply_async(try_compile_config_BM64_BN64_BK256_GM8_nW4_nS2_EU0_kP2_mfma32, args=task_args)]
    config_names += ['BM64_BN64_BK256_GM8_nW4_nS2_EU0_kP2_mfma32']
    results += [thread_pool.apply_async(try_compile_config_BM64_BN64_BK256_GM16_nW4_nS2_EU0_kP1_mfma16, args=task_args)]
    config_names += ['BM64_BN64_BK256_GM16_nW4_nS2_EU0_kP1_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM64_BN64_BK256_GM16_nW4_nS2_EU0_kP2_mfma16, args=task_args)]
    config_names += ['BM64_BN64_BK256_GM16_nW4_nS2_EU0_kP2_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM64_BN64_BK256_GM16_nW4_nS2_EU0_kP1_mfma32, args=task_args)]
    config_names += ['BM64_BN64_BK256_GM16_nW4_nS2_EU0_kP1_mfma32']
    results += [thread_pool.apply_async(try_compile_config_BM64_BN64_BK256_GM16_nW4_nS2_EU0_kP2_mfma32, args=task_args)]
    config_names += ['BM64_BN64_BK256_GM16_nW4_nS2_EU0_kP2_mfma32']
    results += [thread_pool.apply_async(try_compile_config_BM64_BN64_BK256_GM32_nW4_nS2_EU0_kP1_mfma16, args=task_args)]
    config_names += ['BM64_BN64_BK256_GM32_nW4_nS2_EU0_kP1_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM64_BN64_BK256_GM32_nW4_nS2_EU0_kP2_mfma16, args=task_args)]
    config_names += ['BM64_BN64_BK256_GM32_nW4_nS2_EU0_kP2_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM64_BN64_BK256_GM32_nW4_nS2_EU0_kP1_mfma32, args=task_args)]
    config_names += ['BM64_BN64_BK256_GM32_nW4_nS2_EU0_kP1_mfma32']
    results += [thread_pool.apply_async(try_compile_config_BM64_BN64_BK256_GM32_nW4_nS2_EU0_kP2_mfma32, args=task_args)]
    config_names += ['BM64_BN64_BK256_GM32_nW4_nS2_EU0_kP2_mfma32']
    results += [thread_pool.apply_async(try_compile_config_BM64_BN64_BK256_GM1_nW8_nS2_EU0_kP1_mfma16, args=task_args)]
    config_names += ['BM64_BN64_BK256_GM1_nW8_nS2_EU0_kP1_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM64_BN64_BK256_GM1_nW8_nS2_EU0_kP2_mfma16, args=task_args)]
    config_names += ['BM64_BN64_BK256_GM1_nW8_nS2_EU0_kP2_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM64_BN64_BK256_GM1_nW8_nS2_EU0_kP1_mfma32, args=task_args)]
    config_names += ['BM64_BN64_BK256_GM1_nW8_nS2_EU0_kP1_mfma32']
    results += [thread_pool.apply_async(try_compile_config_BM64_BN64_BK256_GM1_nW8_nS2_EU0_kP2_mfma32, args=task_args)]
    config_names += ['BM64_BN64_BK256_GM1_nW8_nS2_EU0_kP2_mfma32']
    results += [thread_pool.apply_async(try_compile_config_BM64_BN64_BK256_GM4_nW8_nS2_EU0_kP1_mfma16, args=task_args)]
    config_names += ['BM64_BN64_BK256_GM4_nW8_nS2_EU0_kP1_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM64_BN64_BK256_GM4_nW8_nS2_EU0_kP2_mfma16, args=task_args)]
    config_names += ['BM64_BN64_BK256_GM4_nW8_nS2_EU0_kP2_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM64_BN64_BK256_GM4_nW8_nS2_EU0_kP1_mfma32, args=task_args)]
    config_names += ['BM64_BN64_BK256_GM4_nW8_nS2_EU0_kP1_mfma32']
    results += [thread_pool.apply_async(try_compile_config_BM64_BN64_BK256_GM4_nW8_nS2_EU0_kP2_mfma32, args=task_args)]
    config_names += ['BM64_BN64_BK256_GM4_nW8_nS2_EU0_kP2_mfma32']
    results += [thread_pool.apply_async(try_compile_config_BM64_BN64_BK256_GM8_nW8_nS2_EU0_kP1_mfma16, args=task_args)]
    config_names += ['BM64_BN64_BK256_GM8_nW8_nS2_EU0_kP1_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM64_BN64_BK256_GM8_nW8_nS2_EU0_kP2_mfma16, args=task_args)]
    config_names += ['BM64_BN64_BK256_GM8_nW8_nS2_EU0_kP2_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM64_BN64_BK256_GM8_nW8_nS2_EU0_kP1_mfma32, args=task_args)]
    config_names += ['BM64_BN64_BK256_GM8_nW8_nS2_EU0_kP1_mfma32']
    results += [thread_pool.apply_async(try_compile_config_BM64_BN64_BK256_GM8_nW8_nS2_EU0_kP2_mfma32, args=task_args)]
    config_names += ['BM64_BN64_BK256_GM8_nW8_nS2_EU0_kP2_mfma32']
    results += [thread_pool.apply_async(try_compile_config_BM64_BN64_BK256_GM16_nW8_nS2_EU0_kP1_mfma16, args=task_args)]
    config_names += ['BM64_BN64_BK256_GM16_nW8_nS2_EU0_kP1_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM64_BN64_BK256_GM16_nW8_nS2_EU0_kP2_mfma16, args=task_args)]
    config_names += ['BM64_BN64_BK256_GM16_nW8_nS2_EU0_kP2_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM64_BN64_BK256_GM16_nW8_nS2_EU0_kP1_mfma32, args=task_args)]
    config_names += ['BM64_BN64_BK256_GM16_nW8_nS2_EU0_kP1_mfma32']
    results += [thread_pool.apply_async(try_compile_config_BM64_BN64_BK256_GM16_nW8_nS2_EU0_kP2_mfma32, args=task_args)]
    config_names += ['BM64_BN64_BK256_GM16_nW8_nS2_EU0_kP2_mfma32']
    results += [thread_pool.apply_async(try_compile_config_BM64_BN64_BK256_GM32_nW8_nS2_EU0_kP1_mfma16, args=task_args)]
    config_names += ['BM64_BN64_BK256_GM32_nW8_nS2_EU0_kP1_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM64_BN64_BK256_GM32_nW8_nS2_EU0_kP2_mfma16, args=task_args)]
    config_names += ['BM64_BN64_BK256_GM32_nW8_nS2_EU0_kP2_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM64_BN64_BK256_GM32_nW8_nS2_EU0_kP1_mfma32, args=task_args)]
    config_names += ['BM64_BN64_BK256_GM32_nW8_nS2_EU0_kP1_mfma32']
    results += [thread_pool.apply_async(try_compile_config_BM64_BN64_BK256_GM32_nW8_nS2_EU0_kP2_mfma32, args=task_args)]
    config_names += ['BM64_BN64_BK256_GM32_nW8_nS2_EU0_kP2_mfma32']
    results += [thread_pool.apply_async(try_compile_config_BM64_BN128_BK64_GM1_nW4_nS2_EU0_kP1_mfma16, args=task_args)]
    config_names += ['BM64_BN128_BK64_GM1_nW4_nS2_EU0_kP1_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM64_BN128_BK64_GM1_nW4_nS2_EU0_kP2_mfma16, args=task_args)]
    config_names += ['BM64_BN128_BK64_GM1_nW4_nS2_EU0_kP2_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM64_BN128_BK64_GM1_nW4_nS2_EU0_kP1_mfma32, args=task_args)]
    config_names += ['BM64_BN128_BK64_GM1_nW4_nS2_EU0_kP1_mfma32']
    results += [thread_pool.apply_async(try_compile_config_BM64_BN128_BK64_GM1_nW4_nS2_EU0_kP2_mfma32, args=task_args)]
    config_names += ['BM64_BN128_BK64_GM1_nW4_nS2_EU0_kP2_mfma32']
    results += [thread_pool.apply_async(try_compile_config_BM64_BN128_BK64_GM4_nW4_nS2_EU0_kP1_mfma16, args=task_args)]
    config_names += ['BM64_BN128_BK64_GM4_nW4_nS2_EU0_kP1_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM64_BN128_BK64_GM4_nW4_nS2_EU0_kP2_mfma16, args=task_args)]
    config_names += ['BM64_BN128_BK64_GM4_nW4_nS2_EU0_kP2_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM64_BN128_BK64_GM4_nW4_nS2_EU0_kP1_mfma32, args=task_args)]
    config_names += ['BM64_BN128_BK64_GM4_nW4_nS2_EU0_kP1_mfma32']
    results += [thread_pool.apply_async(try_compile_config_BM64_BN128_BK64_GM4_nW4_nS2_EU0_kP2_mfma32, args=task_args)]
    config_names += ['BM64_BN128_BK64_GM4_nW4_nS2_EU0_kP2_mfma32']
    results += [thread_pool.apply_async(try_compile_config_BM64_BN128_BK64_GM8_nW4_nS2_EU0_kP1_mfma16, args=task_args)]
    config_names += ['BM64_BN128_BK64_GM8_nW4_nS2_EU0_kP1_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM64_BN128_BK64_GM8_nW4_nS2_EU0_kP2_mfma16, args=task_args)]
    config_names += ['BM64_BN128_BK64_GM8_nW4_nS2_EU0_kP2_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM64_BN128_BK64_GM8_nW4_nS2_EU0_kP1_mfma32, args=task_args)]
    config_names += ['BM64_BN128_BK64_GM8_nW4_nS2_EU0_kP1_mfma32']
    results += [thread_pool.apply_async(try_compile_config_BM64_BN128_BK64_GM8_nW4_nS2_EU0_kP2_mfma32, args=task_args)]
    config_names += ['BM64_BN128_BK64_GM8_nW4_nS2_EU0_kP2_mfma32']
    results += [thread_pool.apply_async(try_compile_config_BM64_BN128_BK64_GM16_nW4_nS2_EU0_kP1_mfma16, args=task_args)]
    config_names += ['BM64_BN128_BK64_GM16_nW4_nS2_EU0_kP1_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM64_BN128_BK64_GM16_nW4_nS2_EU0_kP2_mfma16, args=task_args)]
    config_names += ['BM64_BN128_BK64_GM16_nW4_nS2_EU0_kP2_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM64_BN128_BK64_GM16_nW4_nS2_EU0_kP1_mfma32, args=task_args)]
    config_names += ['BM64_BN128_BK64_GM16_nW4_nS2_EU0_kP1_mfma32']
    results += [thread_pool.apply_async(try_compile_config_BM64_BN128_BK64_GM16_nW4_nS2_EU0_kP2_mfma32, args=task_args)]
    config_names += ['BM64_BN128_BK64_GM16_nW4_nS2_EU0_kP2_mfma32']
    results += [thread_pool.apply_async(try_compile_config_BM64_BN128_BK64_GM32_nW4_nS2_EU0_kP1_mfma16, args=task_args)]
    config_names += ['BM64_BN128_BK64_GM32_nW4_nS2_EU0_kP1_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM64_BN128_BK64_GM32_nW4_nS2_EU0_kP2_mfma16, args=task_args)]
    config_names += ['BM64_BN128_BK64_GM32_nW4_nS2_EU0_kP2_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM64_BN128_BK64_GM32_nW4_nS2_EU0_kP1_mfma32, args=task_args)]
    config_names += ['BM64_BN128_BK64_GM32_nW4_nS2_EU0_kP1_mfma32']
    results += [thread_pool.apply_async(try_compile_config_BM64_BN128_BK64_GM32_nW4_nS2_EU0_kP2_mfma32, args=task_args)]
    config_names += ['BM64_BN128_BK64_GM32_nW4_nS2_EU0_kP2_mfma32']
    results += [thread_pool.apply_async(try_compile_config_BM64_BN128_BK64_GM1_nW8_nS2_EU0_kP1_mfma16, args=task_args)]
    config_names += ['BM64_BN128_BK64_GM1_nW8_nS2_EU0_kP1_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM64_BN128_BK64_GM1_nW8_nS2_EU0_kP2_mfma16, args=task_args)]
    config_names += ['BM64_BN128_BK64_GM1_nW8_nS2_EU0_kP2_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM64_BN128_BK64_GM1_nW8_nS2_EU0_kP1_mfma32, args=task_args)]
    config_names += ['BM64_BN128_BK64_GM1_nW8_nS2_EU0_kP1_mfma32']
    results += [thread_pool.apply_async(try_compile_config_BM64_BN128_BK64_GM1_nW8_nS2_EU0_kP2_mfma32, args=task_args)]
    config_names += ['BM64_BN128_BK64_GM1_nW8_nS2_EU0_kP2_mfma32']
    results += [thread_pool.apply_async(try_compile_config_BM64_BN128_BK64_GM4_nW8_nS2_EU0_kP1_mfma16, args=task_args)]
    config_names += ['BM64_BN128_BK64_GM4_nW8_nS2_EU0_kP1_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM64_BN128_BK64_GM4_nW8_nS2_EU0_kP2_mfma16, args=task_args)]
    config_names += ['BM64_BN128_BK64_GM4_nW8_nS2_EU0_kP2_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM64_BN128_BK64_GM4_nW8_nS2_EU0_kP1_mfma32, args=task_args)]
    config_names += ['BM64_BN128_BK64_GM4_nW8_nS2_EU0_kP1_mfma32']
    results += [thread_pool.apply_async(try_compile_config_BM64_BN128_BK64_GM4_nW8_nS2_EU0_kP2_mfma32, args=task_args)]
    config_names += ['BM64_BN128_BK64_GM4_nW8_nS2_EU0_kP2_mfma32']
    results += [thread_pool.apply_async(try_compile_config_BM64_BN128_BK64_GM8_nW8_nS2_EU0_kP1_mfma16, args=task_args)]
    config_names += ['BM64_BN128_BK64_GM8_nW8_nS2_EU0_kP1_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM64_BN128_BK64_GM8_nW8_nS2_EU0_kP2_mfma16, args=task_args)]
    config_names += ['BM64_BN128_BK64_GM8_nW8_nS2_EU0_kP2_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM64_BN128_BK64_GM8_nW8_nS2_EU0_kP1_mfma32, args=task_args)]
    config_names += ['BM64_BN128_BK64_GM8_nW8_nS2_EU0_kP1_mfma32']
    results += [thread_pool.apply_async(try_compile_config_BM64_BN128_BK64_GM8_nW8_nS2_EU0_kP2_mfma32, args=task_args)]
    config_names += ['BM64_BN128_BK64_GM8_nW8_nS2_EU0_kP2_mfma32']
    results += [thread_pool.apply_async(try_compile_config_BM64_BN128_BK64_GM16_nW8_nS2_EU0_kP1_mfma16, args=task_args)]
    config_names += ['BM64_BN128_BK64_GM16_nW8_nS2_EU0_kP1_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM64_BN128_BK64_GM16_nW8_nS2_EU0_kP2_mfma16, args=task_args)]
    config_names += ['BM64_BN128_BK64_GM16_nW8_nS2_EU0_kP2_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM64_BN128_BK64_GM16_nW8_nS2_EU0_kP1_mfma32, args=task_args)]
    config_names += ['BM64_BN128_BK64_GM16_nW8_nS2_EU0_kP1_mfma32']
    results += [thread_pool.apply_async(try_compile_config_BM64_BN128_BK64_GM16_nW8_nS2_EU0_kP2_mfma32, args=task_args)]
    config_names += ['BM64_BN128_BK64_GM16_nW8_nS2_EU0_kP2_mfma32']
    results += [thread_pool.apply_async(try_compile_config_BM64_BN128_BK64_GM32_nW8_nS2_EU0_kP1_mfma16, args=task_args)]
    config_names += ['BM64_BN128_BK64_GM32_nW8_nS2_EU0_kP1_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM64_BN128_BK64_GM32_nW8_nS2_EU0_kP2_mfma16, args=task_args)]
    config_names += ['BM64_BN128_BK64_GM32_nW8_nS2_EU0_kP2_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM64_BN128_BK64_GM32_nW8_nS2_EU0_kP1_mfma32, args=task_args)]
    config_names += ['BM64_BN128_BK64_GM32_nW8_nS2_EU0_kP1_mfma32']
    results += [thread_pool.apply_async(try_compile_config_BM64_BN128_BK64_GM32_nW8_nS2_EU0_kP2_mfma32, args=task_args)]
    config_names += ['BM64_BN128_BK64_GM32_nW8_nS2_EU0_kP2_mfma32']
    results += [thread_pool.apply_async(try_compile_config_BM64_BN128_BK128_GM1_nW4_nS2_EU0_kP1_mfma16, args=task_args)]
    config_names += ['BM64_BN128_BK128_GM1_nW4_nS2_EU0_kP1_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM64_BN128_BK128_GM1_nW4_nS2_EU0_kP2_mfma16, args=task_args)]
    config_names += ['BM64_BN128_BK128_GM1_nW4_nS2_EU0_kP2_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM64_BN128_BK128_GM1_nW4_nS2_EU0_kP1_mfma32, args=task_args)]
    config_names += ['BM64_BN128_BK128_GM1_nW4_nS2_EU0_kP1_mfma32']
    results += [thread_pool.apply_async(try_compile_config_BM64_BN128_BK128_GM1_nW4_nS2_EU0_kP2_mfma32, args=task_args)]
    config_names += ['BM64_BN128_BK128_GM1_nW4_nS2_EU0_kP2_mfma32']
    results += [thread_pool.apply_async(try_compile_config_BM64_BN128_BK128_GM4_nW4_nS2_EU0_kP1_mfma16, args=task_args)]
    config_names += ['BM64_BN128_BK128_GM4_nW4_nS2_EU0_kP1_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM64_BN128_BK128_GM4_nW4_nS2_EU0_kP2_mfma16, args=task_args)]
    config_names += ['BM64_BN128_BK128_GM4_nW4_nS2_EU0_kP2_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM64_BN128_BK128_GM4_nW4_nS2_EU0_kP1_mfma32, args=task_args)]
    config_names += ['BM64_BN128_BK128_GM4_nW4_nS2_EU0_kP1_mfma32']
    results += [thread_pool.apply_async(try_compile_config_BM64_BN128_BK128_GM4_nW4_nS2_EU0_kP2_mfma32, args=task_args)]
    config_names += ['BM64_BN128_BK128_GM4_nW4_nS2_EU0_kP2_mfma32']
    results += [thread_pool.apply_async(try_compile_config_BM64_BN128_BK128_GM8_nW4_nS2_EU0_kP1_mfma16, args=task_args)]
    config_names += ['BM64_BN128_BK128_GM8_nW4_nS2_EU0_kP1_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM64_BN128_BK128_GM8_nW4_nS2_EU0_kP2_mfma16, args=task_args)]
    config_names += ['BM64_BN128_BK128_GM8_nW4_nS2_EU0_kP2_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM64_BN128_BK128_GM8_nW4_nS2_EU0_kP1_mfma32, args=task_args)]
    config_names += ['BM64_BN128_BK128_GM8_nW4_nS2_EU0_kP1_mfma32']
    results += [thread_pool.apply_async(try_compile_config_BM64_BN128_BK128_GM8_nW4_nS2_EU0_kP2_mfma32, args=task_args)]
    config_names += ['BM64_BN128_BK128_GM8_nW4_nS2_EU0_kP2_mfma32']
    results += [thread_pool.apply_async(try_compile_config_BM64_BN128_BK128_GM16_nW4_nS2_EU0_kP1_mfma16, args=task_args)]
    config_names += ['BM64_BN128_BK128_GM16_nW4_nS2_EU0_kP1_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM64_BN128_BK128_GM16_nW4_nS2_EU0_kP2_mfma16, args=task_args)]
    config_names += ['BM64_BN128_BK128_GM16_nW4_nS2_EU0_kP2_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM64_BN128_BK128_GM16_nW4_nS2_EU0_kP1_mfma32, args=task_args)]
    config_names += ['BM64_BN128_BK128_GM16_nW4_nS2_EU0_kP1_mfma32']
    results += [thread_pool.apply_async(try_compile_config_BM64_BN128_BK128_GM16_nW4_nS2_EU0_kP2_mfma32, args=task_args)]
    config_names += ['BM64_BN128_BK128_GM16_nW4_nS2_EU0_kP2_mfma32']
    results += [thread_pool.apply_async(try_compile_config_BM64_BN128_BK128_GM32_nW4_nS2_EU0_kP1_mfma16, args=task_args)]
    config_names += ['BM64_BN128_BK128_GM32_nW4_nS2_EU0_kP1_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM64_BN128_BK128_GM32_nW4_nS2_EU0_kP2_mfma16, args=task_args)]
    config_names += ['BM64_BN128_BK128_GM32_nW4_nS2_EU0_kP2_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM64_BN128_BK128_GM32_nW4_nS2_EU0_kP1_mfma32, args=task_args)]
    config_names += ['BM64_BN128_BK128_GM32_nW4_nS2_EU0_kP1_mfma32']
    results += [thread_pool.apply_async(try_compile_config_BM64_BN128_BK128_GM32_nW4_nS2_EU0_kP2_mfma32, args=task_args)]
    config_names += ['BM64_BN128_BK128_GM32_nW4_nS2_EU0_kP2_mfma32']
    results += [thread_pool.apply_async(try_compile_config_BM64_BN128_BK128_GM1_nW8_nS2_EU0_kP1_mfma16, args=task_args)]
    config_names += ['BM64_BN128_BK128_GM1_nW8_nS2_EU0_kP1_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM64_BN128_BK128_GM1_nW8_nS2_EU0_kP2_mfma16, args=task_args)]
    config_names += ['BM64_BN128_BK128_GM1_nW8_nS2_EU0_kP2_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM64_BN128_BK128_GM1_nW8_nS2_EU0_kP1_mfma32, args=task_args)]
    config_names += ['BM64_BN128_BK128_GM1_nW8_nS2_EU0_kP1_mfma32']
    results += [thread_pool.apply_async(try_compile_config_BM64_BN128_BK128_GM1_nW8_nS2_EU0_kP2_mfma32, args=task_args)]
    config_names += ['BM64_BN128_BK128_GM1_nW8_nS2_EU0_kP2_mfma32']
    results += [thread_pool.apply_async(try_compile_config_BM64_BN128_BK128_GM4_nW8_nS2_EU0_kP1_mfma16, args=task_args)]
    config_names += ['BM64_BN128_BK128_GM4_nW8_nS2_EU0_kP1_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM64_BN128_BK128_GM4_nW8_nS2_EU0_kP2_mfma16, args=task_args)]
    config_names += ['BM64_BN128_BK128_GM4_nW8_nS2_EU0_kP2_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM64_BN128_BK128_GM4_nW8_nS2_EU0_kP1_mfma32, args=task_args)]
    config_names += ['BM64_BN128_BK128_GM4_nW8_nS2_EU0_kP1_mfma32']
    results += [thread_pool.apply_async(try_compile_config_BM64_BN128_BK128_GM4_nW8_nS2_EU0_kP2_mfma32, args=task_args)]
    config_names += ['BM64_BN128_BK128_GM4_nW8_nS2_EU0_kP2_mfma32']
    results += [thread_pool.apply_async(try_compile_config_BM64_BN128_BK128_GM8_nW8_nS2_EU0_kP1_mfma16, args=task_args)]
    config_names += ['BM64_BN128_BK128_GM8_nW8_nS2_EU0_kP1_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM64_BN128_BK128_GM8_nW8_nS2_EU0_kP2_mfma16, args=task_args)]
    config_names += ['BM64_BN128_BK128_GM8_nW8_nS2_EU0_kP2_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM64_BN128_BK128_GM8_nW8_nS2_EU0_kP1_mfma32, args=task_args)]
    config_names += ['BM64_BN128_BK128_GM8_nW8_nS2_EU0_kP1_mfma32']
    results += [thread_pool.apply_async(try_compile_config_BM64_BN128_BK128_GM8_nW8_nS2_EU0_kP2_mfma32, args=task_args)]
    config_names += ['BM64_BN128_BK128_GM8_nW8_nS2_EU0_kP2_mfma32']
    results += [thread_pool.apply_async(try_compile_config_BM64_BN128_BK128_GM16_nW8_nS2_EU0_kP1_mfma16, args=task_args)]
    config_names += ['BM64_BN128_BK128_GM16_nW8_nS2_EU0_kP1_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM64_BN128_BK128_GM16_nW8_nS2_EU0_kP2_mfma16, args=task_args)]
    config_names += ['BM64_BN128_BK128_GM16_nW8_nS2_EU0_kP2_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM64_BN128_BK128_GM16_nW8_nS2_EU0_kP1_mfma32, args=task_args)]
    config_names += ['BM64_BN128_BK128_GM16_nW8_nS2_EU0_kP1_mfma32']
    results += [thread_pool.apply_async(try_compile_config_BM64_BN128_BK128_GM16_nW8_nS2_EU0_kP2_mfma32, args=task_args)]
    config_names += ['BM64_BN128_BK128_GM16_nW8_nS2_EU0_kP2_mfma32']
    results += [thread_pool.apply_async(try_compile_config_BM64_BN128_BK128_GM32_nW8_nS2_EU0_kP1_mfma16, args=task_args)]
    config_names += ['BM64_BN128_BK128_GM32_nW8_nS2_EU0_kP1_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM64_BN128_BK128_GM32_nW8_nS2_EU0_kP2_mfma16, args=task_args)]
    config_names += ['BM64_BN128_BK128_GM32_nW8_nS2_EU0_kP2_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM64_BN128_BK128_GM32_nW8_nS2_EU0_kP1_mfma32, args=task_args)]
    config_names += ['BM64_BN128_BK128_GM32_nW8_nS2_EU0_kP1_mfma32']
    results += [thread_pool.apply_async(try_compile_config_BM64_BN128_BK128_GM32_nW8_nS2_EU0_kP2_mfma32, args=task_args)]
    config_names += ['BM64_BN128_BK128_GM32_nW8_nS2_EU0_kP2_mfma32']
    results += [thread_pool.apply_async(try_compile_config_BM64_BN256_BK64_GM1_nW4_nS2_EU0_kP1_mfma16, args=task_args)]
    config_names += ['BM64_BN256_BK64_GM1_nW4_nS2_EU0_kP1_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM64_BN256_BK64_GM1_nW4_nS2_EU0_kP2_mfma16, args=task_args)]
    config_names += ['BM64_BN256_BK64_GM1_nW4_nS2_EU0_kP2_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM64_BN256_BK64_GM1_nW4_nS2_EU0_kP1_mfma32, args=task_args)]
    config_names += ['BM64_BN256_BK64_GM1_nW4_nS2_EU0_kP1_mfma32']
    results += [thread_pool.apply_async(try_compile_config_BM64_BN256_BK64_GM1_nW4_nS2_EU0_kP2_mfma32, args=task_args)]
    config_names += ['BM64_BN256_BK64_GM1_nW4_nS2_EU0_kP2_mfma32']
    results += [thread_pool.apply_async(try_compile_config_BM64_BN256_BK64_GM4_nW4_nS2_EU0_kP1_mfma16, args=task_args)]
    config_names += ['BM64_BN256_BK64_GM4_nW4_nS2_EU0_kP1_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM64_BN256_BK64_GM4_nW4_nS2_EU0_kP2_mfma16, args=task_args)]
    config_names += ['BM64_BN256_BK64_GM4_nW4_nS2_EU0_kP2_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM64_BN256_BK64_GM4_nW4_nS2_EU0_kP1_mfma32, args=task_args)]
    config_names += ['BM64_BN256_BK64_GM4_nW4_nS2_EU0_kP1_mfma32']
    results += [thread_pool.apply_async(try_compile_config_BM64_BN256_BK64_GM4_nW4_nS2_EU0_kP2_mfma32, args=task_args)]
    config_names += ['BM64_BN256_BK64_GM4_nW4_nS2_EU0_kP2_mfma32']
    results += [thread_pool.apply_async(try_compile_config_BM64_BN256_BK64_GM8_nW4_nS2_EU0_kP1_mfma16, args=task_args)]
    config_names += ['BM64_BN256_BK64_GM8_nW4_nS2_EU0_kP1_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM64_BN256_BK64_GM8_nW4_nS2_EU0_kP2_mfma16, args=task_args)]
    config_names += ['BM64_BN256_BK64_GM8_nW4_nS2_EU0_kP2_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM64_BN256_BK64_GM8_nW4_nS2_EU0_kP1_mfma32, args=task_args)]
    config_names += ['BM64_BN256_BK64_GM8_nW4_nS2_EU0_kP1_mfma32']
    results += [thread_pool.apply_async(try_compile_config_BM64_BN256_BK64_GM8_nW4_nS2_EU0_kP2_mfma32, args=task_args)]
    config_names += ['BM64_BN256_BK64_GM8_nW4_nS2_EU0_kP2_mfma32']
    results += [thread_pool.apply_async(try_compile_config_BM64_BN256_BK64_GM16_nW4_nS2_EU0_kP1_mfma16, args=task_args)]
    config_names += ['BM64_BN256_BK64_GM16_nW4_nS2_EU0_kP1_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM64_BN256_BK64_GM16_nW4_nS2_EU0_kP2_mfma16, args=task_args)]
    config_names += ['BM64_BN256_BK64_GM16_nW4_nS2_EU0_kP2_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM64_BN256_BK64_GM16_nW4_nS2_EU0_kP1_mfma32, args=task_args)]
    config_names += ['BM64_BN256_BK64_GM16_nW4_nS2_EU0_kP1_mfma32']
    results += [thread_pool.apply_async(try_compile_config_BM64_BN256_BK64_GM16_nW4_nS2_EU0_kP2_mfma32, args=task_args)]
    config_names += ['BM64_BN256_BK64_GM16_nW4_nS2_EU0_kP2_mfma32']
    results += [thread_pool.apply_async(try_compile_config_BM64_BN256_BK64_GM32_nW4_nS2_EU0_kP1_mfma16, args=task_args)]
    config_names += ['BM64_BN256_BK64_GM32_nW4_nS2_EU0_kP1_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM64_BN256_BK64_GM32_nW4_nS2_EU0_kP2_mfma16, args=task_args)]
    config_names += ['BM64_BN256_BK64_GM32_nW4_nS2_EU0_kP2_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM64_BN256_BK64_GM32_nW4_nS2_EU0_kP1_mfma32, args=task_args)]
    config_names += ['BM64_BN256_BK64_GM32_nW4_nS2_EU0_kP1_mfma32']
    results += [thread_pool.apply_async(try_compile_config_BM64_BN256_BK64_GM32_nW4_nS2_EU0_kP2_mfma32, args=task_args)]
    config_names += ['BM64_BN256_BK64_GM32_nW4_nS2_EU0_kP2_mfma32']
    results += [thread_pool.apply_async(try_compile_config_BM64_BN256_BK64_GM1_nW8_nS2_EU0_kP1_mfma16, args=task_args)]
    config_names += ['BM64_BN256_BK64_GM1_nW8_nS2_EU0_kP1_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM64_BN256_BK64_GM1_nW8_nS2_EU0_kP2_mfma16, args=task_args)]
    config_names += ['BM64_BN256_BK64_GM1_nW8_nS2_EU0_kP2_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM64_BN256_BK64_GM1_nW8_nS2_EU0_kP1_mfma32, args=task_args)]
    config_names += ['BM64_BN256_BK64_GM1_nW8_nS2_EU0_kP1_mfma32']
    results += [thread_pool.apply_async(try_compile_config_BM64_BN256_BK64_GM1_nW8_nS2_EU0_kP2_mfma32, args=task_args)]
    config_names += ['BM64_BN256_BK64_GM1_nW8_nS2_EU0_kP2_mfma32']
    results += [thread_pool.apply_async(try_compile_config_BM64_BN256_BK64_GM4_nW8_nS2_EU0_kP1_mfma16, args=task_args)]
    config_names += ['BM64_BN256_BK64_GM4_nW8_nS2_EU0_kP1_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM64_BN256_BK64_GM4_nW8_nS2_EU0_kP2_mfma16, args=task_args)]
    config_names += ['BM64_BN256_BK64_GM4_nW8_nS2_EU0_kP2_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM64_BN256_BK64_GM4_nW8_nS2_EU0_kP1_mfma32, args=task_args)]
    config_names += ['BM64_BN256_BK64_GM4_nW8_nS2_EU0_kP1_mfma32']
    results += [thread_pool.apply_async(try_compile_config_BM64_BN256_BK64_GM4_nW8_nS2_EU0_kP2_mfma32, args=task_args)]
    config_names += ['BM64_BN256_BK64_GM4_nW8_nS2_EU0_kP2_mfma32']
    results += [thread_pool.apply_async(try_compile_config_BM64_BN256_BK64_GM8_nW8_nS2_EU0_kP1_mfma16, args=task_args)]
    config_names += ['BM64_BN256_BK64_GM8_nW8_nS2_EU0_kP1_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM64_BN256_BK64_GM8_nW8_nS2_EU0_kP2_mfma16, args=task_args)]
    config_names += ['BM64_BN256_BK64_GM8_nW8_nS2_EU0_kP2_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM64_BN256_BK64_GM8_nW8_nS2_EU0_kP1_mfma32, args=task_args)]
    config_names += ['BM64_BN256_BK64_GM8_nW8_nS2_EU0_kP1_mfma32']
    results += [thread_pool.apply_async(try_compile_config_BM64_BN256_BK64_GM8_nW8_nS2_EU0_kP2_mfma32, args=task_args)]
    config_names += ['BM64_BN256_BK64_GM8_nW8_nS2_EU0_kP2_mfma32']
    results += [thread_pool.apply_async(try_compile_config_BM64_BN256_BK64_GM16_nW8_nS2_EU0_kP1_mfma16, args=task_args)]
    config_names += ['BM64_BN256_BK64_GM16_nW8_nS2_EU0_kP1_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM64_BN256_BK64_GM16_nW8_nS2_EU0_kP2_mfma16, args=task_args)]
    config_names += ['BM64_BN256_BK64_GM16_nW8_nS2_EU0_kP2_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM64_BN256_BK64_GM16_nW8_nS2_EU0_kP1_mfma32, args=task_args)]
    config_names += ['BM64_BN256_BK64_GM16_nW8_nS2_EU0_kP1_mfma32']
    results += [thread_pool.apply_async(try_compile_config_BM64_BN256_BK64_GM16_nW8_nS2_EU0_kP2_mfma32, args=task_args)]
    config_names += ['BM64_BN256_BK64_GM16_nW8_nS2_EU0_kP2_mfma32']
    results += [thread_pool.apply_async(try_compile_config_BM64_BN256_BK64_GM32_nW8_nS2_EU0_kP1_mfma16, args=task_args)]
    config_names += ['BM64_BN256_BK64_GM32_nW8_nS2_EU0_kP1_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM64_BN256_BK64_GM32_nW8_nS2_EU0_kP2_mfma16, args=task_args)]
    config_names += ['BM64_BN256_BK64_GM32_nW8_nS2_EU0_kP2_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM64_BN256_BK64_GM32_nW8_nS2_EU0_kP1_mfma32, args=task_args)]
    config_names += ['BM64_BN256_BK64_GM32_nW8_nS2_EU0_kP1_mfma32']
    results += [thread_pool.apply_async(try_compile_config_BM64_BN256_BK64_GM32_nW8_nS2_EU0_kP2_mfma32, args=task_args)]
    config_names += ['BM64_BN256_BK64_GM32_nW8_nS2_EU0_kP2_mfma32']
    results += [thread_pool.apply_async(try_compile_config_BM128_BN64_BK64_GM1_nW4_nS2_EU0_kP1_mfma16, args=task_args)]
    config_names += ['BM128_BN64_BK64_GM1_nW4_nS2_EU0_kP1_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM128_BN64_BK64_GM1_nW4_nS2_EU0_kP2_mfma16, args=task_args)]
    config_names += ['BM128_BN64_BK64_GM1_nW4_nS2_EU0_kP2_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM128_BN64_BK64_GM1_nW4_nS2_EU0_kP1_mfma32, args=task_args)]
    config_names += ['BM128_BN64_BK64_GM1_nW4_nS2_EU0_kP1_mfma32']
    results += [thread_pool.apply_async(try_compile_config_BM128_BN64_BK64_GM1_nW4_nS2_EU0_kP2_mfma32, args=task_args)]
    config_names += ['BM128_BN64_BK64_GM1_nW4_nS2_EU0_kP2_mfma32']
    results += [thread_pool.apply_async(try_compile_config_BM128_BN64_BK64_GM4_nW4_nS2_EU0_kP1_mfma16, args=task_args)]
    config_names += ['BM128_BN64_BK64_GM4_nW4_nS2_EU0_kP1_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM128_BN64_BK64_GM4_nW4_nS2_EU0_kP2_mfma16, args=task_args)]
    config_names += ['BM128_BN64_BK64_GM4_nW4_nS2_EU0_kP2_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM128_BN64_BK64_GM4_nW4_nS2_EU0_kP1_mfma32, args=task_args)]
    config_names += ['BM128_BN64_BK64_GM4_nW4_nS2_EU0_kP1_mfma32']
    results += [thread_pool.apply_async(try_compile_config_BM128_BN64_BK64_GM4_nW4_nS2_EU0_kP2_mfma32, args=task_args)]
    config_names += ['BM128_BN64_BK64_GM4_nW4_nS2_EU0_kP2_mfma32']
    results += [thread_pool.apply_async(try_compile_config_BM128_BN64_BK64_GM8_nW4_nS2_EU0_kP1_mfma16, args=task_args)]
    config_names += ['BM128_BN64_BK64_GM8_nW4_nS2_EU0_kP1_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM128_BN64_BK64_GM8_nW4_nS2_EU0_kP2_mfma16, args=task_args)]
    config_names += ['BM128_BN64_BK64_GM8_nW4_nS2_EU0_kP2_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM128_BN64_BK64_GM8_nW4_nS2_EU0_kP1_mfma32, args=task_args)]
    config_names += ['BM128_BN64_BK64_GM8_nW4_nS2_EU0_kP1_mfma32']
    results += [thread_pool.apply_async(try_compile_config_BM128_BN64_BK64_GM8_nW4_nS2_EU0_kP2_mfma32, args=task_args)]
    config_names += ['BM128_BN64_BK64_GM8_nW4_nS2_EU0_kP2_mfma32']
    results += [thread_pool.apply_async(try_compile_config_BM128_BN64_BK64_GM16_nW4_nS2_EU0_kP1_mfma16, args=task_args)]
    config_names += ['BM128_BN64_BK64_GM16_nW4_nS2_EU0_kP1_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM128_BN64_BK64_GM16_nW4_nS2_EU0_kP2_mfma16, args=task_args)]
    config_names += ['BM128_BN64_BK64_GM16_nW4_nS2_EU0_kP2_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM128_BN64_BK64_GM16_nW4_nS2_EU0_kP1_mfma32, args=task_args)]
    config_names += ['BM128_BN64_BK64_GM16_nW4_nS2_EU0_kP1_mfma32']
    results += [thread_pool.apply_async(try_compile_config_BM128_BN64_BK64_GM16_nW4_nS2_EU0_kP2_mfma32, args=task_args)]
    config_names += ['BM128_BN64_BK64_GM16_nW4_nS2_EU0_kP2_mfma32']
    results += [thread_pool.apply_async(try_compile_config_BM128_BN64_BK64_GM32_nW4_nS2_EU0_kP1_mfma16, args=task_args)]
    config_names += ['BM128_BN64_BK64_GM32_nW4_nS2_EU0_kP1_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM128_BN64_BK64_GM32_nW4_nS2_EU0_kP2_mfma16, args=task_args)]
    config_names += ['BM128_BN64_BK64_GM32_nW4_nS2_EU0_kP2_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM128_BN64_BK64_GM32_nW4_nS2_EU0_kP1_mfma32, args=task_args)]
    config_names += ['BM128_BN64_BK64_GM32_nW4_nS2_EU0_kP1_mfma32']
    results += [thread_pool.apply_async(try_compile_config_BM128_BN64_BK64_GM32_nW4_nS2_EU0_kP2_mfma32, args=task_args)]
    config_names += ['BM128_BN64_BK64_GM32_nW4_nS2_EU0_kP2_mfma32']
    results += [thread_pool.apply_async(try_compile_config_BM128_BN64_BK64_GM1_nW8_nS2_EU0_kP1_mfma16, args=task_args)]
    config_names += ['BM128_BN64_BK64_GM1_nW8_nS2_EU0_kP1_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM128_BN64_BK64_GM1_nW8_nS2_EU0_kP2_mfma16, args=task_args)]
    config_names += ['BM128_BN64_BK64_GM1_nW8_nS2_EU0_kP2_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM128_BN64_BK64_GM1_nW8_nS2_EU0_kP1_mfma32, args=task_args)]
    config_names += ['BM128_BN64_BK64_GM1_nW8_nS2_EU0_kP1_mfma32']
    results += [thread_pool.apply_async(try_compile_config_BM128_BN64_BK64_GM1_nW8_nS2_EU0_kP2_mfma32, args=task_args)]
    config_names += ['BM128_BN64_BK64_GM1_nW8_nS2_EU0_kP2_mfma32']
    results += [thread_pool.apply_async(try_compile_config_BM128_BN64_BK64_GM4_nW8_nS2_EU0_kP1_mfma16, args=task_args)]
    config_names += ['BM128_BN64_BK64_GM4_nW8_nS2_EU0_kP1_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM128_BN64_BK64_GM4_nW8_nS2_EU0_kP2_mfma16, args=task_args)]
    config_names += ['BM128_BN64_BK64_GM4_nW8_nS2_EU0_kP2_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM128_BN64_BK64_GM4_nW8_nS2_EU0_kP1_mfma32, args=task_args)]
    config_names += ['BM128_BN64_BK64_GM4_nW8_nS2_EU0_kP1_mfma32']
    results += [thread_pool.apply_async(try_compile_config_BM128_BN64_BK64_GM4_nW8_nS2_EU0_kP2_mfma32, args=task_args)]
    config_names += ['BM128_BN64_BK64_GM4_nW8_nS2_EU0_kP2_mfma32']
    results += [thread_pool.apply_async(try_compile_config_BM128_BN64_BK64_GM8_nW8_nS2_EU0_kP1_mfma16, args=task_args)]
    config_names += ['BM128_BN64_BK64_GM8_nW8_nS2_EU0_kP1_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM128_BN64_BK64_GM8_nW8_nS2_EU0_kP2_mfma16, args=task_args)]
    config_names += ['BM128_BN64_BK64_GM8_nW8_nS2_EU0_kP2_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM128_BN64_BK64_GM8_nW8_nS2_EU0_kP1_mfma32, args=task_args)]
    config_names += ['BM128_BN64_BK64_GM8_nW8_nS2_EU0_kP1_mfma32']
    results += [thread_pool.apply_async(try_compile_config_BM128_BN64_BK64_GM8_nW8_nS2_EU0_kP2_mfma32, args=task_args)]
    config_names += ['BM128_BN64_BK64_GM8_nW8_nS2_EU0_kP2_mfma32']
    results += [thread_pool.apply_async(try_compile_config_BM128_BN64_BK64_GM16_nW8_nS2_EU0_kP1_mfma16, args=task_args)]
    config_names += ['BM128_BN64_BK64_GM16_nW8_nS2_EU0_kP1_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM128_BN64_BK64_GM16_nW8_nS2_EU0_kP2_mfma16, args=task_args)]
    config_names += ['BM128_BN64_BK64_GM16_nW8_nS2_EU0_kP2_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM128_BN64_BK64_GM16_nW8_nS2_EU0_kP1_mfma32, args=task_args)]
    config_names += ['BM128_BN64_BK64_GM16_nW8_nS2_EU0_kP1_mfma32']
    results += [thread_pool.apply_async(try_compile_config_BM128_BN64_BK64_GM16_nW8_nS2_EU0_kP2_mfma32, args=task_args)]
    config_names += ['BM128_BN64_BK64_GM16_nW8_nS2_EU0_kP2_mfma32']
    results += [thread_pool.apply_async(try_compile_config_BM128_BN64_BK64_GM32_nW8_nS2_EU0_kP1_mfma16, args=task_args)]
    config_names += ['BM128_BN64_BK64_GM32_nW8_nS2_EU0_kP1_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM128_BN64_BK64_GM32_nW8_nS2_EU0_kP2_mfma16, args=task_args)]
    config_names += ['BM128_BN64_BK64_GM32_nW8_nS2_EU0_kP2_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM128_BN64_BK64_GM32_nW8_nS2_EU0_kP1_mfma32, args=task_args)]
    config_names += ['BM128_BN64_BK64_GM32_nW8_nS2_EU0_kP1_mfma32']
    results += [thread_pool.apply_async(try_compile_config_BM128_BN64_BK64_GM32_nW8_nS2_EU0_kP2_mfma32, args=task_args)]
    config_names += ['BM128_BN64_BK64_GM32_nW8_nS2_EU0_kP2_mfma32']
    results += [thread_pool.apply_async(try_compile_config_BM128_BN64_BK128_GM1_nW4_nS2_EU0_kP1_mfma16, args=task_args)]
    config_names += ['BM128_BN64_BK128_GM1_nW4_nS2_EU0_kP1_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM128_BN64_BK128_GM1_nW4_nS2_EU0_kP2_mfma16, args=task_args)]
    config_names += ['BM128_BN64_BK128_GM1_nW4_nS2_EU0_kP2_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM128_BN64_BK128_GM1_nW4_nS2_EU0_kP1_mfma32, args=task_args)]
    config_names += ['BM128_BN64_BK128_GM1_nW4_nS2_EU0_kP1_mfma32']
    results += [thread_pool.apply_async(try_compile_config_BM128_BN64_BK128_GM1_nW4_nS2_EU0_kP2_mfma32, args=task_args)]
    config_names += ['BM128_BN64_BK128_GM1_nW4_nS2_EU0_kP2_mfma32']
    results += [thread_pool.apply_async(try_compile_config_BM128_BN64_BK128_GM4_nW4_nS2_EU0_kP1_mfma16, args=task_args)]
    config_names += ['BM128_BN64_BK128_GM4_nW4_nS2_EU0_kP1_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM128_BN64_BK128_GM4_nW4_nS2_EU0_kP2_mfma16, args=task_args)]
    config_names += ['BM128_BN64_BK128_GM4_nW4_nS2_EU0_kP2_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM128_BN64_BK128_GM4_nW4_nS2_EU0_kP1_mfma32, args=task_args)]
    config_names += ['BM128_BN64_BK128_GM4_nW4_nS2_EU0_kP1_mfma32']
    results += [thread_pool.apply_async(try_compile_config_BM128_BN64_BK128_GM4_nW4_nS2_EU0_kP2_mfma32, args=task_args)]
    config_names += ['BM128_BN64_BK128_GM4_nW4_nS2_EU0_kP2_mfma32']
    results += [thread_pool.apply_async(try_compile_config_BM128_BN64_BK128_GM8_nW4_nS2_EU0_kP1_mfma16, args=task_args)]
    config_names += ['BM128_BN64_BK128_GM8_nW4_nS2_EU0_kP1_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM128_BN64_BK128_GM8_nW4_nS2_EU0_kP2_mfma16, args=task_args)]
    config_names += ['BM128_BN64_BK128_GM8_nW4_nS2_EU0_kP2_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM128_BN64_BK128_GM8_nW4_nS2_EU0_kP1_mfma32, args=task_args)]
    config_names += ['BM128_BN64_BK128_GM8_nW4_nS2_EU0_kP1_mfma32']
    results += [thread_pool.apply_async(try_compile_config_BM128_BN64_BK128_GM8_nW4_nS2_EU0_kP2_mfma32, args=task_args)]
    config_names += ['BM128_BN64_BK128_GM8_nW4_nS2_EU0_kP2_mfma32']
    results += [thread_pool.apply_async(try_compile_config_BM128_BN64_BK128_GM16_nW4_nS2_EU0_kP1_mfma16, args=task_args)]
    config_names += ['BM128_BN64_BK128_GM16_nW4_nS2_EU0_kP1_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM128_BN64_BK128_GM16_nW4_nS2_EU0_kP2_mfma16, args=task_args)]
    config_names += ['BM128_BN64_BK128_GM16_nW4_nS2_EU0_kP2_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM128_BN64_BK128_GM16_nW4_nS2_EU0_kP1_mfma32, args=task_args)]
    config_names += ['BM128_BN64_BK128_GM16_nW4_nS2_EU0_kP1_mfma32']
    results += [thread_pool.apply_async(try_compile_config_BM128_BN64_BK128_GM16_nW4_nS2_EU0_kP2_mfma32, args=task_args)]
    config_names += ['BM128_BN64_BK128_GM16_nW4_nS2_EU0_kP2_mfma32']
    results += [thread_pool.apply_async(try_compile_config_BM128_BN64_BK128_GM32_nW4_nS2_EU0_kP1_mfma16, args=task_args)]
    config_names += ['BM128_BN64_BK128_GM32_nW4_nS2_EU0_kP1_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM128_BN64_BK128_GM32_nW4_nS2_EU0_kP2_mfma16, args=task_args)]
    config_names += ['BM128_BN64_BK128_GM32_nW4_nS2_EU0_kP2_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM128_BN64_BK128_GM32_nW4_nS2_EU0_kP1_mfma32, args=task_args)]
    config_names += ['BM128_BN64_BK128_GM32_nW4_nS2_EU0_kP1_mfma32']
    results += [thread_pool.apply_async(try_compile_config_BM128_BN64_BK128_GM32_nW4_nS2_EU0_kP2_mfma32, args=task_args)]
    config_names += ['BM128_BN64_BK128_GM32_nW4_nS2_EU0_kP2_mfma32']
    results += [thread_pool.apply_async(try_compile_config_BM128_BN64_BK128_GM1_nW8_nS2_EU0_kP1_mfma16, args=task_args)]
    config_names += ['BM128_BN64_BK128_GM1_nW8_nS2_EU0_kP1_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM128_BN64_BK128_GM1_nW8_nS2_EU0_kP2_mfma16, args=task_args)]
    config_names += ['BM128_BN64_BK128_GM1_nW8_nS2_EU0_kP2_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM128_BN64_BK128_GM1_nW8_nS2_EU0_kP1_mfma32, args=task_args)]
    config_names += ['BM128_BN64_BK128_GM1_nW8_nS2_EU0_kP1_mfma32']
    results += [thread_pool.apply_async(try_compile_config_BM128_BN64_BK128_GM1_nW8_nS2_EU0_kP2_mfma32, args=task_args)]
    config_names += ['BM128_BN64_BK128_GM1_nW8_nS2_EU0_kP2_mfma32']
    results += [thread_pool.apply_async(try_compile_config_BM128_BN64_BK128_GM4_nW8_nS2_EU0_kP1_mfma16, args=task_args)]
    config_names += ['BM128_BN64_BK128_GM4_nW8_nS2_EU0_kP1_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM128_BN64_BK128_GM4_nW8_nS2_EU0_kP2_mfma16, args=task_args)]
    config_names += ['BM128_BN64_BK128_GM4_nW8_nS2_EU0_kP2_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM128_BN64_BK128_GM4_nW8_nS2_EU0_kP1_mfma32, args=task_args)]
    config_names += ['BM128_BN64_BK128_GM4_nW8_nS2_EU0_kP1_mfma32']
    results += [thread_pool.apply_async(try_compile_config_BM128_BN64_BK128_GM4_nW8_nS2_EU0_kP2_mfma32, args=task_args)]
    config_names += ['BM128_BN64_BK128_GM4_nW8_nS2_EU0_kP2_mfma32']
    results += [thread_pool.apply_async(try_compile_config_BM128_BN64_BK128_GM8_nW8_nS2_EU0_kP1_mfma16, args=task_args)]
    config_names += ['BM128_BN64_BK128_GM8_nW8_nS2_EU0_kP1_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM128_BN64_BK128_GM8_nW8_nS2_EU0_kP2_mfma16, args=task_args)]
    config_names += ['BM128_BN64_BK128_GM8_nW8_nS2_EU0_kP2_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM128_BN64_BK128_GM8_nW8_nS2_EU0_kP1_mfma32, args=task_args)]
    config_names += ['BM128_BN64_BK128_GM8_nW8_nS2_EU0_kP1_mfma32']
    results += [thread_pool.apply_async(try_compile_config_BM128_BN64_BK128_GM8_nW8_nS2_EU0_kP2_mfma32, args=task_args)]
    config_names += ['BM128_BN64_BK128_GM8_nW8_nS2_EU0_kP2_mfma32']
    results += [thread_pool.apply_async(try_compile_config_BM128_BN64_BK128_GM16_nW8_nS2_EU0_kP1_mfma16, args=task_args)]
    config_names += ['BM128_BN64_BK128_GM16_nW8_nS2_EU0_kP1_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM128_BN64_BK128_GM16_nW8_nS2_EU0_kP2_mfma16, args=task_args)]
    config_names += ['BM128_BN64_BK128_GM16_nW8_nS2_EU0_kP2_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM128_BN64_BK128_GM16_nW8_nS2_EU0_kP1_mfma32, args=task_args)]
    config_names += ['BM128_BN64_BK128_GM16_nW8_nS2_EU0_kP1_mfma32']
    results += [thread_pool.apply_async(try_compile_config_BM128_BN64_BK128_GM16_nW8_nS2_EU0_kP2_mfma32, args=task_args)]
    config_names += ['BM128_BN64_BK128_GM16_nW8_nS2_EU0_kP2_mfma32']
    results += [thread_pool.apply_async(try_compile_config_BM128_BN64_BK128_GM32_nW8_nS2_EU0_kP1_mfma16, args=task_args)]
    config_names += ['BM128_BN64_BK128_GM32_nW8_nS2_EU0_kP1_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM128_BN64_BK128_GM32_nW8_nS2_EU0_kP2_mfma16, args=task_args)]
    config_names += ['BM128_BN64_BK128_GM32_nW8_nS2_EU0_kP2_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM128_BN64_BK128_GM32_nW8_nS2_EU0_kP1_mfma32, args=task_args)]
    config_names += ['BM128_BN64_BK128_GM32_nW8_nS2_EU0_kP1_mfma32']
    results += [thread_pool.apply_async(try_compile_config_BM128_BN64_BK128_GM32_nW8_nS2_EU0_kP2_mfma32, args=task_args)]
    config_names += ['BM128_BN64_BK128_GM32_nW8_nS2_EU0_kP2_mfma32']
    results += [thread_pool.apply_async(try_compile_config_BM128_BN128_BK64_GM1_nW4_nS2_EU0_kP1_mfma16, args=task_args)]
    config_names += ['BM128_BN128_BK64_GM1_nW4_nS2_EU0_kP1_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM128_BN128_BK64_GM1_nW4_nS2_EU0_kP2_mfma16, args=task_args)]
    config_names += ['BM128_BN128_BK64_GM1_nW4_nS2_EU0_kP2_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM128_BN128_BK64_GM1_nW4_nS2_EU0_kP1_mfma32, args=task_args)]
    config_names += ['BM128_BN128_BK64_GM1_nW4_nS2_EU0_kP1_mfma32']
    results += [thread_pool.apply_async(try_compile_config_BM128_BN128_BK64_GM1_nW4_nS2_EU0_kP2_mfma32, args=task_args)]
    config_names += ['BM128_BN128_BK64_GM1_nW4_nS2_EU0_kP2_mfma32']
    results += [thread_pool.apply_async(try_compile_config_BM128_BN128_BK64_GM4_nW4_nS2_EU0_kP1_mfma16, args=task_args)]
    config_names += ['BM128_BN128_BK64_GM4_nW4_nS2_EU0_kP1_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM128_BN128_BK64_GM4_nW4_nS2_EU0_kP2_mfma16, args=task_args)]
    config_names += ['BM128_BN128_BK64_GM4_nW4_nS2_EU0_kP2_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM128_BN128_BK64_GM4_nW4_nS2_EU0_kP1_mfma32, args=task_args)]
    config_names += ['BM128_BN128_BK64_GM4_nW4_nS2_EU0_kP1_mfma32']
    results += [thread_pool.apply_async(try_compile_config_BM128_BN128_BK64_GM4_nW4_nS2_EU0_kP2_mfma32, args=task_args)]
    config_names += ['BM128_BN128_BK64_GM4_nW4_nS2_EU0_kP2_mfma32']
    results += [thread_pool.apply_async(try_compile_config_BM128_BN128_BK64_GM8_nW4_nS2_EU0_kP1_mfma16, args=task_args)]
    config_names += ['BM128_BN128_BK64_GM8_nW4_nS2_EU0_kP1_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM128_BN128_BK64_GM8_nW4_nS2_EU0_kP2_mfma16, args=task_args)]
    config_names += ['BM128_BN128_BK64_GM8_nW4_nS2_EU0_kP2_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM128_BN128_BK64_GM8_nW4_nS2_EU0_kP1_mfma32, args=task_args)]
    config_names += ['BM128_BN128_BK64_GM8_nW4_nS2_EU0_kP1_mfma32']
    results += [thread_pool.apply_async(try_compile_config_BM128_BN128_BK64_GM8_nW4_nS2_EU0_kP2_mfma32, args=task_args)]
    config_names += ['BM128_BN128_BK64_GM8_nW4_nS2_EU0_kP2_mfma32']
    results += [thread_pool.apply_async(try_compile_config_BM128_BN128_BK64_GM16_nW4_nS2_EU0_kP1_mfma16, args=task_args)]
    config_names += ['BM128_BN128_BK64_GM16_nW4_nS2_EU0_kP1_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM128_BN128_BK64_GM16_nW4_nS2_EU0_kP2_mfma16, args=task_args)]
    config_names += ['BM128_BN128_BK64_GM16_nW4_nS2_EU0_kP2_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM128_BN128_BK64_GM16_nW4_nS2_EU0_kP1_mfma32, args=task_args)]
    config_names += ['BM128_BN128_BK64_GM16_nW4_nS2_EU0_kP1_mfma32']
    results += [thread_pool.apply_async(try_compile_config_BM128_BN128_BK64_GM16_nW4_nS2_EU0_kP2_mfma32, args=task_args)]
    config_names += ['BM128_BN128_BK64_GM16_nW4_nS2_EU0_kP2_mfma32']
    results += [thread_pool.apply_async(try_compile_config_BM128_BN128_BK64_GM32_nW4_nS2_EU0_kP1_mfma16, args=task_args)]
    config_names += ['BM128_BN128_BK64_GM32_nW4_nS2_EU0_kP1_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM128_BN128_BK64_GM32_nW4_nS2_EU0_kP2_mfma16, args=task_args)]
    config_names += ['BM128_BN128_BK64_GM32_nW4_nS2_EU0_kP2_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM128_BN128_BK64_GM32_nW4_nS2_EU0_kP1_mfma32, args=task_args)]
    config_names += ['BM128_BN128_BK64_GM32_nW4_nS2_EU0_kP1_mfma32']
    results += [thread_pool.apply_async(try_compile_config_BM128_BN128_BK64_GM32_nW4_nS2_EU0_kP2_mfma32, args=task_args)]
    config_names += ['BM128_BN128_BK64_GM32_nW4_nS2_EU0_kP2_mfma32']
    results += [thread_pool.apply_async(try_compile_config_BM128_BN128_BK64_GM1_nW8_nS2_EU0_kP1_mfma16, args=task_args)]
    config_names += ['BM128_BN128_BK64_GM1_nW8_nS2_EU0_kP1_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM128_BN128_BK64_GM1_nW8_nS2_EU0_kP2_mfma16, args=task_args)]
    config_names += ['BM128_BN128_BK64_GM1_nW8_nS2_EU0_kP2_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM128_BN128_BK64_GM1_nW8_nS2_EU0_kP1_mfma32, args=task_args)]
    config_names += ['BM128_BN128_BK64_GM1_nW8_nS2_EU0_kP1_mfma32']
    results += [thread_pool.apply_async(try_compile_config_BM128_BN128_BK64_GM1_nW8_nS2_EU0_kP2_mfma32, args=task_args)]
    config_names += ['BM128_BN128_BK64_GM1_nW8_nS2_EU0_kP2_mfma32']
    results += [thread_pool.apply_async(try_compile_config_BM128_BN128_BK64_GM4_nW8_nS2_EU0_kP1_mfma16, args=task_args)]
    config_names += ['BM128_BN128_BK64_GM4_nW8_nS2_EU0_kP1_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM128_BN128_BK64_GM4_nW8_nS2_EU0_kP2_mfma16, args=task_args)]
    config_names += ['BM128_BN128_BK64_GM4_nW8_nS2_EU0_kP2_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM128_BN128_BK64_GM4_nW8_nS2_EU0_kP1_mfma32, args=task_args)]
    config_names += ['BM128_BN128_BK64_GM4_nW8_nS2_EU0_kP1_mfma32']
    results += [thread_pool.apply_async(try_compile_config_BM128_BN128_BK64_GM4_nW8_nS2_EU0_kP2_mfma32, args=task_args)]
    config_names += ['BM128_BN128_BK64_GM4_nW8_nS2_EU0_kP2_mfma32']
    results += [thread_pool.apply_async(try_compile_config_BM128_BN128_BK64_GM8_nW8_nS2_EU0_kP1_mfma16, args=task_args)]
    config_names += ['BM128_BN128_BK64_GM8_nW8_nS2_EU0_kP1_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM128_BN128_BK64_GM8_nW8_nS2_EU0_kP2_mfma16, args=task_args)]
    config_names += ['BM128_BN128_BK64_GM8_nW8_nS2_EU0_kP2_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM128_BN128_BK64_GM8_nW8_nS2_EU0_kP1_mfma32, args=task_args)]
    config_names += ['BM128_BN128_BK64_GM8_nW8_nS2_EU0_kP1_mfma32']
    results += [thread_pool.apply_async(try_compile_config_BM128_BN128_BK64_GM8_nW8_nS2_EU0_kP2_mfma32, args=task_args)]
    config_names += ['BM128_BN128_BK64_GM8_nW8_nS2_EU0_kP2_mfma32']
    results += [thread_pool.apply_async(try_compile_config_BM128_BN128_BK64_GM16_nW8_nS2_EU0_kP1_mfma16, args=task_args)]
    config_names += ['BM128_BN128_BK64_GM16_nW8_nS2_EU0_kP1_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM128_BN128_BK64_GM16_nW8_nS2_EU0_kP2_mfma16, args=task_args)]
    config_names += ['BM128_BN128_BK64_GM16_nW8_nS2_EU0_kP2_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM128_BN128_BK64_GM16_nW8_nS2_EU0_kP1_mfma32, args=task_args)]
    config_names += ['BM128_BN128_BK64_GM16_nW8_nS2_EU0_kP1_mfma32']
    results += [thread_pool.apply_async(try_compile_config_BM128_BN128_BK64_GM16_nW8_nS2_EU0_kP2_mfma32, args=task_args)]
    config_names += ['BM128_BN128_BK64_GM16_nW8_nS2_EU0_kP2_mfma32']
    results += [thread_pool.apply_async(try_compile_config_BM128_BN128_BK64_GM32_nW8_nS2_EU0_kP1_mfma16, args=task_args)]
    config_names += ['BM128_BN128_BK64_GM32_nW8_nS2_EU0_kP1_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM128_BN128_BK64_GM32_nW8_nS2_EU0_kP2_mfma16, args=task_args)]
    config_names += ['BM128_BN128_BK64_GM32_nW8_nS2_EU0_kP2_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM128_BN128_BK64_GM32_nW8_nS2_EU0_kP1_mfma32, args=task_args)]
    config_names += ['BM128_BN128_BK64_GM32_nW8_nS2_EU0_kP1_mfma32']
    results += [thread_pool.apply_async(try_compile_config_BM128_BN128_BK64_GM32_nW8_nS2_EU0_kP2_mfma32, args=task_args)]
    config_names += ['BM128_BN128_BK64_GM32_nW8_nS2_EU0_kP2_mfma32']
    results += [thread_pool.apply_async(try_compile_config_BM128_BN128_BK128_GM1_nW4_nS2_EU0_kP1_mfma16, args=task_args)]
    config_names += ['BM128_BN128_BK128_GM1_nW4_nS2_EU0_kP1_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM128_BN128_BK128_GM1_nW4_nS2_EU0_kP2_mfma16, args=task_args)]
    config_names += ['BM128_BN128_BK128_GM1_nW4_nS2_EU0_kP2_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM128_BN128_BK128_GM1_nW4_nS2_EU0_kP1_mfma32, args=task_args)]
    config_names += ['BM128_BN128_BK128_GM1_nW4_nS2_EU0_kP1_mfma32']
    results += [thread_pool.apply_async(try_compile_config_BM128_BN128_BK128_GM4_nW4_nS2_EU0_kP1_mfma16, args=task_args)]
    config_names += ['BM128_BN128_BK128_GM4_nW4_nS2_EU0_kP1_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM128_BN128_BK128_GM4_nW4_nS2_EU0_kP2_mfma16, args=task_args)]
    config_names += ['BM128_BN128_BK128_GM4_nW4_nS2_EU0_kP2_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM128_BN128_BK128_GM4_nW4_nS2_EU0_kP1_mfma32, args=task_args)]
    config_names += ['BM128_BN128_BK128_GM4_nW4_nS2_EU0_kP1_mfma32']
    results += [thread_pool.apply_async(try_compile_config_BM128_BN128_BK128_GM8_nW4_nS2_EU0_kP1_mfma16, args=task_args)]
    config_names += ['BM128_BN128_BK128_GM8_nW4_nS2_EU0_kP1_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM128_BN128_BK128_GM8_nW4_nS2_EU0_kP2_mfma16, args=task_args)]
    config_names += ['BM128_BN128_BK128_GM8_nW4_nS2_EU0_kP2_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM128_BN128_BK128_GM8_nW4_nS2_EU0_kP1_mfma32, args=task_args)]
    config_names += ['BM128_BN128_BK128_GM8_nW4_nS2_EU0_kP1_mfma32']
    results += [thread_pool.apply_async(try_compile_config_BM128_BN128_BK128_GM16_nW4_nS2_EU0_kP1_mfma16, args=task_args)]
    config_names += ['BM128_BN128_BK128_GM16_nW4_nS2_EU0_kP1_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM128_BN128_BK128_GM16_nW4_nS2_EU0_kP2_mfma16, args=task_args)]
    config_names += ['BM128_BN128_BK128_GM16_nW4_nS2_EU0_kP2_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM128_BN128_BK128_GM16_nW4_nS2_EU0_kP1_mfma32, args=task_args)]
    config_names += ['BM128_BN128_BK128_GM16_nW4_nS2_EU0_kP1_mfma32']
    results += [thread_pool.apply_async(try_compile_config_BM128_BN128_BK128_GM32_nW4_nS2_EU0_kP1_mfma16, args=task_args)]
    config_names += ['BM128_BN128_BK128_GM32_nW4_nS2_EU0_kP1_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM128_BN128_BK128_GM32_nW4_nS2_EU0_kP2_mfma16, args=task_args)]
    config_names += ['BM128_BN128_BK128_GM32_nW4_nS2_EU0_kP2_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM128_BN128_BK128_GM32_nW4_nS2_EU0_kP1_mfma32, args=task_args)]
    config_names += ['BM128_BN128_BK128_GM32_nW4_nS2_EU0_kP1_mfma32']
    results += [thread_pool.apply_async(try_compile_config_BM128_BN128_BK128_GM1_nW8_nS2_EU0_kP1_mfma16, args=task_args)]
    config_names += ['BM128_BN128_BK128_GM1_nW8_nS2_EU0_kP1_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM128_BN128_BK128_GM1_nW8_nS2_EU0_kP2_mfma16, args=task_args)]
    config_names += ['BM128_BN128_BK128_GM1_nW8_nS2_EU0_kP2_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM128_BN128_BK128_GM1_nW8_nS2_EU0_kP1_mfma32, args=task_args)]
    config_names += ['BM128_BN128_BK128_GM1_nW8_nS2_EU0_kP1_mfma32']
    results += [thread_pool.apply_async(try_compile_config_BM128_BN128_BK128_GM4_nW8_nS2_EU0_kP1_mfma16, args=task_args)]
    config_names += ['BM128_BN128_BK128_GM4_nW8_nS2_EU0_kP1_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM128_BN128_BK128_GM4_nW8_nS2_EU0_kP2_mfma16, args=task_args)]
    config_names += ['BM128_BN128_BK128_GM4_nW8_nS2_EU0_kP2_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM128_BN128_BK128_GM4_nW8_nS2_EU0_kP1_mfma32, args=task_args)]
    config_names += ['BM128_BN128_BK128_GM4_nW8_nS2_EU0_kP1_mfma32']
    results += [thread_pool.apply_async(try_compile_config_BM128_BN128_BK128_GM8_nW8_nS2_EU0_kP1_mfma16, args=task_args)]
    config_names += ['BM128_BN128_BK128_GM8_nW8_nS2_EU0_kP1_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM128_BN128_BK128_GM8_nW8_nS2_EU0_kP2_mfma16, args=task_args)]
    config_names += ['BM128_BN128_BK128_GM8_nW8_nS2_EU0_kP2_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM128_BN128_BK128_GM8_nW8_nS2_EU0_kP1_mfma32, args=task_args)]
    config_names += ['BM128_BN128_BK128_GM8_nW8_nS2_EU0_kP1_mfma32']
    results += [thread_pool.apply_async(try_compile_config_BM128_BN128_BK128_GM16_nW8_nS2_EU0_kP1_mfma16, args=task_args)]
    config_names += ['BM128_BN128_BK128_GM16_nW8_nS2_EU0_kP1_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM128_BN128_BK128_GM16_nW8_nS2_EU0_kP2_mfma16, args=task_args)]
    config_names += ['BM128_BN128_BK128_GM16_nW8_nS2_EU0_kP2_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM128_BN128_BK128_GM16_nW8_nS2_EU0_kP1_mfma32, args=task_args)]
    config_names += ['BM128_BN128_BK128_GM16_nW8_nS2_EU0_kP1_mfma32']
    results += [thread_pool.apply_async(try_compile_config_BM128_BN128_BK128_GM32_nW8_nS2_EU0_kP1_mfma16, args=task_args)]
    config_names += ['BM128_BN128_BK128_GM32_nW8_nS2_EU0_kP1_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM128_BN128_BK128_GM32_nW8_nS2_EU0_kP2_mfma16, args=task_args)]
    config_names += ['BM128_BN128_BK128_GM32_nW8_nS2_EU0_kP2_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM128_BN128_BK128_GM32_nW8_nS2_EU0_kP1_mfma32, args=task_args)]
    config_names += ['BM128_BN128_BK128_GM32_nW8_nS2_EU0_kP1_mfma32']
    results += [thread_pool.apply_async(try_compile_config_BM128_BN256_BK64_GM1_nW4_nS2_EU0_kP1_mfma16, args=task_args)]
    config_names += ['BM128_BN256_BK64_GM1_nW4_nS2_EU0_kP1_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM128_BN256_BK64_GM1_nW4_nS2_EU0_kP2_mfma16, args=task_args)]
    config_names += ['BM128_BN256_BK64_GM1_nW4_nS2_EU0_kP2_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM128_BN256_BK64_GM1_nW4_nS2_EU0_kP1_mfma32, args=task_args)]
    config_names += ['BM128_BN256_BK64_GM1_nW4_nS2_EU0_kP1_mfma32']
    results += [thread_pool.apply_async(try_compile_config_BM128_BN256_BK64_GM1_nW4_nS2_EU0_kP2_mfma32, args=task_args)]
    config_names += ['BM128_BN256_BK64_GM1_nW4_nS2_EU0_kP2_mfma32']
    results += [thread_pool.apply_async(try_compile_config_BM128_BN256_BK64_GM4_nW4_nS2_EU0_kP1_mfma16, args=task_args)]
    config_names += ['BM128_BN256_BK64_GM4_nW4_nS2_EU0_kP1_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM128_BN256_BK64_GM4_nW4_nS2_EU0_kP2_mfma16, args=task_args)]
    config_names += ['BM128_BN256_BK64_GM4_nW4_nS2_EU0_kP2_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM128_BN256_BK64_GM4_nW4_nS2_EU0_kP1_mfma32, args=task_args)]
    config_names += ['BM128_BN256_BK64_GM4_nW4_nS2_EU0_kP1_mfma32']
    results += [thread_pool.apply_async(try_compile_config_BM128_BN256_BK64_GM4_nW4_nS2_EU0_kP2_mfma32, args=task_args)]
    config_names += ['BM128_BN256_BK64_GM4_nW4_nS2_EU0_kP2_mfma32']
    results += [thread_pool.apply_async(try_compile_config_BM128_BN256_BK64_GM8_nW4_nS2_EU0_kP1_mfma16, args=task_args)]
    config_names += ['BM128_BN256_BK64_GM8_nW4_nS2_EU0_kP1_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM128_BN256_BK64_GM8_nW4_nS2_EU0_kP2_mfma16, args=task_args)]
    config_names += ['BM128_BN256_BK64_GM8_nW4_nS2_EU0_kP2_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM128_BN256_BK64_GM8_nW4_nS2_EU0_kP1_mfma32, args=task_args)]
    config_names += ['BM128_BN256_BK64_GM8_nW4_nS2_EU0_kP1_mfma32']
    results += [thread_pool.apply_async(try_compile_config_BM128_BN256_BK64_GM8_nW4_nS2_EU0_kP2_mfma32, args=task_args)]
    config_names += ['BM128_BN256_BK64_GM8_nW4_nS2_EU0_kP2_mfma32']
    results += [thread_pool.apply_async(try_compile_config_BM128_BN256_BK64_GM16_nW4_nS2_EU0_kP1_mfma16, args=task_args)]
    config_names += ['BM128_BN256_BK64_GM16_nW4_nS2_EU0_kP1_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM128_BN256_BK64_GM16_nW4_nS2_EU0_kP2_mfma16, args=task_args)]
    config_names += ['BM128_BN256_BK64_GM16_nW4_nS2_EU0_kP2_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM128_BN256_BK64_GM16_nW4_nS2_EU0_kP1_mfma32, args=task_args)]
    config_names += ['BM128_BN256_BK64_GM16_nW4_nS2_EU0_kP1_mfma32']
    results += [thread_pool.apply_async(try_compile_config_BM128_BN256_BK64_GM16_nW4_nS2_EU0_kP2_mfma32, args=task_args)]
    config_names += ['BM128_BN256_BK64_GM16_nW4_nS2_EU0_kP2_mfma32']
    results += [thread_pool.apply_async(try_compile_config_BM128_BN256_BK64_GM32_nW4_nS2_EU0_kP1_mfma16, args=task_args)]
    config_names += ['BM128_BN256_BK64_GM32_nW4_nS2_EU0_kP1_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM128_BN256_BK64_GM32_nW4_nS2_EU0_kP2_mfma16, args=task_args)]
    config_names += ['BM128_BN256_BK64_GM32_nW4_nS2_EU0_kP2_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM128_BN256_BK64_GM32_nW4_nS2_EU0_kP1_mfma32, args=task_args)]
    config_names += ['BM128_BN256_BK64_GM32_nW4_nS2_EU0_kP1_mfma32']
    results += [thread_pool.apply_async(try_compile_config_BM128_BN256_BK64_GM32_nW4_nS2_EU0_kP2_mfma32, args=task_args)]
    config_names += ['BM128_BN256_BK64_GM32_nW4_nS2_EU0_kP2_mfma32']
    results += [thread_pool.apply_async(try_compile_config_BM128_BN256_BK64_GM1_nW8_nS2_EU0_kP1_mfma16, args=task_args)]
    config_names += ['BM128_BN256_BK64_GM1_nW8_nS2_EU0_kP1_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM128_BN256_BK64_GM1_nW8_nS2_EU0_kP2_mfma16, args=task_args)]
    config_names += ['BM128_BN256_BK64_GM1_nW8_nS2_EU0_kP2_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM128_BN256_BK64_GM1_nW8_nS2_EU0_kP1_mfma32, args=task_args)]
    config_names += ['BM128_BN256_BK64_GM1_nW8_nS2_EU0_kP1_mfma32']
    results += [thread_pool.apply_async(try_compile_config_BM128_BN256_BK64_GM1_nW8_nS2_EU0_kP2_mfma32, args=task_args)]
    config_names += ['BM128_BN256_BK64_GM1_nW8_nS2_EU0_kP2_mfma32']
    results += [thread_pool.apply_async(try_compile_config_BM128_BN256_BK64_GM4_nW8_nS2_EU0_kP1_mfma16, args=task_args)]
    config_names += ['BM128_BN256_BK64_GM4_nW8_nS2_EU0_kP1_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM128_BN256_BK64_GM4_nW8_nS2_EU0_kP2_mfma16, args=task_args)]
    config_names += ['BM128_BN256_BK64_GM4_nW8_nS2_EU0_kP2_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM128_BN256_BK64_GM4_nW8_nS2_EU0_kP1_mfma32, args=task_args)]
    config_names += ['BM128_BN256_BK64_GM4_nW8_nS2_EU0_kP1_mfma32']
    results += [thread_pool.apply_async(try_compile_config_BM128_BN256_BK64_GM4_nW8_nS2_EU0_kP2_mfma32, args=task_args)]
    config_names += ['BM128_BN256_BK64_GM4_nW8_nS2_EU0_kP2_mfma32']
    results += [thread_pool.apply_async(try_compile_config_BM128_BN256_BK64_GM8_nW8_nS2_EU0_kP1_mfma16, args=task_args)]
    config_names += ['BM128_BN256_BK64_GM8_nW8_nS2_EU0_kP1_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM128_BN256_BK64_GM8_nW8_nS2_EU0_kP2_mfma16, args=task_args)]
    config_names += ['BM128_BN256_BK64_GM8_nW8_nS2_EU0_kP2_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM128_BN256_BK64_GM8_nW8_nS2_EU0_kP1_mfma32, args=task_args)]
    config_names += ['BM128_BN256_BK64_GM8_nW8_nS2_EU0_kP1_mfma32']
    results += [thread_pool.apply_async(try_compile_config_BM128_BN256_BK64_GM8_nW8_nS2_EU0_kP2_mfma32, args=task_args)]
    config_names += ['BM128_BN256_BK64_GM8_nW8_nS2_EU0_kP2_mfma32']
    results += [thread_pool.apply_async(try_compile_config_BM128_BN256_BK64_GM16_nW8_nS2_EU0_kP1_mfma16, args=task_args)]
    config_names += ['BM128_BN256_BK64_GM16_nW8_nS2_EU0_kP1_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM128_BN256_BK64_GM16_nW8_nS2_EU0_kP2_mfma16, args=task_args)]
    config_names += ['BM128_BN256_BK64_GM16_nW8_nS2_EU0_kP2_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM128_BN256_BK64_GM16_nW8_nS2_EU0_kP1_mfma32, args=task_args)]
    config_names += ['BM128_BN256_BK64_GM16_nW8_nS2_EU0_kP1_mfma32']
    results += [thread_pool.apply_async(try_compile_config_BM128_BN256_BK64_GM16_nW8_nS2_EU0_kP2_mfma32, args=task_args)]
    config_names += ['BM128_BN256_BK64_GM16_nW8_nS2_EU0_kP2_mfma32']
    results += [thread_pool.apply_async(try_compile_config_BM128_BN256_BK64_GM32_nW8_nS2_EU0_kP1_mfma16, args=task_args)]
    config_names += ['BM128_BN256_BK64_GM32_nW8_nS2_EU0_kP1_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM128_BN256_BK64_GM32_nW8_nS2_EU0_kP2_mfma16, args=task_args)]
    config_names += ['BM128_BN256_BK64_GM32_nW8_nS2_EU0_kP2_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM128_BN256_BK64_GM32_nW8_nS2_EU0_kP1_mfma32, args=task_args)]
    config_names += ['BM128_BN256_BK64_GM32_nW8_nS2_EU0_kP1_mfma32']
    results += [thread_pool.apply_async(try_compile_config_BM128_BN256_BK64_GM32_nW8_nS2_EU0_kP2_mfma32, args=task_args)]
    config_names += ['BM128_BN256_BK64_GM32_nW8_nS2_EU0_kP2_mfma32']
    results += [thread_pool.apply_async(try_compile_config_BM256_BN64_BK64_GM1_nW4_nS2_EU0_kP1_mfma16, args=task_args)]
    config_names += ['BM256_BN64_BK64_GM1_nW4_nS2_EU0_kP1_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM256_BN64_BK64_GM1_nW4_nS2_EU0_kP2_mfma16, args=task_args)]
    config_names += ['BM256_BN64_BK64_GM1_nW4_nS2_EU0_kP2_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM256_BN64_BK64_GM1_nW4_nS2_EU0_kP1_mfma32, args=task_args)]
    config_names += ['BM256_BN64_BK64_GM1_nW4_nS2_EU0_kP1_mfma32']
    results += [thread_pool.apply_async(try_compile_config_BM256_BN64_BK64_GM1_nW4_nS2_EU0_kP2_mfma32, args=task_args)]
    config_names += ['BM256_BN64_BK64_GM1_nW4_nS2_EU0_kP2_mfma32']
    results += [thread_pool.apply_async(try_compile_config_BM256_BN64_BK64_GM4_nW4_nS2_EU0_kP1_mfma16, args=task_args)]
    config_names += ['BM256_BN64_BK64_GM4_nW4_nS2_EU0_kP1_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM256_BN64_BK64_GM4_nW4_nS2_EU0_kP2_mfma16, args=task_args)]
    config_names += ['BM256_BN64_BK64_GM4_nW4_nS2_EU0_kP2_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM256_BN64_BK64_GM4_nW4_nS2_EU0_kP1_mfma32, args=task_args)]
    config_names += ['BM256_BN64_BK64_GM4_nW4_nS2_EU0_kP1_mfma32']
    results += [thread_pool.apply_async(try_compile_config_BM256_BN64_BK64_GM4_nW4_nS2_EU0_kP2_mfma32, args=task_args)]
    config_names += ['BM256_BN64_BK64_GM4_nW4_nS2_EU0_kP2_mfma32']
    results += [thread_pool.apply_async(try_compile_config_BM256_BN64_BK64_GM8_nW4_nS2_EU0_kP1_mfma16, args=task_args)]
    config_names += ['BM256_BN64_BK64_GM8_nW4_nS2_EU0_kP1_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM256_BN64_BK64_GM8_nW4_nS2_EU0_kP2_mfma16, args=task_args)]
    config_names += ['BM256_BN64_BK64_GM8_nW4_nS2_EU0_kP2_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM256_BN64_BK64_GM8_nW4_nS2_EU0_kP1_mfma32, args=task_args)]
    config_names += ['BM256_BN64_BK64_GM8_nW4_nS2_EU0_kP1_mfma32']
    results += [thread_pool.apply_async(try_compile_config_BM256_BN64_BK64_GM8_nW4_nS2_EU0_kP2_mfma32, args=task_args)]
    config_names += ['BM256_BN64_BK64_GM8_nW4_nS2_EU0_kP2_mfma32']
    results += [thread_pool.apply_async(try_compile_config_BM256_BN64_BK64_GM16_nW4_nS2_EU0_kP1_mfma16, args=task_args)]
    config_names += ['BM256_BN64_BK64_GM16_nW4_nS2_EU0_kP1_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM256_BN64_BK64_GM16_nW4_nS2_EU0_kP2_mfma16, args=task_args)]
    config_names += ['BM256_BN64_BK64_GM16_nW4_nS2_EU0_kP2_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM256_BN64_BK64_GM16_nW4_nS2_EU0_kP1_mfma32, args=task_args)]
    config_names += ['BM256_BN64_BK64_GM16_nW4_nS2_EU0_kP1_mfma32']
    results += [thread_pool.apply_async(try_compile_config_BM256_BN64_BK64_GM16_nW4_nS2_EU0_kP2_mfma32, args=task_args)]
    config_names += ['BM256_BN64_BK64_GM16_nW4_nS2_EU0_kP2_mfma32']
    results += [thread_pool.apply_async(try_compile_config_BM256_BN64_BK64_GM1_nW8_nS2_EU0_kP1_mfma16, args=task_args)]
    config_names += ['BM256_BN64_BK64_GM1_nW8_nS2_EU0_kP1_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM256_BN64_BK64_GM1_nW8_nS2_EU0_kP2_mfma16, args=task_args)]
    config_names += ['BM256_BN64_BK64_GM1_nW8_nS2_EU0_kP2_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM256_BN64_BK64_GM1_nW8_nS2_EU0_kP1_mfma32, args=task_args)]
    config_names += ['BM256_BN64_BK64_GM1_nW8_nS2_EU0_kP1_mfma32']
    results += [thread_pool.apply_async(try_compile_config_BM256_BN64_BK64_GM1_nW8_nS2_EU0_kP2_mfma32, args=task_args)]
    config_names += ['BM256_BN64_BK64_GM1_nW8_nS2_EU0_kP2_mfma32']
    results += [thread_pool.apply_async(try_compile_config_BM256_BN64_BK64_GM4_nW8_nS2_EU0_kP1_mfma16, args=task_args)]
    config_names += ['BM256_BN64_BK64_GM4_nW8_nS2_EU0_kP1_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM256_BN64_BK64_GM4_nW8_nS2_EU0_kP2_mfma16, args=task_args)]
    config_names += ['BM256_BN64_BK64_GM4_nW8_nS2_EU0_kP2_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM256_BN64_BK64_GM4_nW8_nS2_EU0_kP1_mfma32, args=task_args)]
    config_names += ['BM256_BN64_BK64_GM4_nW8_nS2_EU0_kP1_mfma32']
    results += [thread_pool.apply_async(try_compile_config_BM256_BN64_BK64_GM4_nW8_nS2_EU0_kP2_mfma32, args=task_args)]
    config_names += ['BM256_BN64_BK64_GM4_nW8_nS2_EU0_kP2_mfma32']
    results += [thread_pool.apply_async(try_compile_config_BM256_BN64_BK64_GM8_nW8_nS2_EU0_kP1_mfma16, args=task_args)]
    config_names += ['BM256_BN64_BK64_GM8_nW8_nS2_EU0_kP1_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM256_BN64_BK64_GM8_nW8_nS2_EU0_kP2_mfma16, args=task_args)]
    config_names += ['BM256_BN64_BK64_GM8_nW8_nS2_EU0_kP2_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM256_BN64_BK64_GM8_nW8_nS2_EU0_kP1_mfma32, args=task_args)]
    config_names += ['BM256_BN64_BK64_GM8_nW8_nS2_EU0_kP1_mfma32']
    results += [thread_pool.apply_async(try_compile_config_BM256_BN64_BK64_GM8_nW8_nS2_EU0_kP2_mfma32, args=task_args)]
    config_names += ['BM256_BN64_BK64_GM8_nW8_nS2_EU0_kP2_mfma32']
    results += [thread_pool.apply_async(try_compile_config_BM256_BN64_BK64_GM16_nW8_nS2_EU0_kP1_mfma16, args=task_args)]
    config_names += ['BM256_BN64_BK64_GM16_nW8_nS2_EU0_kP1_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM256_BN64_BK64_GM16_nW8_nS2_EU0_kP2_mfma16, args=task_args)]
    config_names += ['BM256_BN64_BK64_GM16_nW8_nS2_EU0_kP2_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM256_BN64_BK64_GM16_nW8_nS2_EU0_kP1_mfma32, args=task_args)]
    config_names += ['BM256_BN64_BK64_GM16_nW8_nS2_EU0_kP1_mfma32']
    results += [thread_pool.apply_async(try_compile_config_BM256_BN64_BK64_GM16_nW8_nS2_EU0_kP2_mfma32, args=task_args)]
    config_names += ['BM256_BN64_BK64_GM16_nW8_nS2_EU0_kP2_mfma32']
    results += [thread_pool.apply_async(try_compile_config_BM256_BN128_BK64_GM1_nW4_nS2_EU0_kP1_mfma16, args=task_args)]
    config_names += ['BM256_BN128_BK64_GM1_nW4_nS2_EU0_kP1_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM256_BN128_BK64_GM1_nW4_nS2_EU0_kP2_mfma16, args=task_args)]
    config_names += ['BM256_BN128_BK64_GM1_nW4_nS2_EU0_kP2_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM256_BN128_BK64_GM1_nW4_nS2_EU0_kP1_mfma32, args=task_args)]
    config_names += ['BM256_BN128_BK64_GM1_nW4_nS2_EU0_kP1_mfma32']
    results += [thread_pool.apply_async(try_compile_config_BM256_BN128_BK64_GM1_nW4_nS2_EU0_kP2_mfma32, args=task_args)]
    config_names += ['BM256_BN128_BK64_GM1_nW4_nS2_EU0_kP2_mfma32']
    results += [thread_pool.apply_async(try_compile_config_BM256_BN128_BK64_GM4_nW4_nS2_EU0_kP1_mfma16, args=task_args)]
    config_names += ['BM256_BN128_BK64_GM4_nW4_nS2_EU0_kP1_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM256_BN128_BK64_GM4_nW4_nS2_EU0_kP2_mfma16, args=task_args)]
    config_names += ['BM256_BN128_BK64_GM4_nW4_nS2_EU0_kP2_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM256_BN128_BK64_GM4_nW4_nS2_EU0_kP1_mfma32, args=task_args)]
    config_names += ['BM256_BN128_BK64_GM4_nW4_nS2_EU0_kP1_mfma32']
    results += [thread_pool.apply_async(try_compile_config_BM256_BN128_BK64_GM4_nW4_nS2_EU0_kP2_mfma32, args=task_args)]
    config_names += ['BM256_BN128_BK64_GM4_nW4_nS2_EU0_kP2_mfma32']
    results += [thread_pool.apply_async(try_compile_config_BM256_BN128_BK64_GM8_nW4_nS2_EU0_kP1_mfma16, args=task_args)]
    config_names += ['BM256_BN128_BK64_GM8_nW4_nS2_EU0_kP1_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM256_BN128_BK64_GM8_nW4_nS2_EU0_kP2_mfma16, args=task_args)]
    config_names += ['BM256_BN128_BK64_GM8_nW4_nS2_EU0_kP2_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM256_BN128_BK64_GM8_nW4_nS2_EU0_kP1_mfma32, args=task_args)]
    config_names += ['BM256_BN128_BK64_GM8_nW4_nS2_EU0_kP1_mfma32']
    results += [thread_pool.apply_async(try_compile_config_BM256_BN128_BK64_GM8_nW4_nS2_EU0_kP2_mfma32, args=task_args)]
    config_names += ['BM256_BN128_BK64_GM8_nW4_nS2_EU0_kP2_mfma32']
    results += [thread_pool.apply_async(try_compile_config_BM256_BN128_BK64_GM16_nW4_nS2_EU0_kP1_mfma16, args=task_args)]
    config_names += ['BM256_BN128_BK64_GM16_nW4_nS2_EU0_kP1_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM256_BN128_BK64_GM16_nW4_nS2_EU0_kP2_mfma16, args=task_args)]
    config_names += ['BM256_BN128_BK64_GM16_nW4_nS2_EU0_kP2_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM256_BN128_BK64_GM16_nW4_nS2_EU0_kP1_mfma32, args=task_args)]
    config_names += ['BM256_BN128_BK64_GM16_nW4_nS2_EU0_kP1_mfma32']
    results += [thread_pool.apply_async(try_compile_config_BM256_BN128_BK64_GM16_nW4_nS2_EU0_kP2_mfma32, args=task_args)]
    config_names += ['BM256_BN128_BK64_GM16_nW4_nS2_EU0_kP2_mfma32']
    results += [thread_pool.apply_async(try_compile_config_BM256_BN128_BK64_GM1_nW8_nS2_EU0_kP1_mfma16, args=task_args)]
    config_names += ['BM256_BN128_BK64_GM1_nW8_nS2_EU0_kP1_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM256_BN128_BK64_GM1_nW8_nS2_EU0_kP2_mfma16, args=task_args)]
    config_names += ['BM256_BN128_BK64_GM1_nW8_nS2_EU0_kP2_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM256_BN128_BK64_GM1_nW8_nS2_EU0_kP1_mfma32, args=task_args)]
    config_names += ['BM256_BN128_BK64_GM1_nW8_nS2_EU0_kP1_mfma32']
    results += [thread_pool.apply_async(try_compile_config_BM256_BN128_BK64_GM1_nW8_nS2_EU0_kP2_mfma32, args=task_args)]
    config_names += ['BM256_BN128_BK64_GM1_nW8_nS2_EU0_kP2_mfma32']
    results += [thread_pool.apply_async(try_compile_config_BM256_BN128_BK64_GM4_nW8_nS2_EU0_kP1_mfma16, args=task_args)]
    config_names += ['BM256_BN128_BK64_GM4_nW8_nS2_EU0_kP1_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM256_BN128_BK64_GM4_nW8_nS2_EU0_kP2_mfma16, args=task_args)]
    config_names += ['BM256_BN128_BK64_GM4_nW8_nS2_EU0_kP2_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM256_BN128_BK64_GM4_nW8_nS2_EU0_kP1_mfma32, args=task_args)]
    config_names += ['BM256_BN128_BK64_GM4_nW8_nS2_EU0_kP1_mfma32']
    results += [thread_pool.apply_async(try_compile_config_BM256_BN128_BK64_GM4_nW8_nS2_EU0_kP2_mfma32, args=task_args)]
    config_names += ['BM256_BN128_BK64_GM4_nW8_nS2_EU0_kP2_mfma32']
    results += [thread_pool.apply_async(try_compile_config_BM256_BN128_BK64_GM8_nW8_nS2_EU0_kP1_mfma16, args=task_args)]
    config_names += ['BM256_BN128_BK64_GM8_nW8_nS2_EU0_kP1_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM256_BN128_BK64_GM8_nW8_nS2_EU0_kP2_mfma16, args=task_args)]
    config_names += ['BM256_BN128_BK64_GM8_nW8_nS2_EU0_kP2_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM256_BN128_BK64_GM8_nW8_nS2_EU0_kP1_mfma32, args=task_args)]
    config_names += ['BM256_BN128_BK64_GM8_nW8_nS2_EU0_kP1_mfma32']
    results += [thread_pool.apply_async(try_compile_config_BM256_BN128_BK64_GM8_nW8_nS2_EU0_kP2_mfma32, args=task_args)]
    config_names += ['BM256_BN128_BK64_GM8_nW8_nS2_EU0_kP2_mfma32']
    results += [thread_pool.apply_async(try_compile_config_BM256_BN128_BK64_GM16_nW8_nS2_EU0_kP1_mfma16, args=task_args)]
    config_names += ['BM256_BN128_BK64_GM16_nW8_nS2_EU0_kP1_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM256_BN128_BK64_GM16_nW8_nS2_EU0_kP2_mfma16, args=task_args)]
    config_names += ['BM256_BN128_BK64_GM16_nW8_nS2_EU0_kP2_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM256_BN128_BK64_GM16_nW8_nS2_EU0_kP1_mfma32, args=task_args)]
    config_names += ['BM256_BN128_BK64_GM16_nW8_nS2_EU0_kP1_mfma32']
    results += [thread_pool.apply_async(try_compile_config_BM256_BN128_BK64_GM16_nW8_nS2_EU0_kP2_mfma32, args=task_args)]
    config_names += ['BM256_BN128_BK64_GM16_nW8_nS2_EU0_kP2_mfma32']
    results += [thread_pool.apply_async(try_compile_config_BM256_BN256_BK64_GM1_nW4_nS2_EU0_kP1_mfma16, args=task_args)]
    config_names += ['BM256_BN256_BK64_GM1_nW4_nS2_EU0_kP1_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM256_BN256_BK64_GM1_nW4_nS2_EU0_kP2_mfma16, args=task_args)]
    config_names += ['BM256_BN256_BK64_GM1_nW4_nS2_EU0_kP2_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM256_BN256_BK64_GM1_nW4_nS2_EU0_kP1_mfma32, args=task_args)]
    config_names += ['BM256_BN256_BK64_GM1_nW4_nS2_EU0_kP1_mfma32']
    results += [thread_pool.apply_async(try_compile_config_BM256_BN256_BK64_GM1_nW4_nS2_EU0_kP2_mfma32, args=task_args)]
    config_names += ['BM256_BN256_BK64_GM1_nW4_nS2_EU0_kP2_mfma32']
    results += [thread_pool.apply_async(try_compile_config_BM256_BN256_BK64_GM4_nW4_nS2_EU0_kP1_mfma16, args=task_args)]
    config_names += ['BM256_BN256_BK64_GM4_nW4_nS2_EU0_kP1_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM256_BN256_BK64_GM4_nW4_nS2_EU0_kP2_mfma16, args=task_args)]
    config_names += ['BM256_BN256_BK64_GM4_nW4_nS2_EU0_kP2_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM256_BN256_BK64_GM4_nW4_nS2_EU0_kP1_mfma32, args=task_args)]
    config_names += ['BM256_BN256_BK64_GM4_nW4_nS2_EU0_kP1_mfma32']
    results += [thread_pool.apply_async(try_compile_config_BM256_BN256_BK64_GM4_nW4_nS2_EU0_kP2_mfma32, args=task_args)]
    config_names += ['BM256_BN256_BK64_GM4_nW4_nS2_EU0_kP2_mfma32']
    results += [thread_pool.apply_async(try_compile_config_BM256_BN256_BK64_GM8_nW4_nS2_EU0_kP1_mfma16, args=task_args)]
    config_names += ['BM256_BN256_BK64_GM8_nW4_nS2_EU0_kP1_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM256_BN256_BK64_GM8_nW4_nS2_EU0_kP2_mfma16, args=task_args)]
    config_names += ['BM256_BN256_BK64_GM8_nW4_nS2_EU0_kP2_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM256_BN256_BK64_GM8_nW4_nS2_EU0_kP1_mfma32, args=task_args)]
    config_names += ['BM256_BN256_BK64_GM8_nW4_nS2_EU0_kP1_mfma32']
    results += [thread_pool.apply_async(try_compile_config_BM256_BN256_BK64_GM8_nW4_nS2_EU0_kP2_mfma32, args=task_args)]
    config_names += ['BM256_BN256_BK64_GM8_nW4_nS2_EU0_kP2_mfma32']
    results += [thread_pool.apply_async(try_compile_config_BM256_BN256_BK64_GM16_nW4_nS2_EU0_kP1_mfma16, args=task_args)]
    config_names += ['BM256_BN256_BK64_GM16_nW4_nS2_EU0_kP1_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM256_BN256_BK64_GM16_nW4_nS2_EU0_kP2_mfma16, args=task_args)]
    config_names += ['BM256_BN256_BK64_GM16_nW4_nS2_EU0_kP2_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM256_BN256_BK64_GM16_nW4_nS2_EU0_kP1_mfma32, args=task_args)]
    config_names += ['BM256_BN256_BK64_GM16_nW4_nS2_EU0_kP1_mfma32']
    results += [thread_pool.apply_async(try_compile_config_BM256_BN256_BK64_GM16_nW4_nS2_EU0_kP2_mfma32, args=task_args)]
    config_names += ['BM256_BN256_BK64_GM16_nW4_nS2_EU0_kP2_mfma32']
    results += [thread_pool.apply_async(try_compile_config_BM256_BN256_BK64_GM1_nW8_nS2_EU0_kP1_mfma16, args=task_args)]
    config_names += ['BM256_BN256_BK64_GM1_nW8_nS2_EU0_kP1_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM256_BN256_BK64_GM1_nW8_nS2_EU0_kP2_mfma16, args=task_args)]
    config_names += ['BM256_BN256_BK64_GM1_nW8_nS2_EU0_kP2_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM256_BN256_BK64_GM1_nW8_nS2_EU0_kP1_mfma32, args=task_args)]
    config_names += ['BM256_BN256_BK64_GM1_nW8_nS2_EU0_kP1_mfma32']
    results += [thread_pool.apply_async(try_compile_config_BM256_BN256_BK64_GM1_nW8_nS2_EU0_kP2_mfma32, args=task_args)]
    config_names += ['BM256_BN256_BK64_GM1_nW8_nS2_EU0_kP2_mfma32']
    results += [thread_pool.apply_async(try_compile_config_BM256_BN256_BK64_GM4_nW8_nS2_EU0_kP1_mfma16, args=task_args)]
    config_names += ['BM256_BN256_BK64_GM4_nW8_nS2_EU0_kP1_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM256_BN256_BK64_GM4_nW8_nS2_EU0_kP2_mfma16, args=task_args)]
    config_names += ['BM256_BN256_BK64_GM4_nW8_nS2_EU0_kP2_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM256_BN256_BK64_GM4_nW8_nS2_EU0_kP1_mfma32, args=task_args)]
    config_names += ['BM256_BN256_BK64_GM4_nW8_nS2_EU0_kP1_mfma32']
    results += [thread_pool.apply_async(try_compile_config_BM256_BN256_BK64_GM4_nW8_nS2_EU0_kP2_mfma32, args=task_args)]
    config_names += ['BM256_BN256_BK64_GM4_nW8_nS2_EU0_kP2_mfma32']
    results += [thread_pool.apply_async(try_compile_config_BM256_BN256_BK64_GM8_nW8_nS2_EU0_kP1_mfma16, args=task_args)]
    config_names += ['BM256_BN256_BK64_GM8_nW8_nS2_EU0_kP1_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM256_BN256_BK64_GM8_nW8_nS2_EU0_kP2_mfma16, args=task_args)]
    config_names += ['BM256_BN256_BK64_GM8_nW8_nS2_EU0_kP2_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM256_BN256_BK64_GM8_nW8_nS2_EU0_kP1_mfma32, args=task_args)]
    config_names += ['BM256_BN256_BK64_GM8_nW8_nS2_EU0_kP1_mfma32']
    results += [thread_pool.apply_async(try_compile_config_BM256_BN256_BK64_GM8_nW8_nS2_EU0_kP2_mfma32, args=task_args)]
    config_names += ['BM256_BN256_BK64_GM8_nW8_nS2_EU0_kP2_mfma32']
    results += [thread_pool.apply_async(try_compile_config_BM256_BN256_BK64_GM16_nW8_nS2_EU0_kP1_mfma16, args=task_args)]
    config_names += ['BM256_BN256_BK64_GM16_nW8_nS2_EU0_kP1_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM256_BN256_BK64_GM16_nW8_nS2_EU0_kP2_mfma16, args=task_args)]
    config_names += ['BM256_BN256_BK64_GM16_nW8_nS2_EU0_kP2_mfma16']
    results += [thread_pool.apply_async(try_compile_config_BM256_BN256_BK64_GM16_nW8_nS2_EU0_kP1_mfma32, args=task_args)]
    config_names += ['BM256_BN256_BK64_GM16_nW8_nS2_EU0_kP1_mfma32']
    results += [thread_pool.apply_async(try_compile_config_BM256_BN256_BK64_GM16_nW8_nS2_EU0_kP2_mfma32, args=task_args)]
    config_names += ['BM256_BN256_BK64_GM16_nW8_nS2_EU0_kP2_mfma32']

    failed_configs = []
    for i in range(len(results)):
        results[i].wait()
        res = results[i].get()
        if not res:
            failed_configs += [config_names[i]]
    thread_pool.close()
    thread_pool.join()
    if failed_configs:
        with open("/home/work/persistent-kernels/tune_streamk/utils/../compile_driver.py.failed_configs", "w") as f:
            for cfg in failed_configs:
                f.write(cfg + "\n")

def main():
    parser = argparse.ArgumentParser(
        prog="tune a specific gemm size",
        allow_abbrev=False,)
    parser.add_argument("-n", type=int, default=32, help='number of threads')
    parser.add_argument("-rotating_tensor", type=int, default=0, help='size of rotating buffer (MB), default: 0')
    args = parser.parse_args()
    numThreads = args.n
    rotating_buffer_size = args.rotating_tensor
    compile_kernels(4864, 8192, 4160, 304, rotating_buffer_size, 0, numThreads)

if __name__ == '__main__':
   sys.exit(main())