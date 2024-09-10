import torch
import triton
import triton.language as tl
import argparse
import sys
import multiprocessing
from tune_streamk import gen_rotating_tensors




@triton.jit()
def streamk_gemm_BM64_BN64_BK64_GM32_nW4_nS2_EU0_kP2_mfma16(
        A, B, C, bias_ptr, P, locks,
        M, N, K, num_sms,
        stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn, stride_bias,
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
        GROUP_SIZE_M: tl.constexpr, BIAS: tl.constexpr, EVEN_K: tl.constexpr,
):
    pid = tl.program_id(0)
    pid = (pid % 8) * (num_sms // 8) + (pid // 8)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    iters_per_tile = tl.cdiv(K, BLOCK_SIZE_K)
    total_tiles = num_pid_m * num_pid_n
    if num_sms > 0 and total_tiles > num_sms:
        total_streamk_tiles = total_tiles % num_sms
        total_streamk_tiles = total_streamk_tiles + num_sms
        total_full_tiles = total_tiles - total_streamk_tiles
        total_streamk_iters = total_streamk_tiles * iters_per_tile
        streamk_iters_pcu = total_streamk_iters // num_sms
        streamk_remainder_iters = total_streamk_iters % num_sms
    else:
        total_full_tiles = total_tiles
        total_streamk_tiles = 0
        streamk_iters_pcu = 0
        streamk_remainder_iters = 0
        total_streamk_iters = 0

    acc_dtype = tl.float32 if C.type.element_ty != tl.int8 else tl.int32
    rk = tl.arange(0, BLOCK_SIZE_K)

    for tile_id in range(pid, total_full_tiles, num_sms):
        if GROUP_SIZE_M == 1:
            pid_m = tile_id // num_pid_n
            pid_n = tile_id % num_pid_n
        else:
            num_pid_in_group = GROUP_SIZE_M * num_pid_n
            group_id = tile_id // num_pid_in_group
            first_pid_m = group_id * GROUP_SIZE_M
            group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
            pid_m = first_pid_m + (tile_id % group_size_m)
            pid_n = (tile_id % num_pid_in_group) // group_size_m

        rm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M))%M
        rn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))%N
        rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
        rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)
        A_BASE = A + rm[:, None] * stride_am + rk[None, :] * stride_ak
        B_BASE = B + rk[:, None] * stride_bk + rn[None, :] * stride_bn

        if BIAS:
            bias_ = bias_ptr + rm * stride_bias
            bias = tl.load(bias_, mask=rm < M, other=0.0)

        loop_k = tl.cdiv(K, BLOCK_SIZE_K)
        if not EVEN_K:
            loop_k -= 1

        acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype)
        for k in range(0, loop_k):
            a = tl.load(tl.multiple_of(A_BASE, (1, 16)))
            b = tl.load(tl.multiple_of(B_BASE, (16, 1)))
            acc += tl.dot(a, b)
            A_BASE += BLOCK_SIZE_K * stride_ak
            B_BASE += BLOCK_SIZE_K * stride_bk

        if not EVEN_K:
            k = loop_k
            rk = k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
            A_BASE = A + rm[:, None] * stride_am + rk[None, :] * stride_ak
            B_BASE = B + rk[:, None] * stride_bk + rn[None, :] * stride_bn
            a = tl.load(A_BASE, mask=rk[None, :] < K, other=0.0)
            b = tl.load(B_BASE, mask=rk[:, None] < K, other=0.0)

        c = acc.to(C.type.element_ty)
        if BIAS:
             c += bias[:, None]

        rm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        C_ = C + rm[:, None] * stride_cm + rn[None, :] * stride_cn
        mask = (rm < M)[:, None] & (rn < N)[None, :]
        tl.store(C_, c, mask=mask)

    start_iter = total_full_tiles * iters_per_tile + pid * streamk_iters_pcu + tl.minimum(pid, streamk_remainder_iters)
    last_iter = total_full_tiles * iters_per_tile + (pid + 1) * streamk_iters_pcu + tl.minimum(pid + 1, streamk_remainder_iters)
    while start_iter < last_iter:
        remainder = start_iter % iters_per_tile
        end_iter = tl.minimum(start_iter + (iters_per_tile - remainder), last_iter)
        tile_id = start_iter // iters_per_tile
        if GROUP_SIZE_M == 1:
            pid_m = tile_id // num_pid_n
            pid_n = tile_id % num_pid_n
        else:
            num_pid_in_group = GROUP_SIZE_M * num_pid_n
            group_id = tile_id // num_pid_in_group
            first_pid_m = group_id * GROUP_SIZE_M
            group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
            pid_m = first_pid_m + (tile_id % group_size_m)
            pid_n = (tile_id % num_pid_in_group) // group_size_m

        rm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M))%M
        rn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))%N
        rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
        rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)
        A_BASE = A + rm[:, None] * stride_am + rk[None, :] * stride_ak + BLOCK_SIZE_K * stride_ak * remainder
        B_BASE = B + rk[:, None] * stride_bk + rn[None, :] * stride_bn + BLOCK_SIZE_K * stride_bk * remainder

        if BIAS:
            bias_ = bias_ptr + rm * stride_bias
            bias = tl.load(bias_, mask=rm < M, other=0.0)

        acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype)
        for current_iter in range(start_iter, end_iter):
            if EVEN_K:
                a = tl.load(tl.multiple_of(A_BASE, (1, 16)))
                b = tl.load(tl.multiple_of(B_BASE, (16, 1)))
            else:
                global_k_offset = (current_iter % iters_per_tile) * BLOCK_SIZE_K
                k_mask = global_k_offset + rk < K
                a = tl.load(A_BASE, mask=k_mask[None, :], other=0.0)
                b = tl.load(B_BASE, mask=k_mask[:, None], other=0.0)
            acc += tl.dot(a, b)
            A_BASE += BLOCK_SIZE_K * stride_ak
            B_BASE += BLOCK_SIZE_K * stride_bk

        tile_iter = tile_id * iters_per_tile
        if start_iter == tile_iter:
            tile_iter_end = tile_iter + iters_per_tile
            next_pid = pid + 1
            end = end_iter
            while (end < tile_iter_end and next_pid < num_sms):
                while tl.atomic_cas(locks + next_pid, 1, 1) != 1:
              #  while tl.load(locks + next_pid, cache_modifier = ".cg") != 1:
                    pass
                rm1 = tl.arange(0, BLOCK_SIZE_M)
                rn1 = tl.arange(0, BLOCK_SIZE_N)
                rm1 = tl.max_contiguous(tl.multiple_of(rm1, BLOCK_SIZE_M), BLOCK_SIZE_M)
                rn1 = tl.max_contiguous(tl.multiple_of(rn1, BLOCK_SIZE_N), BLOCK_SIZE_N)
                P_ = P + next_pid * BLOCK_SIZE_M * BLOCK_SIZE_N + rm1[:, None] * BLOCK_SIZE_N + rn1[None, :]
                acc += tl.load(tl.multiple_of(P_, (1, 16)))
              #  acc += tl.load(P_)
                end += streamk_iters_pcu + (next_pid < streamk_remainder_iters)
                next_pid += 1

            c = acc.to(C.type.element_ty)
            if BIAS:
                 c += bias[:, None]

            rm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M))%M
            rn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))%N
            rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
            rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)
            C_ = C + rm[:, None] * stride_cm + rn[None, :] * stride_cn
            mask = (rm < M)[:, None] & (rn < N)[None, :]
            tl.store(C_, c, mask=mask)
        else:
            rm1 = tl.arange(0, BLOCK_SIZE_M)
            rn1 = tl.arange(0, BLOCK_SIZE_N)
            rm1 = tl.max_contiguous(tl.multiple_of(rm1, BLOCK_SIZE_M), BLOCK_SIZE_M)
            rn1 = tl.max_contiguous(tl.multiple_of(rn1, BLOCK_SIZE_N), BLOCK_SIZE_N)
            P_ = P + pid * BLOCK_SIZE_M * BLOCK_SIZE_N +  rm1[:, None] * BLOCK_SIZE_N + rn1[None, :]
            tl.store(P_, acc)
            tl.debug_barrier()
            tl.atomic_xchg(locks + pid, 1)
     #       tl.store(locks + pid, 1, cache_modifier=".wt")

        start_iter = end_iter



def matmul_BM64_BN64_BK64_GM32_nW4_nS2_EU0_kP2_mfma16(a, b, c, bias, P, locks, M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    grid = num_sms
    streamk_gemm_BM64_BN64_BK64_GM32_nW4_nS2_EU0_kP2_mfma16[grid,](
        a, b, c, bias, P, locks,
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
        BIAS = False,
        EVEN_K = True
    )
    return c

def try_matmul_config_BM64_BN64_BK64_GM32_nW4_nS2_EU0_kP2_mfma16(a, b, c, bias, M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        locks = torch.zeros((num_sms,), device = "cuda", dtype = torch.int32)
        P = torch.zeros((num_sms,  64*64), device="cuda", dtype=torch.float32)
        matmul_BM64_BN64_BK64_GM32_nW4_nS2_EU0_kP2_mfma16(a, b, c, bias, P, locks, M, N, K, num_sms, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1), bias_stride)
        return True
    except Exception as e:
        print(f'invalid config(runtime): BM64_BN64_BK64_GM32_nW4_nS2_EU0_kP2_mfma16: ', e, flush=True)
        return False




@triton.jit()
def streamk_gemm_BM64_BN64_BK128_GM1_nW4_nS2_EU0_kP2_mfma16(
        A, B, C, bias_ptr, P, locks,
        M, N, K, num_sms,
        stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn, stride_bias,
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
        GROUP_SIZE_M: tl.constexpr, BIAS: tl.constexpr, EVEN_K: tl.constexpr,
):
    pid = tl.program_id(0)
    pid = (pid % 8) * (num_sms // 8) + (pid // 8)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    iters_per_tile = tl.cdiv(K, BLOCK_SIZE_K)
    total_tiles = num_pid_m * num_pid_n
    if num_sms > 0 and total_tiles > num_sms:
        total_streamk_tiles = total_tiles % num_sms
        total_streamk_tiles = total_streamk_tiles + num_sms
        total_full_tiles = total_tiles - total_streamk_tiles
        total_streamk_iters = total_streamk_tiles * iters_per_tile
        streamk_iters_pcu = total_streamk_iters // num_sms
        streamk_remainder_iters = total_streamk_iters % num_sms
    else:
        total_full_tiles = total_tiles
        total_streamk_tiles = 0
        streamk_iters_pcu = 0
        streamk_remainder_iters = 0
        total_streamk_iters = 0

    acc_dtype = tl.float32 if C.type.element_ty != tl.int8 else tl.int32
    rk = tl.arange(0, BLOCK_SIZE_K)

    for tile_id in range(pid, total_full_tiles, num_sms):
        if GROUP_SIZE_M == 1:
            pid_m = tile_id // num_pid_n
            pid_n = tile_id % num_pid_n
        else:
            num_pid_in_group = GROUP_SIZE_M * num_pid_n
            group_id = tile_id // num_pid_in_group
            first_pid_m = group_id * GROUP_SIZE_M
            group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
            pid_m = first_pid_m + (tile_id % group_size_m)
            pid_n = (tile_id % num_pid_in_group) // group_size_m

        rm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M))%M
        rn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))%N
        rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
        rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)
        A_BASE = A + rm[:, None] * stride_am + rk[None, :] * stride_ak
        B_BASE = B + rk[:, None] * stride_bk + rn[None, :] * stride_bn

        if BIAS:
            bias_ = bias_ptr + rm * stride_bias
            bias = tl.load(bias_, mask=rm < M, other=0.0)

        loop_k = tl.cdiv(K, BLOCK_SIZE_K)
        if not EVEN_K:
            loop_k -= 1

        acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype)
        for k in range(0, loop_k):
            a = tl.load(tl.multiple_of(A_BASE, (1, 16)))
            b = tl.load(tl.multiple_of(B_BASE, (16, 1)))
            acc += tl.dot(a, b)
            A_BASE += BLOCK_SIZE_K * stride_ak
            B_BASE += BLOCK_SIZE_K * stride_bk

        if not EVEN_K:
            k = loop_k
            rk = k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
            A_BASE = A + rm[:, None] * stride_am + rk[None, :] * stride_ak
            B_BASE = B + rk[:, None] * stride_bk + rn[None, :] * stride_bn
            a = tl.load(A_BASE, mask=rk[None, :] < K, other=0.0)
            b = tl.load(B_BASE, mask=rk[:, None] < K, other=0.0)

        c = acc.to(C.type.element_ty)
        if BIAS:
             c += bias[:, None]

        rm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        C_ = C + rm[:, None] * stride_cm + rn[None, :] * stride_cn
        mask = (rm < M)[:, None] & (rn < N)[None, :]
        tl.store(C_, c, mask=mask)

    start_iter = total_full_tiles * iters_per_tile + pid * streamk_iters_pcu + tl.minimum(pid, streamk_remainder_iters)
    last_iter = total_full_tiles * iters_per_tile + (pid + 1) * streamk_iters_pcu + tl.minimum(pid + 1, streamk_remainder_iters)
    while start_iter < last_iter:
        remainder = start_iter % iters_per_tile
        end_iter = tl.minimum(start_iter + (iters_per_tile - remainder), last_iter)
        tile_id = start_iter // iters_per_tile
        if GROUP_SIZE_M == 1:
            pid_m = tile_id // num_pid_n
            pid_n = tile_id % num_pid_n
        else:
            num_pid_in_group = GROUP_SIZE_M * num_pid_n
            group_id = tile_id // num_pid_in_group
            first_pid_m = group_id * GROUP_SIZE_M
            group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
            pid_m = first_pid_m + (tile_id % group_size_m)
            pid_n = (tile_id % num_pid_in_group) // group_size_m

        rm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M))%M
        rn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))%N
        rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
        rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)
        A_BASE = A + rm[:, None] * stride_am + rk[None, :] * stride_ak + BLOCK_SIZE_K * stride_ak * remainder
        B_BASE = B + rk[:, None] * stride_bk + rn[None, :] * stride_bn + BLOCK_SIZE_K * stride_bk * remainder

        if BIAS:
            bias_ = bias_ptr + rm * stride_bias
            bias = tl.load(bias_, mask=rm < M, other=0.0)

        acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype)
        for current_iter in range(start_iter, end_iter):
            if EVEN_K:
                a = tl.load(tl.multiple_of(A_BASE, (1, 16)))
                b = tl.load(tl.multiple_of(B_BASE, (16, 1)))
            else:
                global_k_offset = (current_iter % iters_per_tile) * BLOCK_SIZE_K
                k_mask = global_k_offset + rk < K
                a = tl.load(A_BASE, mask=k_mask[None, :], other=0.0)
                b = tl.load(B_BASE, mask=k_mask[:, None], other=0.0)
            acc += tl.dot(a, b)
            A_BASE += BLOCK_SIZE_K * stride_ak
            B_BASE += BLOCK_SIZE_K * stride_bk

        tile_iter = tile_id * iters_per_tile
        if start_iter == tile_iter:
            tile_iter_end = tile_iter + iters_per_tile
            next_pid = pid + 1
            end = end_iter
            while (end < tile_iter_end and next_pid < num_sms):
                while tl.atomic_cas(locks + next_pid, 1, 1) != 1:
              #  while tl.load(locks + next_pid, cache_modifier = ".cg") != 1:
                    pass
                rm1 = tl.arange(0, BLOCK_SIZE_M)
                rn1 = tl.arange(0, BLOCK_SIZE_N)
                rm1 = tl.max_contiguous(tl.multiple_of(rm1, BLOCK_SIZE_M), BLOCK_SIZE_M)
                rn1 = tl.max_contiguous(tl.multiple_of(rn1, BLOCK_SIZE_N), BLOCK_SIZE_N)
                P_ = P + next_pid * BLOCK_SIZE_M * BLOCK_SIZE_N + rm1[:, None] * BLOCK_SIZE_N + rn1[None, :]
                acc += tl.load(tl.multiple_of(P_, (1, 16)))
              #  acc += tl.load(P_)
                end += streamk_iters_pcu + (next_pid < streamk_remainder_iters)
                next_pid += 1

            c = acc.to(C.type.element_ty)
            if BIAS:
                 c += bias[:, None]

            rm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M))%M
            rn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))%N
            rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
            rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)
            C_ = C + rm[:, None] * stride_cm + rn[None, :] * stride_cn
            mask = (rm < M)[:, None] & (rn < N)[None, :]
            tl.store(C_, c, mask=mask)
        else:
            rm1 = tl.arange(0, BLOCK_SIZE_M)
            rn1 = tl.arange(0, BLOCK_SIZE_N)
            rm1 = tl.max_contiguous(tl.multiple_of(rm1, BLOCK_SIZE_M), BLOCK_SIZE_M)
            rn1 = tl.max_contiguous(tl.multiple_of(rn1, BLOCK_SIZE_N), BLOCK_SIZE_N)
            P_ = P + pid * BLOCK_SIZE_M * BLOCK_SIZE_N +  rm1[:, None] * BLOCK_SIZE_N + rn1[None, :]
            tl.store(P_, acc)
            tl.debug_barrier()
            tl.atomic_xchg(locks + pid, 1)
     #       tl.store(locks + pid, 1, cache_modifier=".wt")

        start_iter = end_iter



def matmul_BM64_BN64_BK128_GM1_nW4_nS2_EU0_kP2_mfma16(a, b, c, bias, P, locks, M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    grid = num_sms
    streamk_gemm_BM64_BN64_BK128_GM1_nW4_nS2_EU0_kP2_mfma16[grid,](
        a, b, c, bias, P, locks,
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
        BIAS = False,
        EVEN_K = False
    )
    return c

def try_matmul_config_BM64_BN64_BK128_GM1_nW4_nS2_EU0_kP2_mfma16(a, b, c, bias, M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        locks = torch.zeros((num_sms,), device = "cuda", dtype = torch.int32)
        P = torch.zeros((num_sms,  64*64), device="cuda", dtype=torch.float32)
        matmul_BM64_BN64_BK128_GM1_nW4_nS2_EU0_kP2_mfma16(a, b, c, bias, P, locks, M, N, K, num_sms, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1), bias_stride)
        return True
    except Exception as e:
        print(f'invalid config(runtime): BM64_BN64_BK128_GM1_nW4_nS2_EU0_kP2_mfma16: ', e, flush=True)
        return False




@triton.jit()
def streamk_gemm_BM64_BN64_BK128_GM4_nW8_nS2_EU0_kP2_mfma16(
        A, B, C, bias_ptr, P, locks,
        M, N, K, num_sms,
        stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn, stride_bias,
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
        GROUP_SIZE_M: tl.constexpr, BIAS: tl.constexpr, EVEN_K: tl.constexpr,
):
    pid = tl.program_id(0)
    pid = (pid % 8) * (num_sms // 8) + (pid // 8)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    iters_per_tile = tl.cdiv(K, BLOCK_SIZE_K)
    total_tiles = num_pid_m * num_pid_n
    if num_sms > 0 and total_tiles > num_sms:
        total_streamk_tiles = total_tiles % num_sms
        total_streamk_tiles = total_streamk_tiles + num_sms
        total_full_tiles = total_tiles - total_streamk_tiles
        total_streamk_iters = total_streamk_tiles * iters_per_tile
        streamk_iters_pcu = total_streamk_iters // num_sms
        streamk_remainder_iters = total_streamk_iters % num_sms
    else:
        total_full_tiles = total_tiles
        total_streamk_tiles = 0
        streamk_iters_pcu = 0
        streamk_remainder_iters = 0
        total_streamk_iters = 0

    acc_dtype = tl.float32 if C.type.element_ty != tl.int8 else tl.int32
    rk = tl.arange(0, BLOCK_SIZE_K)

    for tile_id in range(pid, total_full_tiles, num_sms):
        if GROUP_SIZE_M == 1:
            pid_m = tile_id // num_pid_n
            pid_n = tile_id % num_pid_n
        else:
            num_pid_in_group = GROUP_SIZE_M * num_pid_n
            group_id = tile_id // num_pid_in_group
            first_pid_m = group_id * GROUP_SIZE_M
            group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
            pid_m = first_pid_m + (tile_id % group_size_m)
            pid_n = (tile_id % num_pid_in_group) // group_size_m

        rm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M))%M
        rn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))%N
        rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
        rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)
        A_BASE = A + rm[:, None] * stride_am + rk[None, :] * stride_ak
        B_BASE = B + rk[:, None] * stride_bk + rn[None, :] * stride_bn

        if BIAS:
            bias_ = bias_ptr + rm * stride_bias
            bias = tl.load(bias_, mask=rm < M, other=0.0)

        loop_k = tl.cdiv(K, BLOCK_SIZE_K)
        if not EVEN_K:
            loop_k -= 1

        acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype)
        for k in range(0, loop_k):
            a = tl.load(tl.multiple_of(A_BASE, (1, 16)))
            b = tl.load(tl.multiple_of(B_BASE, (16, 1)))
            acc += tl.dot(a, b)
            A_BASE += BLOCK_SIZE_K * stride_ak
            B_BASE += BLOCK_SIZE_K * stride_bk

        if not EVEN_K:
            k = loop_k
            rk = k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
            A_BASE = A + rm[:, None] * stride_am + rk[None, :] * stride_ak
            B_BASE = B + rk[:, None] * stride_bk + rn[None, :] * stride_bn
            a = tl.load(A_BASE, mask=rk[None, :] < K, other=0.0)
            b = tl.load(B_BASE, mask=rk[:, None] < K, other=0.0)

        c = acc.to(C.type.element_ty)
        if BIAS:
             c += bias[:, None]

        rm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        C_ = C + rm[:, None] * stride_cm + rn[None, :] * stride_cn
        mask = (rm < M)[:, None] & (rn < N)[None, :]
        tl.store(C_, c, mask=mask)

    start_iter = total_full_tiles * iters_per_tile + pid * streamk_iters_pcu + tl.minimum(pid, streamk_remainder_iters)
    last_iter = total_full_tiles * iters_per_tile + (pid + 1) * streamk_iters_pcu + tl.minimum(pid + 1, streamk_remainder_iters)
    while start_iter < last_iter:
        remainder = start_iter % iters_per_tile
        end_iter = tl.minimum(start_iter + (iters_per_tile - remainder), last_iter)
        tile_id = start_iter // iters_per_tile
        if GROUP_SIZE_M == 1:
            pid_m = tile_id // num_pid_n
            pid_n = tile_id % num_pid_n
        else:
            num_pid_in_group = GROUP_SIZE_M * num_pid_n
            group_id = tile_id // num_pid_in_group
            first_pid_m = group_id * GROUP_SIZE_M
            group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
            pid_m = first_pid_m + (tile_id % group_size_m)
            pid_n = (tile_id % num_pid_in_group) // group_size_m

        rm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M))%M
        rn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))%N
        rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
        rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)
        A_BASE = A + rm[:, None] * stride_am + rk[None, :] * stride_ak + BLOCK_SIZE_K * stride_ak * remainder
        B_BASE = B + rk[:, None] * stride_bk + rn[None, :] * stride_bn + BLOCK_SIZE_K * stride_bk * remainder

        if BIAS:
            bias_ = bias_ptr + rm * stride_bias
            bias = tl.load(bias_, mask=rm < M, other=0.0)

        acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype)
        for current_iter in range(start_iter, end_iter):
            if EVEN_K:
                a = tl.load(tl.multiple_of(A_BASE, (1, 16)))
                b = tl.load(tl.multiple_of(B_BASE, (16, 1)))
            else:
                global_k_offset = (current_iter % iters_per_tile) * BLOCK_SIZE_K
                k_mask = global_k_offset + rk < K
                a = tl.load(A_BASE, mask=k_mask[None, :], other=0.0)
                b = tl.load(B_BASE, mask=k_mask[:, None], other=0.0)
            acc += tl.dot(a, b)
            A_BASE += BLOCK_SIZE_K * stride_ak
            B_BASE += BLOCK_SIZE_K * stride_bk

        tile_iter = tile_id * iters_per_tile
        if start_iter == tile_iter:
            tile_iter_end = tile_iter + iters_per_tile
            next_pid = pid + 1
            end = end_iter
            while (end < tile_iter_end and next_pid < num_sms):
                while tl.atomic_cas(locks + next_pid, 1, 1) != 1:
              #  while tl.load(locks + next_pid, cache_modifier = ".cg") != 1:
                    pass
                rm1 = tl.arange(0, BLOCK_SIZE_M)
                rn1 = tl.arange(0, BLOCK_SIZE_N)
                rm1 = tl.max_contiguous(tl.multiple_of(rm1, BLOCK_SIZE_M), BLOCK_SIZE_M)
                rn1 = tl.max_contiguous(tl.multiple_of(rn1, BLOCK_SIZE_N), BLOCK_SIZE_N)
                P_ = P + next_pid * BLOCK_SIZE_M * BLOCK_SIZE_N + rm1[:, None] * BLOCK_SIZE_N + rn1[None, :]
                acc += tl.load(tl.multiple_of(P_, (1, 16)))
              #  acc += tl.load(P_)
                end += streamk_iters_pcu + (next_pid < streamk_remainder_iters)
                next_pid += 1

            c = acc.to(C.type.element_ty)
            if BIAS:
                 c += bias[:, None]

            rm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M))%M
            rn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))%N
            rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
            rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)
            C_ = C + rm[:, None] * stride_cm + rn[None, :] * stride_cn
            mask = (rm < M)[:, None] & (rn < N)[None, :]
            tl.store(C_, c, mask=mask)
        else:
            rm1 = tl.arange(0, BLOCK_SIZE_M)
            rn1 = tl.arange(0, BLOCK_SIZE_N)
            rm1 = tl.max_contiguous(tl.multiple_of(rm1, BLOCK_SIZE_M), BLOCK_SIZE_M)
            rn1 = tl.max_contiguous(tl.multiple_of(rn1, BLOCK_SIZE_N), BLOCK_SIZE_N)
            P_ = P + pid * BLOCK_SIZE_M * BLOCK_SIZE_N +  rm1[:, None] * BLOCK_SIZE_N + rn1[None, :]
            tl.store(P_, acc)
            tl.debug_barrier()
            tl.atomic_xchg(locks + pid, 1)
     #       tl.store(locks + pid, 1, cache_modifier=".wt")

        start_iter = end_iter



def matmul_BM64_BN64_BK128_GM4_nW8_nS2_EU0_kP2_mfma16(a, b, c, bias, P, locks, M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    grid = num_sms
    streamk_gemm_BM64_BN64_BK128_GM4_nW8_nS2_EU0_kP2_mfma16[grid,](
        a, b, c, bias, P, locks,
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
        BIAS = False,
        EVEN_K = False
    )
    return c

def try_matmul_config_BM64_BN64_BK128_GM4_nW8_nS2_EU0_kP2_mfma16(a, b, c, bias, M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        locks = torch.zeros((num_sms,), device = "cuda", dtype = torch.int32)
        P = torch.zeros((num_sms,  64*64), device="cuda", dtype=torch.float32)
        matmul_BM64_BN64_BK128_GM4_nW8_nS2_EU0_kP2_mfma16(a, b, c, bias, P, locks, M, N, K, num_sms, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1), bias_stride)
        return True
    except Exception as e:
        print(f'invalid config(runtime): BM64_BN64_BK128_GM4_nW8_nS2_EU0_kP2_mfma16: ', e, flush=True)
        return False




@triton.jit()
def streamk_gemm_BM64_BN64_BK256_GM8_nW4_nS2_EU0_kP2_mfma16(
        A, B, C, bias_ptr, P, locks,
        M, N, K, num_sms,
        stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn, stride_bias,
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
        GROUP_SIZE_M: tl.constexpr, BIAS: tl.constexpr, EVEN_K: tl.constexpr,
):
    pid = tl.program_id(0)
    pid = (pid % 8) * (num_sms // 8) + (pid // 8)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    iters_per_tile = tl.cdiv(K, BLOCK_SIZE_K)
    total_tiles = num_pid_m * num_pid_n
    if num_sms > 0 and total_tiles > num_sms:
        total_streamk_tiles = total_tiles % num_sms
        total_streamk_tiles = total_streamk_tiles + num_sms
        total_full_tiles = total_tiles - total_streamk_tiles
        total_streamk_iters = total_streamk_tiles * iters_per_tile
        streamk_iters_pcu = total_streamk_iters // num_sms
        streamk_remainder_iters = total_streamk_iters % num_sms
    else:
        total_full_tiles = total_tiles
        total_streamk_tiles = 0
        streamk_iters_pcu = 0
        streamk_remainder_iters = 0
        total_streamk_iters = 0

    acc_dtype = tl.float32 if C.type.element_ty != tl.int8 else tl.int32
    rk = tl.arange(0, BLOCK_SIZE_K)

    for tile_id in range(pid, total_full_tiles, num_sms):
        if GROUP_SIZE_M == 1:
            pid_m = tile_id // num_pid_n
            pid_n = tile_id % num_pid_n
        else:
            num_pid_in_group = GROUP_SIZE_M * num_pid_n
            group_id = tile_id // num_pid_in_group
            first_pid_m = group_id * GROUP_SIZE_M
            group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
            pid_m = first_pid_m + (tile_id % group_size_m)
            pid_n = (tile_id % num_pid_in_group) // group_size_m

        rm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M))%M
        rn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))%N
        rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
        rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)
        A_BASE = A + rm[:, None] * stride_am + rk[None, :] * stride_ak
        B_BASE = B + rk[:, None] * stride_bk + rn[None, :] * stride_bn

        if BIAS:
            bias_ = bias_ptr + rm * stride_bias
            bias = tl.load(bias_, mask=rm < M, other=0.0)

        loop_k = tl.cdiv(K, BLOCK_SIZE_K)
        if not EVEN_K:
            loop_k -= 1

        acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype)
        for k in range(0, loop_k):
            a = tl.load(tl.multiple_of(A_BASE, (1, 16)))
            b = tl.load(tl.multiple_of(B_BASE, (16, 1)))
            acc += tl.dot(a, b)
            A_BASE += BLOCK_SIZE_K * stride_ak
            B_BASE += BLOCK_SIZE_K * stride_bk

        if not EVEN_K:
            k = loop_k
            rk = k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
            A_BASE = A + rm[:, None] * stride_am + rk[None, :] * stride_ak
            B_BASE = B + rk[:, None] * stride_bk + rn[None, :] * stride_bn
            a = tl.load(A_BASE, mask=rk[None, :] < K, other=0.0)
            b = tl.load(B_BASE, mask=rk[:, None] < K, other=0.0)

        c = acc.to(C.type.element_ty)
        if BIAS:
             c += bias[:, None]

        rm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        C_ = C + rm[:, None] * stride_cm + rn[None, :] * stride_cn
        mask = (rm < M)[:, None] & (rn < N)[None, :]
        tl.store(C_, c, mask=mask)

    start_iter = total_full_tiles * iters_per_tile + pid * streamk_iters_pcu + tl.minimum(pid, streamk_remainder_iters)
    last_iter = total_full_tiles * iters_per_tile + (pid + 1) * streamk_iters_pcu + tl.minimum(pid + 1, streamk_remainder_iters)
    while start_iter < last_iter:
        remainder = start_iter % iters_per_tile
        end_iter = tl.minimum(start_iter + (iters_per_tile - remainder), last_iter)
        tile_id = start_iter // iters_per_tile
        if GROUP_SIZE_M == 1:
            pid_m = tile_id // num_pid_n
            pid_n = tile_id % num_pid_n
        else:
            num_pid_in_group = GROUP_SIZE_M * num_pid_n
            group_id = tile_id // num_pid_in_group
            first_pid_m = group_id * GROUP_SIZE_M
            group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
            pid_m = first_pid_m + (tile_id % group_size_m)
            pid_n = (tile_id % num_pid_in_group) // group_size_m

        rm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M))%M
        rn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))%N
        rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
        rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)
        A_BASE = A + rm[:, None] * stride_am + rk[None, :] * stride_ak + BLOCK_SIZE_K * stride_ak * remainder
        B_BASE = B + rk[:, None] * stride_bk + rn[None, :] * stride_bn + BLOCK_SIZE_K * stride_bk * remainder

        if BIAS:
            bias_ = bias_ptr + rm * stride_bias
            bias = tl.load(bias_, mask=rm < M, other=0.0)

        acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype)
        for current_iter in range(start_iter, end_iter):
            if EVEN_K:
                a = tl.load(tl.multiple_of(A_BASE, (1, 16)))
                b = tl.load(tl.multiple_of(B_BASE, (16, 1)))
            else:
                global_k_offset = (current_iter % iters_per_tile) * BLOCK_SIZE_K
                k_mask = global_k_offset + rk < K
                a = tl.load(A_BASE, mask=k_mask[None, :], other=0.0)
                b = tl.load(B_BASE, mask=k_mask[:, None], other=0.0)
            acc += tl.dot(a, b)
            A_BASE += BLOCK_SIZE_K * stride_ak
            B_BASE += BLOCK_SIZE_K * stride_bk

        tile_iter = tile_id * iters_per_tile
        if start_iter == tile_iter:
            tile_iter_end = tile_iter + iters_per_tile
            next_pid = pid + 1
            end = end_iter
            while (end < tile_iter_end and next_pid < num_sms):
                while tl.atomic_cas(locks + next_pid, 1, 1) != 1:
              #  while tl.load(locks + next_pid, cache_modifier = ".cg") != 1:
                    pass
                rm1 = tl.arange(0, BLOCK_SIZE_M)
                rn1 = tl.arange(0, BLOCK_SIZE_N)
                rm1 = tl.max_contiguous(tl.multiple_of(rm1, BLOCK_SIZE_M), BLOCK_SIZE_M)
                rn1 = tl.max_contiguous(tl.multiple_of(rn1, BLOCK_SIZE_N), BLOCK_SIZE_N)
                P_ = P + next_pid * BLOCK_SIZE_M * BLOCK_SIZE_N + rm1[:, None] * BLOCK_SIZE_N + rn1[None, :]
                acc += tl.load(tl.multiple_of(P_, (1, 16)))
              #  acc += tl.load(P_)
                end += streamk_iters_pcu + (next_pid < streamk_remainder_iters)
                next_pid += 1

            c = acc.to(C.type.element_ty)
            if BIAS:
                 c += bias[:, None]

            rm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M))%M
            rn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))%N
            rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
            rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)
            C_ = C + rm[:, None] * stride_cm + rn[None, :] * stride_cn
            mask = (rm < M)[:, None] & (rn < N)[None, :]
            tl.store(C_, c, mask=mask)
        else:
            rm1 = tl.arange(0, BLOCK_SIZE_M)
            rn1 = tl.arange(0, BLOCK_SIZE_N)
            rm1 = tl.max_contiguous(tl.multiple_of(rm1, BLOCK_SIZE_M), BLOCK_SIZE_M)
            rn1 = tl.max_contiguous(tl.multiple_of(rn1, BLOCK_SIZE_N), BLOCK_SIZE_N)
            P_ = P + pid * BLOCK_SIZE_M * BLOCK_SIZE_N +  rm1[:, None] * BLOCK_SIZE_N + rn1[None, :]
            tl.store(P_, acc)
            tl.debug_barrier()
            tl.atomic_xchg(locks + pid, 1)
     #       tl.store(locks + pid, 1, cache_modifier=".wt")

        start_iter = end_iter



def matmul_BM64_BN64_BK256_GM8_nW4_nS2_EU0_kP2_mfma16(a, b, c, bias, P, locks, M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    grid = num_sms
    streamk_gemm_BM64_BN64_BK256_GM8_nW4_nS2_EU0_kP2_mfma16[grid,](
        a, b, c, bias, P, locks,
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
        BIAS = False,
        EVEN_K = False
    )
    return c

def try_matmul_config_BM64_BN64_BK256_GM8_nW4_nS2_EU0_kP2_mfma16(a, b, c, bias, M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        locks = torch.zeros((num_sms,), device = "cuda", dtype = torch.int32)
        P = torch.zeros((num_sms,  64*64), device="cuda", dtype=torch.float32)
        matmul_BM64_BN64_BK256_GM8_nW4_nS2_EU0_kP2_mfma16(a, b, c, bias, P, locks, M, N, K, num_sms, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1), bias_stride)
        return True
    except Exception as e:
        print(f'invalid config(runtime): BM64_BN64_BK256_GM8_nW4_nS2_EU0_kP2_mfma16: ', e, flush=True)
        return False




@triton.jit()
def streamk_gemm_BM64_BN64_BK256_GM16_nW8_nS2_EU0_kP2_mfma16(
        A, B, C, bias_ptr, P, locks,
        M, N, K, num_sms,
        stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn, stride_bias,
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
        GROUP_SIZE_M: tl.constexpr, BIAS: tl.constexpr, EVEN_K: tl.constexpr,
):
    pid = tl.program_id(0)
    pid = (pid % 8) * (num_sms // 8) + (pid // 8)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    iters_per_tile = tl.cdiv(K, BLOCK_SIZE_K)
    total_tiles = num_pid_m * num_pid_n
    if num_sms > 0 and total_tiles > num_sms:
        total_streamk_tiles = total_tiles % num_sms
        total_streamk_tiles = total_streamk_tiles + num_sms
        total_full_tiles = total_tiles - total_streamk_tiles
        total_streamk_iters = total_streamk_tiles * iters_per_tile
        streamk_iters_pcu = total_streamk_iters // num_sms
        streamk_remainder_iters = total_streamk_iters % num_sms
    else:
        total_full_tiles = total_tiles
        total_streamk_tiles = 0
        streamk_iters_pcu = 0
        streamk_remainder_iters = 0
        total_streamk_iters = 0

    acc_dtype = tl.float32 if C.type.element_ty != tl.int8 else tl.int32
    rk = tl.arange(0, BLOCK_SIZE_K)

    for tile_id in range(pid, total_full_tiles, num_sms):
        if GROUP_SIZE_M == 1:
            pid_m = tile_id // num_pid_n
            pid_n = tile_id % num_pid_n
        else:
            num_pid_in_group = GROUP_SIZE_M * num_pid_n
            group_id = tile_id // num_pid_in_group
            first_pid_m = group_id * GROUP_SIZE_M
            group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
            pid_m = first_pid_m + (tile_id % group_size_m)
            pid_n = (tile_id % num_pid_in_group) // group_size_m

        rm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M))%M
        rn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))%N
        rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
        rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)
        A_BASE = A + rm[:, None] * stride_am + rk[None, :] * stride_ak
        B_BASE = B + rk[:, None] * stride_bk + rn[None, :] * stride_bn

        if BIAS:
            bias_ = bias_ptr + rm * stride_bias
            bias = tl.load(bias_, mask=rm < M, other=0.0)

        loop_k = tl.cdiv(K, BLOCK_SIZE_K)
        if not EVEN_K:
            loop_k -= 1

        acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype)
        for k in range(0, loop_k):
            a = tl.load(tl.multiple_of(A_BASE, (1, 16)))
            b = tl.load(tl.multiple_of(B_BASE, (16, 1)))
            acc += tl.dot(a, b)
            A_BASE += BLOCK_SIZE_K * stride_ak
            B_BASE += BLOCK_SIZE_K * stride_bk

        if not EVEN_K:
            k = loop_k
            rk = k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
            A_BASE = A + rm[:, None] * stride_am + rk[None, :] * stride_ak
            B_BASE = B + rk[:, None] * stride_bk + rn[None, :] * stride_bn
            a = tl.load(A_BASE, mask=rk[None, :] < K, other=0.0)
            b = tl.load(B_BASE, mask=rk[:, None] < K, other=0.0)

        c = acc.to(C.type.element_ty)
        if BIAS:
             c += bias[:, None]

        rm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        C_ = C + rm[:, None] * stride_cm + rn[None, :] * stride_cn
        mask = (rm < M)[:, None] & (rn < N)[None, :]
        tl.store(C_, c, mask=mask)

    start_iter = total_full_tiles * iters_per_tile + pid * streamk_iters_pcu + tl.minimum(pid, streamk_remainder_iters)
    last_iter = total_full_tiles * iters_per_tile + (pid + 1) * streamk_iters_pcu + tl.minimum(pid + 1, streamk_remainder_iters)
    while start_iter < last_iter:
        remainder = start_iter % iters_per_tile
        end_iter = tl.minimum(start_iter + (iters_per_tile - remainder), last_iter)
        tile_id = start_iter // iters_per_tile
        if GROUP_SIZE_M == 1:
            pid_m = tile_id // num_pid_n
            pid_n = tile_id % num_pid_n
        else:
            num_pid_in_group = GROUP_SIZE_M * num_pid_n
            group_id = tile_id // num_pid_in_group
            first_pid_m = group_id * GROUP_SIZE_M
            group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
            pid_m = first_pid_m + (tile_id % group_size_m)
            pid_n = (tile_id % num_pid_in_group) // group_size_m

        rm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M))%M
        rn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))%N
        rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
        rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)
        A_BASE = A + rm[:, None] * stride_am + rk[None, :] * stride_ak + BLOCK_SIZE_K * stride_ak * remainder
        B_BASE = B + rk[:, None] * stride_bk + rn[None, :] * stride_bn + BLOCK_SIZE_K * stride_bk * remainder

        if BIAS:
            bias_ = bias_ptr + rm * stride_bias
            bias = tl.load(bias_, mask=rm < M, other=0.0)

        acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype)
        for current_iter in range(start_iter, end_iter):
            if EVEN_K:
                a = tl.load(tl.multiple_of(A_BASE, (1, 16)))
                b = tl.load(tl.multiple_of(B_BASE, (16, 1)))
            else:
                global_k_offset = (current_iter % iters_per_tile) * BLOCK_SIZE_K
                k_mask = global_k_offset + rk < K
                a = tl.load(A_BASE, mask=k_mask[None, :], other=0.0)
                b = tl.load(B_BASE, mask=k_mask[:, None], other=0.0)
            acc += tl.dot(a, b)
            A_BASE += BLOCK_SIZE_K * stride_ak
            B_BASE += BLOCK_SIZE_K * stride_bk

        tile_iter = tile_id * iters_per_tile
        if start_iter == tile_iter:
            tile_iter_end = tile_iter + iters_per_tile
            next_pid = pid + 1
            end = end_iter
            while (end < tile_iter_end and next_pid < num_sms):
                while tl.atomic_cas(locks + next_pid, 1, 1) != 1:
              #  while tl.load(locks + next_pid, cache_modifier = ".cg") != 1:
                    pass
                rm1 = tl.arange(0, BLOCK_SIZE_M)
                rn1 = tl.arange(0, BLOCK_SIZE_N)
                rm1 = tl.max_contiguous(tl.multiple_of(rm1, BLOCK_SIZE_M), BLOCK_SIZE_M)
                rn1 = tl.max_contiguous(tl.multiple_of(rn1, BLOCK_SIZE_N), BLOCK_SIZE_N)
                P_ = P + next_pid * BLOCK_SIZE_M * BLOCK_SIZE_N + rm1[:, None] * BLOCK_SIZE_N + rn1[None, :]
                acc += tl.load(tl.multiple_of(P_, (1, 16)))
              #  acc += tl.load(P_)
                end += streamk_iters_pcu + (next_pid < streamk_remainder_iters)
                next_pid += 1

            c = acc.to(C.type.element_ty)
            if BIAS:
                 c += bias[:, None]

            rm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M))%M
            rn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))%N
            rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
            rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)
            C_ = C + rm[:, None] * stride_cm + rn[None, :] * stride_cn
            mask = (rm < M)[:, None] & (rn < N)[None, :]
            tl.store(C_, c, mask=mask)
        else:
            rm1 = tl.arange(0, BLOCK_SIZE_M)
            rn1 = tl.arange(0, BLOCK_SIZE_N)
            rm1 = tl.max_contiguous(tl.multiple_of(rm1, BLOCK_SIZE_M), BLOCK_SIZE_M)
            rn1 = tl.max_contiguous(tl.multiple_of(rn1, BLOCK_SIZE_N), BLOCK_SIZE_N)
            P_ = P + pid * BLOCK_SIZE_M * BLOCK_SIZE_N +  rm1[:, None] * BLOCK_SIZE_N + rn1[None, :]
            tl.store(P_, acc)
            tl.debug_barrier()
            tl.atomic_xchg(locks + pid, 1)
     #       tl.store(locks + pid, 1, cache_modifier=".wt")

        start_iter = end_iter



def matmul_BM64_BN64_BK256_GM16_nW8_nS2_EU0_kP2_mfma16(a, b, c, bias, P, locks, M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    grid = num_sms
    streamk_gemm_BM64_BN64_BK256_GM16_nW8_nS2_EU0_kP2_mfma16[grid,](
        a, b, c, bias, P, locks,
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
        BIAS = False,
        EVEN_K = False
    )
    return c

def try_matmul_config_BM64_BN64_BK256_GM16_nW8_nS2_EU0_kP2_mfma16(a, b, c, bias, M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        locks = torch.zeros((num_sms,), device = "cuda", dtype = torch.int32)
        P = torch.zeros((num_sms,  64*64), device="cuda", dtype=torch.float32)
        matmul_BM64_BN64_BK256_GM16_nW8_nS2_EU0_kP2_mfma16(a, b, c, bias, P, locks, M, N, K, num_sms, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1), bias_stride)
        return True
    except Exception as e:
        print(f'invalid config(runtime): BM64_BN64_BK256_GM16_nW8_nS2_EU0_kP2_mfma16: ', e, flush=True)
        return False




@triton.jit()
def streamk_gemm_BM64_BN128_BK64_GM32_nW4_nS2_EU0_kP2_mfma16(
        A, B, C, bias_ptr, P, locks,
        M, N, K, num_sms,
        stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn, stride_bias,
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
        GROUP_SIZE_M: tl.constexpr, BIAS: tl.constexpr, EVEN_K: tl.constexpr,
):
    pid = tl.program_id(0)
    pid = (pid % 8) * (num_sms // 8) + (pid // 8)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    iters_per_tile = tl.cdiv(K, BLOCK_SIZE_K)
    total_tiles = num_pid_m * num_pid_n
    if num_sms > 0 and total_tiles > num_sms:
        total_streamk_tiles = total_tiles % num_sms
        total_streamk_tiles = total_streamk_tiles + num_sms
        total_full_tiles = total_tiles - total_streamk_tiles
        total_streamk_iters = total_streamk_tiles * iters_per_tile
        streamk_iters_pcu = total_streamk_iters // num_sms
        streamk_remainder_iters = total_streamk_iters % num_sms
    else:
        total_full_tiles = total_tiles
        total_streamk_tiles = 0
        streamk_iters_pcu = 0
        streamk_remainder_iters = 0
        total_streamk_iters = 0

    acc_dtype = tl.float32 if C.type.element_ty != tl.int8 else tl.int32
    rk = tl.arange(0, BLOCK_SIZE_K)

    for tile_id in range(pid, total_full_tiles, num_sms):
        if GROUP_SIZE_M == 1:
            pid_m = tile_id // num_pid_n
            pid_n = tile_id % num_pid_n
        else:
            num_pid_in_group = GROUP_SIZE_M * num_pid_n
            group_id = tile_id // num_pid_in_group
            first_pid_m = group_id * GROUP_SIZE_M
            group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
            pid_m = first_pid_m + (tile_id % group_size_m)
            pid_n = (tile_id % num_pid_in_group) // group_size_m

        rm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M))%M
        rn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))%N
        rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
        rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)
        A_BASE = A + rm[:, None] * stride_am + rk[None, :] * stride_ak
        B_BASE = B + rk[:, None] * stride_bk + rn[None, :] * stride_bn

        if BIAS:
            bias_ = bias_ptr + rm * stride_bias
            bias = tl.load(bias_, mask=rm < M, other=0.0)

        loop_k = tl.cdiv(K, BLOCK_SIZE_K)
        if not EVEN_K:
            loop_k -= 1

        acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype)
        for k in range(0, loop_k):
            a = tl.load(tl.multiple_of(A_BASE, (1, 16)))
            b = tl.load(tl.multiple_of(B_BASE, (16, 1)))
            acc += tl.dot(a, b)
            A_BASE += BLOCK_SIZE_K * stride_ak
            B_BASE += BLOCK_SIZE_K * stride_bk

        if not EVEN_K:
            k = loop_k
            rk = k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
            A_BASE = A + rm[:, None] * stride_am + rk[None, :] * stride_ak
            B_BASE = B + rk[:, None] * stride_bk + rn[None, :] * stride_bn
            a = tl.load(A_BASE, mask=rk[None, :] < K, other=0.0)
            b = tl.load(B_BASE, mask=rk[:, None] < K, other=0.0)

        c = acc.to(C.type.element_ty)
        if BIAS:
             c += bias[:, None]

        rm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        C_ = C + rm[:, None] * stride_cm + rn[None, :] * stride_cn
        mask = (rm < M)[:, None] & (rn < N)[None, :]
        tl.store(C_, c, mask=mask)

    start_iter = total_full_tiles * iters_per_tile + pid * streamk_iters_pcu + tl.minimum(pid, streamk_remainder_iters)
    last_iter = total_full_tiles * iters_per_tile + (pid + 1) * streamk_iters_pcu + tl.minimum(pid + 1, streamk_remainder_iters)
    while start_iter < last_iter:
        remainder = start_iter % iters_per_tile
        end_iter = tl.minimum(start_iter + (iters_per_tile - remainder), last_iter)
        tile_id = start_iter // iters_per_tile
        if GROUP_SIZE_M == 1:
            pid_m = tile_id // num_pid_n
            pid_n = tile_id % num_pid_n
        else:
            num_pid_in_group = GROUP_SIZE_M * num_pid_n
            group_id = tile_id // num_pid_in_group
            first_pid_m = group_id * GROUP_SIZE_M
            group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
            pid_m = first_pid_m + (tile_id % group_size_m)
            pid_n = (tile_id % num_pid_in_group) // group_size_m

        rm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M))%M
        rn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))%N
        rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
        rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)
        A_BASE = A + rm[:, None] * stride_am + rk[None, :] * stride_ak + BLOCK_SIZE_K * stride_ak * remainder
        B_BASE = B + rk[:, None] * stride_bk + rn[None, :] * stride_bn + BLOCK_SIZE_K * stride_bk * remainder

        if BIAS:
            bias_ = bias_ptr + rm * stride_bias
            bias = tl.load(bias_, mask=rm < M, other=0.0)

        acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype)
        for current_iter in range(start_iter, end_iter):
            if EVEN_K:
                a = tl.load(tl.multiple_of(A_BASE, (1, 16)))
                b = tl.load(tl.multiple_of(B_BASE, (16, 1)))
            else:
                global_k_offset = (current_iter % iters_per_tile) * BLOCK_SIZE_K
                k_mask = global_k_offset + rk < K
                a = tl.load(A_BASE, mask=k_mask[None, :], other=0.0)
                b = tl.load(B_BASE, mask=k_mask[:, None], other=0.0)
            acc += tl.dot(a, b)
            A_BASE += BLOCK_SIZE_K * stride_ak
            B_BASE += BLOCK_SIZE_K * stride_bk

        tile_iter = tile_id * iters_per_tile
        if start_iter == tile_iter:
            tile_iter_end = tile_iter + iters_per_tile
            next_pid = pid + 1
            end = end_iter
            while (end < tile_iter_end and next_pid < num_sms):
                while tl.atomic_cas(locks + next_pid, 1, 1) != 1:
              #  while tl.load(locks + next_pid, cache_modifier = ".cg") != 1:
                    pass
                rm1 = tl.arange(0, BLOCK_SIZE_M)
                rn1 = tl.arange(0, BLOCK_SIZE_N)
                rm1 = tl.max_contiguous(tl.multiple_of(rm1, BLOCK_SIZE_M), BLOCK_SIZE_M)
                rn1 = tl.max_contiguous(tl.multiple_of(rn1, BLOCK_SIZE_N), BLOCK_SIZE_N)
                P_ = P + next_pid * BLOCK_SIZE_M * BLOCK_SIZE_N + rm1[:, None] * BLOCK_SIZE_N + rn1[None, :]
                acc += tl.load(tl.multiple_of(P_, (1, 16)))
              #  acc += tl.load(P_)
                end += streamk_iters_pcu + (next_pid < streamk_remainder_iters)
                next_pid += 1

            c = acc.to(C.type.element_ty)
            if BIAS:
                 c += bias[:, None]

            rm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M))%M
            rn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))%N
            rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
            rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)
            C_ = C + rm[:, None] * stride_cm + rn[None, :] * stride_cn
            mask = (rm < M)[:, None] & (rn < N)[None, :]
            tl.store(C_, c, mask=mask)
        else:
            rm1 = tl.arange(0, BLOCK_SIZE_M)
            rn1 = tl.arange(0, BLOCK_SIZE_N)
            rm1 = tl.max_contiguous(tl.multiple_of(rm1, BLOCK_SIZE_M), BLOCK_SIZE_M)
            rn1 = tl.max_contiguous(tl.multiple_of(rn1, BLOCK_SIZE_N), BLOCK_SIZE_N)
            P_ = P + pid * BLOCK_SIZE_M * BLOCK_SIZE_N +  rm1[:, None] * BLOCK_SIZE_N + rn1[None, :]
            tl.store(P_, acc)
            tl.debug_barrier()
            tl.atomic_xchg(locks + pid, 1)
     #       tl.store(locks + pid, 1, cache_modifier=".wt")

        start_iter = end_iter



def matmul_BM64_BN128_BK64_GM32_nW4_nS2_EU0_kP2_mfma16(a, b, c, bias, P, locks, M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    grid = num_sms
    streamk_gemm_BM64_BN128_BK64_GM32_nW4_nS2_EU0_kP2_mfma16[grid,](
        a, b, c, bias, P, locks,
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
        BIAS = False,
        EVEN_K = True
    )
    return c

def try_matmul_config_BM64_BN128_BK64_GM32_nW4_nS2_EU0_kP2_mfma16(a, b, c, bias, M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        locks = torch.zeros((num_sms,), device = "cuda", dtype = torch.int32)
        P = torch.zeros((num_sms,  64*128), device="cuda", dtype=torch.float32)
        matmul_BM64_BN128_BK64_GM32_nW4_nS2_EU0_kP2_mfma16(a, b, c, bias, P, locks, M, N, K, num_sms, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1), bias_stride)
        return True
    except Exception as e:
        print(f'invalid config(runtime): BM64_BN128_BK64_GM32_nW4_nS2_EU0_kP2_mfma16: ', e, flush=True)
        return False




@triton.jit()
def streamk_gemm_BM64_BN128_BK128_GM1_nW4_nS2_EU0_kP2_mfma16(
        A, B, C, bias_ptr, P, locks,
        M, N, K, num_sms,
        stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn, stride_bias,
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
        GROUP_SIZE_M: tl.constexpr, BIAS: tl.constexpr, EVEN_K: tl.constexpr,
):
    pid = tl.program_id(0)
    pid = (pid % 8) * (num_sms // 8) + (pid // 8)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    iters_per_tile = tl.cdiv(K, BLOCK_SIZE_K)
    total_tiles = num_pid_m * num_pid_n
    if num_sms > 0 and total_tiles > num_sms:
        total_streamk_tiles = total_tiles % num_sms
        total_streamk_tiles = total_streamk_tiles + num_sms
        total_full_tiles = total_tiles - total_streamk_tiles
        total_streamk_iters = total_streamk_tiles * iters_per_tile
        streamk_iters_pcu = total_streamk_iters // num_sms
        streamk_remainder_iters = total_streamk_iters % num_sms
    else:
        total_full_tiles = total_tiles
        total_streamk_tiles = 0
        streamk_iters_pcu = 0
        streamk_remainder_iters = 0
        total_streamk_iters = 0

    acc_dtype = tl.float32 if C.type.element_ty != tl.int8 else tl.int32
    rk = tl.arange(0, BLOCK_SIZE_K)

    for tile_id in range(pid, total_full_tiles, num_sms):
        if GROUP_SIZE_M == 1:
            pid_m = tile_id // num_pid_n
            pid_n = tile_id % num_pid_n
        else:
            num_pid_in_group = GROUP_SIZE_M * num_pid_n
            group_id = tile_id // num_pid_in_group
            first_pid_m = group_id * GROUP_SIZE_M
            group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
            pid_m = first_pid_m + (tile_id % group_size_m)
            pid_n = (tile_id % num_pid_in_group) // group_size_m

        rm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M))%M
        rn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))%N
        rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
        rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)
        A_BASE = A + rm[:, None] * stride_am + rk[None, :] * stride_ak
        B_BASE = B + rk[:, None] * stride_bk + rn[None, :] * stride_bn

        if BIAS:
            bias_ = bias_ptr + rm * stride_bias
            bias = tl.load(bias_, mask=rm < M, other=0.0)

        loop_k = tl.cdiv(K, BLOCK_SIZE_K)
        if not EVEN_K:
            loop_k -= 1

        acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype)
        for k in range(0, loop_k):
            a = tl.load(tl.multiple_of(A_BASE, (1, 16)))
            b = tl.load(tl.multiple_of(B_BASE, (16, 1)))
            acc += tl.dot(a, b)
            A_BASE += BLOCK_SIZE_K * stride_ak
            B_BASE += BLOCK_SIZE_K * stride_bk

        if not EVEN_K:
            k = loop_k
            rk = k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
            A_BASE = A + rm[:, None] * stride_am + rk[None, :] * stride_ak
            B_BASE = B + rk[:, None] * stride_bk + rn[None, :] * stride_bn
            a = tl.load(A_BASE, mask=rk[None, :] < K, other=0.0)
            b = tl.load(B_BASE, mask=rk[:, None] < K, other=0.0)

        c = acc.to(C.type.element_ty)
        if BIAS:
             c += bias[:, None]

        rm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        C_ = C + rm[:, None] * stride_cm + rn[None, :] * stride_cn
        mask = (rm < M)[:, None] & (rn < N)[None, :]
        tl.store(C_, c, mask=mask)

    start_iter = total_full_tiles * iters_per_tile + pid * streamk_iters_pcu + tl.minimum(pid, streamk_remainder_iters)
    last_iter = total_full_tiles * iters_per_tile + (pid + 1) * streamk_iters_pcu + tl.minimum(pid + 1, streamk_remainder_iters)
    while start_iter < last_iter:
        remainder = start_iter % iters_per_tile
        end_iter = tl.minimum(start_iter + (iters_per_tile - remainder), last_iter)
        tile_id = start_iter // iters_per_tile
        if GROUP_SIZE_M == 1:
            pid_m = tile_id // num_pid_n
            pid_n = tile_id % num_pid_n
        else:
            num_pid_in_group = GROUP_SIZE_M * num_pid_n
            group_id = tile_id // num_pid_in_group
            first_pid_m = group_id * GROUP_SIZE_M
            group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
            pid_m = first_pid_m + (tile_id % group_size_m)
            pid_n = (tile_id % num_pid_in_group) // group_size_m

        rm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M))%M
        rn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))%N
        rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
        rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)
        A_BASE = A + rm[:, None] * stride_am + rk[None, :] * stride_ak + BLOCK_SIZE_K * stride_ak * remainder
        B_BASE = B + rk[:, None] * stride_bk + rn[None, :] * stride_bn + BLOCK_SIZE_K * stride_bk * remainder

        if BIAS:
            bias_ = bias_ptr + rm * stride_bias
            bias = tl.load(bias_, mask=rm < M, other=0.0)

        acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype)
        for current_iter in range(start_iter, end_iter):
            if EVEN_K:
                a = tl.load(tl.multiple_of(A_BASE, (1, 16)))
                b = tl.load(tl.multiple_of(B_BASE, (16, 1)))
            else:
                global_k_offset = (current_iter % iters_per_tile) * BLOCK_SIZE_K
                k_mask = global_k_offset + rk < K
                a = tl.load(A_BASE, mask=k_mask[None, :], other=0.0)
                b = tl.load(B_BASE, mask=k_mask[:, None], other=0.0)
            acc += tl.dot(a, b)
            A_BASE += BLOCK_SIZE_K * stride_ak
            B_BASE += BLOCK_SIZE_K * stride_bk

        tile_iter = tile_id * iters_per_tile
        if start_iter == tile_iter:
            tile_iter_end = tile_iter + iters_per_tile
            next_pid = pid + 1
            end = end_iter
            while (end < tile_iter_end and next_pid < num_sms):
                while tl.atomic_cas(locks + next_pid, 1, 1) != 1:
              #  while tl.load(locks + next_pid, cache_modifier = ".cg") != 1:
                    pass
                rm1 = tl.arange(0, BLOCK_SIZE_M)
                rn1 = tl.arange(0, BLOCK_SIZE_N)
                rm1 = tl.max_contiguous(tl.multiple_of(rm1, BLOCK_SIZE_M), BLOCK_SIZE_M)
                rn1 = tl.max_contiguous(tl.multiple_of(rn1, BLOCK_SIZE_N), BLOCK_SIZE_N)
                P_ = P + next_pid * BLOCK_SIZE_M * BLOCK_SIZE_N + rm1[:, None] * BLOCK_SIZE_N + rn1[None, :]
                acc += tl.load(tl.multiple_of(P_, (1, 16)))
              #  acc += tl.load(P_)
                end += streamk_iters_pcu + (next_pid < streamk_remainder_iters)
                next_pid += 1

            c = acc.to(C.type.element_ty)
            if BIAS:
                 c += bias[:, None]

            rm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M))%M
            rn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))%N
            rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
            rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)
            C_ = C + rm[:, None] * stride_cm + rn[None, :] * stride_cn
            mask = (rm < M)[:, None] & (rn < N)[None, :]
            tl.store(C_, c, mask=mask)
        else:
            rm1 = tl.arange(0, BLOCK_SIZE_M)
            rn1 = tl.arange(0, BLOCK_SIZE_N)
            rm1 = tl.max_contiguous(tl.multiple_of(rm1, BLOCK_SIZE_M), BLOCK_SIZE_M)
            rn1 = tl.max_contiguous(tl.multiple_of(rn1, BLOCK_SIZE_N), BLOCK_SIZE_N)
            P_ = P + pid * BLOCK_SIZE_M * BLOCK_SIZE_N +  rm1[:, None] * BLOCK_SIZE_N + rn1[None, :]
            tl.store(P_, acc)
            tl.debug_barrier()
            tl.atomic_xchg(locks + pid, 1)
     #       tl.store(locks + pid, 1, cache_modifier=".wt")

        start_iter = end_iter



def matmul_BM64_BN128_BK128_GM1_nW4_nS2_EU0_kP2_mfma16(a, b, c, bias, P, locks, M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    grid = num_sms
    streamk_gemm_BM64_BN128_BK128_GM1_nW4_nS2_EU0_kP2_mfma16[grid,](
        a, b, c, bias, P, locks,
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
        BIAS = False,
        EVEN_K = False
    )
    return c

def try_matmul_config_BM64_BN128_BK128_GM1_nW4_nS2_EU0_kP2_mfma16(a, b, c, bias, M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        locks = torch.zeros((num_sms,), device = "cuda", dtype = torch.int32)
        P = torch.zeros((num_sms,  64*128), device="cuda", dtype=torch.float32)
        matmul_BM64_BN128_BK128_GM1_nW4_nS2_EU0_kP2_mfma16(a, b, c, bias, P, locks, M, N, K, num_sms, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1), bias_stride)
        return True
    except Exception as e:
        print(f'invalid config(runtime): BM64_BN128_BK128_GM1_nW4_nS2_EU0_kP2_mfma16: ', e, flush=True)
        return False




@triton.jit()
def streamk_gemm_BM64_BN128_BK128_GM4_nW8_nS2_EU0_kP2_mfma16(
        A, B, C, bias_ptr, P, locks,
        M, N, K, num_sms,
        stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn, stride_bias,
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
        GROUP_SIZE_M: tl.constexpr, BIAS: tl.constexpr, EVEN_K: tl.constexpr,
):
    pid = tl.program_id(0)
    pid = (pid % 8) * (num_sms // 8) + (pid // 8)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    iters_per_tile = tl.cdiv(K, BLOCK_SIZE_K)
    total_tiles = num_pid_m * num_pid_n
    if num_sms > 0 and total_tiles > num_sms:
        total_streamk_tiles = total_tiles % num_sms
        total_streamk_tiles = total_streamk_tiles + num_sms
        total_full_tiles = total_tiles - total_streamk_tiles
        total_streamk_iters = total_streamk_tiles * iters_per_tile
        streamk_iters_pcu = total_streamk_iters // num_sms
        streamk_remainder_iters = total_streamk_iters % num_sms
    else:
        total_full_tiles = total_tiles
        total_streamk_tiles = 0
        streamk_iters_pcu = 0
        streamk_remainder_iters = 0
        total_streamk_iters = 0

    acc_dtype = tl.float32 if C.type.element_ty != tl.int8 else tl.int32
    rk = tl.arange(0, BLOCK_SIZE_K)

    for tile_id in range(pid, total_full_tiles, num_sms):
        if GROUP_SIZE_M == 1:
            pid_m = tile_id // num_pid_n
            pid_n = tile_id % num_pid_n
        else:
            num_pid_in_group = GROUP_SIZE_M * num_pid_n
            group_id = tile_id // num_pid_in_group
            first_pid_m = group_id * GROUP_SIZE_M
            group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
            pid_m = first_pid_m + (tile_id % group_size_m)
            pid_n = (tile_id % num_pid_in_group) // group_size_m

        rm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M))%M
        rn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))%N
        rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
        rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)
        A_BASE = A + rm[:, None] * stride_am + rk[None, :] * stride_ak
        B_BASE = B + rk[:, None] * stride_bk + rn[None, :] * stride_bn

        if BIAS:
            bias_ = bias_ptr + rm * stride_bias
            bias = tl.load(bias_, mask=rm < M, other=0.0)

        loop_k = tl.cdiv(K, BLOCK_SIZE_K)
        if not EVEN_K:
            loop_k -= 1

        acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype)
        for k in range(0, loop_k):
            a = tl.load(tl.multiple_of(A_BASE, (1, 16)))
            b = tl.load(tl.multiple_of(B_BASE, (16, 1)))
            acc += tl.dot(a, b)
            A_BASE += BLOCK_SIZE_K * stride_ak
            B_BASE += BLOCK_SIZE_K * stride_bk

        if not EVEN_K:
            k = loop_k
            rk = k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
            A_BASE = A + rm[:, None] * stride_am + rk[None, :] * stride_ak
            B_BASE = B + rk[:, None] * stride_bk + rn[None, :] * stride_bn
            a = tl.load(A_BASE, mask=rk[None, :] < K, other=0.0)
            b = tl.load(B_BASE, mask=rk[:, None] < K, other=0.0)

        c = acc.to(C.type.element_ty)
        if BIAS:
             c += bias[:, None]

        rm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        C_ = C + rm[:, None] * stride_cm + rn[None, :] * stride_cn
        mask = (rm < M)[:, None] & (rn < N)[None, :]
        tl.store(C_, c, mask=mask)

    start_iter = total_full_tiles * iters_per_tile + pid * streamk_iters_pcu + tl.minimum(pid, streamk_remainder_iters)
    last_iter = total_full_tiles * iters_per_tile + (pid + 1) * streamk_iters_pcu + tl.minimum(pid + 1, streamk_remainder_iters)
    while start_iter < last_iter:
        remainder = start_iter % iters_per_tile
        end_iter = tl.minimum(start_iter + (iters_per_tile - remainder), last_iter)
        tile_id = start_iter // iters_per_tile
        if GROUP_SIZE_M == 1:
            pid_m = tile_id // num_pid_n
            pid_n = tile_id % num_pid_n
        else:
            num_pid_in_group = GROUP_SIZE_M * num_pid_n
            group_id = tile_id // num_pid_in_group
            first_pid_m = group_id * GROUP_SIZE_M
            group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
            pid_m = first_pid_m + (tile_id % group_size_m)
            pid_n = (tile_id % num_pid_in_group) // group_size_m

        rm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M))%M
        rn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))%N
        rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
        rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)
        A_BASE = A + rm[:, None] * stride_am + rk[None, :] * stride_ak + BLOCK_SIZE_K * stride_ak * remainder
        B_BASE = B + rk[:, None] * stride_bk + rn[None, :] * stride_bn + BLOCK_SIZE_K * stride_bk * remainder

        if BIAS:
            bias_ = bias_ptr + rm * stride_bias
            bias = tl.load(bias_, mask=rm < M, other=0.0)

        acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype)
        for current_iter in range(start_iter, end_iter):
            if EVEN_K:
                a = tl.load(tl.multiple_of(A_BASE, (1, 16)))
                b = tl.load(tl.multiple_of(B_BASE, (16, 1)))
            else:
                global_k_offset = (current_iter % iters_per_tile) * BLOCK_SIZE_K
                k_mask = global_k_offset + rk < K
                a = tl.load(A_BASE, mask=k_mask[None, :], other=0.0)
                b = tl.load(B_BASE, mask=k_mask[:, None], other=0.0)
            acc += tl.dot(a, b)
            A_BASE += BLOCK_SIZE_K * stride_ak
            B_BASE += BLOCK_SIZE_K * stride_bk

        tile_iter = tile_id * iters_per_tile
        if start_iter == tile_iter:
            tile_iter_end = tile_iter + iters_per_tile
            next_pid = pid + 1
            end = end_iter
            while (end < tile_iter_end and next_pid < num_sms):
                while tl.atomic_cas(locks + next_pid, 1, 1) != 1:
              #  while tl.load(locks + next_pid, cache_modifier = ".cg") != 1:
                    pass
                rm1 = tl.arange(0, BLOCK_SIZE_M)
                rn1 = tl.arange(0, BLOCK_SIZE_N)
                rm1 = tl.max_contiguous(tl.multiple_of(rm1, BLOCK_SIZE_M), BLOCK_SIZE_M)
                rn1 = tl.max_contiguous(tl.multiple_of(rn1, BLOCK_SIZE_N), BLOCK_SIZE_N)
                P_ = P + next_pid * BLOCK_SIZE_M * BLOCK_SIZE_N + rm1[:, None] * BLOCK_SIZE_N + rn1[None, :]
                acc += tl.load(tl.multiple_of(P_, (1, 16)))
              #  acc += tl.load(P_)
                end += streamk_iters_pcu + (next_pid < streamk_remainder_iters)
                next_pid += 1

            c = acc.to(C.type.element_ty)
            if BIAS:
                 c += bias[:, None]

            rm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M))%M
            rn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))%N
            rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
            rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)
            C_ = C + rm[:, None] * stride_cm + rn[None, :] * stride_cn
            mask = (rm < M)[:, None] & (rn < N)[None, :]
            tl.store(C_, c, mask=mask)
        else:
            rm1 = tl.arange(0, BLOCK_SIZE_M)
            rn1 = tl.arange(0, BLOCK_SIZE_N)
            rm1 = tl.max_contiguous(tl.multiple_of(rm1, BLOCK_SIZE_M), BLOCK_SIZE_M)
            rn1 = tl.max_contiguous(tl.multiple_of(rn1, BLOCK_SIZE_N), BLOCK_SIZE_N)
            P_ = P + pid * BLOCK_SIZE_M * BLOCK_SIZE_N +  rm1[:, None] * BLOCK_SIZE_N + rn1[None, :]
            tl.store(P_, acc)
            tl.debug_barrier()
            tl.atomic_xchg(locks + pid, 1)
     #       tl.store(locks + pid, 1, cache_modifier=".wt")

        start_iter = end_iter



def matmul_BM64_BN128_BK128_GM4_nW8_nS2_EU0_kP2_mfma16(a, b, c, bias, P, locks, M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    grid = num_sms
    streamk_gemm_BM64_BN128_BK128_GM4_nW8_nS2_EU0_kP2_mfma16[grid,](
        a, b, c, bias, P, locks,
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
        BIAS = False,
        EVEN_K = False
    )
    return c

def try_matmul_config_BM64_BN128_BK128_GM4_nW8_nS2_EU0_kP2_mfma16(a, b, c, bias, M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        locks = torch.zeros((num_sms,), device = "cuda", dtype = torch.int32)
        P = torch.zeros((num_sms,  64*128), device="cuda", dtype=torch.float32)
        matmul_BM64_BN128_BK128_GM4_nW8_nS2_EU0_kP2_mfma16(a, b, c, bias, P, locks, M, N, K, num_sms, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1), bias_stride)
        return True
    except Exception as e:
        print(f'invalid config(runtime): BM64_BN128_BK128_GM4_nW8_nS2_EU0_kP2_mfma16: ', e, flush=True)
        return False




@triton.jit()
def streamk_gemm_BM64_BN256_BK64_GM8_nW4_nS2_EU0_kP2_mfma16(
        A, B, C, bias_ptr, P, locks,
        M, N, K, num_sms,
        stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn, stride_bias,
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
        GROUP_SIZE_M: tl.constexpr, BIAS: tl.constexpr, EVEN_K: tl.constexpr,
):
    pid = tl.program_id(0)
    pid = (pid % 8) * (num_sms // 8) + (pid // 8)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    iters_per_tile = tl.cdiv(K, BLOCK_SIZE_K)
    total_tiles = num_pid_m * num_pid_n
    if num_sms > 0 and total_tiles > num_sms:
        total_streamk_tiles = total_tiles % num_sms
        total_streamk_tiles = total_streamk_tiles + num_sms
        total_full_tiles = total_tiles - total_streamk_tiles
        total_streamk_iters = total_streamk_tiles * iters_per_tile
        streamk_iters_pcu = total_streamk_iters // num_sms
        streamk_remainder_iters = total_streamk_iters % num_sms
    else:
        total_full_tiles = total_tiles
        total_streamk_tiles = 0
        streamk_iters_pcu = 0
        streamk_remainder_iters = 0
        total_streamk_iters = 0

    acc_dtype = tl.float32 if C.type.element_ty != tl.int8 else tl.int32
    rk = tl.arange(0, BLOCK_SIZE_K)

    for tile_id in range(pid, total_full_tiles, num_sms):
        if GROUP_SIZE_M == 1:
            pid_m = tile_id // num_pid_n
            pid_n = tile_id % num_pid_n
        else:
            num_pid_in_group = GROUP_SIZE_M * num_pid_n
            group_id = tile_id // num_pid_in_group
            first_pid_m = group_id * GROUP_SIZE_M
            group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
            pid_m = first_pid_m + (tile_id % group_size_m)
            pid_n = (tile_id % num_pid_in_group) // group_size_m

        rm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M))%M
        rn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))%N
        rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
        rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)
        A_BASE = A + rm[:, None] * stride_am + rk[None, :] * stride_ak
        B_BASE = B + rk[:, None] * stride_bk + rn[None, :] * stride_bn

        if BIAS:
            bias_ = bias_ptr + rm * stride_bias
            bias = tl.load(bias_, mask=rm < M, other=0.0)

        loop_k = tl.cdiv(K, BLOCK_SIZE_K)
        if not EVEN_K:
            loop_k -= 1

        acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype)
        for k in range(0, loop_k):
            a = tl.load(tl.multiple_of(A_BASE, (1, 16)))
            b = tl.load(tl.multiple_of(B_BASE, (16, 1)))
            acc += tl.dot(a, b)
            A_BASE += BLOCK_SIZE_K * stride_ak
            B_BASE += BLOCK_SIZE_K * stride_bk

        if not EVEN_K:
            k = loop_k
            rk = k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
            A_BASE = A + rm[:, None] * stride_am + rk[None, :] * stride_ak
            B_BASE = B + rk[:, None] * stride_bk + rn[None, :] * stride_bn
            a = tl.load(A_BASE, mask=rk[None, :] < K, other=0.0)
            b = tl.load(B_BASE, mask=rk[:, None] < K, other=0.0)

        c = acc.to(C.type.element_ty)
        if BIAS:
             c += bias[:, None]

        rm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        C_ = C + rm[:, None] * stride_cm + rn[None, :] * stride_cn
        mask = (rm < M)[:, None] & (rn < N)[None, :]
        tl.store(C_, c, mask=mask)

    start_iter = total_full_tiles * iters_per_tile + pid * streamk_iters_pcu + tl.minimum(pid, streamk_remainder_iters)
    last_iter = total_full_tiles * iters_per_tile + (pid + 1) * streamk_iters_pcu + tl.minimum(pid + 1, streamk_remainder_iters)
    while start_iter < last_iter:
        remainder = start_iter % iters_per_tile
        end_iter = tl.minimum(start_iter + (iters_per_tile - remainder), last_iter)
        tile_id = start_iter // iters_per_tile
        if GROUP_SIZE_M == 1:
            pid_m = tile_id // num_pid_n
            pid_n = tile_id % num_pid_n
        else:
            num_pid_in_group = GROUP_SIZE_M * num_pid_n
            group_id = tile_id // num_pid_in_group
            first_pid_m = group_id * GROUP_SIZE_M
            group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
            pid_m = first_pid_m + (tile_id % group_size_m)
            pid_n = (tile_id % num_pid_in_group) // group_size_m

        rm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M))%M
        rn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))%N
        rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
        rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)
        A_BASE = A + rm[:, None] * stride_am + rk[None, :] * stride_ak + BLOCK_SIZE_K * stride_ak * remainder
        B_BASE = B + rk[:, None] * stride_bk + rn[None, :] * stride_bn + BLOCK_SIZE_K * stride_bk * remainder

        if BIAS:
            bias_ = bias_ptr + rm * stride_bias
            bias = tl.load(bias_, mask=rm < M, other=0.0)

        acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype)
        for current_iter in range(start_iter, end_iter):
            if EVEN_K:
                a = tl.load(tl.multiple_of(A_BASE, (1, 16)))
                b = tl.load(tl.multiple_of(B_BASE, (16, 1)))
            else:
                global_k_offset = (current_iter % iters_per_tile) * BLOCK_SIZE_K
                k_mask = global_k_offset + rk < K
                a = tl.load(A_BASE, mask=k_mask[None, :], other=0.0)
                b = tl.load(B_BASE, mask=k_mask[:, None], other=0.0)
            acc += tl.dot(a, b)
            A_BASE += BLOCK_SIZE_K * stride_ak
            B_BASE += BLOCK_SIZE_K * stride_bk

        tile_iter = tile_id * iters_per_tile
        if start_iter == tile_iter:
            tile_iter_end = tile_iter + iters_per_tile
            next_pid = pid + 1
            end = end_iter
            while (end < tile_iter_end and next_pid < num_sms):
                while tl.atomic_cas(locks + next_pid, 1, 1) != 1:
              #  while tl.load(locks + next_pid, cache_modifier = ".cg") != 1:
                    pass
                rm1 = tl.arange(0, BLOCK_SIZE_M)
                rn1 = tl.arange(0, BLOCK_SIZE_N)
                rm1 = tl.max_contiguous(tl.multiple_of(rm1, BLOCK_SIZE_M), BLOCK_SIZE_M)
                rn1 = tl.max_contiguous(tl.multiple_of(rn1, BLOCK_SIZE_N), BLOCK_SIZE_N)
                P_ = P + next_pid * BLOCK_SIZE_M * BLOCK_SIZE_N + rm1[:, None] * BLOCK_SIZE_N + rn1[None, :]
                acc += tl.load(tl.multiple_of(P_, (1, 16)))
              #  acc += tl.load(P_)
                end += streamk_iters_pcu + (next_pid < streamk_remainder_iters)
                next_pid += 1

            c = acc.to(C.type.element_ty)
            if BIAS:
                 c += bias[:, None]

            rm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M))%M
            rn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))%N
            rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
            rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)
            C_ = C + rm[:, None] * stride_cm + rn[None, :] * stride_cn
            mask = (rm < M)[:, None] & (rn < N)[None, :]
            tl.store(C_, c, mask=mask)
        else:
            rm1 = tl.arange(0, BLOCK_SIZE_M)
            rn1 = tl.arange(0, BLOCK_SIZE_N)
            rm1 = tl.max_contiguous(tl.multiple_of(rm1, BLOCK_SIZE_M), BLOCK_SIZE_M)
            rn1 = tl.max_contiguous(tl.multiple_of(rn1, BLOCK_SIZE_N), BLOCK_SIZE_N)
            P_ = P + pid * BLOCK_SIZE_M * BLOCK_SIZE_N +  rm1[:, None] * BLOCK_SIZE_N + rn1[None, :]
            tl.store(P_, acc)
            tl.debug_barrier()
            tl.atomic_xchg(locks + pid, 1)
     #       tl.store(locks + pid, 1, cache_modifier=".wt")

        start_iter = end_iter



def matmul_BM64_BN256_BK64_GM8_nW4_nS2_EU0_kP2_mfma16(a, b, c, bias, P, locks, M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    grid = num_sms
    streamk_gemm_BM64_BN256_BK64_GM8_nW4_nS2_EU0_kP2_mfma16[grid,](
        a, b, c, bias, P, locks,
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
        BIAS = False,
        EVEN_K = True
    )
    return c

def try_matmul_config_BM64_BN256_BK64_GM8_nW4_nS2_EU0_kP2_mfma16(a, b, c, bias, M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        locks = torch.zeros((num_sms,), device = "cuda", dtype = torch.int32)
        P = torch.zeros((num_sms,  64*256), device="cuda", dtype=torch.float32)
        matmul_BM64_BN256_BK64_GM8_nW4_nS2_EU0_kP2_mfma16(a, b, c, bias, P, locks, M, N, K, num_sms, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1), bias_stride)
        return True
    except Exception as e:
        print(f'invalid config(runtime): BM64_BN256_BK64_GM8_nW4_nS2_EU0_kP2_mfma16: ', e, flush=True)
        return False




@triton.jit()
def streamk_gemm_BM64_BN256_BK64_GM16_nW8_nS2_EU0_kP2_mfma16(
        A, B, C, bias_ptr, P, locks,
        M, N, K, num_sms,
        stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn, stride_bias,
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
        GROUP_SIZE_M: tl.constexpr, BIAS: tl.constexpr, EVEN_K: tl.constexpr,
):
    pid = tl.program_id(0)
    pid = (pid % 8) * (num_sms // 8) + (pid // 8)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    iters_per_tile = tl.cdiv(K, BLOCK_SIZE_K)
    total_tiles = num_pid_m * num_pid_n
    if num_sms > 0 and total_tiles > num_sms:
        total_streamk_tiles = total_tiles % num_sms
        total_streamk_tiles = total_streamk_tiles + num_sms
        total_full_tiles = total_tiles - total_streamk_tiles
        total_streamk_iters = total_streamk_tiles * iters_per_tile
        streamk_iters_pcu = total_streamk_iters // num_sms
        streamk_remainder_iters = total_streamk_iters % num_sms
    else:
        total_full_tiles = total_tiles
        total_streamk_tiles = 0
        streamk_iters_pcu = 0
        streamk_remainder_iters = 0
        total_streamk_iters = 0

    acc_dtype = tl.float32 if C.type.element_ty != tl.int8 else tl.int32
    rk = tl.arange(0, BLOCK_SIZE_K)

    for tile_id in range(pid, total_full_tiles, num_sms):
        if GROUP_SIZE_M == 1:
            pid_m = tile_id // num_pid_n
            pid_n = tile_id % num_pid_n
        else:
            num_pid_in_group = GROUP_SIZE_M * num_pid_n
            group_id = tile_id // num_pid_in_group
            first_pid_m = group_id * GROUP_SIZE_M
            group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
            pid_m = first_pid_m + (tile_id % group_size_m)
            pid_n = (tile_id % num_pid_in_group) // group_size_m

        rm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M))%M
        rn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))%N
        rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
        rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)
        A_BASE = A + rm[:, None] * stride_am + rk[None, :] * stride_ak
        B_BASE = B + rk[:, None] * stride_bk + rn[None, :] * stride_bn

        if BIAS:
            bias_ = bias_ptr + rm * stride_bias
            bias = tl.load(bias_, mask=rm < M, other=0.0)

        loop_k = tl.cdiv(K, BLOCK_SIZE_K)
        if not EVEN_K:
            loop_k -= 1

        acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype)
        for k in range(0, loop_k):
            a = tl.load(tl.multiple_of(A_BASE, (1, 16)))
            b = tl.load(tl.multiple_of(B_BASE, (16, 1)))
            acc += tl.dot(a, b)
            A_BASE += BLOCK_SIZE_K * stride_ak
            B_BASE += BLOCK_SIZE_K * stride_bk

        if not EVEN_K:
            k = loop_k
            rk = k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
            A_BASE = A + rm[:, None] * stride_am + rk[None, :] * stride_ak
            B_BASE = B + rk[:, None] * stride_bk + rn[None, :] * stride_bn
            a = tl.load(A_BASE, mask=rk[None, :] < K, other=0.0)
            b = tl.load(B_BASE, mask=rk[:, None] < K, other=0.0)

        c = acc.to(C.type.element_ty)
        if BIAS:
             c += bias[:, None]

        rm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        C_ = C + rm[:, None] * stride_cm + rn[None, :] * stride_cn
        mask = (rm < M)[:, None] & (rn < N)[None, :]
        tl.store(C_, c, mask=mask)

    start_iter = total_full_tiles * iters_per_tile + pid * streamk_iters_pcu + tl.minimum(pid, streamk_remainder_iters)
    last_iter = total_full_tiles * iters_per_tile + (pid + 1) * streamk_iters_pcu + tl.minimum(pid + 1, streamk_remainder_iters)
    while start_iter < last_iter:
        remainder = start_iter % iters_per_tile
        end_iter = tl.minimum(start_iter + (iters_per_tile - remainder), last_iter)
        tile_id = start_iter // iters_per_tile
        if GROUP_SIZE_M == 1:
            pid_m = tile_id // num_pid_n
            pid_n = tile_id % num_pid_n
        else:
            num_pid_in_group = GROUP_SIZE_M * num_pid_n
            group_id = tile_id // num_pid_in_group
            first_pid_m = group_id * GROUP_SIZE_M
            group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
            pid_m = first_pid_m + (tile_id % group_size_m)
            pid_n = (tile_id % num_pid_in_group) // group_size_m

        rm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M))%M
        rn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))%N
        rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
        rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)
        A_BASE = A + rm[:, None] * stride_am + rk[None, :] * stride_ak + BLOCK_SIZE_K * stride_ak * remainder
        B_BASE = B + rk[:, None] * stride_bk + rn[None, :] * stride_bn + BLOCK_SIZE_K * stride_bk * remainder

        if BIAS:
            bias_ = bias_ptr + rm * stride_bias
            bias = tl.load(bias_, mask=rm < M, other=0.0)

        acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype)
        for current_iter in range(start_iter, end_iter):
            if EVEN_K:
                a = tl.load(tl.multiple_of(A_BASE, (1, 16)))
                b = tl.load(tl.multiple_of(B_BASE, (16, 1)))
            else:
                global_k_offset = (current_iter % iters_per_tile) * BLOCK_SIZE_K
                k_mask = global_k_offset + rk < K
                a = tl.load(A_BASE, mask=k_mask[None, :], other=0.0)
                b = tl.load(B_BASE, mask=k_mask[:, None], other=0.0)
            acc += tl.dot(a, b)
            A_BASE += BLOCK_SIZE_K * stride_ak
            B_BASE += BLOCK_SIZE_K * stride_bk

        tile_iter = tile_id * iters_per_tile
        if start_iter == tile_iter:
            tile_iter_end = tile_iter + iters_per_tile
            next_pid = pid + 1
            end = end_iter
            while (end < tile_iter_end and next_pid < num_sms):
                while tl.atomic_cas(locks + next_pid, 1, 1) != 1:
              #  while tl.load(locks + next_pid, cache_modifier = ".cg") != 1:
                    pass
                rm1 = tl.arange(0, BLOCK_SIZE_M)
                rn1 = tl.arange(0, BLOCK_SIZE_N)
                rm1 = tl.max_contiguous(tl.multiple_of(rm1, BLOCK_SIZE_M), BLOCK_SIZE_M)
                rn1 = tl.max_contiguous(tl.multiple_of(rn1, BLOCK_SIZE_N), BLOCK_SIZE_N)
                P_ = P + next_pid * BLOCK_SIZE_M * BLOCK_SIZE_N + rm1[:, None] * BLOCK_SIZE_N + rn1[None, :]
                acc += tl.load(tl.multiple_of(P_, (1, 16)))
              #  acc += tl.load(P_)
                end += streamk_iters_pcu + (next_pid < streamk_remainder_iters)
                next_pid += 1

            c = acc.to(C.type.element_ty)
            if BIAS:
                 c += bias[:, None]

            rm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M))%M
            rn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))%N
            rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
            rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)
            C_ = C + rm[:, None] * stride_cm + rn[None, :] * stride_cn
            mask = (rm < M)[:, None] & (rn < N)[None, :]
            tl.store(C_, c, mask=mask)
        else:
            rm1 = tl.arange(0, BLOCK_SIZE_M)
            rn1 = tl.arange(0, BLOCK_SIZE_N)
            rm1 = tl.max_contiguous(tl.multiple_of(rm1, BLOCK_SIZE_M), BLOCK_SIZE_M)
            rn1 = tl.max_contiguous(tl.multiple_of(rn1, BLOCK_SIZE_N), BLOCK_SIZE_N)
            P_ = P + pid * BLOCK_SIZE_M * BLOCK_SIZE_N +  rm1[:, None] * BLOCK_SIZE_N + rn1[None, :]
            tl.store(P_, acc)
            tl.debug_barrier()
            tl.atomic_xchg(locks + pid, 1)
     #       tl.store(locks + pid, 1, cache_modifier=".wt")

        start_iter = end_iter



def matmul_BM64_BN256_BK64_GM16_nW8_nS2_EU0_kP2_mfma16(a, b, c, bias, P, locks, M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    grid = num_sms
    streamk_gemm_BM64_BN256_BK64_GM16_nW8_nS2_EU0_kP2_mfma16[grid,](
        a, b, c, bias, P, locks,
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
        BIAS = False,
        EVEN_K = True
    )
    return c

def try_matmul_config_BM64_BN256_BK64_GM16_nW8_nS2_EU0_kP2_mfma16(a, b, c, bias, M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        locks = torch.zeros((num_sms,), device = "cuda", dtype = torch.int32)
        P = torch.zeros((num_sms,  64*256), device="cuda", dtype=torch.float32)
        matmul_BM64_BN256_BK64_GM16_nW8_nS2_EU0_kP2_mfma16(a, b, c, bias, P, locks, M, N, K, num_sms, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1), bias_stride)
        return True
    except Exception as e:
        print(f'invalid config(runtime): BM64_BN256_BK64_GM16_nW8_nS2_EU0_kP2_mfma16: ', e, flush=True)
        return False




@triton.jit()
def streamk_gemm_BM128_BN64_BK64_GM32_nW4_nS2_EU0_kP2_mfma16(
        A, B, C, bias_ptr, P, locks,
        M, N, K, num_sms,
        stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn, stride_bias,
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
        GROUP_SIZE_M: tl.constexpr, BIAS: tl.constexpr, EVEN_K: tl.constexpr,
):
    pid = tl.program_id(0)
    pid = (pid % 8) * (num_sms // 8) + (pid // 8)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    iters_per_tile = tl.cdiv(K, BLOCK_SIZE_K)
    total_tiles = num_pid_m * num_pid_n
    if num_sms > 0 and total_tiles > num_sms:
        total_streamk_tiles = total_tiles % num_sms
        total_streamk_tiles = total_streamk_tiles + num_sms
        total_full_tiles = total_tiles - total_streamk_tiles
        total_streamk_iters = total_streamk_tiles * iters_per_tile
        streamk_iters_pcu = total_streamk_iters // num_sms
        streamk_remainder_iters = total_streamk_iters % num_sms
    else:
        total_full_tiles = total_tiles
        total_streamk_tiles = 0
        streamk_iters_pcu = 0
        streamk_remainder_iters = 0
        total_streamk_iters = 0

    acc_dtype = tl.float32 if C.type.element_ty != tl.int8 else tl.int32
    rk = tl.arange(0, BLOCK_SIZE_K)

    for tile_id in range(pid, total_full_tiles, num_sms):
        if GROUP_SIZE_M == 1:
            pid_m = tile_id // num_pid_n
            pid_n = tile_id % num_pid_n
        else:
            num_pid_in_group = GROUP_SIZE_M * num_pid_n
            group_id = tile_id // num_pid_in_group
            first_pid_m = group_id * GROUP_SIZE_M
            group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
            pid_m = first_pid_m + (tile_id % group_size_m)
            pid_n = (tile_id % num_pid_in_group) // group_size_m

        rm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M))%M
        rn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))%N
        rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
        rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)
        A_BASE = A + rm[:, None] * stride_am + rk[None, :] * stride_ak
        B_BASE = B + rk[:, None] * stride_bk + rn[None, :] * stride_bn

        if BIAS:
            bias_ = bias_ptr + rm * stride_bias
            bias = tl.load(bias_, mask=rm < M, other=0.0)

        loop_k = tl.cdiv(K, BLOCK_SIZE_K)
        if not EVEN_K:
            loop_k -= 1

        acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype)
        for k in range(0, loop_k):
            a = tl.load(tl.multiple_of(A_BASE, (1, 16)))
            b = tl.load(tl.multiple_of(B_BASE, (16, 1)))
            acc += tl.dot(a, b)
            A_BASE += BLOCK_SIZE_K * stride_ak
            B_BASE += BLOCK_SIZE_K * stride_bk

        if not EVEN_K:
            k = loop_k
            rk = k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
            A_BASE = A + rm[:, None] * stride_am + rk[None, :] * stride_ak
            B_BASE = B + rk[:, None] * stride_bk + rn[None, :] * stride_bn
            a = tl.load(A_BASE, mask=rk[None, :] < K, other=0.0)
            b = tl.load(B_BASE, mask=rk[:, None] < K, other=0.0)

        c = acc.to(C.type.element_ty)
        if BIAS:
             c += bias[:, None]

        rm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        C_ = C + rm[:, None] * stride_cm + rn[None, :] * stride_cn
        mask = (rm < M)[:, None] & (rn < N)[None, :]
        tl.store(C_, c, mask=mask)

    start_iter = total_full_tiles * iters_per_tile + pid * streamk_iters_pcu + tl.minimum(pid, streamk_remainder_iters)
    last_iter = total_full_tiles * iters_per_tile + (pid + 1) * streamk_iters_pcu + tl.minimum(pid + 1, streamk_remainder_iters)
    while start_iter < last_iter:
        remainder = start_iter % iters_per_tile
        end_iter = tl.minimum(start_iter + (iters_per_tile - remainder), last_iter)
        tile_id = start_iter // iters_per_tile
        if GROUP_SIZE_M == 1:
            pid_m = tile_id // num_pid_n
            pid_n = tile_id % num_pid_n
        else:
            num_pid_in_group = GROUP_SIZE_M * num_pid_n
            group_id = tile_id // num_pid_in_group
            first_pid_m = group_id * GROUP_SIZE_M
            group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
            pid_m = first_pid_m + (tile_id % group_size_m)
            pid_n = (tile_id % num_pid_in_group) // group_size_m

        rm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M))%M
        rn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))%N
        rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
        rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)
        A_BASE = A + rm[:, None] * stride_am + rk[None, :] * stride_ak + BLOCK_SIZE_K * stride_ak * remainder
        B_BASE = B + rk[:, None] * stride_bk + rn[None, :] * stride_bn + BLOCK_SIZE_K * stride_bk * remainder

        if BIAS:
            bias_ = bias_ptr + rm * stride_bias
            bias = tl.load(bias_, mask=rm < M, other=0.0)

        acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype)
        for current_iter in range(start_iter, end_iter):
            if EVEN_K:
                a = tl.load(tl.multiple_of(A_BASE, (1, 16)))
                b = tl.load(tl.multiple_of(B_BASE, (16, 1)))
            else:
                global_k_offset = (current_iter % iters_per_tile) * BLOCK_SIZE_K
                k_mask = global_k_offset + rk < K
                a = tl.load(A_BASE, mask=k_mask[None, :], other=0.0)
                b = tl.load(B_BASE, mask=k_mask[:, None], other=0.0)
            acc += tl.dot(a, b)
            A_BASE += BLOCK_SIZE_K * stride_ak
            B_BASE += BLOCK_SIZE_K * stride_bk

        tile_iter = tile_id * iters_per_tile
        if start_iter == tile_iter:
            tile_iter_end = tile_iter + iters_per_tile
            next_pid = pid + 1
            end = end_iter
            while (end < tile_iter_end and next_pid < num_sms):
                while tl.atomic_cas(locks + next_pid, 1, 1) != 1:
              #  while tl.load(locks + next_pid, cache_modifier = ".cg") != 1:
                    pass
                rm1 = tl.arange(0, BLOCK_SIZE_M)
                rn1 = tl.arange(0, BLOCK_SIZE_N)
                rm1 = tl.max_contiguous(tl.multiple_of(rm1, BLOCK_SIZE_M), BLOCK_SIZE_M)
                rn1 = tl.max_contiguous(tl.multiple_of(rn1, BLOCK_SIZE_N), BLOCK_SIZE_N)
                P_ = P + next_pid * BLOCK_SIZE_M * BLOCK_SIZE_N + rm1[:, None] * BLOCK_SIZE_N + rn1[None, :]
                acc += tl.load(tl.multiple_of(P_, (1, 16)))
              #  acc += tl.load(P_)
                end += streamk_iters_pcu + (next_pid < streamk_remainder_iters)
                next_pid += 1

            c = acc.to(C.type.element_ty)
            if BIAS:
                 c += bias[:, None]

            rm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M))%M
            rn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))%N
            rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
            rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)
            C_ = C + rm[:, None] * stride_cm + rn[None, :] * stride_cn
            mask = (rm < M)[:, None] & (rn < N)[None, :]
            tl.store(C_, c, mask=mask)
        else:
            rm1 = tl.arange(0, BLOCK_SIZE_M)
            rn1 = tl.arange(0, BLOCK_SIZE_N)
            rm1 = tl.max_contiguous(tl.multiple_of(rm1, BLOCK_SIZE_M), BLOCK_SIZE_M)
            rn1 = tl.max_contiguous(tl.multiple_of(rn1, BLOCK_SIZE_N), BLOCK_SIZE_N)
            P_ = P + pid * BLOCK_SIZE_M * BLOCK_SIZE_N +  rm1[:, None] * BLOCK_SIZE_N + rn1[None, :]
            tl.store(P_, acc)
            tl.debug_barrier()
            tl.atomic_xchg(locks + pid, 1)
     #       tl.store(locks + pid, 1, cache_modifier=".wt")

        start_iter = end_iter



def matmul_BM128_BN64_BK64_GM32_nW4_nS2_EU0_kP2_mfma16(a, b, c, bias, P, locks, M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    grid = num_sms
    streamk_gemm_BM128_BN64_BK64_GM32_nW4_nS2_EU0_kP2_mfma16[grid,](
        a, b, c, bias, P, locks,
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
        BIAS = False,
        EVEN_K = True
    )
    return c

def try_matmul_config_BM128_BN64_BK64_GM32_nW4_nS2_EU0_kP2_mfma16(a, b, c, bias, M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        locks = torch.zeros((num_sms,), device = "cuda", dtype = torch.int32)
        P = torch.zeros((num_sms,  128*64), device="cuda", dtype=torch.float32)
        matmul_BM128_BN64_BK64_GM32_nW4_nS2_EU0_kP2_mfma16(a, b, c, bias, P, locks, M, N, K, num_sms, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1), bias_stride)
        return True
    except Exception as e:
        print(f'invalid config(runtime): BM128_BN64_BK64_GM32_nW4_nS2_EU0_kP2_mfma16: ', e, flush=True)
        return False




@triton.jit()
def streamk_gemm_BM128_BN64_BK128_GM1_nW4_nS2_EU0_kP2_mfma16(
        A, B, C, bias_ptr, P, locks,
        M, N, K, num_sms,
        stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn, stride_bias,
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
        GROUP_SIZE_M: tl.constexpr, BIAS: tl.constexpr, EVEN_K: tl.constexpr,
):
    pid = tl.program_id(0)
    pid = (pid % 8) * (num_sms // 8) + (pid // 8)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    iters_per_tile = tl.cdiv(K, BLOCK_SIZE_K)
    total_tiles = num_pid_m * num_pid_n
    if num_sms > 0 and total_tiles > num_sms:
        total_streamk_tiles = total_tiles % num_sms
        total_streamk_tiles = total_streamk_tiles + num_sms
        total_full_tiles = total_tiles - total_streamk_tiles
        total_streamk_iters = total_streamk_tiles * iters_per_tile
        streamk_iters_pcu = total_streamk_iters // num_sms
        streamk_remainder_iters = total_streamk_iters % num_sms
    else:
        total_full_tiles = total_tiles
        total_streamk_tiles = 0
        streamk_iters_pcu = 0
        streamk_remainder_iters = 0
        total_streamk_iters = 0

    acc_dtype = tl.float32 if C.type.element_ty != tl.int8 else tl.int32
    rk = tl.arange(0, BLOCK_SIZE_K)

    for tile_id in range(pid, total_full_tiles, num_sms):
        if GROUP_SIZE_M == 1:
            pid_m = tile_id // num_pid_n
            pid_n = tile_id % num_pid_n
        else:
            num_pid_in_group = GROUP_SIZE_M * num_pid_n
            group_id = tile_id // num_pid_in_group
            first_pid_m = group_id * GROUP_SIZE_M
            group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
            pid_m = first_pid_m + (tile_id % group_size_m)
            pid_n = (tile_id % num_pid_in_group) // group_size_m

        rm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M))%M
        rn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))%N
        rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
        rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)
        A_BASE = A + rm[:, None] * stride_am + rk[None, :] * stride_ak
        B_BASE = B + rk[:, None] * stride_bk + rn[None, :] * stride_bn

        if BIAS:
            bias_ = bias_ptr + rm * stride_bias
            bias = tl.load(bias_, mask=rm < M, other=0.0)

        loop_k = tl.cdiv(K, BLOCK_SIZE_K)
        if not EVEN_K:
            loop_k -= 1

        acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype)
        for k in range(0, loop_k):
            a = tl.load(tl.multiple_of(A_BASE, (1, 16)))
            b = tl.load(tl.multiple_of(B_BASE, (16, 1)))
            acc += tl.dot(a, b)
            A_BASE += BLOCK_SIZE_K * stride_ak
            B_BASE += BLOCK_SIZE_K * stride_bk

        if not EVEN_K:
            k = loop_k
            rk = k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
            A_BASE = A + rm[:, None] * stride_am + rk[None, :] * stride_ak
            B_BASE = B + rk[:, None] * stride_bk + rn[None, :] * stride_bn
            a = tl.load(A_BASE, mask=rk[None, :] < K, other=0.0)
            b = tl.load(B_BASE, mask=rk[:, None] < K, other=0.0)

        c = acc.to(C.type.element_ty)
        if BIAS:
             c += bias[:, None]

        rm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        C_ = C + rm[:, None] * stride_cm + rn[None, :] * stride_cn
        mask = (rm < M)[:, None] & (rn < N)[None, :]
        tl.store(C_, c, mask=mask)

    start_iter = total_full_tiles * iters_per_tile + pid * streamk_iters_pcu + tl.minimum(pid, streamk_remainder_iters)
    last_iter = total_full_tiles * iters_per_tile + (pid + 1) * streamk_iters_pcu + tl.minimum(pid + 1, streamk_remainder_iters)
    while start_iter < last_iter:
        remainder = start_iter % iters_per_tile
        end_iter = tl.minimum(start_iter + (iters_per_tile - remainder), last_iter)
        tile_id = start_iter // iters_per_tile
        if GROUP_SIZE_M == 1:
            pid_m = tile_id // num_pid_n
            pid_n = tile_id % num_pid_n
        else:
            num_pid_in_group = GROUP_SIZE_M * num_pid_n
            group_id = tile_id // num_pid_in_group
            first_pid_m = group_id * GROUP_SIZE_M
            group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
            pid_m = first_pid_m + (tile_id % group_size_m)
            pid_n = (tile_id % num_pid_in_group) // group_size_m

        rm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M))%M
        rn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))%N
        rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
        rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)
        A_BASE = A + rm[:, None] * stride_am + rk[None, :] * stride_ak + BLOCK_SIZE_K * stride_ak * remainder
        B_BASE = B + rk[:, None] * stride_bk + rn[None, :] * stride_bn + BLOCK_SIZE_K * stride_bk * remainder

        if BIAS:
            bias_ = bias_ptr + rm * stride_bias
            bias = tl.load(bias_, mask=rm < M, other=0.0)

        acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype)
        for current_iter in range(start_iter, end_iter):
            if EVEN_K:
                a = tl.load(tl.multiple_of(A_BASE, (1, 16)))
                b = tl.load(tl.multiple_of(B_BASE, (16, 1)))
            else:
                global_k_offset = (current_iter % iters_per_tile) * BLOCK_SIZE_K
                k_mask = global_k_offset + rk < K
                a = tl.load(A_BASE, mask=k_mask[None, :], other=0.0)
                b = tl.load(B_BASE, mask=k_mask[:, None], other=0.0)
            acc += tl.dot(a, b)
            A_BASE += BLOCK_SIZE_K * stride_ak
            B_BASE += BLOCK_SIZE_K * stride_bk

        tile_iter = tile_id * iters_per_tile
        if start_iter == tile_iter:
            tile_iter_end = tile_iter + iters_per_tile
            next_pid = pid + 1
            end = end_iter
            while (end < tile_iter_end and next_pid < num_sms):
                while tl.atomic_cas(locks + next_pid, 1, 1) != 1:
              #  while tl.load(locks + next_pid, cache_modifier = ".cg") != 1:
                    pass
                rm1 = tl.arange(0, BLOCK_SIZE_M)
                rn1 = tl.arange(0, BLOCK_SIZE_N)
                rm1 = tl.max_contiguous(tl.multiple_of(rm1, BLOCK_SIZE_M), BLOCK_SIZE_M)
                rn1 = tl.max_contiguous(tl.multiple_of(rn1, BLOCK_SIZE_N), BLOCK_SIZE_N)
                P_ = P + next_pid * BLOCK_SIZE_M * BLOCK_SIZE_N + rm1[:, None] * BLOCK_SIZE_N + rn1[None, :]
                acc += tl.load(tl.multiple_of(P_, (1, 16)))
              #  acc += tl.load(P_)
                end += streamk_iters_pcu + (next_pid < streamk_remainder_iters)
                next_pid += 1

            c = acc.to(C.type.element_ty)
            if BIAS:
                 c += bias[:, None]

            rm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M))%M
            rn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))%N
            rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
            rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)
            C_ = C + rm[:, None] * stride_cm + rn[None, :] * stride_cn
            mask = (rm < M)[:, None] & (rn < N)[None, :]
            tl.store(C_, c, mask=mask)
        else:
            rm1 = tl.arange(0, BLOCK_SIZE_M)
            rn1 = tl.arange(0, BLOCK_SIZE_N)
            rm1 = tl.max_contiguous(tl.multiple_of(rm1, BLOCK_SIZE_M), BLOCK_SIZE_M)
            rn1 = tl.max_contiguous(tl.multiple_of(rn1, BLOCK_SIZE_N), BLOCK_SIZE_N)
            P_ = P + pid * BLOCK_SIZE_M * BLOCK_SIZE_N +  rm1[:, None] * BLOCK_SIZE_N + rn1[None, :]
            tl.store(P_, acc)
            tl.debug_barrier()
            tl.atomic_xchg(locks + pid, 1)
     #       tl.store(locks + pid, 1, cache_modifier=".wt")

        start_iter = end_iter



def matmul_BM128_BN64_BK128_GM1_nW4_nS2_EU0_kP2_mfma16(a, b, c, bias, P, locks, M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    grid = num_sms
    streamk_gemm_BM128_BN64_BK128_GM1_nW4_nS2_EU0_kP2_mfma16[grid,](
        a, b, c, bias, P, locks,
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
        BIAS = False,
        EVEN_K = False
    )
    return c

def try_matmul_config_BM128_BN64_BK128_GM1_nW4_nS2_EU0_kP2_mfma16(a, b, c, bias, M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        locks = torch.zeros((num_sms,), device = "cuda", dtype = torch.int32)
        P = torch.zeros((num_sms,  128*64), device="cuda", dtype=torch.float32)
        matmul_BM128_BN64_BK128_GM1_nW4_nS2_EU0_kP2_mfma16(a, b, c, bias, P, locks, M, N, K, num_sms, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1), bias_stride)
        return True
    except Exception as e:
        print(f'invalid config(runtime): BM128_BN64_BK128_GM1_nW4_nS2_EU0_kP2_mfma16: ', e, flush=True)
        return False




@triton.jit()
def streamk_gemm_BM128_BN64_BK128_GM4_nW8_nS2_EU0_kP2_mfma16(
        A, B, C, bias_ptr, P, locks,
        M, N, K, num_sms,
        stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn, stride_bias,
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
        GROUP_SIZE_M: tl.constexpr, BIAS: tl.constexpr, EVEN_K: tl.constexpr,
):
    pid = tl.program_id(0)
    pid = (pid % 8) * (num_sms // 8) + (pid // 8)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    iters_per_tile = tl.cdiv(K, BLOCK_SIZE_K)
    total_tiles = num_pid_m * num_pid_n
    if num_sms > 0 and total_tiles > num_sms:
        total_streamk_tiles = total_tiles % num_sms
        total_streamk_tiles = total_streamk_tiles + num_sms
        total_full_tiles = total_tiles - total_streamk_tiles
        total_streamk_iters = total_streamk_tiles * iters_per_tile
        streamk_iters_pcu = total_streamk_iters // num_sms
        streamk_remainder_iters = total_streamk_iters % num_sms
    else:
        total_full_tiles = total_tiles
        total_streamk_tiles = 0
        streamk_iters_pcu = 0
        streamk_remainder_iters = 0
        total_streamk_iters = 0

    acc_dtype = tl.float32 if C.type.element_ty != tl.int8 else tl.int32
    rk = tl.arange(0, BLOCK_SIZE_K)

    for tile_id in range(pid, total_full_tiles, num_sms):
        if GROUP_SIZE_M == 1:
            pid_m = tile_id // num_pid_n
            pid_n = tile_id % num_pid_n
        else:
            num_pid_in_group = GROUP_SIZE_M * num_pid_n
            group_id = tile_id // num_pid_in_group
            first_pid_m = group_id * GROUP_SIZE_M
            group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
            pid_m = first_pid_m + (tile_id % group_size_m)
            pid_n = (tile_id % num_pid_in_group) // group_size_m

        rm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M))%M
        rn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))%N
        rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
        rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)
        A_BASE = A + rm[:, None] * stride_am + rk[None, :] * stride_ak
        B_BASE = B + rk[:, None] * stride_bk + rn[None, :] * stride_bn

        if BIAS:
            bias_ = bias_ptr + rm * stride_bias
            bias = tl.load(bias_, mask=rm < M, other=0.0)

        loop_k = tl.cdiv(K, BLOCK_SIZE_K)
        if not EVEN_K:
            loop_k -= 1

        acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype)
        for k in range(0, loop_k):
            a = tl.load(tl.multiple_of(A_BASE, (1, 16)))
            b = tl.load(tl.multiple_of(B_BASE, (16, 1)))
            acc += tl.dot(a, b)
            A_BASE += BLOCK_SIZE_K * stride_ak
            B_BASE += BLOCK_SIZE_K * stride_bk

        if not EVEN_K:
            k = loop_k
            rk = k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
            A_BASE = A + rm[:, None] * stride_am + rk[None, :] * stride_ak
            B_BASE = B + rk[:, None] * stride_bk + rn[None, :] * stride_bn
            a = tl.load(A_BASE, mask=rk[None, :] < K, other=0.0)
            b = tl.load(B_BASE, mask=rk[:, None] < K, other=0.0)

        c = acc.to(C.type.element_ty)
        if BIAS:
             c += bias[:, None]

        rm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        C_ = C + rm[:, None] * stride_cm + rn[None, :] * stride_cn
        mask = (rm < M)[:, None] & (rn < N)[None, :]
        tl.store(C_, c, mask=mask)

    start_iter = total_full_tiles * iters_per_tile + pid * streamk_iters_pcu + tl.minimum(pid, streamk_remainder_iters)
    last_iter = total_full_tiles * iters_per_tile + (pid + 1) * streamk_iters_pcu + tl.minimum(pid + 1, streamk_remainder_iters)
    while start_iter < last_iter:
        remainder = start_iter % iters_per_tile
        end_iter = tl.minimum(start_iter + (iters_per_tile - remainder), last_iter)
        tile_id = start_iter // iters_per_tile
        if GROUP_SIZE_M == 1:
            pid_m = tile_id // num_pid_n
            pid_n = tile_id % num_pid_n
        else:
            num_pid_in_group = GROUP_SIZE_M * num_pid_n
            group_id = tile_id // num_pid_in_group
            first_pid_m = group_id * GROUP_SIZE_M
            group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
            pid_m = first_pid_m + (tile_id % group_size_m)
            pid_n = (tile_id % num_pid_in_group) // group_size_m

        rm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M))%M
        rn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))%N
        rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
        rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)
        A_BASE = A + rm[:, None] * stride_am + rk[None, :] * stride_ak + BLOCK_SIZE_K * stride_ak * remainder
        B_BASE = B + rk[:, None] * stride_bk + rn[None, :] * stride_bn + BLOCK_SIZE_K * stride_bk * remainder

        if BIAS:
            bias_ = bias_ptr + rm * stride_bias
            bias = tl.load(bias_, mask=rm < M, other=0.0)

        acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype)
        for current_iter in range(start_iter, end_iter):
            if EVEN_K:
                a = tl.load(tl.multiple_of(A_BASE, (1, 16)))
                b = tl.load(tl.multiple_of(B_BASE, (16, 1)))
            else:
                global_k_offset = (current_iter % iters_per_tile) * BLOCK_SIZE_K
                k_mask = global_k_offset + rk < K
                a = tl.load(A_BASE, mask=k_mask[None, :], other=0.0)
                b = tl.load(B_BASE, mask=k_mask[:, None], other=0.0)
            acc += tl.dot(a, b)
            A_BASE += BLOCK_SIZE_K * stride_ak
            B_BASE += BLOCK_SIZE_K * stride_bk

        tile_iter = tile_id * iters_per_tile
        if start_iter == tile_iter:
            tile_iter_end = tile_iter + iters_per_tile
            next_pid = pid + 1
            end = end_iter
            while (end < tile_iter_end and next_pid < num_sms):
                while tl.atomic_cas(locks + next_pid, 1, 1) != 1:
              #  while tl.load(locks + next_pid, cache_modifier = ".cg") != 1:
                    pass
                rm1 = tl.arange(0, BLOCK_SIZE_M)
                rn1 = tl.arange(0, BLOCK_SIZE_N)
                rm1 = tl.max_contiguous(tl.multiple_of(rm1, BLOCK_SIZE_M), BLOCK_SIZE_M)
                rn1 = tl.max_contiguous(tl.multiple_of(rn1, BLOCK_SIZE_N), BLOCK_SIZE_N)
                P_ = P + next_pid * BLOCK_SIZE_M * BLOCK_SIZE_N + rm1[:, None] * BLOCK_SIZE_N + rn1[None, :]
                acc += tl.load(tl.multiple_of(P_, (1, 16)))
              #  acc += tl.load(P_)
                end += streamk_iters_pcu + (next_pid < streamk_remainder_iters)
                next_pid += 1

            c = acc.to(C.type.element_ty)
            if BIAS:
                 c += bias[:, None]

            rm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M))%M
            rn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))%N
            rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
            rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)
            C_ = C + rm[:, None] * stride_cm + rn[None, :] * stride_cn
            mask = (rm < M)[:, None] & (rn < N)[None, :]
            tl.store(C_, c, mask=mask)
        else:
            rm1 = tl.arange(0, BLOCK_SIZE_M)
            rn1 = tl.arange(0, BLOCK_SIZE_N)
            rm1 = tl.max_contiguous(tl.multiple_of(rm1, BLOCK_SIZE_M), BLOCK_SIZE_M)
            rn1 = tl.max_contiguous(tl.multiple_of(rn1, BLOCK_SIZE_N), BLOCK_SIZE_N)
            P_ = P + pid * BLOCK_SIZE_M * BLOCK_SIZE_N +  rm1[:, None] * BLOCK_SIZE_N + rn1[None, :]
            tl.store(P_, acc)
            tl.debug_barrier()
            tl.atomic_xchg(locks + pid, 1)
     #       tl.store(locks + pid, 1, cache_modifier=".wt")

        start_iter = end_iter



def matmul_BM128_BN64_BK128_GM4_nW8_nS2_EU0_kP2_mfma16(a, b, c, bias, P, locks, M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    grid = num_sms
    streamk_gemm_BM128_BN64_BK128_GM4_nW8_nS2_EU0_kP2_mfma16[grid,](
        a, b, c, bias, P, locks,
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
        BIAS = False,
        EVEN_K = False
    )
    return c

def try_matmul_config_BM128_BN64_BK128_GM4_nW8_nS2_EU0_kP2_mfma16(a, b, c, bias, M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        locks = torch.zeros((num_sms,), device = "cuda", dtype = torch.int32)
        P = torch.zeros((num_sms,  128*64), device="cuda", dtype=torch.float32)
        matmul_BM128_BN64_BK128_GM4_nW8_nS2_EU0_kP2_mfma16(a, b, c, bias, P, locks, M, N, K, num_sms, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1), bias_stride)
        return True
    except Exception as e:
        print(f'invalid config(runtime): BM128_BN64_BK128_GM4_nW8_nS2_EU0_kP2_mfma16: ', e, flush=True)
        return False




@triton.jit()
def streamk_gemm_BM128_BN128_BK64_GM8_nW4_nS2_EU0_kP2_mfma16(
        A, B, C, bias_ptr, P, locks,
        M, N, K, num_sms,
        stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn, stride_bias,
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
        GROUP_SIZE_M: tl.constexpr, BIAS: tl.constexpr, EVEN_K: tl.constexpr,
):
    pid = tl.program_id(0)
    pid = (pid % 8) * (num_sms // 8) + (pid // 8)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    iters_per_tile = tl.cdiv(K, BLOCK_SIZE_K)
    total_tiles = num_pid_m * num_pid_n
    if num_sms > 0 and total_tiles > num_sms:
        total_streamk_tiles = total_tiles % num_sms
        total_streamk_tiles = total_streamk_tiles + num_sms
        total_full_tiles = total_tiles - total_streamk_tiles
        total_streamk_iters = total_streamk_tiles * iters_per_tile
        streamk_iters_pcu = total_streamk_iters // num_sms
        streamk_remainder_iters = total_streamk_iters % num_sms
    else:
        total_full_tiles = total_tiles
        total_streamk_tiles = 0
        streamk_iters_pcu = 0
        streamk_remainder_iters = 0
        total_streamk_iters = 0

    acc_dtype = tl.float32 if C.type.element_ty != tl.int8 else tl.int32
    rk = tl.arange(0, BLOCK_SIZE_K)

    for tile_id in range(pid, total_full_tiles, num_sms):
        if GROUP_SIZE_M == 1:
            pid_m = tile_id // num_pid_n
            pid_n = tile_id % num_pid_n
        else:
            num_pid_in_group = GROUP_SIZE_M * num_pid_n
            group_id = tile_id // num_pid_in_group
            first_pid_m = group_id * GROUP_SIZE_M
            group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
            pid_m = first_pid_m + (tile_id % group_size_m)
            pid_n = (tile_id % num_pid_in_group) // group_size_m

        rm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M))%M
        rn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))%N
        rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
        rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)
        A_BASE = A + rm[:, None] * stride_am + rk[None, :] * stride_ak
        B_BASE = B + rk[:, None] * stride_bk + rn[None, :] * stride_bn

        if BIAS:
            bias_ = bias_ptr + rm * stride_bias
            bias = tl.load(bias_, mask=rm < M, other=0.0)

        loop_k = tl.cdiv(K, BLOCK_SIZE_K)
        if not EVEN_K:
            loop_k -= 1

        acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype)
        for k in range(0, loop_k):
            a = tl.load(tl.multiple_of(A_BASE, (1, 16)))
            b = tl.load(tl.multiple_of(B_BASE, (16, 1)))
            acc += tl.dot(a, b)
            A_BASE += BLOCK_SIZE_K * stride_ak
            B_BASE += BLOCK_SIZE_K * stride_bk

        if not EVEN_K:
            k = loop_k
            rk = k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
            A_BASE = A + rm[:, None] * stride_am + rk[None, :] * stride_ak
            B_BASE = B + rk[:, None] * stride_bk + rn[None, :] * stride_bn
            a = tl.load(A_BASE, mask=rk[None, :] < K, other=0.0)
            b = tl.load(B_BASE, mask=rk[:, None] < K, other=0.0)

        c = acc.to(C.type.element_ty)
        if BIAS:
             c += bias[:, None]

        rm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        C_ = C + rm[:, None] * stride_cm + rn[None, :] * stride_cn
        mask = (rm < M)[:, None] & (rn < N)[None, :]
        tl.store(C_, c, mask=mask)

    start_iter = total_full_tiles * iters_per_tile + pid * streamk_iters_pcu + tl.minimum(pid, streamk_remainder_iters)
    last_iter = total_full_tiles * iters_per_tile + (pid + 1) * streamk_iters_pcu + tl.minimum(pid + 1, streamk_remainder_iters)
    while start_iter < last_iter:
        remainder = start_iter % iters_per_tile
        end_iter = tl.minimum(start_iter + (iters_per_tile - remainder), last_iter)
        tile_id = start_iter // iters_per_tile
        if GROUP_SIZE_M == 1:
            pid_m = tile_id // num_pid_n
            pid_n = tile_id % num_pid_n
        else:
            num_pid_in_group = GROUP_SIZE_M * num_pid_n
            group_id = tile_id // num_pid_in_group
            first_pid_m = group_id * GROUP_SIZE_M
            group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
            pid_m = first_pid_m + (tile_id % group_size_m)
            pid_n = (tile_id % num_pid_in_group) // group_size_m

        rm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M))%M
        rn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))%N
        rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
        rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)
        A_BASE = A + rm[:, None] * stride_am + rk[None, :] * stride_ak + BLOCK_SIZE_K * stride_ak * remainder
        B_BASE = B + rk[:, None] * stride_bk + rn[None, :] * stride_bn + BLOCK_SIZE_K * stride_bk * remainder

        if BIAS:
            bias_ = bias_ptr + rm * stride_bias
            bias = tl.load(bias_, mask=rm < M, other=0.0)

        acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype)
        for current_iter in range(start_iter, end_iter):
            if EVEN_K:
                a = tl.load(tl.multiple_of(A_BASE, (1, 16)))
                b = tl.load(tl.multiple_of(B_BASE, (16, 1)))
            else:
                global_k_offset = (current_iter % iters_per_tile) * BLOCK_SIZE_K
                k_mask = global_k_offset + rk < K
                a = tl.load(A_BASE, mask=k_mask[None, :], other=0.0)
                b = tl.load(B_BASE, mask=k_mask[:, None], other=0.0)
            acc += tl.dot(a, b)
            A_BASE += BLOCK_SIZE_K * stride_ak
            B_BASE += BLOCK_SIZE_K * stride_bk

        tile_iter = tile_id * iters_per_tile
        if start_iter == tile_iter:
            tile_iter_end = tile_iter + iters_per_tile
            next_pid = pid + 1
            end = end_iter
            while (end < tile_iter_end and next_pid < num_sms):
                while tl.atomic_cas(locks + next_pid, 1, 1) != 1:
              #  while tl.load(locks + next_pid, cache_modifier = ".cg") != 1:
                    pass
                rm1 = tl.arange(0, BLOCK_SIZE_M)
                rn1 = tl.arange(0, BLOCK_SIZE_N)
                rm1 = tl.max_contiguous(tl.multiple_of(rm1, BLOCK_SIZE_M), BLOCK_SIZE_M)
                rn1 = tl.max_contiguous(tl.multiple_of(rn1, BLOCK_SIZE_N), BLOCK_SIZE_N)
                P_ = P + next_pid * BLOCK_SIZE_M * BLOCK_SIZE_N + rm1[:, None] * BLOCK_SIZE_N + rn1[None, :]
                acc += tl.load(tl.multiple_of(P_, (1, 16)))
              #  acc += tl.load(P_)
                end += streamk_iters_pcu + (next_pid < streamk_remainder_iters)
                next_pid += 1

            c = acc.to(C.type.element_ty)
            if BIAS:
                 c += bias[:, None]

            rm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M))%M
            rn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))%N
            rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
            rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)
            C_ = C + rm[:, None] * stride_cm + rn[None, :] * stride_cn
            mask = (rm < M)[:, None] & (rn < N)[None, :]
            tl.store(C_, c, mask=mask)
        else:
            rm1 = tl.arange(0, BLOCK_SIZE_M)
            rn1 = tl.arange(0, BLOCK_SIZE_N)
            rm1 = tl.max_contiguous(tl.multiple_of(rm1, BLOCK_SIZE_M), BLOCK_SIZE_M)
            rn1 = tl.max_contiguous(tl.multiple_of(rn1, BLOCK_SIZE_N), BLOCK_SIZE_N)
            P_ = P + pid * BLOCK_SIZE_M * BLOCK_SIZE_N +  rm1[:, None] * BLOCK_SIZE_N + rn1[None, :]
            tl.store(P_, acc)
            tl.debug_barrier()
            tl.atomic_xchg(locks + pid, 1)
     #       tl.store(locks + pid, 1, cache_modifier=".wt")

        start_iter = end_iter



def matmul_BM128_BN128_BK64_GM8_nW4_nS2_EU0_kP2_mfma16(a, b, c, bias, P, locks, M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    grid = num_sms
    streamk_gemm_BM128_BN128_BK64_GM8_nW4_nS2_EU0_kP2_mfma16[grid,](
        a, b, c, bias, P, locks,
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
        BIAS = False,
        EVEN_K = True
    )
    return c

def try_matmul_config_BM128_BN128_BK64_GM8_nW4_nS2_EU0_kP2_mfma16(a, b, c, bias, M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        locks = torch.zeros((num_sms,), device = "cuda", dtype = torch.int32)
        P = torch.zeros((num_sms,  128*128), device="cuda", dtype=torch.float32)
        matmul_BM128_BN128_BK64_GM8_nW4_nS2_EU0_kP2_mfma16(a, b, c, bias, P, locks, M, N, K, num_sms, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1), bias_stride)
        return True
    except Exception as e:
        print(f'invalid config(runtime): BM128_BN128_BK64_GM8_nW4_nS2_EU0_kP2_mfma16: ', e, flush=True)
        return False




@triton.jit()
def streamk_gemm_BM128_BN128_BK64_GM16_nW8_nS2_EU0_kP2_mfma16(
        A, B, C, bias_ptr, P, locks,
        M, N, K, num_sms,
        stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn, stride_bias,
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
        GROUP_SIZE_M: tl.constexpr, BIAS: tl.constexpr, EVEN_K: tl.constexpr,
):
    pid = tl.program_id(0)
    pid = (pid % 8) * (num_sms // 8) + (pid // 8)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    iters_per_tile = tl.cdiv(K, BLOCK_SIZE_K)
    total_tiles = num_pid_m * num_pid_n
    if num_sms > 0 and total_tiles > num_sms:
        total_streamk_tiles = total_tiles % num_sms
        total_streamk_tiles = total_streamk_tiles + num_sms
        total_full_tiles = total_tiles - total_streamk_tiles
        total_streamk_iters = total_streamk_tiles * iters_per_tile
        streamk_iters_pcu = total_streamk_iters // num_sms
        streamk_remainder_iters = total_streamk_iters % num_sms
    else:
        total_full_tiles = total_tiles
        total_streamk_tiles = 0
        streamk_iters_pcu = 0
        streamk_remainder_iters = 0
        total_streamk_iters = 0

    acc_dtype = tl.float32 if C.type.element_ty != tl.int8 else tl.int32
    rk = tl.arange(0, BLOCK_SIZE_K)

    for tile_id in range(pid, total_full_tiles, num_sms):
        if GROUP_SIZE_M == 1:
            pid_m = tile_id // num_pid_n
            pid_n = tile_id % num_pid_n
        else:
            num_pid_in_group = GROUP_SIZE_M * num_pid_n
            group_id = tile_id // num_pid_in_group
            first_pid_m = group_id * GROUP_SIZE_M
            group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
            pid_m = first_pid_m + (tile_id % group_size_m)
            pid_n = (tile_id % num_pid_in_group) // group_size_m

        rm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M))%M
        rn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))%N
        rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
        rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)
        A_BASE = A + rm[:, None] * stride_am + rk[None, :] * stride_ak
        B_BASE = B + rk[:, None] * stride_bk + rn[None, :] * stride_bn

        if BIAS:
            bias_ = bias_ptr + rm * stride_bias
            bias = tl.load(bias_, mask=rm < M, other=0.0)

        loop_k = tl.cdiv(K, BLOCK_SIZE_K)
        if not EVEN_K:
            loop_k -= 1

        acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype)
        for k in range(0, loop_k):
            a = tl.load(tl.multiple_of(A_BASE, (1, 16)))
            b = tl.load(tl.multiple_of(B_BASE, (16, 1)))
            acc += tl.dot(a, b)
            A_BASE += BLOCK_SIZE_K * stride_ak
            B_BASE += BLOCK_SIZE_K * stride_bk

        if not EVEN_K:
            k = loop_k
            rk = k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
            A_BASE = A + rm[:, None] * stride_am + rk[None, :] * stride_ak
            B_BASE = B + rk[:, None] * stride_bk + rn[None, :] * stride_bn
            a = tl.load(A_BASE, mask=rk[None, :] < K, other=0.0)
            b = tl.load(B_BASE, mask=rk[:, None] < K, other=0.0)

        c = acc.to(C.type.element_ty)
        if BIAS:
             c += bias[:, None]

        rm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        C_ = C + rm[:, None] * stride_cm + rn[None, :] * stride_cn
        mask = (rm < M)[:, None] & (rn < N)[None, :]
        tl.store(C_, c, mask=mask)

    start_iter = total_full_tiles * iters_per_tile + pid * streamk_iters_pcu + tl.minimum(pid, streamk_remainder_iters)
    last_iter = total_full_tiles * iters_per_tile + (pid + 1) * streamk_iters_pcu + tl.minimum(pid + 1, streamk_remainder_iters)
    while start_iter < last_iter:
        remainder = start_iter % iters_per_tile
        end_iter = tl.minimum(start_iter + (iters_per_tile - remainder), last_iter)
        tile_id = start_iter // iters_per_tile
        if GROUP_SIZE_M == 1:
            pid_m = tile_id // num_pid_n
            pid_n = tile_id % num_pid_n
        else:
            num_pid_in_group = GROUP_SIZE_M * num_pid_n
            group_id = tile_id // num_pid_in_group
            first_pid_m = group_id * GROUP_SIZE_M
            group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
            pid_m = first_pid_m + (tile_id % group_size_m)
            pid_n = (tile_id % num_pid_in_group) // group_size_m

        rm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M))%M
        rn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))%N
        rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
        rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)
        A_BASE = A + rm[:, None] * stride_am + rk[None, :] * stride_ak + BLOCK_SIZE_K * stride_ak * remainder
        B_BASE = B + rk[:, None] * stride_bk + rn[None, :] * stride_bn + BLOCK_SIZE_K * stride_bk * remainder

        if BIAS:
            bias_ = bias_ptr + rm * stride_bias
            bias = tl.load(bias_, mask=rm < M, other=0.0)

        acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype)
        for current_iter in range(start_iter, end_iter):
            if EVEN_K:
                a = tl.load(tl.multiple_of(A_BASE, (1, 16)))
                b = tl.load(tl.multiple_of(B_BASE, (16, 1)))
            else:
                global_k_offset = (current_iter % iters_per_tile) * BLOCK_SIZE_K
                k_mask = global_k_offset + rk < K
                a = tl.load(A_BASE, mask=k_mask[None, :], other=0.0)
                b = tl.load(B_BASE, mask=k_mask[:, None], other=0.0)
            acc += tl.dot(a, b)
            A_BASE += BLOCK_SIZE_K * stride_ak
            B_BASE += BLOCK_SIZE_K * stride_bk

        tile_iter = tile_id * iters_per_tile
        if start_iter == tile_iter:
            tile_iter_end = tile_iter + iters_per_tile
            next_pid = pid + 1
            end = end_iter
            while (end < tile_iter_end and next_pid < num_sms):
                while tl.atomic_cas(locks + next_pid, 1, 1) != 1:
              #  while tl.load(locks + next_pid, cache_modifier = ".cg") != 1:
                    pass
                rm1 = tl.arange(0, BLOCK_SIZE_M)
                rn1 = tl.arange(0, BLOCK_SIZE_N)
                rm1 = tl.max_contiguous(tl.multiple_of(rm1, BLOCK_SIZE_M), BLOCK_SIZE_M)
                rn1 = tl.max_contiguous(tl.multiple_of(rn1, BLOCK_SIZE_N), BLOCK_SIZE_N)
                P_ = P + next_pid * BLOCK_SIZE_M * BLOCK_SIZE_N + rm1[:, None] * BLOCK_SIZE_N + rn1[None, :]
                acc += tl.load(tl.multiple_of(P_, (1, 16)))
              #  acc += tl.load(P_)
                end += streamk_iters_pcu + (next_pid < streamk_remainder_iters)
                next_pid += 1

            c = acc.to(C.type.element_ty)
            if BIAS:
                 c += bias[:, None]

            rm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M))%M
            rn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))%N
            rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
            rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)
            C_ = C + rm[:, None] * stride_cm + rn[None, :] * stride_cn
            mask = (rm < M)[:, None] & (rn < N)[None, :]
            tl.store(C_, c, mask=mask)
        else:
            rm1 = tl.arange(0, BLOCK_SIZE_M)
            rn1 = tl.arange(0, BLOCK_SIZE_N)
            rm1 = tl.max_contiguous(tl.multiple_of(rm1, BLOCK_SIZE_M), BLOCK_SIZE_M)
            rn1 = tl.max_contiguous(tl.multiple_of(rn1, BLOCK_SIZE_N), BLOCK_SIZE_N)
            P_ = P + pid * BLOCK_SIZE_M * BLOCK_SIZE_N +  rm1[:, None] * BLOCK_SIZE_N + rn1[None, :]
            tl.store(P_, acc)
            tl.debug_barrier()
            tl.atomic_xchg(locks + pid, 1)
     #       tl.store(locks + pid, 1, cache_modifier=".wt")

        start_iter = end_iter



def matmul_BM128_BN128_BK64_GM16_nW8_nS2_EU0_kP2_mfma16(a, b, c, bias, P, locks, M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    grid = num_sms
    streamk_gemm_BM128_BN128_BK64_GM16_nW8_nS2_EU0_kP2_mfma16[grid,](
        a, b, c, bias, P, locks,
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
        BIAS = False,
        EVEN_K = True
    )
    return c

def try_matmul_config_BM128_BN128_BK64_GM16_nW8_nS2_EU0_kP2_mfma16(a, b, c, bias, M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        locks = torch.zeros((num_sms,), device = "cuda", dtype = torch.int32)
        P = torch.zeros((num_sms,  128*128), device="cuda", dtype=torch.float32)
        matmul_BM128_BN128_BK64_GM16_nW8_nS2_EU0_kP2_mfma16(a, b, c, bias, P, locks, M, N, K, num_sms, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1), bias_stride)
        return True
    except Exception as e:
        print(f'invalid config(runtime): BM128_BN128_BK64_GM16_nW8_nS2_EU0_kP2_mfma16: ', e, flush=True)
        return False




@triton.jit()
def streamk_gemm_BM128_BN128_BK128_GM1_nW8_nS2_EU0_kP1_mfma32(
        A, B, C, bias_ptr, P, locks,
        M, N, K, num_sms,
        stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn, stride_bias,
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
        GROUP_SIZE_M: tl.constexpr, BIAS: tl.constexpr, EVEN_K: tl.constexpr,
):
    pid = tl.program_id(0)
    pid = (pid % 8) * (num_sms // 8) + (pid // 8)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    iters_per_tile = tl.cdiv(K, BLOCK_SIZE_K)
    total_tiles = num_pid_m * num_pid_n
    if num_sms > 0 and total_tiles > num_sms:
        total_streamk_tiles = total_tiles % num_sms
        total_streamk_tiles = total_streamk_tiles + num_sms
        total_full_tiles = total_tiles - total_streamk_tiles
        total_streamk_iters = total_streamk_tiles * iters_per_tile
        streamk_iters_pcu = total_streamk_iters // num_sms
        streamk_remainder_iters = total_streamk_iters % num_sms
    else:
        total_full_tiles = total_tiles
        total_streamk_tiles = 0
        streamk_iters_pcu = 0
        streamk_remainder_iters = 0
        total_streamk_iters = 0

    acc_dtype = tl.float32 if C.type.element_ty != tl.int8 else tl.int32
    rk = tl.arange(0, BLOCK_SIZE_K)

    for tile_id in range(pid, total_full_tiles, num_sms):
        if GROUP_SIZE_M == 1:
            pid_m = tile_id // num_pid_n
            pid_n = tile_id % num_pid_n
        else:
            num_pid_in_group = GROUP_SIZE_M * num_pid_n
            group_id = tile_id // num_pid_in_group
            first_pid_m = group_id * GROUP_SIZE_M
            group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
            pid_m = first_pid_m + (tile_id % group_size_m)
            pid_n = (tile_id % num_pid_in_group) // group_size_m

        rm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M))%M
        rn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))%N
        rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
        rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)
        A_BASE = A + rm[:, None] * stride_am + rk[None, :] * stride_ak
        B_BASE = B + rk[:, None] * stride_bk + rn[None, :] * stride_bn

        if BIAS:
            bias_ = bias_ptr + rm * stride_bias
            bias = tl.load(bias_, mask=rm < M, other=0.0)

        loop_k = tl.cdiv(K, BLOCK_SIZE_K)
        if not EVEN_K:
            loop_k -= 1

        acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype)
        for k in range(0, loop_k):
            a = tl.load(tl.multiple_of(A_BASE, (1, 16)))
            b = tl.load(tl.multiple_of(B_BASE, (16, 1)))
            acc += tl.dot(a, b)
            A_BASE += BLOCK_SIZE_K * stride_ak
            B_BASE += BLOCK_SIZE_K * stride_bk

        if not EVEN_K:
            k = loop_k
            rk = k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
            A_BASE = A + rm[:, None] * stride_am + rk[None, :] * stride_ak
            B_BASE = B + rk[:, None] * stride_bk + rn[None, :] * stride_bn
            a = tl.load(A_BASE, mask=rk[None, :] < K, other=0.0)
            b = tl.load(B_BASE, mask=rk[:, None] < K, other=0.0)

        c = acc.to(C.type.element_ty)
        if BIAS:
             c += bias[:, None]

        rm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        C_ = C + rm[:, None] * stride_cm + rn[None, :] * stride_cn
        mask = (rm < M)[:, None] & (rn < N)[None, :]
        tl.store(C_, c, mask=mask)

    start_iter = total_full_tiles * iters_per_tile + pid * streamk_iters_pcu + tl.minimum(pid, streamk_remainder_iters)
    last_iter = total_full_tiles * iters_per_tile + (pid + 1) * streamk_iters_pcu + tl.minimum(pid + 1, streamk_remainder_iters)
    while start_iter < last_iter:
        remainder = start_iter % iters_per_tile
        end_iter = tl.minimum(start_iter + (iters_per_tile - remainder), last_iter)
        tile_id = start_iter // iters_per_tile
        if GROUP_SIZE_M == 1:
            pid_m = tile_id // num_pid_n
            pid_n = tile_id % num_pid_n
        else:
            num_pid_in_group = GROUP_SIZE_M * num_pid_n
            group_id = tile_id // num_pid_in_group
            first_pid_m = group_id * GROUP_SIZE_M
            group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
            pid_m = first_pid_m + (tile_id % group_size_m)
            pid_n = (tile_id % num_pid_in_group) // group_size_m

        rm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M))%M
        rn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))%N
        rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
        rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)
        A_BASE = A + rm[:, None] * stride_am + rk[None, :] * stride_ak + BLOCK_SIZE_K * stride_ak * remainder
        B_BASE = B + rk[:, None] * stride_bk + rn[None, :] * stride_bn + BLOCK_SIZE_K * stride_bk * remainder

        if BIAS:
            bias_ = bias_ptr + rm * stride_bias
            bias = tl.load(bias_, mask=rm < M, other=0.0)

        acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype)
        for current_iter in range(start_iter, end_iter):
            if EVEN_K:
                a = tl.load(tl.multiple_of(A_BASE, (1, 16)))
                b = tl.load(tl.multiple_of(B_BASE, (16, 1)))
            else:
                global_k_offset = (current_iter % iters_per_tile) * BLOCK_SIZE_K
                k_mask = global_k_offset + rk < K
                a = tl.load(A_BASE, mask=k_mask[None, :], other=0.0)
                b = tl.load(B_BASE, mask=k_mask[:, None], other=0.0)
            acc += tl.dot(a, b)
            A_BASE += BLOCK_SIZE_K * stride_ak
            B_BASE += BLOCK_SIZE_K * stride_bk

        tile_iter = tile_id * iters_per_tile
        if start_iter == tile_iter:
            tile_iter_end = tile_iter + iters_per_tile
            next_pid = pid + 1
            end = end_iter
            while (end < tile_iter_end and next_pid < num_sms):
                while tl.atomic_cas(locks + next_pid, 1, 1) != 1:
              #  while tl.load(locks + next_pid, cache_modifier = ".cg") != 1:
                    pass
                rm1 = tl.arange(0, BLOCK_SIZE_M)
                rn1 = tl.arange(0, BLOCK_SIZE_N)
                rm1 = tl.max_contiguous(tl.multiple_of(rm1, BLOCK_SIZE_M), BLOCK_SIZE_M)
                rn1 = tl.max_contiguous(tl.multiple_of(rn1, BLOCK_SIZE_N), BLOCK_SIZE_N)
                P_ = P + next_pid * BLOCK_SIZE_M * BLOCK_SIZE_N + rm1[:, None] * BLOCK_SIZE_N + rn1[None, :]
                acc += tl.load(tl.multiple_of(P_, (1, 16)))
              #  acc += tl.load(P_)
                end += streamk_iters_pcu + (next_pid < streamk_remainder_iters)
                next_pid += 1

            c = acc.to(C.type.element_ty)
            if BIAS:
                 c += bias[:, None]

            rm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M))%M
            rn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))%N
            rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
            rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)
            C_ = C + rm[:, None] * stride_cm + rn[None, :] * stride_cn
            mask = (rm < M)[:, None] & (rn < N)[None, :]
            tl.store(C_, c, mask=mask)
        else:
            rm1 = tl.arange(0, BLOCK_SIZE_M)
            rn1 = tl.arange(0, BLOCK_SIZE_N)
            rm1 = tl.max_contiguous(tl.multiple_of(rm1, BLOCK_SIZE_M), BLOCK_SIZE_M)
            rn1 = tl.max_contiguous(tl.multiple_of(rn1, BLOCK_SIZE_N), BLOCK_SIZE_N)
            P_ = P + pid * BLOCK_SIZE_M * BLOCK_SIZE_N +  rm1[:, None] * BLOCK_SIZE_N + rn1[None, :]
            tl.store(P_, acc)
            tl.debug_barrier()
            tl.atomic_xchg(locks + pid, 1)
     #       tl.store(locks + pid, 1, cache_modifier=".wt")

        start_iter = end_iter



def matmul_BM128_BN128_BK128_GM1_nW8_nS2_EU0_kP1_mfma32(a, b, c, bias, P, locks, M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    grid = num_sms
    streamk_gemm_BM128_BN128_BK128_GM1_nW8_nS2_EU0_kP1_mfma32[grid,](
        a, b, c, bias, P, locks,
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
        BIAS = False,
        EVEN_K = False
    )
    return c

def try_matmul_config_BM128_BN128_BK128_GM1_nW8_nS2_EU0_kP1_mfma32(a, b, c, bias, M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        locks = torch.zeros((num_sms,), device = "cuda", dtype = torch.int32)
        P = torch.zeros((num_sms,  128*128), device="cuda", dtype=torch.float32)
        matmul_BM128_BN128_BK128_GM1_nW8_nS2_EU0_kP1_mfma32(a, b, c, bias, P, locks, M, N, K, num_sms, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1), bias_stride)
        return True
    except Exception as e:
        print(f'invalid config(runtime): BM128_BN128_BK128_GM1_nW8_nS2_EU0_kP1_mfma32: ', e, flush=True)
        return False




@triton.jit()
def streamk_gemm_BM128_BN256_BK64_GM8_nW4_nS2_EU0_kP2_mfma32(
        A, B, C, bias_ptr, P, locks,
        M, N, K, num_sms,
        stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn, stride_bias,
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
        GROUP_SIZE_M: tl.constexpr, BIAS: tl.constexpr, EVEN_K: tl.constexpr,
):
    pid = tl.program_id(0)
    pid = (pid % 8) * (num_sms // 8) + (pid // 8)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    iters_per_tile = tl.cdiv(K, BLOCK_SIZE_K)
    total_tiles = num_pid_m * num_pid_n
    if num_sms > 0 and total_tiles > num_sms:
        total_streamk_tiles = total_tiles % num_sms
        total_streamk_tiles = total_streamk_tiles + num_sms
        total_full_tiles = total_tiles - total_streamk_tiles
        total_streamk_iters = total_streamk_tiles * iters_per_tile
        streamk_iters_pcu = total_streamk_iters // num_sms
        streamk_remainder_iters = total_streamk_iters % num_sms
    else:
        total_full_tiles = total_tiles
        total_streamk_tiles = 0
        streamk_iters_pcu = 0
        streamk_remainder_iters = 0
        total_streamk_iters = 0

    acc_dtype = tl.float32 if C.type.element_ty != tl.int8 else tl.int32
    rk = tl.arange(0, BLOCK_SIZE_K)

    for tile_id in range(pid, total_full_tiles, num_sms):
        if GROUP_SIZE_M == 1:
            pid_m = tile_id // num_pid_n
            pid_n = tile_id % num_pid_n
        else:
            num_pid_in_group = GROUP_SIZE_M * num_pid_n
            group_id = tile_id // num_pid_in_group
            first_pid_m = group_id * GROUP_SIZE_M
            group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
            pid_m = first_pid_m + (tile_id % group_size_m)
            pid_n = (tile_id % num_pid_in_group) // group_size_m

        rm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M))%M
        rn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))%N
        rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
        rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)
        A_BASE = A + rm[:, None] * stride_am + rk[None, :] * stride_ak
        B_BASE = B + rk[:, None] * stride_bk + rn[None, :] * stride_bn

        if BIAS:
            bias_ = bias_ptr + rm * stride_bias
            bias = tl.load(bias_, mask=rm < M, other=0.0)

        loop_k = tl.cdiv(K, BLOCK_SIZE_K)
        if not EVEN_K:
            loop_k -= 1

        acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype)
        for k in range(0, loop_k):
            a = tl.load(tl.multiple_of(A_BASE, (1, 16)))
            b = tl.load(tl.multiple_of(B_BASE, (16, 1)))
            acc += tl.dot(a, b)
            A_BASE += BLOCK_SIZE_K * stride_ak
            B_BASE += BLOCK_SIZE_K * stride_bk

        if not EVEN_K:
            k = loop_k
            rk = k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
            A_BASE = A + rm[:, None] * stride_am + rk[None, :] * stride_ak
            B_BASE = B + rk[:, None] * stride_bk + rn[None, :] * stride_bn
            a = tl.load(A_BASE, mask=rk[None, :] < K, other=0.0)
            b = tl.load(B_BASE, mask=rk[:, None] < K, other=0.0)

        c = acc.to(C.type.element_ty)
        if BIAS:
             c += bias[:, None]

        rm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        C_ = C + rm[:, None] * stride_cm + rn[None, :] * stride_cn
        mask = (rm < M)[:, None] & (rn < N)[None, :]
        tl.store(C_, c, mask=mask)

    start_iter = total_full_tiles * iters_per_tile + pid * streamk_iters_pcu + tl.minimum(pid, streamk_remainder_iters)
    last_iter = total_full_tiles * iters_per_tile + (pid + 1) * streamk_iters_pcu + tl.minimum(pid + 1, streamk_remainder_iters)
    while start_iter < last_iter:
        remainder = start_iter % iters_per_tile
        end_iter = tl.minimum(start_iter + (iters_per_tile - remainder), last_iter)
        tile_id = start_iter // iters_per_tile
        if GROUP_SIZE_M == 1:
            pid_m = tile_id // num_pid_n
            pid_n = tile_id % num_pid_n
        else:
            num_pid_in_group = GROUP_SIZE_M * num_pid_n
            group_id = tile_id // num_pid_in_group
            first_pid_m = group_id * GROUP_SIZE_M
            group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
            pid_m = first_pid_m + (tile_id % group_size_m)
            pid_n = (tile_id % num_pid_in_group) // group_size_m

        rm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M))%M
        rn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))%N
        rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
        rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)
        A_BASE = A + rm[:, None] * stride_am + rk[None, :] * stride_ak + BLOCK_SIZE_K * stride_ak * remainder
        B_BASE = B + rk[:, None] * stride_bk + rn[None, :] * stride_bn + BLOCK_SIZE_K * stride_bk * remainder

        if BIAS:
            bias_ = bias_ptr + rm * stride_bias
            bias = tl.load(bias_, mask=rm < M, other=0.0)

        acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype)
        for current_iter in range(start_iter, end_iter):
            if EVEN_K:
                a = tl.load(tl.multiple_of(A_BASE, (1, 16)))
                b = tl.load(tl.multiple_of(B_BASE, (16, 1)))
            else:
                global_k_offset = (current_iter % iters_per_tile) * BLOCK_SIZE_K
                k_mask = global_k_offset + rk < K
                a = tl.load(A_BASE, mask=k_mask[None, :], other=0.0)
                b = tl.load(B_BASE, mask=k_mask[:, None], other=0.0)
            acc += tl.dot(a, b)
            A_BASE += BLOCK_SIZE_K * stride_ak
            B_BASE += BLOCK_SIZE_K * stride_bk

        tile_iter = tile_id * iters_per_tile
        if start_iter == tile_iter:
            tile_iter_end = tile_iter + iters_per_tile
            next_pid = pid + 1
            end = end_iter
            while (end < tile_iter_end and next_pid < num_sms):
                while tl.atomic_cas(locks + next_pid, 1, 1) != 1:
              #  while tl.load(locks + next_pid, cache_modifier = ".cg") != 1:
                    pass
                rm1 = tl.arange(0, BLOCK_SIZE_M)
                rn1 = tl.arange(0, BLOCK_SIZE_N)
                rm1 = tl.max_contiguous(tl.multiple_of(rm1, BLOCK_SIZE_M), BLOCK_SIZE_M)
                rn1 = tl.max_contiguous(tl.multiple_of(rn1, BLOCK_SIZE_N), BLOCK_SIZE_N)
                P_ = P + next_pid * BLOCK_SIZE_M * BLOCK_SIZE_N + rm1[:, None] * BLOCK_SIZE_N + rn1[None, :]
                acc += tl.load(tl.multiple_of(P_, (1, 16)))
              #  acc += tl.load(P_)
                end += streamk_iters_pcu + (next_pid < streamk_remainder_iters)
                next_pid += 1

            c = acc.to(C.type.element_ty)
            if BIAS:
                 c += bias[:, None]

            rm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M))%M
            rn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))%N
            rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
            rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)
            C_ = C + rm[:, None] * stride_cm + rn[None, :] * stride_cn
            mask = (rm < M)[:, None] & (rn < N)[None, :]
            tl.store(C_, c, mask=mask)
        else:
            rm1 = tl.arange(0, BLOCK_SIZE_M)
            rn1 = tl.arange(0, BLOCK_SIZE_N)
            rm1 = tl.max_contiguous(tl.multiple_of(rm1, BLOCK_SIZE_M), BLOCK_SIZE_M)
            rn1 = tl.max_contiguous(tl.multiple_of(rn1, BLOCK_SIZE_N), BLOCK_SIZE_N)
            P_ = P + pid * BLOCK_SIZE_M * BLOCK_SIZE_N +  rm1[:, None] * BLOCK_SIZE_N + rn1[None, :]
            tl.store(P_, acc)
            tl.debug_barrier()
            tl.atomic_xchg(locks + pid, 1)
     #       tl.store(locks + pid, 1, cache_modifier=".wt")

        start_iter = end_iter



def matmul_BM128_BN256_BK64_GM8_nW4_nS2_EU0_kP2_mfma32(a, b, c, bias, P, locks, M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    grid = num_sms
    streamk_gemm_BM128_BN256_BK64_GM8_nW4_nS2_EU0_kP2_mfma32[grid,](
        a, b, c, bias, P, locks,
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
        BIAS = False,
        EVEN_K = True
    )
    return c

def try_matmul_config_BM128_BN256_BK64_GM8_nW4_nS2_EU0_kP2_mfma32(a, b, c, bias, M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        locks = torch.zeros((num_sms,), device = "cuda", dtype = torch.int32)
        P = torch.zeros((num_sms,  128*256), device="cuda", dtype=torch.float32)
        matmul_BM128_BN256_BK64_GM8_nW4_nS2_EU0_kP2_mfma32(a, b, c, bias, P, locks, M, N, K, num_sms, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1), bias_stride)
        return True
    except Exception as e:
        print(f'invalid config(runtime): BM128_BN256_BK64_GM8_nW4_nS2_EU0_kP2_mfma32: ', e, flush=True)
        return False




@triton.jit()
def streamk_gemm_BM128_BN256_BK64_GM16_nW8_nS2_EU0_kP2_mfma32(
        A, B, C, bias_ptr, P, locks,
        M, N, K, num_sms,
        stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn, stride_bias,
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
        GROUP_SIZE_M: tl.constexpr, BIAS: tl.constexpr, EVEN_K: tl.constexpr,
):
    pid = tl.program_id(0)
    pid = (pid % 8) * (num_sms // 8) + (pid // 8)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    iters_per_tile = tl.cdiv(K, BLOCK_SIZE_K)
    total_tiles = num_pid_m * num_pid_n
    if num_sms > 0 and total_tiles > num_sms:
        total_streamk_tiles = total_tiles % num_sms
        total_streamk_tiles = total_streamk_tiles + num_sms
        total_full_tiles = total_tiles - total_streamk_tiles
        total_streamk_iters = total_streamk_tiles * iters_per_tile
        streamk_iters_pcu = total_streamk_iters // num_sms
        streamk_remainder_iters = total_streamk_iters % num_sms
    else:
        total_full_tiles = total_tiles
        total_streamk_tiles = 0
        streamk_iters_pcu = 0
        streamk_remainder_iters = 0
        total_streamk_iters = 0

    acc_dtype = tl.float32 if C.type.element_ty != tl.int8 else tl.int32
    rk = tl.arange(0, BLOCK_SIZE_K)

    for tile_id in range(pid, total_full_tiles, num_sms):
        if GROUP_SIZE_M == 1:
            pid_m = tile_id // num_pid_n
            pid_n = tile_id % num_pid_n
        else:
            num_pid_in_group = GROUP_SIZE_M * num_pid_n
            group_id = tile_id // num_pid_in_group
            first_pid_m = group_id * GROUP_SIZE_M
            group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
            pid_m = first_pid_m + (tile_id % group_size_m)
            pid_n = (tile_id % num_pid_in_group) // group_size_m

        rm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M))%M
        rn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))%N
        rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
        rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)
        A_BASE = A + rm[:, None] * stride_am + rk[None, :] * stride_ak
        B_BASE = B + rk[:, None] * stride_bk + rn[None, :] * stride_bn

        if BIAS:
            bias_ = bias_ptr + rm * stride_bias
            bias = tl.load(bias_, mask=rm < M, other=0.0)

        loop_k = tl.cdiv(K, BLOCK_SIZE_K)
        if not EVEN_K:
            loop_k -= 1

        acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype)
        for k in range(0, loop_k):
            a = tl.load(tl.multiple_of(A_BASE, (1, 16)))
            b = tl.load(tl.multiple_of(B_BASE, (16, 1)))
            acc += tl.dot(a, b)
            A_BASE += BLOCK_SIZE_K * stride_ak
            B_BASE += BLOCK_SIZE_K * stride_bk

        if not EVEN_K:
            k = loop_k
            rk = k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
            A_BASE = A + rm[:, None] * stride_am + rk[None, :] * stride_ak
            B_BASE = B + rk[:, None] * stride_bk + rn[None, :] * stride_bn
            a = tl.load(A_BASE, mask=rk[None, :] < K, other=0.0)
            b = tl.load(B_BASE, mask=rk[:, None] < K, other=0.0)

        c = acc.to(C.type.element_ty)
        if BIAS:
             c += bias[:, None]

        rm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        C_ = C + rm[:, None] * stride_cm + rn[None, :] * stride_cn
        mask = (rm < M)[:, None] & (rn < N)[None, :]
        tl.store(C_, c, mask=mask)

    start_iter = total_full_tiles * iters_per_tile + pid * streamk_iters_pcu + tl.minimum(pid, streamk_remainder_iters)
    last_iter = total_full_tiles * iters_per_tile + (pid + 1) * streamk_iters_pcu + tl.minimum(pid + 1, streamk_remainder_iters)
    while start_iter < last_iter:
        remainder = start_iter % iters_per_tile
        end_iter = tl.minimum(start_iter + (iters_per_tile - remainder), last_iter)
        tile_id = start_iter // iters_per_tile
        if GROUP_SIZE_M == 1:
            pid_m = tile_id // num_pid_n
            pid_n = tile_id % num_pid_n
        else:
            num_pid_in_group = GROUP_SIZE_M * num_pid_n
            group_id = tile_id // num_pid_in_group
            first_pid_m = group_id * GROUP_SIZE_M
            group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
            pid_m = first_pid_m + (tile_id % group_size_m)
            pid_n = (tile_id % num_pid_in_group) // group_size_m

        rm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M))%M
        rn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))%N
        rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
        rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)
        A_BASE = A + rm[:, None] * stride_am + rk[None, :] * stride_ak + BLOCK_SIZE_K * stride_ak * remainder
        B_BASE = B + rk[:, None] * stride_bk + rn[None, :] * stride_bn + BLOCK_SIZE_K * stride_bk * remainder

        if BIAS:
            bias_ = bias_ptr + rm * stride_bias
            bias = tl.load(bias_, mask=rm < M, other=0.0)

        acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype)
        for current_iter in range(start_iter, end_iter):
            if EVEN_K:
                a = tl.load(tl.multiple_of(A_BASE, (1, 16)))
                b = tl.load(tl.multiple_of(B_BASE, (16, 1)))
            else:
                global_k_offset = (current_iter % iters_per_tile) * BLOCK_SIZE_K
                k_mask = global_k_offset + rk < K
                a = tl.load(A_BASE, mask=k_mask[None, :], other=0.0)
                b = tl.load(B_BASE, mask=k_mask[:, None], other=0.0)
            acc += tl.dot(a, b)
            A_BASE += BLOCK_SIZE_K * stride_ak
            B_BASE += BLOCK_SIZE_K * stride_bk

        tile_iter = tile_id * iters_per_tile
        if start_iter == tile_iter:
            tile_iter_end = tile_iter + iters_per_tile
            next_pid = pid + 1
            end = end_iter
            while (end < tile_iter_end and next_pid < num_sms):
                while tl.atomic_cas(locks + next_pid, 1, 1) != 1:
              #  while tl.load(locks + next_pid, cache_modifier = ".cg") != 1:
                    pass
                rm1 = tl.arange(0, BLOCK_SIZE_M)
                rn1 = tl.arange(0, BLOCK_SIZE_N)
                rm1 = tl.max_contiguous(tl.multiple_of(rm1, BLOCK_SIZE_M), BLOCK_SIZE_M)
                rn1 = tl.max_contiguous(tl.multiple_of(rn1, BLOCK_SIZE_N), BLOCK_SIZE_N)
                P_ = P + next_pid * BLOCK_SIZE_M * BLOCK_SIZE_N + rm1[:, None] * BLOCK_SIZE_N + rn1[None, :]
                acc += tl.load(tl.multiple_of(P_, (1, 16)))
              #  acc += tl.load(P_)
                end += streamk_iters_pcu + (next_pid < streamk_remainder_iters)
                next_pid += 1

            c = acc.to(C.type.element_ty)
            if BIAS:
                 c += bias[:, None]

            rm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M))%M
            rn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))%N
            rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
            rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)
            C_ = C + rm[:, None] * stride_cm + rn[None, :] * stride_cn
            mask = (rm < M)[:, None] & (rn < N)[None, :]
            tl.store(C_, c, mask=mask)
        else:
            rm1 = tl.arange(0, BLOCK_SIZE_M)
            rn1 = tl.arange(0, BLOCK_SIZE_N)
            rm1 = tl.max_contiguous(tl.multiple_of(rm1, BLOCK_SIZE_M), BLOCK_SIZE_M)
            rn1 = tl.max_contiguous(tl.multiple_of(rn1, BLOCK_SIZE_N), BLOCK_SIZE_N)
            P_ = P + pid * BLOCK_SIZE_M * BLOCK_SIZE_N +  rm1[:, None] * BLOCK_SIZE_N + rn1[None, :]
            tl.store(P_, acc)
            tl.debug_barrier()
            tl.atomic_xchg(locks + pid, 1)
     #       tl.store(locks + pid, 1, cache_modifier=".wt")

        start_iter = end_iter



def matmul_BM128_BN256_BK64_GM16_nW8_nS2_EU0_kP2_mfma32(a, b, c, bias, P, locks, M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    grid = num_sms
    streamk_gemm_BM128_BN256_BK64_GM16_nW8_nS2_EU0_kP2_mfma32[grid,](
        a, b, c, bias, P, locks,
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
        BIAS = False,
        EVEN_K = True
    )
    return c

def try_matmul_config_BM128_BN256_BK64_GM16_nW8_nS2_EU0_kP2_mfma32(a, b, c, bias, M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        locks = torch.zeros((num_sms,), device = "cuda", dtype = torch.int32)
        P = torch.zeros((num_sms,  128*256), device="cuda", dtype=torch.float32)
        matmul_BM128_BN256_BK64_GM16_nW8_nS2_EU0_kP2_mfma32(a, b, c, bias, P, locks, M, N, K, num_sms, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1), bias_stride)
        return True
    except Exception as e:
        print(f'invalid config(runtime): BM128_BN256_BK64_GM16_nW8_nS2_EU0_kP2_mfma32: ', e, flush=True)
        return False




@triton.jit()
def streamk_gemm_BM256_BN64_BK64_GM1_nW8_nS2_EU0_kP2_mfma32(
        A, B, C, bias_ptr, P, locks,
        M, N, K, num_sms,
        stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn, stride_bias,
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
        GROUP_SIZE_M: tl.constexpr, BIAS: tl.constexpr, EVEN_K: tl.constexpr,
):
    pid = tl.program_id(0)
    pid = (pid % 8) * (num_sms // 8) + (pid // 8)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    iters_per_tile = tl.cdiv(K, BLOCK_SIZE_K)
    total_tiles = num_pid_m * num_pid_n
    if num_sms > 0 and total_tiles > num_sms:
        total_streamk_tiles = total_tiles % num_sms
        total_streamk_tiles = total_streamk_tiles + num_sms
        total_full_tiles = total_tiles - total_streamk_tiles
        total_streamk_iters = total_streamk_tiles * iters_per_tile
        streamk_iters_pcu = total_streamk_iters // num_sms
        streamk_remainder_iters = total_streamk_iters % num_sms
    else:
        total_full_tiles = total_tiles
        total_streamk_tiles = 0
        streamk_iters_pcu = 0
        streamk_remainder_iters = 0
        total_streamk_iters = 0

    acc_dtype = tl.float32 if C.type.element_ty != tl.int8 else tl.int32
    rk = tl.arange(0, BLOCK_SIZE_K)

    for tile_id in range(pid, total_full_tiles, num_sms):
        if GROUP_SIZE_M == 1:
            pid_m = tile_id // num_pid_n
            pid_n = tile_id % num_pid_n
        else:
            num_pid_in_group = GROUP_SIZE_M * num_pid_n
            group_id = tile_id // num_pid_in_group
            first_pid_m = group_id * GROUP_SIZE_M
            group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
            pid_m = first_pid_m + (tile_id % group_size_m)
            pid_n = (tile_id % num_pid_in_group) // group_size_m

        rm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M))%M
        rn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))%N
        rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
        rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)
        A_BASE = A + rm[:, None] * stride_am + rk[None, :] * stride_ak
        B_BASE = B + rk[:, None] * stride_bk + rn[None, :] * stride_bn

        if BIAS:
            bias_ = bias_ptr + rm * stride_bias
            bias = tl.load(bias_, mask=rm < M, other=0.0)

        loop_k = tl.cdiv(K, BLOCK_SIZE_K)
        if not EVEN_K:
            loop_k -= 1

        acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype)
        for k in range(0, loop_k):
            a = tl.load(tl.multiple_of(A_BASE, (1, 16)))
            b = tl.load(tl.multiple_of(B_BASE, (16, 1)))
            acc += tl.dot(a, b)
            A_BASE += BLOCK_SIZE_K * stride_ak
            B_BASE += BLOCK_SIZE_K * stride_bk

        if not EVEN_K:
            k = loop_k
            rk = k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
            A_BASE = A + rm[:, None] * stride_am + rk[None, :] * stride_ak
            B_BASE = B + rk[:, None] * stride_bk + rn[None, :] * stride_bn
            a = tl.load(A_BASE, mask=rk[None, :] < K, other=0.0)
            b = tl.load(B_BASE, mask=rk[:, None] < K, other=0.0)

        c = acc.to(C.type.element_ty)
        if BIAS:
             c += bias[:, None]

        rm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        C_ = C + rm[:, None] * stride_cm + rn[None, :] * stride_cn
        mask = (rm < M)[:, None] & (rn < N)[None, :]
        tl.store(C_, c, mask=mask)

    start_iter = total_full_tiles * iters_per_tile + pid * streamk_iters_pcu + tl.minimum(pid, streamk_remainder_iters)
    last_iter = total_full_tiles * iters_per_tile + (pid + 1) * streamk_iters_pcu + tl.minimum(pid + 1, streamk_remainder_iters)
    while start_iter < last_iter:
        remainder = start_iter % iters_per_tile
        end_iter = tl.minimum(start_iter + (iters_per_tile - remainder), last_iter)
        tile_id = start_iter // iters_per_tile
        if GROUP_SIZE_M == 1:
            pid_m = tile_id // num_pid_n
            pid_n = tile_id % num_pid_n
        else:
            num_pid_in_group = GROUP_SIZE_M * num_pid_n
            group_id = tile_id // num_pid_in_group
            first_pid_m = group_id * GROUP_SIZE_M
            group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
            pid_m = first_pid_m + (tile_id % group_size_m)
            pid_n = (tile_id % num_pid_in_group) // group_size_m

        rm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M))%M
        rn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))%N
        rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
        rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)
        A_BASE = A + rm[:, None] * stride_am + rk[None, :] * stride_ak + BLOCK_SIZE_K * stride_ak * remainder
        B_BASE = B + rk[:, None] * stride_bk + rn[None, :] * stride_bn + BLOCK_SIZE_K * stride_bk * remainder

        if BIAS:
            bias_ = bias_ptr + rm * stride_bias
            bias = tl.load(bias_, mask=rm < M, other=0.0)

        acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype)
        for current_iter in range(start_iter, end_iter):
            if EVEN_K:
                a = tl.load(tl.multiple_of(A_BASE, (1, 16)))
                b = tl.load(tl.multiple_of(B_BASE, (16, 1)))
            else:
                global_k_offset = (current_iter % iters_per_tile) * BLOCK_SIZE_K
                k_mask = global_k_offset + rk < K
                a = tl.load(A_BASE, mask=k_mask[None, :], other=0.0)
                b = tl.load(B_BASE, mask=k_mask[:, None], other=0.0)
            acc += tl.dot(a, b)
            A_BASE += BLOCK_SIZE_K * stride_ak
            B_BASE += BLOCK_SIZE_K * stride_bk

        tile_iter = tile_id * iters_per_tile
        if start_iter == tile_iter:
            tile_iter_end = tile_iter + iters_per_tile
            next_pid = pid + 1
            end = end_iter
            while (end < tile_iter_end and next_pid < num_sms):
                while tl.atomic_cas(locks + next_pid, 1, 1) != 1:
              #  while tl.load(locks + next_pid, cache_modifier = ".cg") != 1:
                    pass
                rm1 = tl.arange(0, BLOCK_SIZE_M)
                rn1 = tl.arange(0, BLOCK_SIZE_N)
                rm1 = tl.max_contiguous(tl.multiple_of(rm1, BLOCK_SIZE_M), BLOCK_SIZE_M)
                rn1 = tl.max_contiguous(tl.multiple_of(rn1, BLOCK_SIZE_N), BLOCK_SIZE_N)
                P_ = P + next_pid * BLOCK_SIZE_M * BLOCK_SIZE_N + rm1[:, None] * BLOCK_SIZE_N + rn1[None, :]
                acc += tl.load(tl.multiple_of(P_, (1, 16)))
              #  acc += tl.load(P_)
                end += streamk_iters_pcu + (next_pid < streamk_remainder_iters)
                next_pid += 1

            c = acc.to(C.type.element_ty)
            if BIAS:
                 c += bias[:, None]

            rm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M))%M
            rn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))%N
            rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
            rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)
            C_ = C + rm[:, None] * stride_cm + rn[None, :] * stride_cn
            mask = (rm < M)[:, None] & (rn < N)[None, :]
            tl.store(C_, c, mask=mask)
        else:
            rm1 = tl.arange(0, BLOCK_SIZE_M)
            rn1 = tl.arange(0, BLOCK_SIZE_N)
            rm1 = tl.max_contiguous(tl.multiple_of(rm1, BLOCK_SIZE_M), BLOCK_SIZE_M)
            rn1 = tl.max_contiguous(tl.multiple_of(rn1, BLOCK_SIZE_N), BLOCK_SIZE_N)
            P_ = P + pid * BLOCK_SIZE_M * BLOCK_SIZE_N +  rm1[:, None] * BLOCK_SIZE_N + rn1[None, :]
            tl.store(P_, acc)
            tl.debug_barrier()
            tl.atomic_xchg(locks + pid, 1)
     #       tl.store(locks + pid, 1, cache_modifier=".wt")

        start_iter = end_iter



def matmul_BM256_BN64_BK64_GM1_nW8_nS2_EU0_kP2_mfma32(a, b, c, bias, P, locks, M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    grid = num_sms
    streamk_gemm_BM256_BN64_BK64_GM1_nW8_nS2_EU0_kP2_mfma32[grid,](
        a, b, c, bias, P, locks,
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
        BIAS = False,
        EVEN_K = True
    )
    return c

def try_matmul_config_BM256_BN64_BK64_GM1_nW8_nS2_EU0_kP2_mfma32(a, b, c, bias, M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        locks = torch.zeros((num_sms,), device = "cuda", dtype = torch.int32)
        P = torch.zeros((num_sms,  256*64), device="cuda", dtype=torch.float32)
        matmul_BM256_BN64_BK64_GM1_nW8_nS2_EU0_kP2_mfma32(a, b, c, bias, P, locks, M, N, K, num_sms, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1), bias_stride)
        return True
    except Exception as e:
        print(f'invalid config(runtime): BM256_BN64_BK64_GM1_nW8_nS2_EU0_kP2_mfma32: ', e, flush=True)
        return False




@triton.jit()
def streamk_gemm_BM256_BN128_BK64_GM8_nW4_nS2_EU0_kP2_mfma32(
        A, B, C, bias_ptr, P, locks,
        M, N, K, num_sms,
        stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn, stride_bias,
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
        GROUP_SIZE_M: tl.constexpr, BIAS: tl.constexpr, EVEN_K: tl.constexpr,
):
    pid = tl.program_id(0)
    pid = (pid % 8) * (num_sms // 8) + (pid // 8)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    iters_per_tile = tl.cdiv(K, BLOCK_SIZE_K)
    total_tiles = num_pid_m * num_pid_n
    if num_sms > 0 and total_tiles > num_sms:
        total_streamk_tiles = total_tiles % num_sms
        total_streamk_tiles = total_streamk_tiles + num_sms
        total_full_tiles = total_tiles - total_streamk_tiles
        total_streamk_iters = total_streamk_tiles * iters_per_tile
        streamk_iters_pcu = total_streamk_iters // num_sms
        streamk_remainder_iters = total_streamk_iters % num_sms
    else:
        total_full_tiles = total_tiles
        total_streamk_tiles = 0
        streamk_iters_pcu = 0
        streamk_remainder_iters = 0
        total_streamk_iters = 0

    acc_dtype = tl.float32 if C.type.element_ty != tl.int8 else tl.int32
    rk = tl.arange(0, BLOCK_SIZE_K)

    for tile_id in range(pid, total_full_tiles, num_sms):
        if GROUP_SIZE_M == 1:
            pid_m = tile_id // num_pid_n
            pid_n = tile_id % num_pid_n
        else:
            num_pid_in_group = GROUP_SIZE_M * num_pid_n
            group_id = tile_id // num_pid_in_group
            first_pid_m = group_id * GROUP_SIZE_M
            group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
            pid_m = first_pid_m + (tile_id % group_size_m)
            pid_n = (tile_id % num_pid_in_group) // group_size_m

        rm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M))%M
        rn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))%N
        rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
        rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)
        A_BASE = A + rm[:, None] * stride_am + rk[None, :] * stride_ak
        B_BASE = B + rk[:, None] * stride_bk + rn[None, :] * stride_bn

        if BIAS:
            bias_ = bias_ptr + rm * stride_bias
            bias = tl.load(bias_, mask=rm < M, other=0.0)

        loop_k = tl.cdiv(K, BLOCK_SIZE_K)
        if not EVEN_K:
            loop_k -= 1

        acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype)
        for k in range(0, loop_k):
            a = tl.load(tl.multiple_of(A_BASE, (1, 16)))
            b = tl.load(tl.multiple_of(B_BASE, (16, 1)))
            acc += tl.dot(a, b)
            A_BASE += BLOCK_SIZE_K * stride_ak
            B_BASE += BLOCK_SIZE_K * stride_bk

        if not EVEN_K:
            k = loop_k
            rk = k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
            A_BASE = A + rm[:, None] * stride_am + rk[None, :] * stride_ak
            B_BASE = B + rk[:, None] * stride_bk + rn[None, :] * stride_bn
            a = tl.load(A_BASE, mask=rk[None, :] < K, other=0.0)
            b = tl.load(B_BASE, mask=rk[:, None] < K, other=0.0)

        c = acc.to(C.type.element_ty)
        if BIAS:
             c += bias[:, None]

        rm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        C_ = C + rm[:, None] * stride_cm + rn[None, :] * stride_cn
        mask = (rm < M)[:, None] & (rn < N)[None, :]
        tl.store(C_, c, mask=mask)

    start_iter = total_full_tiles * iters_per_tile + pid * streamk_iters_pcu + tl.minimum(pid, streamk_remainder_iters)
    last_iter = total_full_tiles * iters_per_tile + (pid + 1) * streamk_iters_pcu + tl.minimum(pid + 1, streamk_remainder_iters)
    while start_iter < last_iter:
        remainder = start_iter % iters_per_tile
        end_iter = tl.minimum(start_iter + (iters_per_tile - remainder), last_iter)
        tile_id = start_iter // iters_per_tile
        if GROUP_SIZE_M == 1:
            pid_m = tile_id // num_pid_n
            pid_n = tile_id % num_pid_n
        else:
            num_pid_in_group = GROUP_SIZE_M * num_pid_n
            group_id = tile_id // num_pid_in_group
            first_pid_m = group_id * GROUP_SIZE_M
            group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
            pid_m = first_pid_m + (tile_id % group_size_m)
            pid_n = (tile_id % num_pid_in_group) // group_size_m

        rm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M))%M
        rn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))%N
        rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
        rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)
        A_BASE = A + rm[:, None] * stride_am + rk[None, :] * stride_ak + BLOCK_SIZE_K * stride_ak * remainder
        B_BASE = B + rk[:, None] * stride_bk + rn[None, :] * stride_bn + BLOCK_SIZE_K * stride_bk * remainder

        if BIAS:
            bias_ = bias_ptr + rm * stride_bias
            bias = tl.load(bias_, mask=rm < M, other=0.0)

        acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype)
        for current_iter in range(start_iter, end_iter):
            if EVEN_K:
                a = tl.load(tl.multiple_of(A_BASE, (1, 16)))
                b = tl.load(tl.multiple_of(B_BASE, (16, 1)))
            else:
                global_k_offset = (current_iter % iters_per_tile) * BLOCK_SIZE_K
                k_mask = global_k_offset + rk < K
                a = tl.load(A_BASE, mask=k_mask[None, :], other=0.0)
                b = tl.load(B_BASE, mask=k_mask[:, None], other=0.0)
            acc += tl.dot(a, b)
            A_BASE += BLOCK_SIZE_K * stride_ak
            B_BASE += BLOCK_SIZE_K * stride_bk

        tile_iter = tile_id * iters_per_tile
        if start_iter == tile_iter:
            tile_iter_end = tile_iter + iters_per_tile
            next_pid = pid + 1
            end = end_iter
            while (end < tile_iter_end and next_pid < num_sms):
                while tl.atomic_cas(locks + next_pid, 1, 1) != 1:
              #  while tl.load(locks + next_pid, cache_modifier = ".cg") != 1:
                    pass
                rm1 = tl.arange(0, BLOCK_SIZE_M)
                rn1 = tl.arange(0, BLOCK_SIZE_N)
                rm1 = tl.max_contiguous(tl.multiple_of(rm1, BLOCK_SIZE_M), BLOCK_SIZE_M)
                rn1 = tl.max_contiguous(tl.multiple_of(rn1, BLOCK_SIZE_N), BLOCK_SIZE_N)
                P_ = P + next_pid * BLOCK_SIZE_M * BLOCK_SIZE_N + rm1[:, None] * BLOCK_SIZE_N + rn1[None, :]
                acc += tl.load(tl.multiple_of(P_, (1, 16)))
              #  acc += tl.load(P_)
                end += streamk_iters_pcu + (next_pid < streamk_remainder_iters)
                next_pid += 1

            c = acc.to(C.type.element_ty)
            if BIAS:
                 c += bias[:, None]

            rm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M))%M
            rn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))%N
            rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
            rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)
            C_ = C + rm[:, None] * stride_cm + rn[None, :] * stride_cn
            mask = (rm < M)[:, None] & (rn < N)[None, :]
            tl.store(C_, c, mask=mask)
        else:
            rm1 = tl.arange(0, BLOCK_SIZE_M)
            rn1 = tl.arange(0, BLOCK_SIZE_N)
            rm1 = tl.max_contiguous(tl.multiple_of(rm1, BLOCK_SIZE_M), BLOCK_SIZE_M)
            rn1 = tl.max_contiguous(tl.multiple_of(rn1, BLOCK_SIZE_N), BLOCK_SIZE_N)
            P_ = P + pid * BLOCK_SIZE_M * BLOCK_SIZE_N +  rm1[:, None] * BLOCK_SIZE_N + rn1[None, :]
            tl.store(P_, acc)
            tl.debug_barrier()
            tl.atomic_xchg(locks + pid, 1)
     #       tl.store(locks + pid, 1, cache_modifier=".wt")

        start_iter = end_iter



def matmul_BM256_BN128_BK64_GM8_nW4_nS2_EU0_kP2_mfma32(a, b, c, bias, P, locks, M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    grid = num_sms
    streamk_gemm_BM256_BN128_BK64_GM8_nW4_nS2_EU0_kP2_mfma32[grid,](
        a, b, c, bias, P, locks,
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
        BIAS = False,
        EVEN_K = True
    )
    return c

def try_matmul_config_BM256_BN128_BK64_GM8_nW4_nS2_EU0_kP2_mfma32(a, b, c, bias, M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        locks = torch.zeros((num_sms,), device = "cuda", dtype = torch.int32)
        P = torch.zeros((num_sms,  256*128), device="cuda", dtype=torch.float32)
        matmul_BM256_BN128_BK64_GM8_nW4_nS2_EU0_kP2_mfma32(a, b, c, bias, P, locks, M, N, K, num_sms, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1), bias_stride)
        return True
    except Exception as e:
        print(f'invalid config(runtime): BM256_BN128_BK64_GM8_nW4_nS2_EU0_kP2_mfma32: ', e, flush=True)
        return False




@triton.jit()
def streamk_gemm_BM256_BN256_BK64_GM1_nW4_nS2_EU0_kP2_mfma32(
        A, B, C, bias_ptr, P, locks,
        M, N, K, num_sms,
        stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn, stride_bias,
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
        GROUP_SIZE_M: tl.constexpr, BIAS: tl.constexpr, EVEN_K: tl.constexpr,
):
    pid = tl.program_id(0)
    pid = (pid % 8) * (num_sms // 8) + (pid // 8)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    iters_per_tile = tl.cdiv(K, BLOCK_SIZE_K)
    total_tiles = num_pid_m * num_pid_n
    if num_sms > 0 and total_tiles > num_sms:
        total_streamk_tiles = total_tiles % num_sms
        total_streamk_tiles = total_streamk_tiles + num_sms
        total_full_tiles = total_tiles - total_streamk_tiles
        total_streamk_iters = total_streamk_tiles * iters_per_tile
        streamk_iters_pcu = total_streamk_iters // num_sms
        streamk_remainder_iters = total_streamk_iters % num_sms
    else:
        total_full_tiles = total_tiles
        total_streamk_tiles = 0
        streamk_iters_pcu = 0
        streamk_remainder_iters = 0
        total_streamk_iters = 0

    acc_dtype = tl.float32 if C.type.element_ty != tl.int8 else tl.int32
    rk = tl.arange(0, BLOCK_SIZE_K)

    for tile_id in range(pid, total_full_tiles, num_sms):
        if GROUP_SIZE_M == 1:
            pid_m = tile_id // num_pid_n
            pid_n = tile_id % num_pid_n
        else:
            num_pid_in_group = GROUP_SIZE_M * num_pid_n
            group_id = tile_id // num_pid_in_group
            first_pid_m = group_id * GROUP_SIZE_M
            group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
            pid_m = first_pid_m + (tile_id % group_size_m)
            pid_n = (tile_id % num_pid_in_group) // group_size_m

        rm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M))%M
        rn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))%N
        rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
        rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)
        A_BASE = A + rm[:, None] * stride_am + rk[None, :] * stride_ak
        B_BASE = B + rk[:, None] * stride_bk + rn[None, :] * stride_bn

        if BIAS:
            bias_ = bias_ptr + rm * stride_bias
            bias = tl.load(bias_, mask=rm < M, other=0.0)

        loop_k = tl.cdiv(K, BLOCK_SIZE_K)
        if not EVEN_K:
            loop_k -= 1

        acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype)
        for k in range(0, loop_k):
            a = tl.load(tl.multiple_of(A_BASE, (1, 16)))
            b = tl.load(tl.multiple_of(B_BASE, (16, 1)))
            acc += tl.dot(a, b)
            A_BASE += BLOCK_SIZE_K * stride_ak
            B_BASE += BLOCK_SIZE_K * stride_bk

        if not EVEN_K:
            k = loop_k
            rk = k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
            A_BASE = A + rm[:, None] * stride_am + rk[None, :] * stride_ak
            B_BASE = B + rk[:, None] * stride_bk + rn[None, :] * stride_bn
            a = tl.load(A_BASE, mask=rk[None, :] < K, other=0.0)
            b = tl.load(B_BASE, mask=rk[:, None] < K, other=0.0)

        c = acc.to(C.type.element_ty)
        if BIAS:
             c += bias[:, None]

        rm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        C_ = C + rm[:, None] * stride_cm + rn[None, :] * stride_cn
        mask = (rm < M)[:, None] & (rn < N)[None, :]
        tl.store(C_, c, mask=mask)

    start_iter = total_full_tiles * iters_per_tile + pid * streamk_iters_pcu + tl.minimum(pid, streamk_remainder_iters)
    last_iter = total_full_tiles * iters_per_tile + (pid + 1) * streamk_iters_pcu + tl.minimum(pid + 1, streamk_remainder_iters)
    while start_iter < last_iter:
        remainder = start_iter % iters_per_tile
        end_iter = tl.minimum(start_iter + (iters_per_tile - remainder), last_iter)
        tile_id = start_iter // iters_per_tile
        if GROUP_SIZE_M == 1:
            pid_m = tile_id // num_pid_n
            pid_n = tile_id % num_pid_n
        else:
            num_pid_in_group = GROUP_SIZE_M * num_pid_n
            group_id = tile_id // num_pid_in_group
            first_pid_m = group_id * GROUP_SIZE_M
            group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
            pid_m = first_pid_m + (tile_id % group_size_m)
            pid_n = (tile_id % num_pid_in_group) // group_size_m

        rm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M))%M
        rn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))%N
        rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
        rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)
        A_BASE = A + rm[:, None] * stride_am + rk[None, :] * stride_ak + BLOCK_SIZE_K * stride_ak * remainder
        B_BASE = B + rk[:, None] * stride_bk + rn[None, :] * stride_bn + BLOCK_SIZE_K * stride_bk * remainder

        if BIAS:
            bias_ = bias_ptr + rm * stride_bias
            bias = tl.load(bias_, mask=rm < M, other=0.0)

        acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype)
        for current_iter in range(start_iter, end_iter):
            if EVEN_K:
                a = tl.load(tl.multiple_of(A_BASE, (1, 16)))
                b = tl.load(tl.multiple_of(B_BASE, (16, 1)))
            else:
                global_k_offset = (current_iter % iters_per_tile) * BLOCK_SIZE_K
                k_mask = global_k_offset + rk < K
                a = tl.load(A_BASE, mask=k_mask[None, :], other=0.0)
                b = tl.load(B_BASE, mask=k_mask[:, None], other=0.0)
            acc += tl.dot(a, b)
            A_BASE += BLOCK_SIZE_K * stride_ak
            B_BASE += BLOCK_SIZE_K * stride_bk

        tile_iter = tile_id * iters_per_tile
        if start_iter == tile_iter:
            tile_iter_end = tile_iter + iters_per_tile
            next_pid = pid + 1
            end = end_iter
            while (end < tile_iter_end and next_pid < num_sms):
                while tl.atomic_cas(locks + next_pid, 1, 1) != 1:
              #  while tl.load(locks + next_pid, cache_modifier = ".cg") != 1:
                    pass
                rm1 = tl.arange(0, BLOCK_SIZE_M)
                rn1 = tl.arange(0, BLOCK_SIZE_N)
                rm1 = tl.max_contiguous(tl.multiple_of(rm1, BLOCK_SIZE_M), BLOCK_SIZE_M)
                rn1 = tl.max_contiguous(tl.multiple_of(rn1, BLOCK_SIZE_N), BLOCK_SIZE_N)
                P_ = P + next_pid * BLOCK_SIZE_M * BLOCK_SIZE_N + rm1[:, None] * BLOCK_SIZE_N + rn1[None, :]
                acc += tl.load(tl.multiple_of(P_, (1, 16)))
              #  acc += tl.load(P_)
                end += streamk_iters_pcu + (next_pid < streamk_remainder_iters)
                next_pid += 1

            c = acc.to(C.type.element_ty)
            if BIAS:
                 c += bias[:, None]

            rm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M))%M
            rn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))%N
            rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
            rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)
            C_ = C + rm[:, None] * stride_cm + rn[None, :] * stride_cn
            mask = (rm < M)[:, None] & (rn < N)[None, :]
            tl.store(C_, c, mask=mask)
        else:
            rm1 = tl.arange(0, BLOCK_SIZE_M)
            rn1 = tl.arange(0, BLOCK_SIZE_N)
            rm1 = tl.max_contiguous(tl.multiple_of(rm1, BLOCK_SIZE_M), BLOCK_SIZE_M)
            rn1 = tl.max_contiguous(tl.multiple_of(rn1, BLOCK_SIZE_N), BLOCK_SIZE_N)
            P_ = P + pid * BLOCK_SIZE_M * BLOCK_SIZE_N +  rm1[:, None] * BLOCK_SIZE_N + rn1[None, :]
            tl.store(P_, acc)
            tl.debug_barrier()
            tl.atomic_xchg(locks + pid, 1)
     #       tl.store(locks + pid, 1, cache_modifier=".wt")

        start_iter = end_iter



def matmul_BM256_BN256_BK64_GM1_nW4_nS2_EU0_kP2_mfma32(a, b, c, bias, P, locks, M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    grid = num_sms
    streamk_gemm_BM256_BN256_BK64_GM1_nW4_nS2_EU0_kP2_mfma32[grid,](
        a, b, c, bias, P, locks,
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
        BIAS = False,
        EVEN_K = True
    )
    return c

def try_matmul_config_BM256_BN256_BK64_GM1_nW4_nS2_EU0_kP2_mfma32(a, b, c, bias, M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        locks = torch.zeros((num_sms,), device = "cuda", dtype = torch.int32)
        P = torch.zeros((num_sms,  256*256), device="cuda", dtype=torch.float32)
        matmul_BM256_BN256_BK64_GM1_nW4_nS2_EU0_kP2_mfma32(a, b, c, bias, P, locks, M, N, K, num_sms, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1), bias_stride)
        return True
    except Exception as e:
        print(f'invalid config(runtime): BM256_BN256_BK64_GM1_nW4_nS2_EU0_kP2_mfma32: ', e, flush=True)
        return False




@triton.jit()
def streamk_gemm_BM256_BN256_BK64_GM8_nW8_nS2_EU0_kP2_mfma32(
        A, B, C, bias_ptr, P, locks,
        M, N, K, num_sms,
        stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn, stride_bias,
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
        GROUP_SIZE_M: tl.constexpr, BIAS: tl.constexpr, EVEN_K: tl.constexpr,
):
    pid = tl.program_id(0)
    pid = (pid % 8) * (num_sms // 8) + (pid // 8)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    iters_per_tile = tl.cdiv(K, BLOCK_SIZE_K)
    total_tiles = num_pid_m * num_pid_n
    if num_sms > 0 and total_tiles > num_sms:
        total_streamk_tiles = total_tiles % num_sms
        total_streamk_tiles = total_streamk_tiles + num_sms
        total_full_tiles = total_tiles - total_streamk_tiles
        total_streamk_iters = total_streamk_tiles * iters_per_tile
        streamk_iters_pcu = total_streamk_iters // num_sms
        streamk_remainder_iters = total_streamk_iters % num_sms
    else:
        total_full_tiles = total_tiles
        total_streamk_tiles = 0
        streamk_iters_pcu = 0
        streamk_remainder_iters = 0
        total_streamk_iters = 0

    acc_dtype = tl.float32 if C.type.element_ty != tl.int8 else tl.int32
    rk = tl.arange(0, BLOCK_SIZE_K)

    for tile_id in range(pid, total_full_tiles, num_sms):
        if GROUP_SIZE_M == 1:
            pid_m = tile_id // num_pid_n
            pid_n = tile_id % num_pid_n
        else:
            num_pid_in_group = GROUP_SIZE_M * num_pid_n
            group_id = tile_id // num_pid_in_group
            first_pid_m = group_id * GROUP_SIZE_M
            group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
            pid_m = first_pid_m + (tile_id % group_size_m)
            pid_n = (tile_id % num_pid_in_group) // group_size_m

        rm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M))%M
        rn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))%N
        rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
        rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)
        A_BASE = A + rm[:, None] * stride_am + rk[None, :] * stride_ak
        B_BASE = B + rk[:, None] * stride_bk + rn[None, :] * stride_bn

        if BIAS:
            bias_ = bias_ptr + rm * stride_bias
            bias = tl.load(bias_, mask=rm < M, other=0.0)

        loop_k = tl.cdiv(K, BLOCK_SIZE_K)
        if not EVEN_K:
            loop_k -= 1

        acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype)
        for k in range(0, loop_k):
            a = tl.load(tl.multiple_of(A_BASE, (1, 16)))
            b = tl.load(tl.multiple_of(B_BASE, (16, 1)))
            acc += tl.dot(a, b)
            A_BASE += BLOCK_SIZE_K * stride_ak
            B_BASE += BLOCK_SIZE_K * stride_bk

        if not EVEN_K:
            k = loop_k
            rk = k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
            A_BASE = A + rm[:, None] * stride_am + rk[None, :] * stride_ak
            B_BASE = B + rk[:, None] * stride_bk + rn[None, :] * stride_bn
            a = tl.load(A_BASE, mask=rk[None, :] < K, other=0.0)
            b = tl.load(B_BASE, mask=rk[:, None] < K, other=0.0)

        c = acc.to(C.type.element_ty)
        if BIAS:
             c += bias[:, None]

        rm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        C_ = C + rm[:, None] * stride_cm + rn[None, :] * stride_cn
        mask = (rm < M)[:, None] & (rn < N)[None, :]
        tl.store(C_, c, mask=mask)

    start_iter = total_full_tiles * iters_per_tile + pid * streamk_iters_pcu + tl.minimum(pid, streamk_remainder_iters)
    last_iter = total_full_tiles * iters_per_tile + (pid + 1) * streamk_iters_pcu + tl.minimum(pid + 1, streamk_remainder_iters)
    while start_iter < last_iter:
        remainder = start_iter % iters_per_tile
        end_iter = tl.minimum(start_iter + (iters_per_tile - remainder), last_iter)
        tile_id = start_iter // iters_per_tile
        if GROUP_SIZE_M == 1:
            pid_m = tile_id // num_pid_n
            pid_n = tile_id % num_pid_n
        else:
            num_pid_in_group = GROUP_SIZE_M * num_pid_n
            group_id = tile_id // num_pid_in_group
            first_pid_m = group_id * GROUP_SIZE_M
            group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
            pid_m = first_pid_m + (tile_id % group_size_m)
            pid_n = (tile_id % num_pid_in_group) // group_size_m

        rm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M))%M
        rn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))%N
        rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
        rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)
        A_BASE = A + rm[:, None] * stride_am + rk[None, :] * stride_ak + BLOCK_SIZE_K * stride_ak * remainder
        B_BASE = B + rk[:, None] * stride_bk + rn[None, :] * stride_bn + BLOCK_SIZE_K * stride_bk * remainder

        if BIAS:
            bias_ = bias_ptr + rm * stride_bias
            bias = tl.load(bias_, mask=rm < M, other=0.0)

        acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype)
        for current_iter in range(start_iter, end_iter):
            if EVEN_K:
                a = tl.load(tl.multiple_of(A_BASE, (1, 16)))
                b = tl.load(tl.multiple_of(B_BASE, (16, 1)))
            else:
                global_k_offset = (current_iter % iters_per_tile) * BLOCK_SIZE_K
                k_mask = global_k_offset + rk < K
                a = tl.load(A_BASE, mask=k_mask[None, :], other=0.0)
                b = tl.load(B_BASE, mask=k_mask[:, None], other=0.0)
            acc += tl.dot(a, b)
            A_BASE += BLOCK_SIZE_K * stride_ak
            B_BASE += BLOCK_SIZE_K * stride_bk

        tile_iter = tile_id * iters_per_tile
        if start_iter == tile_iter:
            tile_iter_end = tile_iter + iters_per_tile
            next_pid = pid + 1
            end = end_iter
            while (end < tile_iter_end and next_pid < num_sms):
                while tl.atomic_cas(locks + next_pid, 1, 1) != 1:
              #  while tl.load(locks + next_pid, cache_modifier = ".cg") != 1:
                    pass
                rm1 = tl.arange(0, BLOCK_SIZE_M)
                rn1 = tl.arange(0, BLOCK_SIZE_N)
                rm1 = tl.max_contiguous(tl.multiple_of(rm1, BLOCK_SIZE_M), BLOCK_SIZE_M)
                rn1 = tl.max_contiguous(tl.multiple_of(rn1, BLOCK_SIZE_N), BLOCK_SIZE_N)
                P_ = P + next_pid * BLOCK_SIZE_M * BLOCK_SIZE_N + rm1[:, None] * BLOCK_SIZE_N + rn1[None, :]
                acc += tl.load(tl.multiple_of(P_, (1, 16)))
              #  acc += tl.load(P_)
                end += streamk_iters_pcu + (next_pid < streamk_remainder_iters)
                next_pid += 1

            c = acc.to(C.type.element_ty)
            if BIAS:
                 c += bias[:, None]

            rm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M))%M
            rn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))%N
            rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
            rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)
            C_ = C + rm[:, None] * stride_cm + rn[None, :] * stride_cn
            mask = (rm < M)[:, None] & (rn < N)[None, :]
            tl.store(C_, c, mask=mask)
        else:
            rm1 = tl.arange(0, BLOCK_SIZE_M)
            rn1 = tl.arange(0, BLOCK_SIZE_N)
            rm1 = tl.max_contiguous(tl.multiple_of(rm1, BLOCK_SIZE_M), BLOCK_SIZE_M)
            rn1 = tl.max_contiguous(tl.multiple_of(rn1, BLOCK_SIZE_N), BLOCK_SIZE_N)
            P_ = P + pid * BLOCK_SIZE_M * BLOCK_SIZE_N +  rm1[:, None] * BLOCK_SIZE_N + rn1[None, :]
            tl.store(P_, acc)
            tl.debug_barrier()
            tl.atomic_xchg(locks + pid, 1)
     #       tl.store(locks + pid, 1, cache_modifier=".wt")

        start_iter = end_iter



def matmul_BM256_BN256_BK64_GM8_nW8_nS2_EU0_kP2_mfma32(a, b, c, bias, P, locks, M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    grid = num_sms
    streamk_gemm_BM256_BN256_BK64_GM8_nW8_nS2_EU0_kP2_mfma32[grid,](
        a, b, c, bias, P, locks,
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
        BIAS = False,
        EVEN_K = True
    )
    return c

def try_matmul_config_BM256_BN256_BK64_GM8_nW8_nS2_EU0_kP2_mfma32(a, b, c, bias, M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        locks = torch.zeros((num_sms,), device = "cuda", dtype = torch.int32)
        P = torch.zeros((num_sms,  256*256), device="cuda", dtype=torch.float32)
        matmul_BM256_BN256_BK64_GM8_nW8_nS2_EU0_kP2_mfma32(a, b, c, bias, P, locks, M, N, K, num_sms, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1), bias_stride)
        return True
    except Exception as e:
        print(f'invalid config(runtime): BM256_BN256_BK64_GM8_nW8_nS2_EU0_kP2_mfma32: ', e, flush=True)
        return False

def test_gemm(M, N, K, num_sms, rotating_buffer_size, bias_size):
    tensors = gen_rotating_tensors(M, N, K, 'fp16', False, 'fp16', True, 'fp16',
                                   1, 'randn', rotating_buffer_size, bias_size, device='cuda')

    a = tensors['input_a'][0]
    b = tensors['input_b'][0]
    c = tensors['output_c'][0]
    assert bias_size == M or bias_size == 0

    stride_bias = tensors['bias'][0].stride(0) if bias_size > 0 else 0

    try:
        with open("/home/work/persistent-kernels/tune_streamk/utils/../compile_driver.py.failed_configs", "r") as f:
            failed_configs = [cfg.strip() for cfg in f.readlines()]
    except Exception:
        failed_configs = []


    if 'BM64_BN64_BK64_GM32_nW4_nS2_EU0_kP2_mfma16' not in failed_configs:
        rotating_num = tensors['rotating_num']
        locks = torch.zeros((120, num_sms), device = "cuda", dtype = torch.int32)
        P = torch.zeros((120, num_sms,  64*64), device="cuda", dtype=torch.float32)
        for i in range(120):
            a = tensors['input_a'][i % rotating_num]
            b = tensors['input_b'][i % rotating_num]
            c = tensors['output_c'][i % rotating_num]
            bias = tensors['bias'][i % rotating_num] if bias_size > 0 else None
            bias_stride = bias.stride(0) if bias_size > 0 else 0
            current_locks = locks[i]
            current_P = P[i]
            d = matmul_BM64_BN64_BK64_GM32_nW4_nS2_EU0_kP2_mfma16(a, b, c, bias, current_P, current_locks, M, N, K, num_sms, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1), bias_stride)

    if 'BM64_BN64_BK128_GM1_nW4_nS2_EU0_kP2_mfma16' not in failed_configs:
        rotating_num = tensors['rotating_num']
        locks = torch.zeros((120, num_sms), device = "cuda", dtype = torch.int32)
        P = torch.zeros((120, num_sms,  64*64), device="cuda", dtype=torch.float32)
        for i in range(120):
            a = tensors['input_a'][i % rotating_num]
            b = tensors['input_b'][i % rotating_num]
            c = tensors['output_c'][i % rotating_num]
            bias = tensors['bias'][i % rotating_num] if bias_size > 0 else None
            bias_stride = bias.stride(0) if bias_size > 0 else 0
            current_locks = locks[i]
            current_P = P[i]
            d = matmul_BM64_BN64_BK128_GM1_nW4_nS2_EU0_kP2_mfma16(a, b, c, bias, current_P, current_locks, M, N, K, num_sms, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1), bias_stride)

    if 'BM64_BN64_BK128_GM4_nW8_nS2_EU0_kP2_mfma16' not in failed_configs:
        rotating_num = tensors['rotating_num']
        locks = torch.zeros((120, num_sms), device = "cuda", dtype = torch.int32)
        P = torch.zeros((120, num_sms,  64*64), device="cuda", dtype=torch.float32)
        for i in range(120):
            a = tensors['input_a'][i % rotating_num]
            b = tensors['input_b'][i % rotating_num]
            c = tensors['output_c'][i % rotating_num]
            bias = tensors['bias'][i % rotating_num] if bias_size > 0 else None
            bias_stride = bias.stride(0) if bias_size > 0 else 0
            current_locks = locks[i]
            current_P = P[i]
            d = matmul_BM64_BN64_BK128_GM4_nW8_nS2_EU0_kP2_mfma16(a, b, c, bias, current_P, current_locks, M, N, K, num_sms, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1), bias_stride)

    if 'BM64_BN64_BK256_GM8_nW4_nS2_EU0_kP2_mfma16' not in failed_configs:
        rotating_num = tensors['rotating_num']
        locks = torch.zeros((120, num_sms), device = "cuda", dtype = torch.int32)
        P = torch.zeros((120, num_sms,  64*64), device="cuda", dtype=torch.float32)
        for i in range(120):
            a = tensors['input_a'][i % rotating_num]
            b = tensors['input_b'][i % rotating_num]
            c = tensors['output_c'][i % rotating_num]
            bias = tensors['bias'][i % rotating_num] if bias_size > 0 else None
            bias_stride = bias.stride(0) if bias_size > 0 else 0
            current_locks = locks[i]
            current_P = P[i]
            d = matmul_BM64_BN64_BK256_GM8_nW4_nS2_EU0_kP2_mfma16(a, b, c, bias, current_P, current_locks, M, N, K, num_sms, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1), bias_stride)

    if 'BM64_BN64_BK256_GM16_nW8_nS2_EU0_kP2_mfma16' not in failed_configs:
        rotating_num = tensors['rotating_num']
        locks = torch.zeros((120, num_sms), device = "cuda", dtype = torch.int32)
        P = torch.zeros((120, num_sms,  64*64), device="cuda", dtype=torch.float32)
        for i in range(120):
            a = tensors['input_a'][i % rotating_num]
            b = tensors['input_b'][i % rotating_num]
            c = tensors['output_c'][i % rotating_num]
            bias = tensors['bias'][i % rotating_num] if bias_size > 0 else None
            bias_stride = bias.stride(0) if bias_size > 0 else 0
            current_locks = locks[i]
            current_P = P[i]
            d = matmul_BM64_BN64_BK256_GM16_nW8_nS2_EU0_kP2_mfma16(a, b, c, bias, current_P, current_locks, M, N, K, num_sms, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1), bias_stride)

    if 'BM64_BN128_BK64_GM32_nW4_nS2_EU0_kP2_mfma16' not in failed_configs:
        rotating_num = tensors['rotating_num']
        locks = torch.zeros((120, num_sms), device = "cuda", dtype = torch.int32)
        P = torch.zeros((120, num_sms,  64*128), device="cuda", dtype=torch.float32)
        for i in range(120):
            a = tensors['input_a'][i % rotating_num]
            b = tensors['input_b'][i % rotating_num]
            c = tensors['output_c'][i % rotating_num]
            bias = tensors['bias'][i % rotating_num] if bias_size > 0 else None
            bias_stride = bias.stride(0) if bias_size > 0 else 0
            current_locks = locks[i]
            current_P = P[i]
            d = matmul_BM64_BN128_BK64_GM32_nW4_nS2_EU0_kP2_mfma16(a, b, c, bias, current_P, current_locks, M, N, K, num_sms, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1), bias_stride)

    if 'BM64_BN128_BK128_GM1_nW4_nS2_EU0_kP2_mfma16' not in failed_configs:
        rotating_num = tensors['rotating_num']
        locks = torch.zeros((120, num_sms), device = "cuda", dtype = torch.int32)
        P = torch.zeros((120, num_sms,  64*128), device="cuda", dtype=torch.float32)
        for i in range(120):
            a = tensors['input_a'][i % rotating_num]
            b = tensors['input_b'][i % rotating_num]
            c = tensors['output_c'][i % rotating_num]
            bias = tensors['bias'][i % rotating_num] if bias_size > 0 else None
            bias_stride = bias.stride(0) if bias_size > 0 else 0
            current_locks = locks[i]
            current_P = P[i]
            d = matmul_BM64_BN128_BK128_GM1_nW4_nS2_EU0_kP2_mfma16(a, b, c, bias, current_P, current_locks, M, N, K, num_sms, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1), bias_stride)

    if 'BM64_BN128_BK128_GM4_nW8_nS2_EU0_kP2_mfma16' not in failed_configs:
        rotating_num = tensors['rotating_num']
        locks = torch.zeros((120, num_sms), device = "cuda", dtype = torch.int32)
        P = torch.zeros((120, num_sms,  64*128), device="cuda", dtype=torch.float32)
        for i in range(120):
            a = tensors['input_a'][i % rotating_num]
            b = tensors['input_b'][i % rotating_num]
            c = tensors['output_c'][i % rotating_num]
            bias = tensors['bias'][i % rotating_num] if bias_size > 0 else None
            bias_stride = bias.stride(0) if bias_size > 0 else 0
            current_locks = locks[i]
            current_P = P[i]
            d = matmul_BM64_BN128_BK128_GM4_nW8_nS2_EU0_kP2_mfma16(a, b, c, bias, current_P, current_locks, M, N, K, num_sms, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1), bias_stride)

    if 'BM64_BN256_BK64_GM8_nW4_nS2_EU0_kP2_mfma16' not in failed_configs:
        rotating_num = tensors['rotating_num']
        locks = torch.zeros((120, num_sms), device = "cuda", dtype = torch.int32)
        P = torch.zeros((120, num_sms,  64*256), device="cuda", dtype=torch.float32)
        for i in range(120):
            a = tensors['input_a'][i % rotating_num]
            b = tensors['input_b'][i % rotating_num]
            c = tensors['output_c'][i % rotating_num]
            bias = tensors['bias'][i % rotating_num] if bias_size > 0 else None
            bias_stride = bias.stride(0) if bias_size > 0 else 0
            current_locks = locks[i]
            current_P = P[i]
            d = matmul_BM64_BN256_BK64_GM8_nW4_nS2_EU0_kP2_mfma16(a, b, c, bias, current_P, current_locks, M, N, K, num_sms, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1), bias_stride)

    if 'BM64_BN256_BK64_GM16_nW8_nS2_EU0_kP2_mfma16' not in failed_configs:
        rotating_num = tensors['rotating_num']
        locks = torch.zeros((120, num_sms), device = "cuda", dtype = torch.int32)
        P = torch.zeros((120, num_sms,  64*256), device="cuda", dtype=torch.float32)
        for i in range(120):
            a = tensors['input_a'][i % rotating_num]
            b = tensors['input_b'][i % rotating_num]
            c = tensors['output_c'][i % rotating_num]
            bias = tensors['bias'][i % rotating_num] if bias_size > 0 else None
            bias_stride = bias.stride(0) if bias_size > 0 else 0
            current_locks = locks[i]
            current_P = P[i]
            d = matmul_BM64_BN256_BK64_GM16_nW8_nS2_EU0_kP2_mfma16(a, b, c, bias, current_P, current_locks, M, N, K, num_sms, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1), bias_stride)

    if 'BM128_BN64_BK64_GM32_nW4_nS2_EU0_kP2_mfma16' not in failed_configs:
        rotating_num = tensors['rotating_num']
        locks = torch.zeros((120, num_sms), device = "cuda", dtype = torch.int32)
        P = torch.zeros((120, num_sms,  128*64), device="cuda", dtype=torch.float32)
        for i in range(120):
            a = tensors['input_a'][i % rotating_num]
            b = tensors['input_b'][i % rotating_num]
            c = tensors['output_c'][i % rotating_num]
            bias = tensors['bias'][i % rotating_num] if bias_size > 0 else None
            bias_stride = bias.stride(0) if bias_size > 0 else 0
            current_locks = locks[i]
            current_P = P[i]
            d = matmul_BM128_BN64_BK64_GM32_nW4_nS2_EU0_kP2_mfma16(a, b, c, bias, current_P, current_locks, M, N, K, num_sms, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1), bias_stride)

    if 'BM128_BN64_BK128_GM1_nW4_nS2_EU0_kP2_mfma16' not in failed_configs:
        rotating_num = tensors['rotating_num']
        locks = torch.zeros((120, num_sms), device = "cuda", dtype = torch.int32)
        P = torch.zeros((120, num_sms,  128*64), device="cuda", dtype=torch.float32)
        for i in range(120):
            a = tensors['input_a'][i % rotating_num]
            b = tensors['input_b'][i % rotating_num]
            c = tensors['output_c'][i % rotating_num]
            bias = tensors['bias'][i % rotating_num] if bias_size > 0 else None
            bias_stride = bias.stride(0) if bias_size > 0 else 0
            current_locks = locks[i]
            current_P = P[i]
            d = matmul_BM128_BN64_BK128_GM1_nW4_nS2_EU0_kP2_mfma16(a, b, c, bias, current_P, current_locks, M, N, K, num_sms, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1), bias_stride)

    if 'BM128_BN64_BK128_GM4_nW8_nS2_EU0_kP2_mfma16' not in failed_configs:
        rotating_num = tensors['rotating_num']
        locks = torch.zeros((120, num_sms), device = "cuda", dtype = torch.int32)
        P = torch.zeros((120, num_sms,  128*64), device="cuda", dtype=torch.float32)
        for i in range(120):
            a = tensors['input_a'][i % rotating_num]
            b = tensors['input_b'][i % rotating_num]
            c = tensors['output_c'][i % rotating_num]
            bias = tensors['bias'][i % rotating_num] if bias_size > 0 else None
            bias_stride = bias.stride(0) if bias_size > 0 else 0
            current_locks = locks[i]
            current_P = P[i]
            d = matmul_BM128_BN64_BK128_GM4_nW8_nS2_EU0_kP2_mfma16(a, b, c, bias, current_P, current_locks, M, N, K, num_sms, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1), bias_stride)

    if 'BM128_BN128_BK64_GM8_nW4_nS2_EU0_kP2_mfma16' not in failed_configs:
        rotating_num = tensors['rotating_num']
        locks = torch.zeros((120, num_sms), device = "cuda", dtype = torch.int32)
        P = torch.zeros((120, num_sms,  128*128), device="cuda", dtype=torch.float32)
        for i in range(120):
            a = tensors['input_a'][i % rotating_num]
            b = tensors['input_b'][i % rotating_num]
            c = tensors['output_c'][i % rotating_num]
            bias = tensors['bias'][i % rotating_num] if bias_size > 0 else None
            bias_stride = bias.stride(0) if bias_size > 0 else 0
            current_locks = locks[i]
            current_P = P[i]
            d = matmul_BM128_BN128_BK64_GM8_nW4_nS2_EU0_kP2_mfma16(a, b, c, bias, current_P, current_locks, M, N, K, num_sms, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1), bias_stride)

    if 'BM128_BN128_BK64_GM16_nW8_nS2_EU0_kP2_mfma16' not in failed_configs:
        rotating_num = tensors['rotating_num']
        locks = torch.zeros((120, num_sms), device = "cuda", dtype = torch.int32)
        P = torch.zeros((120, num_sms,  128*128), device="cuda", dtype=torch.float32)
        for i in range(120):
            a = tensors['input_a'][i % rotating_num]
            b = tensors['input_b'][i % rotating_num]
            c = tensors['output_c'][i % rotating_num]
            bias = tensors['bias'][i % rotating_num] if bias_size > 0 else None
            bias_stride = bias.stride(0) if bias_size > 0 else 0
            current_locks = locks[i]
            current_P = P[i]
            d = matmul_BM128_BN128_BK64_GM16_nW8_nS2_EU0_kP2_mfma16(a, b, c, bias, current_P, current_locks, M, N, K, num_sms, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1), bias_stride)

    if 'BM128_BN128_BK128_GM1_nW8_nS2_EU0_kP1_mfma32' not in failed_configs:
        rotating_num = tensors['rotating_num']
        locks = torch.zeros((120, num_sms), device = "cuda", dtype = torch.int32)
        P = torch.zeros((120, num_sms,  128*128), device="cuda", dtype=torch.float32)
        for i in range(120):
            a = tensors['input_a'][i % rotating_num]
            b = tensors['input_b'][i % rotating_num]
            c = tensors['output_c'][i % rotating_num]
            bias = tensors['bias'][i % rotating_num] if bias_size > 0 else None
            bias_stride = bias.stride(0) if bias_size > 0 else 0
            current_locks = locks[i]
            current_P = P[i]
            d = matmul_BM128_BN128_BK128_GM1_nW8_nS2_EU0_kP1_mfma32(a, b, c, bias, current_P, current_locks, M, N, K, num_sms, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1), bias_stride)

    if 'BM128_BN256_BK64_GM8_nW4_nS2_EU0_kP2_mfma32' not in failed_configs:
        rotating_num = tensors['rotating_num']
        locks = torch.zeros((120, num_sms), device = "cuda", dtype = torch.int32)
        P = torch.zeros((120, num_sms,  128*256), device="cuda", dtype=torch.float32)
        for i in range(120):
            a = tensors['input_a'][i % rotating_num]
            b = tensors['input_b'][i % rotating_num]
            c = tensors['output_c'][i % rotating_num]
            bias = tensors['bias'][i % rotating_num] if bias_size > 0 else None
            bias_stride = bias.stride(0) if bias_size > 0 else 0
            current_locks = locks[i]
            current_P = P[i]
            d = matmul_BM128_BN256_BK64_GM8_nW4_nS2_EU0_kP2_mfma32(a, b, c, bias, current_P, current_locks, M, N, K, num_sms, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1), bias_stride)

    if 'BM128_BN256_BK64_GM16_nW8_nS2_EU0_kP2_mfma32' not in failed_configs:
        rotating_num = tensors['rotating_num']
        locks = torch.zeros((120, num_sms), device = "cuda", dtype = torch.int32)
        P = torch.zeros((120, num_sms,  128*256), device="cuda", dtype=torch.float32)
        for i in range(120):
            a = tensors['input_a'][i % rotating_num]
            b = tensors['input_b'][i % rotating_num]
            c = tensors['output_c'][i % rotating_num]
            bias = tensors['bias'][i % rotating_num] if bias_size > 0 else None
            bias_stride = bias.stride(0) if bias_size > 0 else 0
            current_locks = locks[i]
            current_P = P[i]
            d = matmul_BM128_BN256_BK64_GM16_nW8_nS2_EU0_kP2_mfma32(a, b, c, bias, current_P, current_locks, M, N, K, num_sms, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1), bias_stride)

    if 'BM256_BN64_BK64_GM1_nW8_nS2_EU0_kP2_mfma32' not in failed_configs:
        rotating_num = tensors['rotating_num']
        locks = torch.zeros((120, num_sms), device = "cuda", dtype = torch.int32)
        P = torch.zeros((120, num_sms,  256*64), device="cuda", dtype=torch.float32)
        for i in range(120):
            a = tensors['input_a'][i % rotating_num]
            b = tensors['input_b'][i % rotating_num]
            c = tensors['output_c'][i % rotating_num]
            bias = tensors['bias'][i % rotating_num] if bias_size > 0 else None
            bias_stride = bias.stride(0) if bias_size > 0 else 0
            current_locks = locks[i]
            current_P = P[i]
            d = matmul_BM256_BN64_BK64_GM1_nW8_nS2_EU0_kP2_mfma32(a, b, c, bias, current_P, current_locks, M, N, K, num_sms, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1), bias_stride)

    if 'BM256_BN128_BK64_GM8_nW4_nS2_EU0_kP2_mfma32' not in failed_configs:
        rotating_num = tensors['rotating_num']
        locks = torch.zeros((120, num_sms), device = "cuda", dtype = torch.int32)
        P = torch.zeros((120, num_sms,  256*128), device="cuda", dtype=torch.float32)
        for i in range(120):
            a = tensors['input_a'][i % rotating_num]
            b = tensors['input_b'][i % rotating_num]
            c = tensors['output_c'][i % rotating_num]
            bias = tensors['bias'][i % rotating_num] if bias_size > 0 else None
            bias_stride = bias.stride(0) if bias_size > 0 else 0
            current_locks = locks[i]
            current_P = P[i]
            d = matmul_BM256_BN128_BK64_GM8_nW4_nS2_EU0_kP2_mfma32(a, b, c, bias, current_P, current_locks, M, N, K, num_sms, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1), bias_stride)

    if 'BM256_BN256_BK64_GM1_nW4_nS2_EU0_kP2_mfma32' not in failed_configs:
        rotating_num = tensors['rotating_num']
        locks = torch.zeros((120, num_sms), device = "cuda", dtype = torch.int32)
        P = torch.zeros((120, num_sms,  256*256), device="cuda", dtype=torch.float32)
        for i in range(120):
            a = tensors['input_a'][i % rotating_num]
            b = tensors['input_b'][i % rotating_num]
            c = tensors['output_c'][i % rotating_num]
            bias = tensors['bias'][i % rotating_num] if bias_size > 0 else None
            bias_stride = bias.stride(0) if bias_size > 0 else 0
            current_locks = locks[i]
            current_P = P[i]
            d = matmul_BM256_BN256_BK64_GM1_nW4_nS2_EU0_kP2_mfma32(a, b, c, bias, current_P, current_locks, M, N, K, num_sms, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1), bias_stride)

    if 'BM256_BN256_BK64_GM8_nW8_nS2_EU0_kP2_mfma32' not in failed_configs:
        rotating_num = tensors['rotating_num']
        locks = torch.zeros((120, num_sms), device = "cuda", dtype = torch.int32)
        P = torch.zeros((120, num_sms,  256*256), device="cuda", dtype=torch.float32)
        for i in range(120):
            a = tensors['input_a'][i % rotating_num]
            b = tensors['input_b'][i % rotating_num]
            c = tensors['output_c'][i % rotating_num]
            bias = tensors['bias'][i % rotating_num] if bias_size > 0 else None
            bias_stride = bias.stride(0) if bias_size > 0 else 0
            current_locks = locks[i]
            current_P = P[i]
            d = matmul_BM256_BN256_BK64_GM8_nW8_nS2_EU0_kP2_mfma32(a, b, c, bias, current_P, current_locks, M, N, K, num_sms, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1), bias_stride)

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
    test_gemm(4864, 8192, 4160, num_sms, rotating_buffer_size, 0)

if __name__ == '__main__':
   sys.exit(main())