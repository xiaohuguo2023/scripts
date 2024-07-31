import torch
import triton
import triton.language as tl
import argparse
import sys
import multiprocessing
from datetime import datetime
#from myKernels import *

@triton.jit()
def get_new_pid(current_pid, num_sms):
    # Number of XCDs
    num_xcds = 8
    # Number of pids per XCD in the new arrangement
    pids_per_xcd = num_sms // num_xcds
    # Compute current XCD and local pid within the XCD
    xcd = current_pid % num_xcds
    local_pid = current_pid // num_xcds

    # Calculate new pid based on the new grouping
    new_pid = xcd * pids_per_xcd + local_pid
    return new_pid

@triton.jit()
def get_tiles_config(M, N, K, num_sms,
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
):
    total_blocks_M = tl.cdiv(M, BLOCK_SIZE_M)
    total_blocks_N = tl.cdiv(N, BLOCK_SIZE_N)
    iters_per_tile = tl.cdiv(K, BLOCK_SIZE_K)

    total_tiles = total_blocks_M * total_blocks_N
    if num_sms > 0 and total_tiles > num_sms:  # Stream-K
        total_full_tiles_pcu = total_tiles // num_sms
        total_streamk_tiles = total_tiles % num_sms
        total_full_tiles = total_tiles - total_streamk_tiles
        total_streamk_iters = total_streamk_tiles * iters_per_tile
        # iterations related to full waves
        streamk_iters_pcu = total_streamk_iters // num_sms
        # iterations related to last (partial) wave
        streamk_remainder_iters = total_streamk_iters % num_sms

    else:  # all tiles are computed using classical blocking
        total_full_tiles = total_tiles
        total_streamk_tiles = 0
        streamk_iters_pcu = 0
        streamk_remainder_iters = 0
        total_streamk_iters = 0

    return iters_per_tile, total_full_tiles, total_streamk_tiles, streamk_iters_pcu, streamk_remainder_iters

@triton.jit()
def streamk_gemm_BM256_BN256_BK32_GM1_nW1_nS0_EU0_kP2_mfma16(
        A, B, C, bias_ptr, P, locks,
        M, N, K, num_sms,
        stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn, stride_bias,
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
        GROUP_SIZE_M: tl.constexpr, BIAS: tl.constexpr, EVEN_K: tl.constexpr,
):
    pid = tl.program_id(0)
    pid = get_new_pid(pid, num_sms)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

    iters_per_tile, total_full_tiles, total_streamk_tiles, streamk_iters_pcu, streamk_remainder_iters = get_tiles_config(M, N, K, num_sms, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K)

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

        acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype)
        for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
            if EVEN_K:
                a = tl.load(A_BASE)
                b = tl.load(B_BASE)
            else:
                a = tl.load(A_BASE, mask=rk[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
                b = tl.load(B_BASE, mask=rk[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
            acc += tl.dot(a, b)
            A_BASE += BLOCK_SIZE_K * stride_ak
            B_BASE += BLOCK_SIZE_K * stride_bk

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
        # where are we in the grid
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
    #    rk = tl.arange(0, BLOCK_SIZE_K)
        A_BASE = A + rm[:, None] * stride_am + rk[None, :] * stride_ak + BLOCK_SIZE_K * stride_ak * remainder
        B_BASE = B + rk[:, None] * stride_bk + rn[None, :] * stride_bn + BLOCK_SIZE_K * stride_bk * remainder

        if BIAS:
            bias_ = bias_ptr + rm * stride_bias
            bias = tl.load(bias_, mask=rm < M, other=0.0)

        acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype)
        for current_iter in range(start_iter, end_iter):
            if EVEN_K:
                a = tl.load(A_BASE)
                b = tl.load(B_BASE)
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
         #       while tl.load(locks + next_pid, cache_modifier='.ca') != 1:
                    pass
                rm1 = tl.arange(0, BLOCK_SIZE_M)
                rn1 = tl.arange(0, BLOCK_SIZE_N)
                rm1 = tl.max_contiguous(tl.multiple_of(rm1, BLOCK_SIZE_M), BLOCK_SIZE_M)
                rn1 = tl.max_contiguous(tl.multiple_of(rn1, BLOCK_SIZE_N), BLOCK_SIZE_N)
                P_ = P + next_pid * BLOCK_SIZE_M * BLOCK_SIZE_N + rm1[:, None] * BLOCK_SIZE_N + rn1[None, :]
                acc += tl.load(P_)
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
            tl.atomic_xchg(locks + pid, 1)

        start_iter = end_iter

def matmul_BM256_BN256_BK32_GM1_nW1_nS0_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    streamk_gemm_BM256_BN256_BK32_GM1_nW1_nS0_EU0_kP2_mfma16.warmup(
        torch.float16, torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
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
        BIAS=False,
        EVEN_K=True,
        grid=(1,),
    )
    return None

def try_compile_config_BM256_BN256_BK32_GM1_nW1_nS0_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn):
    try:
        start_time = datetime.now()
        matmul_BM256_BN256_BK32_GM1_nW1_nS0_EU0_kP2_mfma16(M, N, K, num_sms, am, ak, bk, bn, cm, cn, biasn)
        end_time = datetime.now() - start_time
        print("kernel BM256_BN256_BK32_GM1_nW1_nS0_EU0_kP2_mfma16 took ", end_time)
        return True
    except Exception as e:
        print(f'invalid config(compilation): BM256_BN256_BK32_GM1_nW1_nS0_EU0_kP2_mfma16: ', e, flush=True)
        return False

def compile_kernels(M, N, K, num_sms, bias_size, num_threads):
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


    results += [thread_pool.apply_async(try_compile_config_BM256_BN256_BK32_GM1_nW1_nS0_EU0_kP2_mfma16, args=task_args)]
    config_names += ['BM256_BN256_BK32_GM1_nW1_nS0_EU0_kP2_mfma16']

    print("try compile finished")
    failed_configs = []
    for i in range(len(results)):
        results[i].wait()
        res = results[i].get()
        if not res:
            failed_configs += [config_names[i]]
    thread_pool.close()
    thread_pool.join()
    if failed_configs:
        with open("/home/work/stream-k/tune_streamk/utils/../compile_driver.py.failed_configs", "w") as f:
            for cfg in failed_configs:
                f.write(cfg + "\n")
    print("end of compile")

def main():
    parser = argparse.ArgumentParser(
        prog="tune a specific gemm size",
        allow_abbrev=False,)
    parser.add_argument("-n", type=int, default=32, help='number of threads')
    args = parser.parse_args()
    numThreads = args.n
    compile_kernels(512, 512, 512, 304, 0, numThreads)

if __name__ == '__main__':
   sys.exit(main())
