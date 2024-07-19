import torch
import triton
import triton.language as tl
import argparse
import sys
import multiprocessing
from tune_streamk import gen_input

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
def streamk_gemm_M512_N512_K512_BM128_BN128_BK128_GM4_nW2_nS0_EU0_kP2_mfma32(
        A, B, C, P, locks,
        M, N, K, num_sms,
        stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
        GROUP_SIZE_M: tl.constexpr, EVEN_K: tl.constexpr,
):
    pid = tl.program_id(0)
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

        rm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        C_ = C + rm[:, None] * stride_cm + rn[None, :] * stride_cn
        mask = (rm < M)[:, None] & (rn < N)[None, :]
        tl.store(C_, acc, mask=mask)

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
                    pass
                rm1 = tl.arange(0, BLOCK_SIZE_M)
                rn1 = tl.arange(0, BLOCK_SIZE_N)
                rm1 = tl.max_contiguous(tl.multiple_of(rm1, BLOCK_SIZE_M), BLOCK_SIZE_M)
                rn1 = tl.max_contiguous(tl.multiple_of(rn1, BLOCK_SIZE_N), BLOCK_SIZE_N)
                P_ = P + next_pid * BLOCK_SIZE_M * BLOCK_SIZE_N + rm1[:, None] * BLOCK_SIZE_N + rn1[None, :]
                acc += tl.load(P_)
                end += streamk_iters_pcu + (next_pid < streamk_remainder_iters)

                next_pid += 1

            rm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M))%M
            rn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))%N
            rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
            rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)
            C_ = C + rm[:, None] * stride_cm + rn[None, :] * stride_cn
            mask = (rm < M)[:, None] & (rn < N)[None, :]
            tl.store(C_, acc, mask=mask)
        else:
            rm1 = tl.arange(0, BLOCK_SIZE_M)
            rn1 = tl.arange(0, BLOCK_SIZE_N)
            rm1 = tl.max_contiguous(tl.multiple_of(rm1, BLOCK_SIZE_M), BLOCK_SIZE_M)
            rn1 = tl.max_contiguous(tl.multiple_of(rn1, BLOCK_SIZE_N), BLOCK_SIZE_N)
            P_ = P + pid * BLOCK_SIZE_M * BLOCK_SIZE_N +  rm1[:, None] * BLOCK_SIZE_N + rn1[None, :]
            tl.store(P_, acc)
            tl.atomic_xchg(locks + pid, 1)

        start_iter = end_iter



def matmul_M512_N512_K512_BM128_BN128_BK128_GM4_nW2_nS0_EU0_kP2_mfma32(a, b, c, P, locks, M, N, K, num_sms, am, ak, bk, bn, cm, cn, warmup=False):
    grid = num_sms
    #print(f'config: streamk_gemm_M512_N512_K512_BM128_BN128_BK128_GM4_nW2_nS0_EU0_kP2_mfma32', flush=True)
    if warmup:
        streamk_gemm_M512_N512_K512_BM128_BN128_BK128_GM4_nW2_nS0_EU0_kP2_mfma32.warmup(
            torch.float16, torch.float16, torch.float16, torch.float32, torch.int32,
            M, N, K, num_sms,
            am, ak, bk, bn, cm, cn,
            BLOCK_SIZE_M = 128,
            BLOCK_SIZE_N = 128,
            BLOCK_SIZE_K = 128,
            GROUP_SIZE_M = 4,
            num_warps = 2,
            num_stages = 0,
            waves_per_eu = 0,
            matrix_instr_nonkdim = 32,
            kpack = 2,
            EVEN_K = True,
            grid=(1,)
        )
        return None
    else:
        streamk_gemm_M512_N512_K512_BM128_BN128_BK128_GM4_nW2_nS0_EU0_kP2_mfma32[grid,](
            a, b, c, P, locks,
            M, N, K, num_sms,
            am, ak, bk, bn, cm, cn,
            BLOCK_SIZE_M = 128,
            BLOCK_SIZE_N = 128,
            BLOCK_SIZE_K = 128,
            GROUP_SIZE_M = 4,
            num_warps = 2,
            num_stages = 0,
            waves_per_eu = 0,
            matrix_instr_nonkdim = 32,
            kpack = 2,
            EVEN_K = True
        )
        return c

def try_config_M512_N512_K512_BM128_BN128_BK128_GM4_nW2_nS0_EU0_kP2_mfma32(M, N, K, num_sms, am, ak, bk, bn, cm, cn):
    try:
        matmul_M512_N512_K512_BM128_BN128_BK128_GM4_nW2_nS0_EU0_kP2_mfma32(None, None, None, None, None, M, N, K, num_sms, am, ak, bk, bn, cm, cn, True)
        return True
    except Exception as e:
        print(f'invalid config(compilation): M512_N512_K512_BM128_BN128_BK128_GM4_nW2_nS0_EU0_kP2_mfma32: ', e, flush=True)
        return False

def test_gemm(M, N, K, num_sms, num_threads):
    thread_pool = multiprocessing.Pool(processes=num_threads)
    a, a_fp16 = gen_input(M, K, 'fp16', False, 1, 'randn', device='cuda')
    b, b_fp16 = gen_input(K, N, 'fp16', True, 2, 'randn', device='cuda')
    c = torch.zeros((M, N), device=a.device, dtype=torch.float16)
    task_args = (M, N, K, num_sms,
                 a.stride(0), a.stride(1),
                 b.stride(0), b.stride(1),
                 c.stride(0), c.stride(1))

    if num_threads > 1:
        results = []
        config_names = []

        results += [thread_pool.apply_async(try_config_M512_N512_K512_BM128_BN128_BK128_GM4_nW2_nS0_EU0_kP2_mfma32, args=task_args)]
        config_names += ['M512_N512_K512_BM128_BN128_BK128_GM4_nW2_nS0_EU0_kP2_mfma32']

        failed_configs = []
        for i in range(len(results)):
            results[i].wait()
            res = results[i].get()
            if not res:
                failed_configs += [config_names[i]]
        thread_pool.close()
        thread_pool.join()
        with open("generated_kernel512-512-512-7.py.failed_configs", "w") as f:
            for cfg in failed_configs:
                f.write(cfg + "\n")
    else:
        try:
            with open("generated_kernel512-512-512-7.py.failed_configs", "r") as f:
                failed_configs = [cfg.strip() for cfg in f.readlines()]
        except Exception:
            failed_configs = []
        
        if 'M512_N512_K512_BM128_BN128_BK128_GM4_nW2_nS0_EU0_kP2_mfma32' not in failed_configs:
            print(f"M512_N512_K512_BM128_BN128_BK128_GM4_nW2_nS0_EU0_kP2_mfma32")
            for i in range(200):
                locks = torch.zeros((num_sms,), device = "cuda", dtype = torch.int32)
                P = torch.zeros((num_sms,  128*128), device="cuda", dtype=torch.float32)
                d = matmul_M512_N512_K512_BM128_BN128_BK128_GM4_nW2_nS0_EU0_kP2_mfma32(a, b, c, P, locks, M, N, K, num_sms, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1))

def main():
    parser = argparse.ArgumentParser(
        prog="tune a specific gemm size",
        allow_abbrev=False,)
    parser.add_argument("-n", type=int, default=1, help='number of threads')
    args = parser.parse_args()
    numThreads = args.n
    num_sms = 304
    test_gemm(512, 512, 512, num_sms, numThreads)

if __name__ == '__main__':
   sys.exit(main())
