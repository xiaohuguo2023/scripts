import triton
import triton.language as tl

@triton.jit()
def streamk_gemm(
         A, B, C, bias_ptr, P, locks,
         M, N, K,
         stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn, stride_bias,
         BLOCK_SIZE_M: tl.constexpr,
         BLOCK_SIZE_N: tl.constexpr,
         BLOCK_SIZE_K: tl.constexpr,
         GROUP_SIZE_M: tl.constexpr,
         NUM_SMS: tl.constexpr,
         BIAS: tl.constexpr,
         EVEN_K: tl.constexpr,
):
     pid = tl.program_id(0)
     pid = (pid % 8) * (NUM_SMS // 8) + (pid // 8)
     num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
     num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
     iters_per_tile = tl.cdiv(K, BLOCK_SIZE_K)
     total_tiles = num_pid_m * num_pid_n

     acc_dtype = tl.float32 if C.type.element_ty != tl.int8 else tl.int32

     for tile_id in range(pid, 304, NUM_SMS):
         num_pid_in_group = GROUP_SIZE_M * num_pid_n
         group_id = tile_id // num_pid_in_group
         first_pid_m = group_id * GROUP_SIZE_M
         group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
         pid_m = first_pid_m + ((tile_id % num_pid_in_group) % group_size_m)
         pid_n = (tile_id % num_pid_in_group) // group_size_m

         rm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M))%M
         rn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))%N
         rk = tl.arange(0, BLOCK_SIZE_K)
         rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
         rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)
         A_BASE = A + rm[:, None] * stride_am + rk[None, :] * stride_ak
         B_BASE = B + rk[:, None] * stride_bk + rn[None, :] * stride_bn

         loop_k = tl.cdiv(K, BLOCK_SIZE_K)
         acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype)
         for k in range(0, loop_k):
             a = tl.load(tl.multiple_of(A_BASE, (1, 16)))
             b = tl.load(tl.multiple_of(B_BASE, (16, 1)))
             acc += tl.dot(a, b)
             A_BASE += BLOCK_SIZE_K * stride_ak
             B_BASE += BLOCK_SIZE_K * stride_bk

         c = acc.to(C.type.element_ty)

         rm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
         rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
         C_ = C + rm[:, None] * stride_cm + rn[None, :] * stride_cn
         c_mask = (rm[:, None] < M) & (rn[None, :] < N)
         tl.store(C_, c, c_mask)
    #     tl.store(C_, c)
