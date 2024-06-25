import torch
import triton
import triton.language as tl

# Define the Triton kernel
@triton.jit
def spinning_lock_kernel(P, C, locks, num_sms, k, M, N, stride_cm, stride_cn, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr):
    pid = tl.program_id(0)
    pid_m = pid // num_sms
    pid_n = pid % num_sms

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)  # Assuming acc initialization

    # Perform reduction for every kth pid
    for iters in range(1, 10):
        if (pid % k == 0):
            next_pid = pid + 1

            while next_pid < pid + k and next_pid < num_sms:
                while tl.atomic_cas(locks + next_pid, 1, 1) != 1:
                    pass

                rm1 = tl.arange(0, BLOCK_SIZE_M)
                rn1 = tl.arange(0, BLOCK_SIZE_N)
                P_ = P + next_pid * BLOCK_SIZE_M * BLOCK_SIZE_N + rm1[:, None] * BLOCK_SIZE_N + rn1[None, :]
                acc1 = tl.load(P_)
                acc += acc1

                next_pid += 1
              

        # Store results using temporary storage P for every k-1 pids
        else:
            rm1 = tl.arange(0, BLOCK_SIZE_M)
            rn1 = tl.arange(0, BLOCK_SIZE_N)
            P_ = P + pid * BLOCK_SIZE_M * BLOCK_SIZE_N + rm1[:, None] * BLOCK_SIZE_N + rn1[None, :]
            tl.store(P_, acc)
            tl.atomic_xchg(locks + pid, 1)

        # Store final results in C
        rm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        C_ = C + rm[:, None] * stride_cm + rn[None, :] * stride_cn
        mask = (rm < M)[:, None] & (rn < N)[None, :]
        tl.store(C_, acc, mask=mask)


def run_triton_kernel(P, C, locks, num_sms, k, M, N, stride_cm, stride_cn, BLOCK_SIZE_M, BLOCK_SIZE_N):
    grid = (num_sms,)
    spinning_lock_kernel[grid](
        P, C, locks, num_sms, k, M, N, stride_cm, stride_cn, BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N,)

BLOCK_SIZE_M = 128
BLOCK_SIZE_N = 128
BLOCK_SIZE_K = 64
M = 1024
N = 1024
K = 1024
num_sms = 304
total_programs_streamk = num_sms
total_blocks_M = triton.cdiv(M, BLOCK_SIZE_M)
total_blocks_N = triton.cdiv(N, BLOCK_SIZE_N)
iters_per_tile = triton.cdiv(K, BLOCK_SIZE_K)
total_tiles = total_blocks_M * total_blocks_N
total_tiles_streamk = total_tiles % total_programs_streamk
total_blocking_tiles = total_tiles - total_tiles_streamk
total_streamk_iters = total_tiles_streamk * iters_per_tile
streamk_iters_pcu = total_streamk_iters // num_sms
streamk_remainder_iters = total_streamk_iters % num_sms
print(f"M,N,K={M},{N},{K} ; BLK_M,N,K={BLOCK_SIZE_M},{BLOCK_SIZE_N},{BLOCK_SIZE_K}")
print(f"{total_blocks_M=} x {total_blocks_N=} = {total_tiles=}")
print(f"{total_tiles_streamk=} + {total_blocking_tiles=} = {total_tiles=}")
print(f"{total_streamk_iters=} + {streamk_iters_pcu=} = {streamk_remainder_iters=}")
k=3

P = torch.zeros((num_sms * BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=torch.float32, device='cuda')
C = torch.zeros((M, N), dtype=torch.float32, device='cuda')
locks = torch.zeros(num_sms, dtype=torch.int32, device='cuda')

stride_cm = C.stride(0)
stride_cn = C.stride(1)

run_triton_kernel(P, C, locks, num_sms, k, M, N, stride_cm, stride_cn, BLOCK_SIZE_M, BLOCK_SIZE_N)

# Verify the output
print(C)

