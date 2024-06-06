import argparse
import sys
import yaml
import os
import glob
import subprocess
import torch
import triton
import triton.language as tl
from matmul_kernel import matmul_kernel
from datetime import datetime
import pandas as pd
import torch.distributed as dist
from torch.multiprocessing import spawn

def get_full_tuning_space():
    configs = []
    block_mn_range = [16, 32, 64, 128, 256]
    block_k_range = [16, 32, 64, 128, 256]
    split_k_range = [1, 2, 4, 5, 6, 8, 10, 12, 16, 18, 24]
    num_warps_range = [1, 2, 4, 8]
    group_m_range = [1, 4, 8, 16, 32]
    num_stage_range = [0]
    waves_per_eu_range = [0]
    matrix_instr_nonkdim_range = [16, 32]
    kpack_range = [1, 2]

    for block_m in block_mn_range:
        for block_n in block_mn_range:
            for block_k in block_k_range:
                for num_warps in num_warps_range:
                    for group_m in group_m_range:
                        for split_k in split_k_range:
                            for num_stages in num_stage_range:
                                for waves_per_eu in waves_per_eu_range:
                                    for matrix_instr_nonkdim in matrix_instr_nonkdim_range:
                                        for kpack in kpack_range:
                                            configs.append({
                                                'BLOCK_SIZE_M': block_m, 'BLOCK_SIZE_N': block_n, 'BLOCK_SIZE_K': block_k,
                                                'GROUP_SIZE_M': group_m, 'SPLIT_K': split_k, 'num_warps': num_warps,
                                                'num_stages': num_stages, 'waves_per_eu': waves_per_eu,
                                                'matrix_instr_nonkdim': matrix_instr_nonkdim, 'kpack': kpack
                                            })
    return configs

def prune_configs(M, N, K, configs, elemBytes_a, elemBytes_b):
    pruned_configs = []
    if M < 32 or N < 32:
        mfma = 16
    else:
        mfma = 32

    large_gemm = False
    if M >= 2048 and N >=2048:
        large_gemm = True

    for config in configs:
        BLOCK_SIZE_M = config.get("BLOCK_SIZE_M")
        BLOCK_SIZE_N = config.get("BLOCK_SIZE_N")
        BLOCK_SIZE_K = config.get("BLOCK_SIZE_K")
        num_warps = config.get("num_warps")
        matrix_instr_nonkdim = config.get("matrix_instr_nonkdim")
        kpack = config.get("kpack")
        if matrix_instr_nonkdim > mfma:
            continue
        if mfma == 4 and BLOCK_SIZE_K < 64:
            continue
        if BLOCK_SIZE_M * BLOCK_SIZE_N < 64:
            continue
        SPLIT_K = config.get("SPLIT_K")
        GROUP_M = config.get("GROUP_SIZE_M")
        if BLOCK_SIZE_M < matrix_instr_nonkdim or BLOCK_SIZE_N < matrix_instr_nonkdim:
            continue
        if M <= matrix_instr_nonkdim and BLOCK_SIZE_M != matrix_instr_nonkdim:
            continue
        if N <= matrix_instr_nonkdim and BLOCK_SIZE_N != matrix_instr_nonkdim:
            continue
        if BLOCK_SIZE_M > M * 2 and BLOCK_SIZE_M != 16:
            continue
        if BLOCK_SIZE_N > N * 2 and BLOCK_SIZE_N != 16:
            continue
        if SPLIT_K != 1 and not need_split_k(M, N, K):
            continue
        leap = SPLIT_K * BLOCK_SIZE_K
        modv = K % leap
        if modv != 0:
            continue
        if GROUP_M * BLOCK_SIZE_M > M and GROUP_M != 1:
            continue
        LDS = BLOCK_SIZE_K * BLOCK_SIZE_M * elemBytes_a + BLOCK_SIZE_K * BLOCK_SIZE_N * elemBytes_b
        if LDS > 65536:
            continue
        if large_gemm:
            if BLOCK_SIZE_M < 64 or BLOCK_SIZE_N < 64:
                continue
            if BLOCK_SIZE_K < 64:
                continue
            if num_warps < 4:
                continue
        pruned_configs.append(config)
    return pruned_configs

def need_split_k(SIZE_M, SIZE_N, SIZE_K):
    return (SIZE_M < 64 or SIZE_N < 64) and SIZE_K > 1024

def run_bash_command_wrapper(commandstring, capture=True):
    try:
        run_bash_command(commandstring, capture)
    except subprocess.CalledProcessError as e:
        if not capture:
            print(f"running {commandstring} one more time")
        run_bash_command(commandstring, capture)

def run_bash_command(commandstring, capture=True):
    if capture:
        proc = subprocess.run(commandstring, shell=True, check=True, executable='/bin/bash', stdout=subprocess.PIPE)
        return proc.stdout.splitlines()
    proc = subprocess.run(commandstring, shell=True, check=True, executable='/bin/bash')
    return None

def read_config(config):
    block_m = config.get('BLOCK_SIZE_M')
    block_n = config.get('BLOCK_SIZE_N')
    block_k = config.get('BLOCK_SIZE_K')
    group_m = config.get('GROUP_SIZE_M')
    split_k = config.get('SPLIT_K')
    num_warps = config.get('num_warps')
    num_stages = config.get('num_stages')
    waves_per_eu = config.get('waves_per_eu')
    mfma_instr_size = config.get('matrix_instr_nonkdim')
    kpack = config.get('kpack')
    return block_m, block_n, block_k, group_m, split_k, num_warps, num_stages, waves_per_eu, mfma_instr_size, kpack

def gen_kernel_and_configStr_from_config(M, N, K, config, dtype_a, dtype_b, dtype_c):
    block_m, block_n, block_k, group_m, split_k, num_warps, num_stages, waves_per_eu, mfmaInstrSize, kpack = read_config(config)
    torch_dtype_a = 'fp16'
    torch_dtype_b = 'fp16'
    torch_dtype_c = 'fp16'
    if dtype_a:
        torch_dtype_a = tl_to_torch_types[name_to_tl_types[dtype_a]]
    if dtype_b:
        torch_dtype_b = tl_to_torch_types[name_to_tl_types[dtype_b]]
    if dtype_c:
        torch_dtype_c = tl_to_torch_types[name_to_tl_types[dtype_c]]
    configStr = f"M{M}_N{N}_K{K}_BM{block_m}_BN{block_n}_BK{block_k}_GM{group_m}_SK{split_k}_nW{num_warps}_nS{num_stages}_EU{waves_per_eu}_kP{kpack}_mfma{mfmaInstrSize}"

    matmul_def_str = f"""
def matmul_{configStr}(a, b, c, M, N, K, am, ak, bk, bn, cm, cn, warmup=False):
    grid = triton.cdiv(M, {block_m}) * triton.cdiv(N, {block_n}), {split_k}
    if warmup:
        matmul_kernel_{configStr}.warmup(
            {torch_dtype_a}, {torch_dtype_b}, {torch_dtype_c},
            M, N, K,
            am, ak, bk, bn, cm, cn,
            BLOCK_SIZE_M = {block_m},
            BLOCK_SIZE_N = {block_n},
            BLOCK_SIZE_K = {block_k},
            GROUP_SIZE_M = {group_m},
            SPLIT_K = {split_k},
            num_warps = {num_warps},
            num_stages = {num_stages},
            waves_per_eu = {waves_per_eu},
            matrix_instr_nonkdim = {mfmaInstrSize},
            kpack = {kpack},
            grid=(1,)
        )
        return None
    else:
        matmul_kernel_{configStr}[grid](
            a, b, c,
            M, N, K,
            am, ak, bk, bn, cm, cn,
            BLOCK_SIZE_M = {block_m},
            BLOCK_SIZE_N = {block_n},
            BLOCK_SIZE_K = {block_k},
            GROUP_SIZE_M = {group_m},
            SPLIT_K = {split_k},
            num_warps = {num_warps},
            num_stages = {num_stages},
            waves_per_eu = {waves_per_eu},
            matrix_instr_nonkdim = {mfmaInstrSize},
            kpack = {kpack}
        )
        return c

def try_config_{configStr}(M, N, K, am, ak, bk, bn, cm, cn):
    try:
        matmul_{configStr}(None, None, None, M, N, K, am, ak, bk, bn, cm, cn, True)
        return True
    except Exception as e:
        print(f'invalid config(compilation): {configStr}: ', e, flush=True)
        return False
"""
    return configStr, matmul_def_str

def generated_kernel_name(M, N, K, gpu_id):
    path = os.path.dirname(os.path.abspath(__file__))
    return f"{path}/generated_kernel{M}-{N}-{K}-{gpu_id}.py"

def generate_kernel(rank, world_size, M, N, K, col_a, col_b, dtype_a, dtype_b, dtype_c, init_type, configs, jobs, iters, run_bench):
    filenames = [generated_kernel_name(M, N, K, i) for i in range(jobs)]
    f_kernel = [open(path, 'w') for path in filenames]

    import_str = """import torch
import triton
import triton.language as tl
import argparse
import sys
import torch.multiprocessing as mp
from tune_gemm import gen_input
"""
    for fi in range(jobs):
        f_kernel[fi].write(import_str + "\n")

    with open(os.path.dirname(os.path.abspath(__file__))+"/matmul_kernel.py") as file:
        matmul_kernel_code = file.read()
    idx = 0
    for config in configs:
        file_idx = idx % jobs
        configStr, matmul_def_str = gen_kernel_and_configStr_from_config(M, N, K, config, dtype_a, dtype_b, dtype_c)
        matmul_kernel_config = matmul_kernel_code.replace("matmul_kernel", f"matmul_kernel_{configStr}")
        matmul_kernel_config = matmul_kernel_config.replace("import triton.language as tl", "")
        matmul_kernel_config = matmul_kernel_config.replace("import triton", "")
        f_kernel[file_idx].write(matmul_kernel_config + "\n\n")
        f_kernel[file_idx].write(matmul_def_str + "\n")
        idx += 1

    test_gemm_pre_str = f"""def test_gemm(M, N, K, num_threads):
    results = []
    config_names = []
    a, a_fp16 = gen_input(M, K, '{dtype_a}', {col_a}, 1, '{init_type}', device='cuda')
    b, b_fp16 = gen_input(K, N, '{dtype_b}', {col_b}, 2, '{init_type}', device='cuda')
    c = torch.zeros((M, N), device=a.device, dtype={tl_to_torch_types[name_to_tl_types[dtype_c]]})
    task_args = (M, N, K,
                 a.stride(0), a.stride(1),
                 b.stride(0), b.stride(1),
                 c.stride(0), c.stride(1))

"""
    for fi in range(jobs):
        f_kernel[fi].write(test_gemm_pre_str + "\n")

    idx = 0
    for config in configs:
        configStr, _ = gen_kernel_and_configStr_from_config(M, N, K, config, None, None, None)
        task_str = f"    results.append(try_config_{configStr}(*task_args))\n"
        task_str += f"    config_names.append('{configStr}')\n"
        f_kernel[idx % jobs].write(task_str)
        idx += 1

    for fi in range(jobs):
        threadpool_str = """
    failed_configs = []
    for i in range(len(results)):
        if not results[i]:
            failed_configs.append(config_names[i])
    with open("{filename}.failed_configs", "w") as f:
        for cfg in failed_configs:
            f.write(cfg + "\\n")
    else:
        try:
            with open("{filename}.failed_configs", "r") as f:
                failed_configs = [cfg.strip() for cfg in f.readlines()]
        except Exception:
            failed_configs = []
        """.format(filename=filenames[fi])
        f_kernel[fi].write(threadpool_str)

    idx = 0
    runs = iters if run_bench else 200
    for config in configs:
        configStr, _ = gen_kernel_and_configStr_from_config(M, N, K, config, None, None, None)
        matmul_call_str = f"""
    if '{configStr}' not in failed_configs:
        for i in range({runs}):
            d = matmul_{configStr}(a, b, c, M, N, K, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1))"""
        f_kernel[idx % jobs].write(matmul_call_str + "\n")
        idx += 1
    for fi in range(jobs):
        f_kernel[fi].write("        return d\n")

    def_main_str = """
def main():
    parser = argparse.ArgumentParser(
        prog="tune a specific gemm size",
        allow_abbrev=False,)
    parser.add_argument("-n", type=int, default=1, help='number of threads')
    args = parser.parse_args()
    numThreads = args.n
    """
    test_gemm_call_str = f'test_gemm({M}, {N}, {K}, numThreads)'
    for fi in range(jobs):
        f_kernel[fi].write(def_main_str)
        f_kernel[fi].write(test_gemm_call_str + "\n\n")
        f_kernel[fi].write("""if __name__ == '__main__':
   sys.exit(main())""")
        f_kernel[fi].close()

def extract_kernel_time(M, N, K, config, df):
    configStr, _ = gen_kernel_and_configStr_from_config(M, N, K, config, None, None, None)
    df = df[df['KernelName'].str.contains(configStr)]
    meanTime = df['DurationNs'].tail(100).mean()
    return config, meanTime

def profile_batch_kernels(rank, world_size, M, N, K, jobs, verbose):
    ngpus = world_size
    gpu_id = rank
    os.environ['ROCR_VISIBLE_DEVICES'] = str(gpu_id)
    jobId = gpu_id
    while jobId < jobs:
        if verbose:
            print(f"profiling {generated_kernel_name(M, N, K, jobId)} on GPU {gpu_id}")
        run_bash_command_wrapper(f"rocprof --stats -o results-{jobId}.csv python {generated_kernel_name(M, N, K, jobId)}", capture=(verbose < 2))
        jobId += ngpus

def tune_gemm_config(rank, world_size, M, N, K, col_a, col_b, dtype_a, dtype_b, dtype_c, init_type, configs, run_bench, jobs, iters, skipWarmup, verbose=0, num_threads=16):
    setup(rank, world_size)
    
    if rank == 0:
        generate_kernel(rank, world_size, M, N, K, col_a, col_b, dtype_a, dtype_b, dtype_c, init_type, configs, jobs, iters, run_bench)
        run_bash_command("rm -rf ~/.triton/cache")

        start_time = datetime.now()
        if not skipWarmup:
            for i in range(jobs):
                run_bash_command(f"python {generated_kernel_name(M, N, K, i)} -n {num_threads}", capture=(verbose < 2))
        compile_end = datetime.now()
        compile_time = compile_end - start_time
        if verbose:
            print(f"compile time: {compile_time}", flush=True)
    dist.barrier()

    profile_batch_kernels(rank, world_size, M, N, K, jobs, verbose)
    dist.barrier()

    if rank == 0:
        profile_end = datetime.now()
        profile_time = profile_end - compile_end
        if verbose:
            print(f"profile time: {profile_time}", flush=True)

        minTime = 1024 * 1024 * 1024
        tasks = []
        idx = 0
        df_prof = [pd.read_csv(f"results-{i}.csv") for i in range(jobs)]
        for config in configs:
            file_idx = idx % jobs
            tasks.append((M, N, K, config, df_prof[file_idx]))
            idx += 1

        for task in tasks:
            config, myTime = extract_kernel_time(*task)
            if myTime:
                min_us = myTime / 1000
                if min_us < minTime:
                    minTime = min_us
                    bestConfig = config
            else:
                min_us = -1
                print(f"invalid config(post processing): SIZE {M} {N} {K}: {config}", flush=True)
        post_end = datetime.now()
        post_time = post_end - profile_end
        if verbose:
            print(f"post processing time: {post_time}", flush=True)
        return minTime, bestConfig, compile_time, profile_time, post_time
    cleanup()

def gen_input(M, N, ty_name, needTrans, seed, init_type, device='cuda'):
    d_type = name_to_tl_types[ty_name]
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    @triton.jit
    def copy_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        offsets = tl.program_id(axis=0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        input = tl.load(input_ptr + offsets, mask=mask)
        output = input
        tl.store(output_ptr + offsets, output, mask=mask)

    def init_by_size_and_type(size, dtype, init_type):
        if init_type == 'hpl':
            return torch.empty(size, device='cuda', dtype=dtype).uniform_(-0.5, 0.5)
        # This init type has element[i] in row[j] equal to sin(i+j*N)
        elif init_type == 'trig_float':
            M, N = size
            return torch.reshape(torch.arange(0, M*N), (M, N)).sin().to(dtype=dtype, device='cuda')
        elif init_type == 'zeros':
            return torch.zeros(size, dtype=dtype, device='cuda')
        elif init_type == "randn":
            temp = torch.randn(size, dtype=dtype, device='cuda')
            return temp
        else:
            raise ValueError("Bad matrix initialization type.")

    raw_data = init_by_size_and_type((N,M) if needTrans else (M,N), torch.float32, init_type)
    if needTrans:
        raw_data = raw_data.T
    if (d_type == tl.float8e4b8 and TORCH_HAS_FP8E4B8) or \
        (d_type == tl.float8e5b16 and TORCH_HAS_FP8E5B16) or not d_type.is_fp8():
        input = raw_data.to(tl_to_torch_types[d_type])
        input_f16 = input.to(torch.float16)
    else:
        f8_tensor = raw_data.to(torch.int8)
        # keep only two bits of exponent to avoid overflow
        f8_tensor = f8_tensor & 0b00111111
        input = triton.reinterpret(f8_tensor, d_type)
        input_f16 = torch.empty_like(f8_tensor, dtype=torch.float16)
        grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
        n_elements = raw_data.numel()
        copy_kernel[grid](input, input_f16, n_elements, BLOCK_SIZE=1024)

    return input, input_f16

def matmul(a, b, c, block_m, block_n, block_k, group_m, split_k, num_warps, num_stages, waves_per_eu, mfmaInstrSize, kpack):
    # Check constraints.
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    #assert a.is_contiguous(), "Matrix A must be contiguous"
    #assert b.is_contiguous(), "Matrix B must be contiguous"
    M, K = a.shape
    K, N = b.shape
    # 1D launch kernel where each block gets its own program.

    grid = triton.cdiv(M, block_m) * triton.cdiv(N, block_n), split_k

    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_SIZE_M=block_m,
        BLOCK_SIZE_N=block_n,
        BLOCK_SIZE_K=block_k,
        GROUP_SIZE_M=group_m,
        SPLIT_K=split_k,
        num_warps=num_warps,
        num_stages=num_stages,
        waves_per_eu=waves_per_eu,
        matrix_instr_nonkdim = mfmaInstrSize,
        kpack = kpack
    )
    return c


def test_correctness(M, N, K, col_a, col_b, dtype_a, dtype_b, dtype_c, init_type, config, verbose):
    block_m, block_n, block_k, group_m, split_k, num_warps, num_stages, waves_per_eu, mfmaInstrSize, kpack = read_config(config)
    torch.manual_seed(0)
    #a = torch.randn((M, K), device='cuda', dtype=datatype)
    #b = torch.randn((K, N), device='cuda', dtype=datatype)
    a, a_fp16 = gen_input(M, K, dtype_a, col_a, 1, init_type, device='cuda')
    b, b_fp16 = gen_input(K, N, dtype_b, col_b, 2, init_type, device='cuda')
    # Allocates output.
    c = torch.zeros((M, N), device=a.device, dtype=tl_to_torch_types[name_to_tl_types[dtype_c]])
    triton_output = matmul(a, b, c, block_m, block_n, block_k, group_m, split_k, num_warps, num_stages, waves_per_eu, mfmaInstrSize, kpack)
    torch_output = torch.matmul(a_fp16, b_fp16)
    # print(f"triton_output={triton_output}")
    # print(f"torch_output={torch_output}")
    rtol = 0 if torch.version.hip is None else 1e-2
    atol = 1e-3 if split_k == 1 else 4e-2
    row_a_str = 'N' if col_a else 'T'
    row_b_str = 'N' if col_b else 'T'
    size_str = ''
    if verbose:
        size_str = f'SIZE M: {M}, N: {N}, K: {K}, trans: {row_a_str}{row_b_str}'
    if torch.allclose(triton_output.to(torch.float16), torch_output, atol=atol, rtol=rtol):
        print(f'{size_str} Correct✅')
    else:
        print(f'{size_str} Incorrect❌')


def get_default_tuning_result_filename():
    git_branch_name = run_bash_command("git rev-parse --abbrev-ref HEAD")
    git_branch_name = git_branch_name[0].decode()
    git_commit_hash = run_bash_command("git rev-parse --short HEAD")
    git_commit_hash = git_commit_hash[0].decode()

    dt_string = datetime.now().strftime("%m-%d-%Y-%H:%M:%S")
    defaultName = f"tuning_results_{git_branch_name}@{git_commit_hash}_{dt_string}.yaml"
    return defaultName


def parse_args():
    parser = argparse.ArgumentParser(
        prog="tune a specific gemm size",
        allow_abbrev=False,
    )

    parser.add_argument("-m", type=int, default=0)
    parser.add_argument("-n", type=int, default=0)
    parser.add_argument("-k", type=int, default=0)
    parser.add_argument("-col_a", action='store_true', default=False, help='whether matrix a is column major')
    parser.add_argument("-col_b", action='store_true', default=False, help='whether matrix b is column major')
    parser.add_argument("-dtype_a", type=str, default='fp16', help="matrix a element data type")
    parser.add_argument("-dtype_b", type=str, default='fp16', help="matrix b element data type")
    parser.add_argument("-dtype_c", type=str, default='fp16', help="output element data type")
    parser.add_argument("--ngpus", type=int, default=0, help='number of GPUs used in the profiling step')
    parser.add_argument("--gpu_ids", type=lambda s: [int(id) for id in s.split(',')], default=[], help='list of gpu ids to use for tuning')
    parser.add_argument("--gemm_size_file", type=str, default="", help='yaml file to indicate matrix size')
    parser.add_argument("--o", type=str, default='', help='yaml file to store tuning results')
    parser.add_argument("--keep", action='store_true', default=False, help='keep generated files')
    parser.add_argument("--compare", action='store_true', default=False, help="Whether check result correctness")
    parser.add_argument("--compare_wo_tuning", action='store_true', default=False, help="Whether check result correctness")
    parser.add_argument("--benchmark", action='store_true', default=False, help="Benchmark the given config")
    parser.add_argument("--time_breakdown", action='store_true', default=False, help="Show detailed time breakdown of each step during the tuning")
    parser.add_argument("--verbose", action='store_true', default=False, help="enables time_breakdown and additional logging messages")
    parser.add_argument("--num_threads", type=int, default=16, help="number of threads to use for kernel compilation and post processing")
    parser.add_argument("--jobs", type=int, default=1, help="number of generated files")
    parser.add_argument("--iters", type=int, default=1000, help="number of generated files")
    parser.add_argument("--init_type", type=str, default='randn', help="Initialization type for input matrices (default uniform rand [0, 1.0)])")
    parser.add_argument("--no_warmup", action='store_true', default=False, help="Do not call the warmup kernel")
    args = parser.parse_args()
    if not args.o:
        if args.benchmark:
            args.o = "benchmarking_results.csv"
        else:
            args.o = get_default_tuning_result_filename()

    return args


TORCH_HAS_FP8E5B16 = hasattr(torch, 'float8_e5m2fnuz')
TORCH_HAS_FP8E4B8 = hasattr(torch, 'float8_e4m3fnuz')
tl_to_torch_types = {
    tl.float16: torch.float16,
    tl.bfloat16: torch.bfloat16,
    tl.float32: torch.float32,
    tl.int8: torch.int8,
    tl.int32: torch.int32,
}
if TORCH_HAS_FP8E5B16:
    tl_to_torch_types[tl.float8e5b16] = torch.float8_e5m2fnuz
if TORCH_HAS_FP8E4B8:
    tl_to_torch_types[tl.float8e4b8] = torch.float8_e4m3fnuz

name_to_tl_types = {
    'int8': tl.int8,
    'int32': tl.int32,
    'fp16': tl.float16,
    'fp32': tl.float32,
    'bf16': tl.bfloat16,
    'fp8': tl.float8e4b8,
    'bf8': tl.float8e5b16,
}

def process_item(item):
    M = item['M']
    N = item['N']
    K = item['K']
    col_a = False if item['rowMajorA'] == 'T' else True
    col_b = False if item['rowMajorB'] == 'T' else True
    del item['M']
    del item['N']
    del item['K']
    del item['rowMajorA']
    del item['rowMajorB']
    return M, N, K, col_a, col_b, item

def type_name_to_bytes(ty_name):
    if '32' in ty_name:
        return 4
    if '16' in ty_name:
        return 2
    if '8' in ty_name:
        return 1
    else:
        print(f"Unrecognized input type name {ty_name}")
        sys.exit(1)

def format_output(unformatted):
    if unformatted < 0.0001:
        formatted = "{:.3e}".format(unformatted)
    elif unformatted > 1000:
        formatted = "{:.1f}".format(unformatted)
    else:
        formatted = "{:.2f}".format(unformatted)
    return formatted

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    print(f"Rank {rank}/{world_size} process initialized.")

def cleanup():
    dist.destroy_process_group()
    print("Process group destroyed.")

def main():
    args = parse_args()
    world_size = len(args.gpu_ids) if args.gpu_ids else torch.cuda.device_count()
    print(f"Number of GPUs available: {world_size}")
    matrix_size_file = args.gemm_size_file
    output_file = args.o
    keepTmp = args.keep
    run_bench = args.benchmark
    jobs = args.jobs
    iters = args.iters
    skipWarmup = args.no_warmup

    # Get GPU ids
    ngpus = args.ngpus
    gpu_ids = args.gpu_ids
    if ngpus != 0 and gpu_ids:
        print("--ngpus and --gpu_ids are mutually exclusive options")
        return os.EX_USAGE
    if ngpus == 0 and not gpu_ids:
        ngpus = 1
    if ngpus != 0:
        gpus = list(range(ngpus))
    if gpu_ids:
        gpus = gpu_ids

    if run_bench:
        gpus = [gpus[0]]
        jobs = 1

    # Get element type
    dtype_a = args.dtype_a
    dtype_b = args.dtype_b
    dtype_c = args.dtype_c
    if not dtype_a in name_to_tl_types or not dtype_b in name_to_tl_types or not dtype_c in name_to_tl_types:
        print(f"Unsupported dtype_a {args.dtype_a} or dtype_b {args.dtype_b} or dtype_c {args.dtype_c}")
        print("Supported types: ", list(name_to_tl_types.keys()))
        sys.exit(1)

    mnks = []
    # TODO: make it more robust to get user input
    init_type = args.init_type
    if matrix_size_file == "" or not os.path.isfile(matrix_size_file):
        M = args.m
        N = args.n
        K = args.k
        col_a = args.col_a
        col_b = args.col_b
        mnks = [(M, N, K, col_a, col_b, None)]
    else:
        with open(matrix_size_file) as file:
            matrix_sizes = yaml.safe_load(file)
        for item in matrix_sizes:
            M, N, K, col_a, col_b, item = process_item(item)
            mnks.append((M, N, K, col_a, col_b, item))

    # Check correctness from given configs
    if args.compare_wo_tuning:
        for (M, N, K, col_a, col_b, myConfig) in mnks:
            test_correctness(M, N, K, col_a, col_b, dtype_a, dtype_b, dtype_c, init_type, myConfig, True)
        return

    configs_full = get_full_tuning_space()

    start_time = datetime.now()
    # Append to the output file so that we can save all results into one file
    f_results = open(output_file, 'a')
    if run_bench:
        print(f"Benchmarking gemm with {dtype_a} inputs")
        print("trans     M      N      K    TFLOPS   us")
        f_results.write("trans,M,N,K,TFLOPS,us\n")
    else:
        print(f"Tuning {len(mnks)} gemm sizes starts at: {start_time}", flush=True)

    for (M, N, K, col_a, col_b, myConfig) in mnks:
        start_local_time = datetime.now()
        # Obtain a pruned tuning space according to gemm size
        # If running benchmark, use the provided config
        pruned_configs = [myConfig] if run_bench else prune_configs(M, N, K, configs_full, type_name_to_bytes(dtype_a), type_name_to_bytes(dtype_b))

        row_a_str = 'N' if col_a else 'T'
        row_b_str = 'N' if col_b else 'T'
        size_str = f'SIZE: {M} {N} {K} {row_a_str}{row_b_str}'
        if not run_bench:
            print(f"{size_str} nConfigs: {len(pruned_configs)}", end=" ", flush=True)
        else:
            print(f"{row_a_str}{row_b_str}    {M:5d}  {N:5d}  {K:5d}    ", end="")
            f_results.write(f"{row_a_str}{row_b_str},{M},{N},{K},")

        # The main tuning function for one gemm size
        verbose_level = 0
        if args.time_breakdown:
            verbose_level = 1
        if args.verbose:
            verbose_level = 2

        def tune_gemm(rank, world_size, M, N, K, col_a, col_b, dtype_a, dtype_b, dtype_c, init_type, pruned_configs, run_bench, jobs, iters, skipWarmup, verbose, num_threads):
            return tune_gemm_config(rank, world_size, M, N, K, col_a, col_b, dtype_a, dtype_b, dtype_c, init_type, pruned_configs, run_bench, jobs, iters, skipWarmup, verbose, num_threads)

        minTime, bestConfig, compile_time, profile_time, post_time = spawn(
            tune_gemm,
            args=(world_size, M, N, K, col_a, col_b, dtype_a, dtype_b, dtype_c, init_type, pruned_configs, run_bench, jobs, iters, skipWarmup, verbose_level, num_threads),
            nprocs=world_size,
            join=True
        )

        # post processing the numbers
        perf_tflops = lambda us: 2 * M * N * K * 1e-12 / (us * 1e-6)
        tri_tflops = perf_tflops(minTime)
        formatted_tflops = format_output(tri_tflops)
        minTime = format_output(minTime)
        if not run_bench:
            print(f'TFLOPS: {formatted_tflops} time(us): {minTime}', end=" ", flush=True)

        bestConfig_compact_str, _ = gen_kernel_and_configStr_from_config(M, N, K, bestConfig, None, None, None)
        if not run_bench:
            print(f'best_config: {bestConfig_compact_str}', end=" ", flush=True)

        # write best config to tuning_results.yaml
        if run_bench:
            print(f"{formatted_tflops}     {minTime}")
            f_results.write(f"{formatted_tflops},{minTime}\n")

        sizeDict = {'M': M, 'N': N, 'K': K, 'rowMajorA': row_a_str, 'rowMajorB': row_b_str}
        sizeDict.update(bestConfig)
        if not run_bench:
            f_results.write("- " + str(sizeDict) + " ")
            f_results.write(f'# TFLOPS: {formatted_tflops} time(us): {minTime}\n')

        # remove generated files if asked to
        if not keepTmp:
            for i in range(jobs):
                generated_script = generated_kernel_name(M, N, K, i)
                os.remove(generated_script)
                if not skipWarmup:
                    os.remove(generated_script + ".failed_configs")
                for f in glob.glob(f"results-{i}.*"):
                    os.remove(f)

        # Check correctness if asked to
        if args.compare:
            print("correctness: ", end=" ", flush=True)
            test_correctness(M, N, K, col_a, col_b, dtype_a, dtype_b, dtype_c, init_type, bestConfig, False)
        elif not run_bench:
            print("", flush=True)

        end_local_time = datetime.now()
        if not run_bench:
            print(f">>> Elapsed time: {end_local_time - start_local_time} = {compile_time} (compile) + {profile_time} (profile) + {post_time} (post processing)", flush=True)

    f_results.close()

    end_time = datetime.now()
    tuning_time = end_time - start_time
    if not run_bench:
        print(f"Tuning ends at: {end_time}")
        print(f"Total tuning time (h:m:s): {tuning_time}")

if __name__ == '__main__':
    main()
