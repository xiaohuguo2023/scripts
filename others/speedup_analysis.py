import torch

import triton
import triton.language as tl
import sys
import argparse
import pytest
import json
#use 03-matmul tutorial
import 

def read_shapes(filename):
    shapes = []
    with open(filename, 'r') as file:
        for line in file:
            # Strip whitespace and then take the part after the line number
            line_content = line.strip().split(')')[0].split('(')[-1]
            if line_content:  # Check if there is anything to process
                # Convert the numbers to integers and create a tuple out of them
                shape = tuple(map(int, line_content.split(',')))
                shapes.append(shape)
    return shapes

def speedup(filename):

    shapes=read_shapes(filename)
    results = []
    shapes = shapes[0]
    for M, N, K in shapes:
        a = torch.randn((M, K), device='cuda', dtype=torch.float16)
        b = torch.randn((K, N), device='cuda', dtype=torch.float16)
        torch_ms = triton.testing.do_bench(lambda: torch.matmul(a, b))
        triton_ms = triton.testing.do_bench(lambda: matmul(a, b))

        d = {
            "m": M,
            "n": N,
            "k": K,
            "pytorch_ms": torch_ms,
            "speedup": triton_ms / torch_ms,
        }
        results.append(d)
     
    results.sort(key=lambda x: x["speedup"], reverse=False)
    with open("results.json", "w") as f:
        json.dump(results, f, indent=4)

def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark shapes file")
    parser.add_argument("-v", action='store_true', help="Verbose output")
    parser.add_argument("filename", type=str, help="Filename containing the shapes")
    return parser.parse_args()

def main():
    args = parse_args()
    speedup(args.filename)

if __name__ == '__main__':
    sys.exit(main())
