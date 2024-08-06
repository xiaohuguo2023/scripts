
### how to run 
```
root@smc300x-ccs-aus-GPUF292:/home/work/scripts/reproducer# python compile_driver_BM256_BN256_BK32_GM1_nW1_nS0_EU0_kP2_mfma16.py
try compile finished
kernel BM256_BN256_BK32_GM1_nW1_nS0_EU0_kP2_mfma16 took  0:04:16.810674
end of compile
```

### evidence to show slowness

```
root@smc300x-ccs-aus-GPUF292:/home/work/scripts/reproducer# time ~/.triton/llvm/llvm-4713bd4c-ubuntu-x64/bin/llc  -mtriple=amdgcn-hsa-amdhsa -mcpu=gfx942 ./streamk_gemm_BM256_BN256_BK32_GM1_nW1_nS0_EU0_kP2_mfma16.llir -o ./stream_gemm.amdgcn

real    3m49.375s
user    3m49.122s
sys     0m0.108s
```
