#pragma once

// These macros keep IntelliSense / non CUDA compilers calm
#ifndef __CUDACC__
  #ifndef __device__
    #define __device__
  #endif
  #ifndef __global__
    #define __global__
  #endif
#endif

#include <cuda_runtime.h>

extern "C" {
#include "fk_alpha.cuh"   // defines casadi_real, casadi_int, fkeval_*
}

// Device helper that calls CasADi FK
__device__ void device_fk_eval(
    const casadi_real* q,        // i0[4]
    const casadi_real* params1,  // i1[6]
    const casadi_real* params2,  // i2[6]
    casadi_real* out             // o0[6]
)
{
    // Pointers to inputs
    const casadi_real* arg_local[3] = { q, params1, params2 };
    const casadi_real** arg = arg_local;

    // Pointers to outputs
    casadi_real* res_local[1] = { out };
    casadi_real** res = res_local;

    // Work arrays
    casadi_int  iw[fkeval_SZ_IW > 0 ? fkeval_SZ_IW : 1];
    casadi_real w [fkeval_SZ_W  > 0 ? fkeval_SZ_W  : 1];

    fkeval(arg, res, iw, w, 0);
}

// Kernel to compute FK for many candidates in parallel
__global__ void fk_kernel(
    const casadi_real* q_all,    // shape [N, 4]
    const casadi_real* p1,       // shared parameters
    const casadi_real* p2,       // shared parameters
    casadi_real* out_all,        // shape [N, 6]
    int n_candidates
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= n_candidates) {
        return;
    }

    // each candidate has 4 joints
    const int DOF = 4;
    const int OUT_DIM = 6;

    const casadi_real* q_i  = q_all  + DOF * idx;
    casadi_real*       out_i = out_all + OUT_DIM * idx;

    device_fk_eval(q_i, p1, p2, out_i);
}

