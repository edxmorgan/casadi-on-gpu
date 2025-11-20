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
#include "fk_alpha.h"   // defines casadi_real, casadi_int, fkeval_*
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

    // Work arrays (sizes from fk_alpha.h, might be zero)
    casadi_int  iw[fkeval_SZ_IW > 0 ? fkeval_SZ_IW : 1];
    casadi_real w [fkeval_SZ_W  > 0 ? fkeval_SZ_W  : 1];

    fkeval(arg, res, iw, w, 0);
}

// Simple kernel that calls the helper once
__global__ void fk_kernel(
    const casadi_real* q,
    const casadi_real* p1,
    const casadi_real* p2,
    casadi_real* out
)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        device_fk_eval(q, p1, p2, out);
    }
}
