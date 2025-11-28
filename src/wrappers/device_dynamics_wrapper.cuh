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
#include "dynamics_blue.cuh"   // defines casadi_real, casadi_int, Vnext_reg_*
}

// Device helper that calls CasADi Dynamics evaluation function
__device__ void device_dynamics_eval(
    const casadi_real* sim_x,         // i0[12]
    const casadi_real* sim_u,         // i1[6]
    const casadi_real* sim_p,         // i2[33]
    const casadi_real* dt,            // i3[1]
    const casadi_real* f_ext,         // i4[6]
    casadi_real* sim_x_next           // o0[12]
)
{
    // Pointers to inputs (match Vnext_reg signature)
    const casadi_real* arg_local[Vnext_reg_SZ_ARG] = {
        sim_x, sim_u, sim_p, dt, f_ext
    };
    const casadi_real** arg = arg_local;

    // Pointers to outputs
    casadi_real* res_local[Vnext_reg_SZ_RES] = { sim_x_next };
    casadi_real** res = res_local;

    // Work arrays
    casadi_int  iw[Vnext_reg_SZ_IW > 0 ? Vnext_reg_SZ_IW : 1];
    casadi_real w [Vnext_reg_SZ_W  > 0 ? Vnext_reg_SZ_W  : 1];

    Vnext_reg(arg, res, iw, w, 0);
}

// Kernel to compute Dynamics for many candidates in parallel
__global__ void dynamics_kernel(
    const casadi_real* sim_x,        // shared initial state (12)
    const casadi_real* sim_u,        // shared control input (6)
    const casadi_real* sim_p_all,    // shape [N, 33], per-candidate params
    casadi_real dt,                  // shared timestep (1)
    const casadi_real* f_ext,        // shared external force (6)
    casadi_real* sim_x_next_all,     // shape [N, 12]
    int n_candidates
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= n_candidates) {
        return;
    }

    const int PARAM_DIM = 33;
    const int OUT_DIM = 12;

    const casadi_real* sim_p_i = sim_p_all + PARAM_DIM * idx;
    casadi_real* sim_x_next_i = sim_x_next_all + OUT_DIM * idx;

    const casadi_real dt_local = dt;
    device_dynamics_eval(sim_x, sim_u, sim_p_i, &dt_local, f_ext, sim_x_next_i);
}
