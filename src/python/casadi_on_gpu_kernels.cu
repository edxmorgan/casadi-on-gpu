#include "casadi_on_gpu_api.h"

#include <stdexcept>
#include <string>

#include <cuda_runtime.h>

#include "dynamics_blue.cuh"
#include "fk_alpha.cuh"

namespace casadi_on_gpu {
namespace {

void throw_on_cuda_error(cudaError_t err, const char* context) {
    if (err == cudaSuccess) {
        return;
    }
    std::string message = context;
    message += ": ";
    message += cudaGetErrorString(err);
    throw std::runtime_error(message);
}

cudaStream_t stream_from_ptr(std::uintptr_t stream_ptr) {
    return reinterpret_cast<cudaStream_t>(stream_ptr);
}

}  // namespace

void fk_forward(std::uintptr_t q_all_ptr,
                std::uintptr_t p1_ptr,
                std::uintptr_t p2_ptr,
                std::uintptr_t out_ptr,
                int n_candidates,
                int threads_per_block,
                std::uintptr_t stream_ptr,
                bool sync) {
    if (n_candidates <= 0) {
        throw std::invalid_argument("n_candidates must be > 0");
    }
    if (threads_per_block <= 0) {
        throw std::invalid_argument("threads_per_block must be > 0");
    }

    auto* q_all = reinterpret_cast<casadi_real*>(q_all_ptr);
    auto* p1 = reinterpret_cast<casadi_real*>(p1_ptr);
    auto* p2 = reinterpret_cast<casadi_real*>(p2_ptr);
    auto* out_all = reinterpret_cast<casadi_real*>(out_ptr);
    cudaStream_t stream = stream_from_ptr(stream_ptr);

    const int blocks = (n_candidates + threads_per_block - 1) / threads_per_block;
    fkeval_kernel<<<blocks, threads_per_block, 0, stream>>>(
        q_all,
        p1,
        p2,
        out_all,
        n_candidates
    );

    throw_on_cuda_error(cudaGetLastError(), "fkeval_kernel launch failed");
    if (sync) {
        throw_on_cuda_error(cudaStreamSynchronize(stream), "fkeval_kernel sync failed");
    }
}

void dynamics_forward(std::uintptr_t sim_x_ptr,
                      std::uintptr_t sim_u_ptr,
                      std::uintptr_t sim_p_all_ptr,
                      std::uintptr_t dt_ptr,
                      std::uintptr_t f_ext_ptr,
                      std::uintptr_t sim_x_next_all_ptr,
                      int n_candidates,
                      int threads_per_block,
                      std::uintptr_t stream_ptr,
                      bool sync) {
    if (n_candidates <= 0) {
        throw std::invalid_argument("n_candidates must be > 0");
    }
    if (threads_per_block <= 0) {
        throw std::invalid_argument("threads_per_block must be > 0");
    }

    auto* sim_x = reinterpret_cast<casadi_real*>(sim_x_ptr);
    auto* sim_u = reinterpret_cast<casadi_real*>(sim_u_ptr);
    auto* sim_p_all = reinterpret_cast<casadi_real*>(sim_p_all_ptr);
    auto* dt = reinterpret_cast<const casadi_real*>(dt_ptr);
    auto* f_ext = reinterpret_cast<casadi_real*>(f_ext_ptr);
    auto* sim_x_next_all = reinterpret_cast<casadi_real*>(sim_x_next_all_ptr);
    cudaStream_t stream = stream_from_ptr(stream_ptr);

    const int blocks = (n_candidates + threads_per_block - 1) / threads_per_block;
    Vnext_reg_kernel<<<blocks, threads_per_block, 0, stream>>>(
        sim_x,
        sim_u,
        sim_p_all,
        dt,
        f_ext,
        sim_x_next_all,
        n_candidates
    );

    throw_on_cuda_error(cudaGetLastError(), "Vnext_reg_kernel launch failed");
    if (sync) {
        throw_on_cuda_error(cudaStreamSynchronize(stream), "Vnext_reg_kernel sync failed");
    }
}

void device_synchronize() {
    throw_on_cuda_error(cudaDeviceSynchronize(), "cudaDeviceSynchronize failed");
}

}  // namespace casadi_on_gpu
