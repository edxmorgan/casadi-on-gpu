#pragma once

#include <cstdint>

namespace casadi_on_gpu {

void fk_forward(std::uintptr_t q_all_ptr,
                std::uintptr_t p1_ptr,
                std::uintptr_t p2_ptr,
                std::uintptr_t out_ptr,
                int n_candidates,
                int threads_per_block,
                std::uintptr_t stream_ptr,
                bool sync);

void dynamics_forward(std::uintptr_t sim_x_ptr,
                      std::uintptr_t sim_u_ptr,
                      std::uintptr_t sim_p_all_ptr,
                      std::uintptr_t dt_ptr,
                      std::uintptr_t f_ext_ptr,
                      std::uintptr_t sim_x_next_all_ptr,
                      int n_candidates,
                      int threads_per_block,
                      std::uintptr_t stream_ptr,
                      bool sync);

void device_synchronize();

}  // namespace casadi_on_gpu
