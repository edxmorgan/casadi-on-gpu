#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include "device_fk_wrapper.cuh"

int main() {
    // Problem sizes
    const int DOF      = 4;   // i0[4]
    const int OUT_DIM  = 6;   // o0[6]
    const int N        = 80000;   // number of FK candidates to test

    // 1. Host arrays
    std::vector<casadi_real> h_q_all(N * DOF);
    std::vector<casadi_real> h_p1(6);
    std::vector<casadi_real> h_p2(6);
    std::vector<casadi_real> h_out_all(N * OUT_DIM);

    // populating joint angles
    for (int i = 0; i < N; ++i) {
        h_q_all[DOF * i + 0] = 0.1f * i;
        h_q_all[DOF * i + 1] = 0.2f * i;
        h_q_all[DOF * i + 2] = 0.3f * i;
        h_q_all[DOF * i + 3] = 0.4f * i;
    }

    // Set params
    h_p1 = {0.190, 0.000, -0.120, 3.142, 0.000, 0.000};
    h_p2 = {0.000, 0.000, 0.000, 0.000, 0.000, 0.000};

    // Device pointers
    casadi_real *d_q_all  = nullptr;
    casadi_real *d_p1     = nullptr;
    casadi_real *d_p2     = nullptr;
    casadi_real *d_out_all = nullptr;

    cudaMalloc(&d_q_all,   N * DOF     * sizeof(casadi_real));
    cudaMalloc(&d_p1,      6           * sizeof(casadi_real));
    cudaMalloc(&d_p2,      6           * sizeof(casadi_real));
    cudaMalloc(&d_out_all, N * OUT_DIM * sizeof(casadi_real));

    // Copy from host to GPU device
    cudaMemcpy(d_q_all,   h_q_all.data(),
               N * DOF * sizeof(casadi_real),
               cudaMemcpyHostToDevice);

    cudaMemcpy(d_p1, h_p1.data(),
               6 * sizeof(casadi_real),
               cudaMemcpyHostToDevice);

    cudaMemcpy(d_p2, h_p2.data(),
               6 * sizeof(casadi_real),
               cudaMemcpyHostToDevice);

    // Launch kernel
    int threads_per_block = 128;
    int blocks = (N + threads_per_block - 1) / threads_per_block;

    fk_kernel<<<blocks, threads_per_block>>>(
        d_q_all,
        d_p1,
        d_p2,
        d_out_all,
        N
    );

    cudaDeviceSynchronize();

    // Copy results gpu to host
    cudaMemcpy(h_out_all.data(), d_out_all,
               N * OUT_DIM * sizeof(casadi_real),
               cudaMemcpyDeviceToHost);

    // Print results for evaluated candidates
    for (int i = 0; i < N; ++i) {
        std::cout << "Candidate " << i << " output: ";
        for (int j = 0; j < OUT_DIM; ++j) {
            std::cout << h_out_all[OUT_DIM * i + j] << " ";
        }
        std::cout << std::endl;
    }

    // free up device memory
    cudaFree(d_q_all);
    cudaFree(d_p1);
    cudaFree(d_p2);
    cudaFree(d_out_all);

    return 0;
}