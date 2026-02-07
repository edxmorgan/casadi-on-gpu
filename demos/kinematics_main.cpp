#include <iostream>
#include <vector>
#include <algorithm>
#include <cstdlib>
#include <cuda_runtime.h>

#include "fk_alpha.cuh"

static void die_cuda(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::cerr << msg << " : " << cudaGetErrorString(err) << "\n";
        std::exit(1);
    }
}

int main() {
    // Problem sizes
    const int DOF      = 4;      // i0[4]
    const int OUT_DIM  = 6;      // o0[6]
    const int N        = 80000;  // batch size

    // Host arrays
    std::vector<casadi_real> h_q_all(static_cast<size_t>(N) * DOF);
    std::vector<casadi_real> h_p1(6);
    std::vector<casadi_real> h_p2(6);
    std::vector<casadi_real> h_out_all(static_cast<size_t>(N) * OUT_DIM);

    // Populate joint angles
    for (int i = 0; i < N; ++i) {
        h_q_all[static_cast<size_t>(DOF) * i + 0] = 0.1f * i;
        h_q_all[static_cast<size_t>(DOF) * i + 1] = 0.2f * i;
        h_q_all[static_cast<size_t>(DOF) * i + 2] = 0.3f * i;
        h_q_all[static_cast<size_t>(DOF) * i + 3] = 0.4f * i;
    }

    // Set params
    h_p1 = {0.190f, 0.000f, -0.120f, 3.142f, 0.000f, 0.000f};
    h_p2 = {0.000f, 0.000f,  0.000f, 0.000f, 0.000f, 0.000f};

    // Device pointers
    casadi_real *d_q_all   = nullptr;
    casadi_real *d_p1      = nullptr;
    casadi_real *d_p2      = nullptr;
    casadi_real *d_out_all = nullptr;

    die_cuda(cudaMalloc(&d_q_all,   static_cast<size_t>(N) * DOF     * sizeof(casadi_real)), "cudaMalloc d_q_all");
    die_cuda(cudaMalloc(&d_p1,      6                               * sizeof(casadi_real)), "cudaMalloc d_p1");
    die_cuda(cudaMalloc(&d_p2,      6                               * sizeof(casadi_real)), "cudaMalloc d_p2");
    die_cuda(cudaMalloc(&d_out_all, static_cast<size_t>(N) * OUT_DIM * sizeof(casadi_real)), "cudaMalloc d_out_all");

    // Copy H2D
    die_cuda(cudaMemcpy(d_q_all, h_q_all.data(),
                        static_cast<size_t>(N) * DOF * sizeof(casadi_real),
                        cudaMemcpyHostToDevice),
            "cudaMemcpy q_all");
    die_cuda(cudaMemcpy(d_p1, h_p1.data(), 6 * sizeof(casadi_real), cudaMemcpyHostToDevice), "cudaMemcpy p1");
    die_cuda(cudaMemcpy(d_p2, h_p2.data(), 6 * sizeof(casadi_real), cudaMemcpyHostToDevice), "cudaMemcpy p2");

    // Launch config
    const int threads_per_block = 128;
    const int blocks = (N + threads_per_block - 1) / threads_per_block;

    // Warmup
    const int warmup = 50;
    for (int i = 0; i < warmup; ++i) {
        fkeval_kernel<<<blocks, threads_per_block>>>(d_q_all, d_p1, d_p2, d_out_all, N);
    }
    die_cuda(cudaGetLastError(), "Kernel launch (warmup)");
    die_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize (warmup)");

    // Timed runs (kernel only)
    const int reps = 500;
    cudaEvent_t start, stop;
    die_cuda(cudaEventCreate(&start), "cudaEventCreate start");
    die_cuda(cudaEventCreate(&stop), "cudaEventCreate stop");

    die_cuda(cudaEventRecord(start), "cudaEventRecord start");
    for (int i = 0; i < reps; ++i) {
        fkeval_kernel<<<blocks, threads_per_block>>>(d_q_all, d_p1, d_p2, d_out_all, N);
    }
    die_cuda(cudaGetLastError(), "Kernel launch (timed)");
    die_cuda(cudaEventRecord(stop), "cudaEventRecord stop");
    die_cuda(cudaEventSynchronize(stop), "cudaEventSynchronize stop");

    float ms_total = 0.0f;
    die_cuda(cudaEventElapsedTime(&ms_total, start, stop), "cudaEventElapsedTime");

    die_cuda(cudaEventDestroy(start), "cudaEventDestroy start");
    die_cuda(cudaEventDestroy(stop), "cudaEventDestroy stop");

    const double ms_per = static_cast<double>(ms_total) / reps;
    const double evals_per_s = (static_cast<double>(N) / ms_per) * 1000.0;

    std::cout << "\nFK kernel timing\n";
    std::cout << "  GPU batch N=" << N << ", threads_per_block=" << threads_per_block
              << ", blocks=" << blocks << "\n";
    std::cout << "  Total: " << ms_total << " ms for " << reps << " launches\n";
    std::cout << "  Per launch: " << ms_per << " ms\n";
    std::cout << "  Throughput: " << evals_per_s << " eval/s\n\n";

    // Copy results back (not included in kernel timing)
    die_cuda(cudaMemcpy(h_out_all.data(), d_out_all,
                        static_cast<size_t>(N) * OUT_DIM * sizeof(casadi_real),
                        cudaMemcpyDeviceToHost),
            "cudaMemcpy out_all");

    // Print only a few candidates, printing 80k ruins benchmarks
    const int to_print = 3;
    for (int i = 0; i < to_print; ++i) {
        std::cout << "Candidate " << i << " output: ";
        for (int j = 0; j < OUT_DIM; ++j) {
            std::cout << h_out_all[static_cast<size_t>(OUT_DIM) * i + j]
                      << (j + 1 == OUT_DIM ? "" : " ");
        }
        std::cout << "\n";
    }

    // Cleanup
    cudaFree(d_q_all);
    cudaFree(d_p1);
    cudaFree(d_p2);
    cudaFree(d_out_all);

    return 0;
}
