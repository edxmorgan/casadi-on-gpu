#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>

#include "dynamics_blue.cuh"

#ifndef POSTERIOR_BIN_PATH
#define POSTERIOR_BIN_PATH "src/posterior.bin"
#endif

static void die_cuda(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::cerr << msg << " : " << cudaGetErrorString(err) << "\n";
        std::exit(1);
    }
}

// Load posterior.bin from the configured path.
static bool load_posterior(const std::string& path,
                           std::vector<casadi_real>& buffer,
                           size_t expected_count) {
    std::ifstream fin(path, std::ios::binary | std::ios::ate);
    if (!fin) {
        std::cerr << "Could not open posterior bin at " << path << "\n";
        return false;
    }

    const std::streamsize file_size = fin.tellg();
    fin.seekg(0, std::ios::beg);

    const std::streamsize float_bytes  = static_cast<std::streamsize>(expected_count * sizeof(casadi_real));
    const std::streamsize double_bytes = static_cast<std::streamsize>(expected_count * sizeof(double));

    if (file_size == float_bytes) {
        fin.read(reinterpret_cast<char*>(buffer.data()), float_bytes);
        const std::streamsize got = fin.gcount();
        if (got != float_bytes) {
            std::cerr << "Size mismatch while reading " << path
                      << " (got " << got << " bytes, expected "
                      << float_bytes << ")\n";
            return false;
        }
    } else if (file_size == double_bytes) {
        std::vector<double> tmp(expected_count);
        fin.read(reinterpret_cast<char*>(tmp.data()), double_bytes);
        const std::streamsize got = fin.gcount();
        if (got != double_bytes) {
            std::cerr << "Size mismatch while reading " << path
                      << " (got " << got << " bytes, expected "
                      << double_bytes << ")\n";
            return false;
        }
        for (size_t i = 0; i < expected_count; ++i) {
            buffer[i] = static_cast<casadi_real>(tmp[i]);
        }
    } else {
        std::cerr << "Unexpected file size for " << path
                  << " (bytes=" << file_size
                  << ", expected " << float_bytes << " for float or "
                  << double_bytes << " for double)\n";
        return false;
    }

    std::cout << "Loaded posterior from " << path << "\n";
    return true;
}

int main() {
    const int STATE_DIM   = 12;    // i0[12]
    const int CONTROL_DIM = 6;     // i1[6]
    const int PARAM_DIM   = 33;    // i2[33]
    const int OUT_DIM     = 12;    // o0[12]
    const int N           = 80000; // number of candidates

    // Host-side data
    std::vector<casadi_real> h_sim_x(STATE_DIM, 0.0f);
    std::vector<casadi_real> h_sim_u(CONTROL_DIM, 0.0f);
    std::vector<casadi_real> h_sim_p_all(static_cast<size_t>(N) * PARAM_DIM, 0.0f);
    std::vector<casadi_real> h_f_ext(CONTROL_DIM, 0.0f);
    std::vector<casadi_real> h_sim_x_next_all(static_cast<size_t>(N) * OUT_DIM, 0.0f);

    const size_t expected_count = static_cast<size_t>(N) * PARAM_DIM;
    if (!load_posterior(POSTERIOR_BIN_PATH, h_sim_p_all, expected_count)) {
        return 1;
    }

    // Populate a simple pattern
    for (int i = 0; i < STATE_DIM; ++i) h_sim_x[i] = 0.1f * (i + 1);
    for (int i = 0; i < CONTROL_DIM; ++i) {
        h_sim_u[i] = 0.05f * (i + 1);
        h_f_ext[i] = 0.0f;
    }

    const casadi_real dt = 0.04f;

    // Device pointers
    casadi_real *d_sim_x = nullptr;
    casadi_real *d_sim_u = nullptr;
    casadi_real *d_sim_p_all = nullptr;
    casadi_real *d_dt = nullptr;
    casadi_real *d_f_ext = nullptr;
    casadi_real *d_sim_x_next_all = nullptr;

    die_cuda(cudaMalloc(&d_sim_x, STATE_DIM * sizeof(casadi_real)), "cudaMalloc d_sim_x");
    die_cuda(cudaMalloc(&d_sim_u, CONTROL_DIM * sizeof(casadi_real)), "cudaMalloc d_sim_u");
    die_cuda(cudaMalloc(&d_sim_p_all, static_cast<size_t>(N) * PARAM_DIM * sizeof(casadi_real)), "cudaMalloc d_sim_p_all");
    die_cuda(cudaMalloc(&d_dt, sizeof(casadi_real)), "cudaMalloc d_dt");
    die_cuda(cudaMalloc(&d_f_ext, CONTROL_DIM * sizeof(casadi_real)), "cudaMalloc d_f_ext");
    die_cuda(cudaMalloc(&d_sim_x_next_all, static_cast<size_t>(N) * OUT_DIM * sizeof(casadi_real)), "cudaMalloc d_sim_x_next_all");

    // Copy to device
    die_cuda(cudaMemcpy(d_sim_x, h_sim_x.data(), STATE_DIM * sizeof(casadi_real), cudaMemcpyHostToDevice), "cudaMemcpy sim_x");
    die_cuda(cudaMemcpy(d_sim_u, h_sim_u.data(), CONTROL_DIM * sizeof(casadi_real), cudaMemcpyHostToDevice), "cudaMemcpy sim_u");
    die_cuda(cudaMemcpy(d_sim_p_all, h_sim_p_all.data(),
                        static_cast<size_t>(N) * PARAM_DIM * sizeof(casadi_real),
                        cudaMemcpyHostToDevice),
            "cudaMemcpy sim_p_all");
    die_cuda(cudaMemcpy(d_dt, &dt, sizeof(casadi_real), cudaMemcpyHostToDevice), "cudaMemcpy dt");
    die_cuda(cudaMemcpy(d_f_ext, h_f_ext.data(), CONTROL_DIM * sizeof(casadi_real), cudaMemcpyHostToDevice), "cudaMemcpy f_ext");

    // Launch config
    const int threads_per_block = 128;
    const int blocks = (N + threads_per_block - 1) / threads_per_block;

    // Warmup
    const int warmup = 20;
    for (int i = 0; i < warmup; ++i) {
        Vnext_reg_kernel<<<blocks, threads_per_block>>>(
            d_sim_x, d_sim_u, d_sim_p_all, d_dt, d_f_ext, d_sim_x_next_all, N
        );
    }
    die_cuda(cudaGetLastError(), "Kernel launch (warmup)");
    die_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize (warmup)");

    // Timed runs (kernel only)
    const int reps = 200;
    cudaEvent_t start, stop;
    die_cuda(cudaEventCreate(&start), "cudaEventCreate start");
    die_cuda(cudaEventCreate(&stop), "cudaEventCreate stop");

    die_cuda(cudaEventRecord(start), "cudaEventRecord start");
    for (int i = 0; i < reps; ++i) {
        Vnext_reg_kernel<<<blocks, threads_per_block>>>(
            d_sim_x, d_sim_u, d_sim_p_all, d_dt, d_f_ext, d_sim_x_next_all, N
        );
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

    std::cout << "\nDynamics kernel timing\n";
    std::cout << "  GPU batch N=" << N << ", threads_per_block=" << threads_per_block
              << ", blocks=" << blocks << "\n";
    std::cout << "  Total: " << ms_total << " ms for " << reps << " launches\n";
    std::cout << "  Per launch: " << ms_per << " ms\n";
    std::cout << "  Throughput: " << evals_per_s << " eval/s\n\n";

    // Copy results back (not included in kernel timing)
    die_cuda(cudaMemcpy(h_sim_x_next_all.data(), d_sim_x_next_all,
                        static_cast<size_t>(N) * OUT_DIM * sizeof(casadi_real),
                        cudaMemcpyDeviceToHost),
            "cudaMemcpy sim_x_next_all");

    // Print a few candidates only
    const int to_print = 3;
    for (int i = 0; i < to_print; ++i) {
        std::cout << "Candidate " << i << " next state: ";
        for (int j = 0; j < OUT_DIM; ++j) {
            std::cout << h_sim_x_next_all[static_cast<size_t>(i) * OUT_DIM + j]
                      << (j + 1 == OUT_DIM ? "" : " ");
        }
        std::cout << "\n";
    }

    // Cleanup
    cudaFree(d_sim_x);
    cudaFree(d_sim_u);
    cudaFree(d_sim_p_all);
    cudaFree(d_dt);
    cudaFree(d_f_ext);
    cudaFree(d_sim_x_next_all);

    return 0;
}
