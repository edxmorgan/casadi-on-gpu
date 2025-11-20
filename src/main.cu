#include <cstdio>
#include <cuda_runtime.h>
#include "fk_wrapper.cuh"

int main() {
    casadi_real h_q[4]  = {0, 0, 0, 0};
    casadi_real h_p1[6] = {0, 0, 0, 0, 0, 0};
    casadi_real h_p2[6] = {0, 0, 0, 0, 0, 0};
    casadi_real h_out[6];

    casadi_real *d_q, *d_p1, *d_p2, *d_out;
    cudaMalloc(&d_q,   4 * sizeof(casadi_real));
    cudaMalloc(&d_p1,  6 * sizeof(casadi_real));
    cudaMalloc(&d_p2,  6 * sizeof(casadi_real));
    cudaMalloc(&d_out, 6 * sizeof(casadi_real));

    cudaMemcpy(d_q,  h_q,  4 * sizeof(casadi_real), cudaMemcpyHostToDevice);
    cudaMemcpy(d_p1, h_p1, 6 * sizeof(casadi_real), cudaMemcpyHostToDevice);
    cudaMemcpy(d_p2, h_p2, 6 * sizeof(casadi_real), cudaMemcpyHostToDevice);

    fk_kernel<<<1, 1>>>(d_q, d_p1, d_p2, d_out);
    cudaDeviceSynchronize();

    cudaMemcpy(h_out, d_out, 6 * sizeof(casadi_real), cudaMemcpyDeviceToHost);

    printf("FK output:\n");
    for (int i = 0; i < 6; ++i) {
        printf("  out[%d] = %f\n", i, h_out[i]);
    }

    cudaFree(d_q);
    cudaFree(d_p1);
    cudaFree(d_p2);
    cudaFree(d_out);

    return 0;
}
