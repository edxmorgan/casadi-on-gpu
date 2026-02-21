// Author: edward morgan

#include "casadi_on_gpu_api.h"

#include <stdexcept>
#include <string>
#include <utility>

#include <cuda_runtime.h>

#include "casadi_on_gpu_kernel_registry.h"

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

const KernelEntry& find_kernel(const std::string& function_name) {
    std::size_t n = 0;
    const KernelEntry* entries = get_kernel_registry(&n);
    for (std::size_t i = 0; i < n; ++i) {
        if (function_name == entries[i].function_name) {
            return entries[i];
        }
    }
    throw std::invalid_argument("Unknown kernel function: " + function_name);
}

}  // namespace

void launch(const std::string& function_name,
            const std::vector<std::uintptr_t>& input_ptrs,
            const std::vector<std::uintptr_t>& output_ptrs,
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

    const KernelEntry& entry = find_kernel(function_name);
    if (static_cast<int>(input_ptrs.size()) != entry.n_in) {
        throw std::invalid_argument(
            "input_ptrs size mismatch for " + function_name +
            " (got " + std::to_string(input_ptrs.size()) +
            ", expected " + std::to_string(entry.n_in) + ")");
    }
    if (static_cast<int>(output_ptrs.size()) != entry.n_out) {
        throw std::invalid_argument(
            "output_ptrs size mismatch for " + function_name +
            " (got " + std::to_string(output_ptrs.size()) +
            ", expected " + std::to_string(entry.n_out) + ")");
    }

    cudaStream_t stream = stream_from_ptr(stream_ptr);
    const int blocks = (n_candidates + threads_per_block - 1) / threads_per_block;

    entry.launch(input_ptrs.data(), output_ptrs.data(), blocks, threads_per_block, stream, n_candidates);
    throw_on_cuda_error(cudaGetLastError(), "kernel launch failed");

    if (sync) {
        throw_on_cuda_error(cudaStreamSynchronize(stream), "kernel sync failed");
    }
}

std::vector<KernelMetadata> list_kernels() {
    std::size_t n = 0;
    const KernelEntry* entries = get_kernel_registry(&n);

    std::vector<KernelMetadata> out;
    out.reserve(n);
    for (std::size_t i = 0; i < n; ++i) {
        KernelMetadata m;
        m.function_name = entries[i].function_name;
        m.kernel_name = entries[i].kernel_name;
        m.batch_inputs = entries[i].batch_inputs;
        m.input_nnz = entries[i].input_nnz;
        m.output_nnz = entries[i].output_nnz;
        out.push_back(std::move(m));
    }

    return out;
}

void device_synchronize() {
    throw_on_cuda_error(cudaDeviceSynchronize(), "cudaDeviceSynchronize failed");
}

}  // namespace casadi_on_gpu
