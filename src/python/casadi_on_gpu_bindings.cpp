// Author: edward morgan

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "casadi_on_gpu_api.h"

namespace py = pybind11;

namespace {

py::list as_python_kernel_list(const std::vector<casadi_on_gpu::KernelMetadata>& kernels) {
    py::list out;
    for (const auto& k : kernels) {
        py::dict d;
        d["function_name"] = k.function_name;
        d["kernel_name"] = k.kernel_name;
        d["batch_inputs"] = k.batch_inputs;
        d["input_nnz"] = k.input_nnz;
        d["output_nnz"] = k.output_nnz;
        out.append(d);
    }
    return out;
}

}  // namespace

PYBIND11_MODULE(casadi_on_gpu, m) {
    m.doc() = "Manifest-driven CasADi CUDA kernel launcher (PyTorch/CuPy pointer API)";

    m.attr("MANIFEST_FILENAME") = "kernels_manifest.json";

    m.def("launch",
          &casadi_on_gpu::launch,
          py::arg("function_name"),
          py::arg("input_ptrs"),
          py::arg("output_ptrs"),
          py::arg("n_candidates"),
          py::arg("threads_per_block") = 128,
          py::arg("stream_ptr") = 0,
          py::arg("sync") = true,
          "Launch a registered CasADi CUDA kernel by function name.");

    m.def("list_kernels",
          []() { return as_python_kernel_list(casadi_on_gpu::list_kernels()); },
          "Return metadata for all registered kernels.");

    m.def("device_synchronize",
          &casadi_on_gpu::device_synchronize,
          "Synchronize the current CUDA device.");
}
