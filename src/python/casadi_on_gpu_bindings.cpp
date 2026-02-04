#include <pybind11/pybind11.h>

#include "casadi_on_gpu_api.h"

namespace py = pybind11;

PYBIND11_MODULE(casadi_on_gpu, m) {
    m.doc() = "CasADi GPU kernels (FK + Dynamics) exposed via pybind11";

    m.attr("FK_DOF") = 4;
    m.attr("FK_OUT_DIM") = 6;
    m.attr("DYNAMICS_STATE_DIM") = 12;
    m.attr("DYNAMICS_CONTROL_DIM") = 6;
    m.attr("DYNAMICS_PARAM_DIM") = 33;
    m.attr("DYNAMICS_OUT_DIM") = 12;

    m.def("fk_forward",
          &casadi_on_gpu::fk_forward,
          py::arg("q_all_ptr"),
          py::arg("p1_ptr"),
          py::arg("p2_ptr"),
          py::arg("out_ptr"),
          py::arg("n_candidates"),
          py::arg("threads_per_block") = 128,
          py::arg("stream_ptr") = 0,
          py::arg("sync") = true,
          "Launch FK kernel. Pointers must be GPU addresses.");

    m.def("dynamics_forward",
          &casadi_on_gpu::dynamics_forward,
          py::arg("sim_x_ptr"),
          py::arg("sim_u_ptr"),
          py::arg("sim_p_all_ptr"),
          py::arg("dt"),
          py::arg("f_ext_ptr"),
          py::arg("sim_x_next_all_ptr"),
          py::arg("n_candidates"),
          py::arg("threads_per_block") = 128,
          py::arg("stream_ptr") = 0,
          py::arg("sync") = true,
          "Launch dynamics kernel. Pointers must be GPU addresses.");

    m.def("device_synchronize",
          &casadi_on_gpu::device_synchronize,
          "Synchronize the current CUDA device.");
}
