# **casadi on gpu**

<p align="center">
  <img src="header.png" alt="parallel computing with casadi on gpu" width="1500">
</p>

---

This project began with manual CUDA patching; the CUDA codegen effort is now being merged into CasADi to emit `.cu/.cuh` and evaluate functions directly on the GPU. This repo provides:
- generated CUDA kernels for FK and dynamics in `src/generated/`
- C++ demo launchers in `demos/` (compiled as CUDA even though they are `.cpp`)
- a local pybind11 module for PyTorch/CuPy GPU tensors

To demonstrate this, the repo contains two example models:

1) The demo below evaluates **80k** samples of a forward kinematics model for a 4 degree of freedom manipulator in under **three milliseconds**. 
<p align="center">
  <img src="demo.gif" alt="80000 evaluations of forward kinematics" width="3000">
</p>


2) A demo of a robot dynamic model with 33 parameters is also included where posteriors of the parameters are sampled from `src/posterior.bin` and batch evaluated for stochastic forward dynamics.



## **Prereq: CasADi CUDA codegen branch**

This repo expects CasADi built from the CUDA codegen branch (to be merged to main soon):

```bash
git clone https://github.com/edxmorgan/casadi.git
cd casadi
git checkout cuda_codegen
mkdir -p build && cd build
cmake -DWITH_PYTHON=ON -DCMAKE_INSTALL_PREFIX=$HOME/.local ..
make -j8
cmake --install .
```

## **Generate kernels (only if you change the models)**

The codegen notebooks live in `codegen/` and write `.cu/.cuh` into `src/generated/`.
Open and run:
- `codegen/generate_fk_cuda.ipynb`
- `codegen/generate_dynamics_cuda.ipynb`

## **Build casadi-on-gpu**

```bash
cd casadi-on-gpu
mkdir -p build
cd build
cmake -DBUILD_PYTHON=ON -DCMAKE_INSTALL_PREFIX=$HOME/.local ..
make -j8
cmake --install .

# Kinematics demo
./run_kinematics_gpu

# Dynamics demo (expects src/posterior.bin)
./run_dynamics_gpu
```

### Usage (C++)

The C++ demos in `demos/` include the generated headers and launch the generated kernels:
- `demos/kinematics_main.cpp` launches `fkeval_kernel`
- `demos/dynamics_main.cpp` launches `Vnext_reg_kernel`

They are `.cpp` files but compiled as CUDA by CMake (so the `<<< >>>` kernel syntax works).


### Usage (PyTorch)

```python
import torch
import casadi_on_gpu as cog

N = 1024
q_all = torch.zeros((N, cog.FK_DOF), device="cuda", dtype=torch.float32)
p1 = torch.zeros((6,), device="cuda", dtype=torch.float32)
p2 = torch.zeros((6,), device="cuda", dtype=torch.float32)
out = torch.zeros((N, cog.FK_OUT_DIM), device="cuda", dtype=torch.float32)

stream = torch.cuda.current_stream().cuda_stream
cog.fk_forward(q_all.data_ptr(), p1.data_ptr(), p2.data_ptr(), out.data_ptr(),
               N, stream_ptr=stream, sync=False)
```

### Usage (CuPy)

```python
import cupy as cp
import casadi_on_gpu as cog

N = 1024
q_all = cp.zeros((N, cog.FK_DOF), dtype=cp.float32)
p1 = cp.zeros((6,), dtype=cp.float32)
p2 = cp.zeros((6,), dtype=cp.float32)
out = cp.zeros((N, cog.FK_OUT_DIM), dtype=cp.float32)

stream = cp.cuda.get_current_stream().ptr
cog.fk_forward(q_all.data.ptr, p1.data.ptr, p2.data.ptr, out.data.ptr,
               N, stream_ptr=stream, sync=False)
```

Notes:
- Inputs and outputs must be `float32` on the GPU.
- Use `sync=True` if you are not coordinating with your own CUDA streams.
