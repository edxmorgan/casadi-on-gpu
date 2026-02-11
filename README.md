# **casadi on gpu**

<p align="center">
  <img src="header.png" alt="parallel computing with casadi on gpu" width="1500">
</p>

---

This project began as a guide on manually cuda-patching c-generated code from CasADi codegen. Efforts to automate the process have been made and in the pipeline to be merged into CasADi [https://github.com/casadi/casadi/pull/4291]. This allows casadi code to be natively converted to `.cu/.cuh` and evaluated directly on a GPU. Now, this repo is being repurposed to demonstrate `cpp`, `pytorch` and `cupy` interfacing with the generated `.cu/.cuh` enabling `cpp` and `python` batch parallelization of functions.


To demonstrate this, the repo contains two example models:

1) A demo that evaluates **80k** samples of a forward kinematics model for a 4 degree of freedom manipulator in under **three milliseconds** both in `cpp` and `python`. 
<p align="center">
  <img src="demo.gif" alt="80000 evaluations of forward kinematics" width="3000">
</p>


2) Another demo of a robot dynamic model with 33 parameters is also included where posteriors of the parameters are sampled from `src/posterior.bin` and batch evaluated for stochastic forward dynamics.



## **Prereq: CasADi CUDA codegen branch**

```bash
git clone https://github.com/edxmorgan/casadi.git
cd casadi
git checkout cuda_codegen
mkdir -p build && cd build
cmake -DWITH_PYTHON=ON -DCMAKE_INSTALL_PREFIX=$HOME/.local ..
make -j8
cmake --install .
```

## **Generate kernels**
```python
fk_eval = ca.Function(fname, [..], [..])
cg = ca.CodeGenerator("fk_alpha", {
    "with_header": True,
    "casadi_real": "float",
    "cpp": False,
    "cuda": True,
    "cuda_kernels": {
        fname: {
            "batch_inputs": [0],
        }
    },
})
cg.add(fk_eval)
out_path = cg.generate(str(codegen_folder) + "/")
```

Example script can be found in the `codegen/` notebooks. The scripts generate `.cu/.cuh` into `src/generated/`.
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

ctest -V -R casadi_on_gpu_py_smoke
```

### Usage (C++)

The C++ demos in `demos/` include the example interfacing script :
- `demos/kinematics_main.cpp` launches `fkeval_kernel`
- `demos/dynamics_main.cpp` launches `Vnext_reg_kernel`

They are `.cpp` files but compiled as CUDA by CMake (so the `<<< >>>` kernel syntax works).
```bash
cd casadi-on-gpu
# Kinematics demo
./run_kinematics_gpu

# Dynamics demo
./run_dynamics_gpu
```

### Usage (PyTorch)

```python
import torch, casadi_on_gpu as cog

N = 80000
q_all = torch.zeros((N, cog.FK_DOF), device="cuda", dtype=torch.float32)
p1 = torch.zeros((6,), device="cuda", dtype=torch.float32)
p2 = torch.zeros((6,), device="cuda", dtype=torch.float32)
out = torch.zeros((N, cog.FK_OUT_DIM), device="cuda", dtype=torch.float32)
stream = torch.cuda.current_stream().cuda_stream
cog.fk_forward(q_all.data_ptr(), p1.data_ptr(), p2.data_ptr(), out.data_ptr(),
                N, stream_ptr=stream, sync=True)

out

```

### Usage (CuPy)

```python
import cupy as cp
import casadi_on_gpu as cog

N = 80000
q_all = cp.zeros((N, cog.FK_DOF), dtype=cp.float32)
p1 = cp.zeros((6,), dtype=cp.float32)
p2 = cp.zeros((6,), dtype=cp.float32)
out = cp.zeros((N, cog.FK_OUT_DIM), dtype=cp.float32)

stream = cp.cuda.get_current_stream().ptr
cog.fk_forward(q_all.data.ptr, p1.data.ptr, p2.data.ptr, out.data.ptr,
               N, stream_ptr=stream, sync=False)
out
```

Notes:
- Inputs and outputs must be `float32` on the GPU.
- Use `sync=True` if you are not coordinating with your own CUDA streams.


## Performance & Benchmarks

This project targets high-throughput batched evaluation of CasADi-generated CUDA kernels. Benchmarks below reflect realistic large-batch workloads and were measured using CUDA event timing with warmup and repeated launches.

---

## Benchmark Environment

All measurements were collected on:

**Hardware**

- NVIDIA GeForce RTX 5090 Laptop GPU  
- 24 GB VRAM  

**Software**

- NVIDIA driver: 580.105.08  
- CUDA runtime: 13.0  

**Methodology**

- Batched evaluation with N = 80,000 samples  
- Warmup runs performed before timing  
- CUDA events used for accurate GPU timing  
- Hundreds of repeated launches averaged  
- Asynchronous stream execution  

Performance will vary with GPU architecture, clock behavior, thermals, and batch size.

---

## Forward kinematics kernel  
4-DOF manipulator model

Batch size: **N = 80,000**

| Interface | Time / batch | Throughput |
|----------|--------------|------------|
| C++      | 0.00820 ms   | 9.7568 × 10⁹ eval/s |
| PyTorch  | 0.00830 ms   | 9.6766 × 10⁹ eval/s |
| CuPy     | 0.00830 ms   | 9.6922 × 10⁹ eval/s |

This kernel is extremely lightweight, so runtime is dominated by kernel launch overhead. Throughput increases further with larger batches.

---

## Forward dynamics kernel  
12-state, 33-parameter stochastic model

Batch size: **N = 80,000**

| Interface | Time / batch | Throughput |
|----------|--------------|------------|
| C++      | 6.6391 ms    | 1.2050 × 10⁷ eval/s |
| PyTorch  | 6.6731 ms    | 1.1988 × 10⁷ eval/s |
| CuPy     | 6.6523 ms    | 1.2026 × 10⁷ eval/s |

This kernel is compute-heavy and represents realistic stochastic forward dynamics workloads.

---

## Observations

- GPU acceleration is most effective for **large batched evaluations**
- PyTorch and CuPy wrappers introduce **negligible overhead**
- Performance is close to raw CUDA launch speed
- Lightweight kernels approach launch-bound limits
- Heavy kernels scale with compute intensity
- Throughput improves with larger batch sizes

---

## Practical implications

These results demonstrate that CasADi-generated CUDA kernels can support:

- Massive batched simulation  
- Stochastic parameter sampling  
- Real-time trajectory rollout  
- Monte Carlo dynamics evaluation  
- Differentiable robotics pipelines  

Performance scales with GPU architecture and problem size, making this approach well suited for large-scale robotics and control workloads.
