# **casadi on gpu**

<p align="center">
  <img src="header.png" alt="parallel computing with casadi on gpu" width="1500">
</p>

---

This library lets you run CasADi functions on NVIDIA GPUs from Python in large batches at high speed.

You provide CasADi `.casadi` function files, generate CUDA kernel files (`.cu/.cuh`), and then call those kernels directly from `PyTorch` or `CuPy` using GPU pointers. The package provides a single Python module (`casadi_on_gpu`) that exposes kernel discovery and launch APIs, so large batched evaluations can be executed on GPU with very little glue code.

The repository includes two example assets:

1) `fk_eval.casadi`: forward kinematics for a 4-DoF manipulator (high-throughput batched evaluation).
2) `dynamics_eval.casadi`: stochastic forward dynamics model with sample parameters in `examples/assets/posterior.bin`.



## **Prereq: Build CasADi (with CUDA codegen support)**

The flags below are for building the upstream `casadi` repository, not this `casadi-on-gpu` repository.

```bash
git clone -b cuda_codegen https://github.com/edxmorgan/casadi.git casadi
cd casadi
mkdir -p build
cd build
cmake -DWITH_PYTHON=ON -DWITH_PYTHON3=ON ..
make -j8
sudo make install
```

## **Generate kernels**
Example `.casadi` assets are stored in `examples/assets/`. Generated `.cu/.cuh` files are written to `src/generated/`.

After generating kernels, create runtime metadata + dispatch registry:
```bash
./tools/generate_manifest_and_registry.py
```
This emits:
- `src/generated/kernels_manifest.json`
- `src/python/casadi_on_gpu_kernel_registry.cu`

To add custom kernels with a simple CLI:
```bash
./tools/generate_manifest_and_registry.py \
  --casadi examples/assets/fk_eval.casadi \
  --casadi examples/assets/dynamics_eval.casadi \
  --batch-override examples/assets/fk_eval.casadi=0 \
  --batch-override examples/assets/dynamics_eval.casadi=2
```
Most common single-kernel case:
```bash
./tools/generate_manifest_and_registry.py --casadi path/to/my_model.casadi --batch-inputs 0,2
```

Advanced mode (full control) is still available:
```bash
./tools/generate_manifest_and_registry.py \
  --entry path/to/my_model.casadi:my_model_unit:0,2[:kernel_name[:device_name]]
```

## **Build casadi-on-gpu**

Use `BUILD_PYTHON` for this repository.
Do not use `WITH_PYTHON` / `WITH_PYTHON3` here.

```bash
cd casadi-on-gpu
mkdir -p build
cd build
cmake -DBUILD_PYTHON=ON -DCMAKE_INSTALL_PREFIX=$HOME/.local ..
cmake --build . -j
cmake --install .

ctest -V -R casadi_on_gpu_py_smoke
```

### Build as pip package

```bash
cd casadi-on-gpu
python3 -m pip install .
```

### Usage (PyTorch)

```python
import torch, casadi_on_gpu as cog

print(cog.list_kernels())

N = 80000
q_all = torch.zeros((N, 4), device="cuda", dtype=torch.float32)
p1 = torch.zeros((6,), device="cuda", dtype=torch.float32)
p2 = torch.zeros((6,), device="cuda", dtype=torch.float32)
out = torch.zeros((N, 6), device="cuda", dtype=torch.float32)
stream = torch.cuda.current_stream().cuda_stream
cog.launch(
    "fkeval",
    [q_all.data_ptr(), p1.data_ptr(), p2.data_ptr()],
    [out.data_ptr()],
    N,
    stream_ptr=stream,
    sync=True,
)

out

```

### Usage (CuPy)

```python
import cupy as cp
import casadi_on_gpu as cog

N = 80000
q_all = cp.zeros((N, 4), dtype=cp.float32)
p1 = cp.zeros((6,), dtype=cp.float32)
p2 = cp.zeros((6,), dtype=cp.float32)
out = cp.zeros((N, 6), dtype=cp.float32)

stream = cp.cuda.get_current_stream().ptr
cog.launch(
    "fkeval",
    [q_all.data.ptr, p1.data.ptr, p2.data.ptr],
    [out.data.ptr],
    N,
    stream_ptr=stream,
    sync=False,
)
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
