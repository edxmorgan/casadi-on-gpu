# **casadi-on-gpu**

<p align="center">
  <img src="demo.gif" alt="80000 evaluations of forward kinematics" width="600">
</p>

A minimal example showing how to take a CasADi generated C function, patch it for CUDA, and run it directly on the GPU. This enables extremely fast parallel evaluation of forward kinematics, dynamics, or any other CasADi symbolic function on thousands of candidates at once.

This is **not** a library, but a clean template you can copy and adapt whenever you want CasADi functions to run inside CUDA kernels.

---

## **Workflow Overview**

### **1. Create CasADi Function in Python**

Generate your CasADi function as usual:

```python
fk = ca.Function("fkeval", [q, params1, params2], [output])
fk.save("fk_eval.casadi")
```

---

### **2. Generate C Code**

```python
cg = ca.CodeGenerator("fk_alpha", {"with_header": True, "casadi_real": "float"})
cg.add(fk)
cg.generate("src/")
```

This produces:

* `fk_alpha.h`
* `fk_alpha.c`

---

## **3. Patch the Code for CUDA**

### **Header (`fk_alpha.h`) → rename to `fk_alpha.cuh`**

Add safe CUDA qualifiers so the header works on both host and device:

```c
#ifndef __CUDACC__
#define __device__
#endif

__device__ int fkeval(const casadi_real** arg, casadi_real** res, ...);
```

### **Source (`fk_alpha.c`) → rename to `fk_alpha.cu`**

Mark all functions called inside the CasADi kernel as device code:

```c
__device__ casadi_real casadi_sq(casadi_real x) { return x * x; }

static __device__
int casadi_f0(...) { ... }

__device__
int fkeval(...) { return casadi_f0(...); }
```

Any helper functions used by `casadi_f0()` must also be tagged `__device__`.
Everything else can remain CPU only.

---

## **4. Device Wrapper**

`device_fk_eval.cuh`:

```cpp
__device__ void device_fk_eval(
    const casadi_real* q,
    const casadi_real* p1,
    const casadi_real* p2,
    casadi_real* out
) {
    const casadi_real* arg_local[3] = { q, p1, p2 };
    casadi_real* res_local[1]       = { out };

    casadi_int  iw[1];
    casadi_real w[1];

    fkeval(arg_local, res_local, iw, w, 0);
}
```

---

## **5. Evaluate Thousands in Parallel**

Example kernel:

```cpp
fk_kernel<<<blocks, threads>>>(d_q_all, d_p1, d_p2, d_out_all, N);
cudaDeviceSynchronize();
```

Each GPU thread processes one candidate configuration.

---

## **Project Structure**

```
casadi-on-gpu/
│
├── src/
│   ├── fk_alpha.cu          # CasADi-generated and CUDA patched
│   ├── fk_alpha.h
│   ├── device_fk_eval.cuh   # Device wrapper
│   ├── main.cu              # Example CUDA usage
│
└── CMakeLists.txt
```

---

## **Build and Run**

```bash
mkdir build
cd build
cmake ..
make -j8
./run_casadi_gpu
```

---

## **Why Use This**

* CasADi C code is fast but limited to the CPU.
* Robotics and control algorithms often require huge batches of FK or dynamics evaluations.
* CUDA makes these workloads massively parallel.
* The method here lets you drop CasADi functions straight onto the GPU with minimal changes.