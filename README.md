# **casadi-on-gpu**

<p align="center">
  <img src="demo.gif" alt="80000 evaluations of forward kinematics" width="6000">
</p>

A minimal example that shows how to take CasADi generated C code, patch it so it can run inside CUDA kernels, and evaluate it directly on the GPU. The demo above runs forward kinematics for **N = 80000** different joint configurations in parallel within a few milliseconds.

This approach makes it possible to run forward kinematics, dynamics, or any other CasADi symbolic function on thousands of samples at very high speed.

This is not a library. It is a small template that you can copy when you want CasADi functions to run on the GPU.

---


## **Workflow Overview**

### **1. Create a CasADi Function in Python**

```python
fk = ca.Function("fkeval", [q, params1, params2], [output])
fk.save("fk_eval.casadi")
```

---

### **2. Generate C Code**

```python
cg = ca.CodeGenerator(
    "fk_alpha",
    {"with_header": True, "casadi_real": "float"}
)
cg.add(fk)
cg.generate("src/")
```

This creates `fk_alpha.h` and `fk_alpha.c`.

---

## **3. Patch the Code for CUDA**

### **Header (`fk_alpha.h`) → rename to `fk_alpha.cuh`**

Add safe CUDA qualifiers so the header works on both host and device.

```c
#ifndef __CUDACC__
#define __device__
#endif

__device__ int fkeval(const casadi_real** arg, casadi_real** res, ...);
```

### **Source (`fk_alpha.c`) → rename to `fk_alpha.cu`**

Mark all functions that run on the GPU.

```c
__device__ casadi_real casadi_sq(casadi_real x) { return x * x; }

static __device__
int casadi_f0(...) { ... }

__device__
int fkeval(...) { return casadi_f0(...); }
```

Any helper called inside `casadi_f0` must also be tagged `__device__`.

---

## **4. Device Wrapper**

`device_fk_eval.cuh`

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

### What are `iw`, `w`, and `mem`

CasADi functions always receive:

```c
int fkeval(const casadi_real** arg,
           casadi_real** res,
           casadi_int* iw,
           casadi_real* w,
           int mem);
```

* `arg` and `res` are arrays of pointers to inputs and outputs
* `iw` and `w` are small scratch workspaces CasADi may use internally
* The sizes of these arrays are provided in the generated header
* For this FK example they are both zero, so we pass small dummy arrays
* `mem` is a memory slot index used when CasADi maintains internal state
  In this FK example it does nothing, so `0` is fine

If your function has non zero workspace sizes, allocate arrays of the required sizes inside the wrapper.

---

## **5. Evaluate Many Samples in Parallel**

```cpp
fk_kernel<<<blocks, threads>>>(
    d_q_all, d_p1, d_p2, d_out_all, N
);
cudaDeviceSynchronize();
```

Each GPU thread evaluates one forward kinematics call.

---

## **Project Structure**

```
casadi-on-gpu/
│
├── src/
│   ├── fk_alpha.cu          CUDA patched CasADi code
│   ├── fk_alpha.cuh
│   ├── device_fk_eval.cuh   Device wrapper
│   ├── main.cu              Example usage
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