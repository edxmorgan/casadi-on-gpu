# **casadi on gpu**

<p align="center">
  <img src="demo.gif" alt="80000 evaluations of forward kinematics" width="3000">
</p>

This project shows how to take CasADi generated C code, make a small patch so it can run inside CUDA kernels, and evaluate it directly on the GPU. The demo above evaluates a 4 DOF forward kinematics model for **N = 80000** joint configurations in parallel in under **three milliseconds**.

This is not a library. It is a minimal template you can copy whenever you want CasADi functions to run on the GPU.

---

## **Workflow Overview**

### **1. Create a CasADi Function in Python**

```python
fk = ca.Function("fkeval", [q, params1, params2], [output])
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

Add a safe qualifier so the header works both in CUDA and regular C files.

```c
#ifndef __CUDACC__
#define __device__
#endif
```

Declare the function with a device qualifier.

```c
__device__ int fkeval(const casadi_real** arg, casadi_real** res, ...);
```

---

### **Source (`fk_alpha.c`) → rename to `fk_alpha.cu`**

Any function that will be executed on the GPU must have a device qualifier.

```c
__device__ casadi_real casadi_sq(casadi_real x) { return x * x; }

static __device__
int casadi_f0(...) { ... }

__device__
int fkeval(...) { return casadi_f0(...); }
```

Any functions defined in the generated file that are called inside `casadi_f0()` must be marked `__device__` so they can run on the GPU. Built in math functions like sin, cos, or sqrt already have device versions, so they do not need modification. Everything else that is not executed on the GPU can remain unchanged.

---

## **4. Device Wrapper**

`device_fk_eval.cuh`

```cpp
__device__ void device_fk_eval(
    const casadi_real* q,        // i0[4]
    const casadi_real* params1,  // i1[6]
    const casadi_real* params2,  // i2[6]
    casadi_real* out             // o0[6]
)
{
    const casadi_real* arg_local[3] = { q, params1, params2 };
    const casadi_real** arg = arg_local;

    casadi_real* res_local[1] = { out };
    casadi_real** res = res_local;

    casadi_int  iw[fkeval_SZ_IW > 0 ? fkeval_SZ_IW : 1];
    casadi_real w [fkeval_SZ_W  > 0 ? fkeval_SZ_W  : 1];

    fkeval(arg, res, iw, w, 0);
}
```

### What are `iw`, `w`, and `mem`

CasADi generated functions always follow the signature:

```c
int fkeval(const casadi_real** arg,
           casadi_real** res,
           casadi_int* iw,
           casadi_real* w,
           int mem);
```

`arg` and `res` are arrays of pointers to inputs and outputs
`iw` and `w` are small scratch workspaces CasADi may use internally
The sizes of these arrays are provided in the generated header
For this FK example they are both zero, so we pass small dummy arrays
`mem` is a memory slot index used when CasADi maintains internal state In this FK example it does nothing, so 0 is fine
If your function has non zero workspace sizes, allocate arrays of the required sizes inside the wrapper.
---

## **5. Evaluate Many Samples in Parallel**

This kernel assigns one GPU thread to each FK computation.

```cpp
__global__ void fk_kernel(
    const casadi_real* q_all,    // shape [N, 4]
    const casadi_real* p1,
    const casadi_real* p2,
    casadi_real* out_all,        // shape [N, 6]
    int n_candidates
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_candidates) return;

    const int DOF = 4;
    const int OUT_DIM = 6;

    const casadi_real* q_i  = q_all   + DOF     * idx;
    casadi_real*       out_i = out_all + OUT_DIM * idx;

    device_fk_eval(q_i, p1, p2, out_i);
}
```

Launch the kernel:

```cpp
fk_kernel<<<blocks, threads>>>(
    d_q_all,
    d_p1,
    d_p2,
    d_out_all,
    N
);

cudaDeviceSynchronize();
```

Each thread performs one forward kinematics call.
This is what gives the large speedup.

---

## **Project Structure**

```
casadi on gpu/
│
├── src/
│   ├── fk_alpha.cu          CUDA patched CasADi code
│   ├── fk_alpha.cuh
│   ├── device_fk_eval.cuh   Device wrapper
│   ├── main.cu              Example program
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