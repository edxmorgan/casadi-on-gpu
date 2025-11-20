# **casadi-on-gpu**

<p align="center">
  <img src="demo.gif" alt="80000 evaluations of forward kinematics" width="600">
</p>

This project shows how to take a symbolic CasADi function, generate C code, adapt it for CUDA, and run it directly on the GPU using `__device__` kernels. This makes heavy forward kinematics, dynamics, or optimization workloads much faster, especially when evaluating many candidates in parallel.

It shows how to patch the codegen output so it becomes CUDA compatible, how to wrap the function for device side execution, and how to launch batched evaluations directly on the GPU. The goal is not to provide a full library, but to give a clean minimal example that users can copy, adapt, and extend when they need high speed casadi function evaluations running in parallel on the GPU.

---


## **Workflow Overview**

### **1. Create CasADi Function in Python**

Generate your casadi function as usual:

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

### **3. Make the Code CUDA Friendly**

#### **Header (`fk_alpha.h`)**

Add safe CUDA qualifiers:

```c
#ifndef __CUDACC__
#define __device__
#endif

__device__ int fkeval(const casadi_real** arg, casadi_real** res, ...);
```

#### **Source (`fk_alpha.c`)**

Rename to `fk_alpha.cu` and tag key functions:

```c
__device__ casadi_real casadi_sq(casadi_real x) { return x*x; }

static __device__
int casadi_f0(...) { ... }

__device__
int fkeval(...) { return casadi_f0(...); }
```

All other functions non standard function used in casadi_f0 should be __device__ qualified. Everything else can remain on CPU only.

---

## **4. Write a CUDA Wrapper**

`device_fk_eval.cuh`:

```c++
__device__ void device_fk_eval(
    const casadi_real* q,
    const casadi_real* p1,
    const casadi_real* p2,
    casadi_real* out
) {
    const casadi_real* arg_local[3] = { q, p1, p2 };
    casadi_real* res_local[1] = { out };

    casadi_int iw[1];
    casadi_real w[1];

    fkeval(arg_local, res_local, iw, w, 0);
}
```

---

## **5. Run Thousands in Parallel**

`main.cu` example:
```c++
fk_kernel_many<<<blocks, threads>>>(d_q_all, d_p1, d_p2, d_out_all, N);
cudaDeviceSynchronize();
```

---

## **Example Project**

```
casadi-on-gpu/
│
├── src/
│   ├── fk_alpha.cu          # CasADi generated and patched
│   ├── fk_alpha.h
│   ├── device_fk_eval.cuh   # Device wrapper
│   ├── main.cu              # Example usage
│
└── CMakeLists.txt
```

---

## **Build**

```bash
mkdir build
cd build
cmake ..
make -j8
./run_casadi_gpu
```

---

## **Why This Is Useful**

* CasADi produces efficient C code but normally runs on CPU only.
* Many robotics tasks need evaluating FK or Jacobians hundreds or thousands of times.
* GPUs let you do that in parallel.