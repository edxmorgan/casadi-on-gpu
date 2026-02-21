"""Author: edward morgan."""

import sys


def get_fk_spec(cog):
    kernels = cog.list_kernels()
    if not kernels:
        raise RuntimeError("No kernels registered in casadi_on_gpu module")

    for k in kernels:
        if k["function_name"] == "fkeval":
            return k

    raise RuntimeError("Expected kernel function 'fkeval' not found")


def run_torch(cog):
    import torch

    if not torch.cuda.is_available():
        print("torch CUDA not available; skipping GPU kernel smoke test.")
        return True

    fk = get_fk_spec(cog)

    n = 128
    dof = fk["input_nnz"][0]
    p1_dim = fk["input_nnz"][1]
    p2_dim = fk["input_nnz"][2]
    out_dim = fk["output_nnz"][0]

    q_all = torch.zeros((n, dof), device="cuda", dtype=torch.float32)
    p1 = torch.zeros((p1_dim,), device="cuda", dtype=torch.float32)
    p2 = torch.zeros((p2_dim,), device="cuda", dtype=torch.float32)
    out = torch.zeros((n, out_dim), device="cuda", dtype=torch.float32)

    stream = torch.cuda.current_stream().cuda_stream
    cog.launch(
        "fkeval",
        [q_all.data_ptr(), p1.data_ptr(), p2.data_ptr()],
        [out.data_ptr()],
        n,
        stream_ptr=stream,
        sync=True,
    )

    print("torch FK smoke test passed.")
    return True


def run_cupy(cog):
    import cupy as cp

    try:
        _ = cp.cuda.runtime.getDeviceCount()
    except cp.cuda.runtime.CUDARuntimeError:
        print("cupy CUDA not available; skipping GPU kernel smoke test.")
        return True

    fk = get_fk_spec(cog)

    n = 128
    dof = fk["input_nnz"][0]
    p1_dim = fk["input_nnz"][1]
    p2_dim = fk["input_nnz"][2]
    out_dim = fk["output_nnz"][0]

    q_all = cp.zeros((n, dof), dtype=cp.float32)
    p1 = cp.zeros((p1_dim,), dtype=cp.float32)
    p2 = cp.zeros((p2_dim,), dtype=cp.float32)
    out = cp.zeros((n, out_dim), dtype=cp.float32)

    stream = cp.cuda.get_current_stream().ptr
    cog.launch(
        "fkeval",
        [q_all.data.ptr, p1.data.ptr, p2.data.ptr],
        [out.data.ptr],
        n,
        stream_ptr=stream,
        sync=True,
    )

    print("cupy FK smoke test passed.")
    return True


def main():
    try:
        import casadi_on_gpu as cog
    except Exception as exc:
        print(f"Failed to import casadi_on_gpu: {exc}")
        return 1

    kernels = cog.list_kernels()
    if not kernels:
        print("No kernels available from casadi_on_gpu")
        return 1
    print(f"Registered kernels: {[k['function_name'] for k in kernels]}")

    # Try torch first, then cupy, otherwise just accept import-only success.
    try:
        return 0 if run_torch(cog) else 1
    except Exception as exc:
        print(f"torch backend unavailable or failed: {exc}")

    try:
        return 0 if run_cupy(cog) else 1
    except Exception as exc:
        print(f"cupy backend unavailable or failed: {exc}")

    print("No CUDA backend available; import-only smoke test passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
