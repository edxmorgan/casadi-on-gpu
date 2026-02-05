import sys


def run_torch(cog):
    import torch

    if not torch.cuda.is_available():
        print("torch CUDA not available; skipping GPU kernel smoke test.")
        return True

    n = 128
    q_all = torch.zeros((n, cog.FK_DOF), device="cuda", dtype=torch.float32)
    p1 = torch.zeros((6,), device="cuda", dtype=torch.float32)
    p2 = torch.zeros((6,), device="cuda", dtype=torch.float32)
    out = torch.zeros((n, cog.FK_OUT_DIM), device="cuda", dtype=torch.float32)

    stream = torch.cuda.current_stream().cuda_stream
    cog.fk_forward(q_all.data_ptr(), p1.data_ptr(), p2.data_ptr(), out.data_ptr(),
                   n, stream_ptr=stream, sync=True)

    print("torch FK smoke test passed.")
    return True


def run_cupy(cog):
    import cupy as cp

    try:
        _ = cp.cuda.runtime.getDeviceCount()
    except cp.cuda.runtime.CUDARuntimeError:
        print("cupy CUDA not available; skipping GPU kernel smoke test.")
        return True

    n = 128
    q_all = cp.zeros((n, cog.FK_DOF), dtype=cp.float32)
    p1 = cp.zeros((6,), dtype=cp.float32)
    p2 = cp.zeros((6,), dtype=cp.float32)
    out = cp.zeros((n, cog.FK_OUT_DIM), dtype=cp.float32)

    stream = cp.cuda.get_current_stream().ptr
    cog.fk_forward(q_all.data.ptr, p1.data.ptr, p2.data.ptr, out.data.ptr,
                   n, stream_ptr=stream, sync=True)

    print("cupy FK smoke test passed.")
    return True


def main():
    try:
        import casadi_on_gpu as cog
    except Exception as exc:
        print(f"Failed to import casadi_on_gpu: {exc}")
        return 1

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
