#!/usr/bin/env python3
"""Generate CasADi CUDA sources + runtime manifest + CUDA kernel registry.

This script wraps the CasADi CodeGenerator process so metadata is derived from the
same generation inputs (no separate manual spec JSON required).
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import casadi as ca


@dataclass
class KernelEntryConfig:
    casadi_file: Path
    unit_name: str
    batch_inputs: list[int]
    kernel_name: str | None = None
    device_name: str | None = None


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _parse_entry(raw: str, repo_root: Path) -> KernelEntryConfig:
    # Format: casadi_file:unit_name:batch_inputs[:kernel_name[:device_name]]
    # batch_inputs uses commas, e.g. 0 or 2 or 0,2
    parts = raw.split(":")
    if len(parts) < 3:
        raise ValueError(
            f"Invalid --entry '{raw}'. Expected format: casadi_file:unit_name:batch_inputs[:kernel_name[:device_name]]"
        )

    casadi_file = Path(parts[0])
    if not casadi_file.is_absolute():
        casadi_file = (repo_root / casadi_file).resolve()

    unit_name = parts[1].strip()
    if not unit_name:
        raise ValueError(f"Invalid --entry '{raw}': unit_name must be non-empty")

    batch_spec = parts[2].strip()
    batch_inputs = [] if batch_spec == "" else [int(x) for x in batch_spec.split(",") if x]

    kernel_name = parts[3].strip() if len(parts) >= 4 and parts[3].strip() else None
    device_name = parts[4].strip() if len(parts) >= 5 and parts[4].strip() else None

    return KernelEntryConfig(
        casadi_file=casadi_file,
        unit_name=unit_name,
        batch_inputs=batch_inputs,
        kernel_name=kernel_name,
        device_name=device_name,
    )


def _resolve_casadi_file(raw: str, repo_root: Path) -> Path:
    casadi_file = Path(raw)
    if not casadi_file.is_absolute():
        casadi_file = (repo_root / casadi_file).resolve()
    return casadi_file


def _parse_batch_inputs(raw: str) -> list[int]:
    spec = raw.strip()
    if spec == "":
        return []
    return [int(x) for x in spec.split(",") if x.strip()]


def _unit_name_from_casadi_file(casadi_file: Path) -> str:
    stem = casadi_file.stem
    unit_name = re.sub(r"[^0-9A-Za-z_]", "_", stem)
    if not unit_name or not (unit_name[0].isalpha() or unit_name[0] == "_"):
        unit_name = f"unit_{unit_name}"
    return unit_name


def _parse_batch_override(raw: str, repo_root: Path) -> tuple[str, list[int]]:
    # Format: casadi_file=batch_inputs
    if "=" not in raw:
        raise ValueError(
            f"Invalid per-file batch mapping '{raw}'. Expected format: casadi_file=batch_inputs"
        )
    path_part, batch_part = raw.split("=", 1)
    resolved = _resolve_casadi_file(path_part.strip(), repo_root)
    return str(resolved), _parse_batch_inputs(batch_part)


def _entries_from_simple_args(
    casadi_files: list[str],
    batch_inputs_default: list[int],
    batch_overrides: dict[str, list[int]],
    repo_root: Path,
) -> list[KernelEntryConfig]:
    entries: list[KernelEntryConfig] = []
    for raw in casadi_files:
        casadi_file = _resolve_casadi_file(raw, repo_root)
        entries.append(
            KernelEntryConfig(
                casadi_file=casadi_file,
                unit_name=_unit_name_from_casadi_file(casadi_file),
                batch_inputs=batch_overrides.get(str(casadi_file), batch_inputs_default),
            )
        )
    return entries


def _collect_casadi_from_dirs(raw_dirs: list[str], repo_root: Path) -> list[str]:
    files: list[str] = []
    for raw_dir in raw_dirs:
        d = Path(raw_dir)
        if not d.is_absolute():
            d = (repo_root / d).resolve()
        if not d.is_dir():
            raise ValueError(f"--casadi-dir '{raw_dir}' does not exist or is not a directory")
        for p in sorted(d.glob("*.casadi")):
            files.append(str(p))
    return files


def _prune_stale_generated_units(generated_dir: Path, entries: list[KernelEntryConfig]) -> None:
    expected = set()
    for e in entries:
        expected.add(f"{e.unit_name}.cu")
        expected.add(f"{e.unit_name}.cuh")

    for p in generated_dir.iterdir():
        if not p.is_file() or p.suffix not in {".cu", ".cuh"}:
            continue
        if p.name not in expected:
            p.unlink()
            print(f"Removed stale generated file: {p}")


def _generate_and_collect(
    entries: list[KernelEntryConfig],
    generated_dir: Path,
    casadi_real: str,
) -> dict[str, Any]:
    kernels: list[dict[str, Any]] = []

    generated_dir.mkdir(parents=True, exist_ok=True)
    _prune_stale_generated_units(generated_dir, entries)

    seen_units: set[str] = set()
    seen_functions: set[str] = set()
    seen_kernel_names: set[str] = set()
    seen_device_names: set[str] = set()

    for e in entries:
        if e.unit_name in seen_units:
            raise ValueError(f"Duplicate unit name '{e.unit_name}' in generation entries")
        seen_units.add(e.unit_name)

        f = ca.Function.load(str(e.casadi_file))
        function_name = f.name()

        kernel_name = e.kernel_name or f"{function_name}_kernel"
        device_name = e.device_name or f"device_{function_name}_eval"

        if function_name in seen_functions:
            raise ValueError(f"Duplicate CasADi function name '{function_name}' across entries")
        if kernel_name in seen_kernel_names:
            raise ValueError(f"Duplicate kernel name '{kernel_name}' across entries")
        if device_name in seen_device_names:
            raise ValueError(f"Duplicate device function name '{device_name}' across entries")
        seen_functions.add(function_name)
        seen_kernel_names.add(kernel_name)
        seen_device_names.add(device_name)

        cg = ca.CodeGenerator(
            e.unit_name,
            {
                "with_header": True,
                "casadi_real": casadi_real,
                "cpp": False,
                "cuda": True,
                "cuda_kernels": {
                    function_name: {
                        "batch_inputs": e.batch_inputs,
                        "kernel_name": kernel_name,
                        "device_name": device_name,
                    }
                },
            },
        )
        cg.add(f)
        cg.generate(str(generated_dir) + "/")

        header = f"{e.unit_name}.cuh"
        source = f"{e.unit_name}.cu"
        if not (generated_dir / header).exists() or not (generated_dir / source).exists():
            raise RuntimeError(
                f"Code generation did not produce expected files for {function_name}: {source}, {header}"
            )

        inputs = []
        for i in range(f.n_in()):
            sp = f.sparsity_in(i)
            inputs.append(
                {
                    "index": i,
                    "name": f.name_in(i),
                    "shape": [sp.shape[0], sp.shape[1]],
                    "nnz": int(sp.nnz()),
                }
            )

        outputs = []
        for i in range(f.n_out()):
            sp = f.sparsity_out(i)
            outputs.append(
                {
                    "index": i,
                    "name": f.name_out(i),
                    "shape": [sp.shape[0], sp.shape[1]],
                    "nnz": int(sp.nnz()),
                }
            )

        kernels.append(
            {
                "function_name": function_name,
                "kernel_name": kernel_name,
                "device_name": device_name,
                "header": header,
                "source": source,
                "batch_inputs": [int(i) for i in e.batch_inputs],
                "inputs": inputs,
                "outputs": outputs,
                "work": {
                    "sz_arg": int(f.sz_arg()),
                    "sz_res": int(f.sz_res()),
                    "sz_iw": int(f.sz_iw()),
                    "sz_w": int(f.sz_w()),
                    "n_nodes": int(f.n_nodes()),
                    "n_instructions": int(f.n_instructions()),
                },
            }
        )

    return {
        "schema_version": 1,
        "casadi_real": casadi_real,
        "kernels": kernels,
    }


def _generate_registry_source(manifest: dict[str, Any]) -> str:
    kernels = manifest["kernels"]

    include_headers = sorted({k["header"] for k in kernels})

    lines: list[str] = []
    lines.append("// Auto-generated by tools/generate_manifest_and_registry.py")
    lines.append("#include \"casadi_on_gpu_kernel_registry.h\"")
    for header in include_headers:
        lines.append(f"#include \"{header}\"")
    lines.append("")
    lines.append("namespace casadi_on_gpu {")
    lines.append("namespace {")

    for k in kernels:
        fn = k["function_name"]
        kernel_name = k["kernel_name"]
        n_in = len(k["inputs"])
        n_out = len(k["outputs"])

        launch_name = f"launch_{fn}"
        lines.append(
            f"void {launch_name}(const std::uintptr_t* input_ptrs, const std::uintptr_t* output_ptrs, "
            "int blocks, int threads_per_block, cudaStream_t stream, int n_candidates) {"
        )

        for i in range(n_in):
            lines.append(
                f"  auto* i{i} = reinterpret_cast<const casadi_real*>(input_ptrs[{i}]);"
            )
        for i in range(n_out):
            lines.append(f"  auto* o{i} = reinterpret_cast<casadi_real*>(output_ptrs[{i}]);")

        call_args = [f"i{i}" for i in range(n_in)] + [f"o{i}" for i in range(n_out)] + [
            "n_candidates"
        ]
        lines.append(
            f"  {kernel_name}<<<blocks, threads_per_block, 0, stream>>>({', '.join(call_args)});"
        )
        lines.append("}")
        lines.append("")

    lines.append("const KernelEntry kRegistry[] = {")
    for k in kernels:
        fn = k["function_name"]
        kernel_name = k["kernel_name"]
        n_in = len(k["inputs"])
        n_out = len(k["outputs"])
        batch = ", ".join(str(i) for i in k["batch_inputs"])
        in_nnz = ", ".join(str(int(i["nnz"])) for i in k["inputs"])
        out_nnz = ", ".join(str(int(i["nnz"])) for i in k["outputs"])

        lines.append(
            "  {"
            + f"\"{fn}\", \"{kernel_name}\", {n_in}, {n_out}, "
            + "{"
            + batch
            + "}, {"
            + in_nnz
            + "}, {"
            + out_nnz
            + "}, "
            + f"launch_{fn}"
            + "},"
        )
    lines.append("};")
    lines.append("")
    lines.append("}  // namespace")
    lines.append("")
    lines.append("const KernelEntry* get_kernel_registry(std::size_t* count) {")
    lines.append("  *count = sizeof(kRegistry) / sizeof(kRegistry[0]);")
    lines.append("  return kRegistry;")
    lines.append("}")
    lines.append("")
    lines.append("}  // namespace casadi_on_gpu")

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--casadi",
        action="append",
        default=[],
        help=(
            "Path to a .casadi function file. Repeat for multiple kernels. "
            "Unit name is auto-derived from file stem."
        ),
    )
    parser.add_argument(
        "--batch-inputs",
        default="0",
        help=(
            "Comma-separated batched input indices for --casadi/--casadi-dir mode. "
            "Default is '0'. Use empty string for none."
        ),
    )
    parser.add_argument(
        "--batch-inputs-for",
        action="append",
        default=[],
        help=(
            "Set batch inputs for a specific file in --casadi/--casadi-dir mode. "
            "Format: casadi_file=batch_inputs (repeatable)."
        ),
    )
    parser.add_argument(
        "--batch-override",
        action="append",
        default=[],
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--casadi-dir",
        action="append",
        default=[],
        help=(
            "Directory to scan for .casadi files (non-recursive). "
            "Repeat for multiple directories."
        ),
    )
    parser.add_argument(
        "--entry",
        action="append",
        default=[],
        help=(
            "Kernel entry as casadi_file:unit_name:batch_inputs[:kernel_name[:device_name]]. "
            "Repeat for multiple kernels. batch_inputs is comma-separated indices."
        ),
    )
    parser.add_argument(
        "--casadi-real",
        default="float",
        choices=["float", "double"],
        help="casadi_real type used for code generation",
    )
    parser.add_argument(
        "--generated-dir",
        type=Path,
        default=Path("src/generated"),
        help="Directory where .cu/.cuh should be generated",
    )
    parser.add_argument(
        "--manifest-out",
        type=Path,
        default=Path("src/generated/kernels_manifest.json"),
        help="Output manifest JSON",
    )
    parser.add_argument(
        "--registry-out",
        type=Path,
        default=Path("src/python/casadi_on_gpu_kernel_registry.cu"),
        help="Output generated kernel registry source",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent

    try:
        if args.entry and (args.casadi or args.casadi_dir):
            raise ValueError(
                "Use either --entry (advanced) or --casadi/--casadi-dir (simple), not both."
            )

        if args.entry:
            entries = [_parse_entry(e, repo_root) for e in args.entry]
        else:
            casadi_files = list(args.casadi)
            casadi_files.extend(_collect_casadi_from_dirs(args.casadi_dir, repo_root))
            if not casadi_files:
                raise ValueError(
                    "No kernels specified. Provide --casadi <file>, --casadi-dir <dir>, or --entry <spec>."
                )
            batch_inputs_default = _parse_batch_inputs(args.batch_inputs)
            raw_batch_mappings = list(args.batch_inputs_for) + list(args.batch_override)
            batch_overrides = dict(
                _parse_batch_override(raw, repo_root) for raw in raw_batch_mappings
            )
            entries = _entries_from_simple_args(
                casadi_files, batch_inputs_default, batch_overrides, repo_root
            )
    except ValueError as exc:
        parser.error(str(exc))

    generated_dir = (
        args.generated_dir if args.generated_dir.is_absolute() else (repo_root / args.generated_dir)
    ).resolve()
    manifest_out = (
        args.manifest_out if args.manifest_out.is_absolute() else (repo_root / args.manifest_out)
    ).resolve()
    registry_out = (
        args.registry_out if args.registry_out.is_absolute() else (repo_root / args.registry_out)
    ).resolve()

    manifest = _generate_and_collect(entries, generated_dir, args.casadi_real)
    manifest_json = json.dumps(manifest, indent=2, sort_keys=False) + "\n"
    registry_src = _generate_registry_source(manifest)

    _write_text(manifest_out, manifest_json)
    _write_text(registry_out, registry_src)

    print(f"Generated {len(manifest['kernels'])} kernel(s)")
    print(f"Wrote manifest: {manifest_out}")
    print(f"Wrote registry: {registry_out}")


if __name__ == "__main__":
    main()
