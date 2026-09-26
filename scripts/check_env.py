#!/usr/bin/env python
"""Phase 1 smoke test: ROCm reachable, diffusers has LTX-2.5, checkpoint loadable.

Run with no arguments to check the runtime only; pass --model to additionally
try loading the pipeline off disk.

Note: do NOT set HSA_OVERRIDE_GFX_VERSION. torch here is a native gfx1151 build
with its own bundled ROCm runtime; overriding the arch breaks it.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path


def check_runtime() -> bool:
    import torch

    print("== runtime ==")
    print(f"  torch          {torch.__version__}")
    print(f"  hip            {torch.version.hip}")
    if override := os.environ.get("HSA_OVERRIDE_GFX_VERSION"):
        print(f"  WARNING: HSA_OVERRIDE_GFX_VERSION={override} is set; unset it.")

    if not torch.cuda.is_available():
        print("  FAIL: torch.cuda.is_available() is False")
        return False

    props = torch.cuda.get_device_properties(0)
    print(f"  device         {torch.cuda.get_device_name(0)} ({props.gcnArchName})")
    print(f"  VRAM           {props.total_memory / 1024**3:.1f} GiB")

    x = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(10):
        x = x @ x.T / 4096
    torch.cuda.synchronize()
    dt = time.perf_counter() - t
    tflops = 10 * 2 * 4096**3 / dt / 1e12
    print(f"  bf16 matmul    ok ({tflops:.1f} TFLOP/s)")
    return True


def check_diffusers() -> bool:
    import diffusers

    print("== diffusers ==")
    print(f"  version        {diffusers.__version__}")
    required = ["LTX25ModularPipeline", "LTX2Pipeline", "LTX2ImageToVideoPipeline", "LTX2ConditionPipeline"]
    missing = [n for n in required if not hasattr(diffusers, n)]
    for name in required:
        print(f"  {'ok ' if name not in missing else 'MISSING'}        {name}")
    if missing:
        print("  FAIL: install diffusers from git main (LTX-2.5 is not in any release)")
        return False
    return True


def check_model(path: Path) -> bool:
    """Load every component once, one at a time.

    Deliberately *not* a whole-pipeline from_pretrained: the checkpoint is 67 GiB
    in bf16, which fits neither the 45 GiB of system RAM nor the 48 GiB of VRAM.
    Loading serially and freeing as we go proves each component and its config are
    intact without ever holding more than the largest one (transformer, 35 GiB).
    """
    import gc
    import importlib
    import json

    import torch

    print(f"== checkpoint: {path} ==")
    index_file = path / "model_index.json"
    if not index_file.exists():
        print(f"  FAIL: {index_file} missing (run scripts/download_models.py)")
        return False

    index = json.loads(index_file.read_text())
    print(f"  pipeline class {index['_class_name']}")

    ok = True
    for name, spec in sorted(index.items()):
        if name.startswith("_") or not isinstance(spec, list):
            continue
        module_name, class_name = spec
        if module_name is None:  # optional component, not shipped
            print(f"    {name:20s} (absent)")
            continue
        if not (path / name).exists():
            print(f"    {name:20s} FAIL: subfolder missing")
            ok = False
            continue

        # "ltx2" components live in the pipeline package, not the top-level namespace.
        module = importlib.import_module("diffusers.pipelines.ltx2" if module_name == "ltx2" else module_name)
        cls = getattr(module, class_name)

        # Schedulers and tokenizers are config-only; they reject a dtype argument.
        kwargs = {"dtype": torch.bfloat16} if hasattr(cls, "num_parameters") or issubclass(cls, torch.nn.Module) else {}

        t = time.perf_counter()
        try:
            comp = cls.from_pretrained(path, subfolder=name, **kwargs)
        except Exception as e:
            print(f"    {name:20s} FAIL: {type(e).__name__}: {str(e).splitlines()[0][:110]}")
            ok = False
            continue

        params = sum(p.numel() for p in comp.parameters()) if hasattr(comp, "parameters") else 0
        size = f"{params / 1e9:5.2f} B params" if params else "no params"
        print(f"    {name:20s} {class_name:42s} {size}  ({time.perf_counter() - t:.1f}s)")
        del comp
        gc.collect()

    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", type=Path, help="checkpoint dir to try loading (skipped if omitted)")
    args = ap.parse_args()

    ok = check_runtime()
    ok = check_diffusers() and ok
    if args.model:
        ok = check_model(args.model) and ok

    print("\nPhase 1:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
