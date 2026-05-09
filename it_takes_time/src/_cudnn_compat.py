"""Preload the cuDNN bundled inside the torch wheel's `nvidia.cudnn` package.

Background: this Coder/EC2 image ships system cuDNN 9.7 under
`/usr/local/cuda-12.6/lib`, which is on the global `LD_LIBRARY_PATH`. PyTorch
2.9 is compiled against cuDNN 9.8 and refuses to run when it sees a different
runtime version, raising:

    RuntimeError: cuDNN version incompatibility: PyTorch was compiled
    against (9, 8, 0) but found runtime version (9, 7, 0).

We don't want to mutate the system `LD_LIBRARY_PATH` (it is shared with other
tools on the instance). Instead, we `dlopen` the wheel-bundled cuDNN by
absolute path here, before torch first touches cuDNN. Subsequent `dlopen`s by
soname inside torch will return the handle we already loaded, so the wrong
9.7 copy on `LD_LIBRARY_PATH` is never picked up.

Importing this module is a no-op when the bundled cuDNN is missing or already
loaded; safe to call from non-CUDA hosts.
"""

from __future__ import annotations

import ctypes
import os
import sys
from pathlib import Path


def _bundled_cudnn_dir() -> Path | None:
    for site_dir in sys.path:
        cand = Path(site_dir) / "nvidia" / "cudnn" / "lib"
        if cand.is_dir():
            return cand
    try:
        import nvidia.cudnn  # type: ignore
        return Path(nvidia.cudnn.__file__).parent / "lib"
    except Exception:
        return None


def _preload() -> None:
    lib_dir = _bundled_cudnn_dir()
    if lib_dir is None:
        return
    # Load order matters: leaf libs first, then libcudnn.so.9 which depends on them.
    leaf_order = [
        "libcudnn_graph.so.9",
        "libcudnn_ops.so.9",
        "libcudnn_adv.so.9",
        "libcudnn_cnn.so.9",
        "libcudnn_engines_precompiled.so.9",
        "libcudnn_engines_runtime_compiled.so.9",
        "libcudnn_heuristic.so.9",
        "libcudnn.so.9",
    ]
    for name in leaf_order:
        path = lib_dir / name
        if not path.exists():
            continue
        try:
            ctypes.CDLL(str(path), mode=os.RTLD_GLOBAL | os.RTLD_LAZY)
        except OSError:
            # Some sub-libs only resolve at runtime against another already-loaded one;
            # ignore individual failures, the final libcudnn.so.9 load is what matters.
            pass


_preload()
