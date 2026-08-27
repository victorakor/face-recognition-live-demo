"""Re-export models/best.pt to models/best.onnx.

The server runs the ONNX graph, so this only needs running when the YOLO weights change:

    pip install -r requirements-export.txt
    python scripts/export_onnx.py

Exported with dynamic height/width so one graph serves both pipeline modes at whatever
input size each asks for -- see detector.py. `--imgsz` only sets the shape used to trace the
export; it is not baked in. Keep --dynamic on unless you have a reason not to: measured on
the deployment container the dynamic graph is *faster* than a fixed-shape one at the same
size (127 ms vs 144 ms at 640), so there is nothing to trade away.

Class names ride along in the ONNX metadata, which is where detector.py reads them from --
there is no hardcoded class list to keep in sync.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any

DEFAULT_WEIGHTS = os.path.join("models", "best.pt")
# The size the weights were trained at, and what the server serves them at. Below this the
# detector loses covered faces badly: on a 640x360 frame with a 180 px subject, `no_mask`
# scores 0.86 at 640 and 0.06 at 480 -- a miss.
DEFAULT_IMGSZ = 640
DEFAULT_OPSET = 12


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS)
    parser.add_argument(
        "--imgsz",
        type=int,
        default=DEFAULT_IMGSZ,
        help="tracing size; with --dynamic (the default) the graph accepts any multiple of 32",
    )
    parser.add_argument("--opset", type=int, default=DEFAULT_OPSET)
    parser.add_argument(
        "--no-dynamic",
        dest="dynamic",
        action="store_false",
        help="bake imgsz into the graph (detector.py will then ignore per-mode sizes)",
    )
    parser.set_defaults(dynamic=True)
    args = parser.parse_args()

    if not os.path.exists(args.weights):
        print(f"[error] weights not found: {args.weights}", file=sys.stderr)
        return 1

    try:
        import torch
        from ultralytics import YOLO
    except ImportError as exc:
        print(f"[error] {exc}. Install with: pip install -r requirements-export.txt", file=sys.stderr)
        return 1

    # torch>=2.6 defaults torch.load to weights_only=True, which refuses to unpickle an
    # ultralytics checkpoint. These are our own weights.
    original_load = torch.load

    def permissive_load(*call_args: Any, **kwargs: Any):
        kwargs.setdefault("weights_only", False)
        return original_load(*call_args, **kwargs)

    torch.load = permissive_load
    try:
        model = YOLO(args.weights)
        print(f"[info] classes: {model.names}")
        path = model.export(
            format="onnx",
            imgsz=args.imgsz,
            opset=args.opset,
            simplify=False,
            dynamic=args.dynamic,
            half=False,
        )
    finally:
        torch.load = original_load

    size_mb = os.path.getsize(path) / (1024 * 1024)
    shape = "dynamic HxW" if args.dynamic else f"fixed {args.imgsz}x{args.imgsz}"
    print(f"[ok] {path} ({size_mb:.1f} MB), input {shape}, traced at {args.imgsz}, opset {args.opset}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
