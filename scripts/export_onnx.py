"""Re-export models/best.pt to models/best.onnx.

The server runs the ONNX graph, so this only needs running when the YOLO weights change:

    pip install -r requirements-export.txt
    python scripts/export_onnx.py

The exported input size is baked into the graph, so keep it in step with YOLO_IMGSZ (480).
Class names ride along in the ONNX metadata, which is where detector.py reads them from --
there is no hardcoded class list to keep in sync.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any

DEFAULT_WEIGHTS = os.path.join("models", "best.pt")
DEFAULT_IMGSZ = 480
DEFAULT_OPSET = 12


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS)
    parser.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ)
    parser.add_argument("--opset", type=int, default=DEFAULT_OPSET)
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
            dynamic=False,
            half=False,
        )
    finally:
        torch.load = original_load

    size_mb = os.path.getsize(path) / (1024 * 1024)
    print(f"[ok] {path} ({size_mb:.1f} MB), input {args.imgsz}x{args.imgsz}, opset {args.opset}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
