"""Detection backends.

Three of them, tried in preference order:

    onnx   models/best.onnx through onnxruntime. Same weights as best.pt, ~80 MB resident.
    torch  models/best.pt through ultralytics. Identical output, but importing torch costs
           ~400 MB, which does not fit a 512 MB free-tier container.
    hog    dlib's HOG face detector. Finds faces and nothing else -- no covering classes, no
           weapons. The last resort.

Each backend takes a BGR frame and returns a list of `Detection`. Everything above this
layer (splitting faces from objects, embedding, matching) is the recogniser's job.

The ONNX path is the interesting one: onnxruntime hands back YOLOv8's raw head output, so
letterboxing, score extraction, NMS and coordinate un-mapping are done here rather than by
ultralytics. Getting the letterbox arithmetic wrong shifts every box, so it deliberately
mirrors ultralytics' own rounding.
"""

from __future__ import annotations

import ast
import os
from typing import Any, NamedTuple, Protocol

import cv2
import numpy as np

# Ultralytics pads with this grey and splits the padding across both sides.
_PAD_VALUE = 114

# NMS IoU. 0.7 is the ultralytics predict default; keeping it means the ONNX path suppresses
# the same overlaps the torch path would.
DEFAULT_IOU = float(os.getenv("YOLO_IOU", "0.7"))

# Only used if a model somehow arrives without the names metadata ultralytics writes.
_FALLBACK_CLASSES = ["no_mask", "mask", "other_coverings", "weapon"]


class Detection(NamedTuple):
    box: tuple[int, int, int, int]
    label: str
    score: float | None


class Detector(Protocol):
    name: str
    classes: list[str]

    def __call__(self, frame_bgr: np.ndarray) -> list[Detection]: ...


# --------------------------------------------------------------------------------------
# ONNX
# --------------------------------------------------------------------------------------


class OnnxDetector:
    name = "yolov8-onnx"

    def __init__(self, model_path: str, conf: float, iou: float = DEFAULT_IOU, threads: int = 2):
        import onnxruntime as ort

        options = ort.SessionOptions()
        # One frame at a time, so intra-op parallelism is all that helps. Letting onnxruntime
        # spin up its default thread pool on a 0.1-CPU container just adds contention.
        options.intra_op_num_threads = max(1, threads)
        options.inter_op_num_threads = 1
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = ort.InferenceSession(
            model_path, sess_options=options, providers=["CPUExecutionProvider"]
        )
        self.conf = conf
        self.iou = iou

        spec = self.session.get_inputs()[0]
        self.input_name = spec.name
        # [1, 3, H, W]; the export is fixed-shape, but fall back to 480 if it ever is not.
        _, _, height, width = spec.shape
        self.height = int(height) if isinstance(height, int) else 480
        self.width = int(width) if isinstance(width, int) else 480

        self.classes = self._read_classes()

    def _read_classes(self) -> list[str]:
        """Pull the class list out of the metadata ultralytics writes at export time."""
        meta = self.session.get_modelmeta().custom_metadata_map or {}
        raw = meta.get("names")
        if raw:
            try:
                names = ast.literal_eval(raw)
                if isinstance(names, dict):
                    return [str(names[key]) for key in sorted(names)]
                if isinstance(names, (list, tuple)):
                    return [str(name) for name in names]
            except (ValueError, SyntaxError):
                pass
        return list(_FALLBACK_CLASSES)

    def _letterbox(self, frame_bgr: np.ndarray) -> tuple[np.ndarray, float, int, int]:
        """Resize preserving aspect ratio and centre-pad to the model's input size.

        The `round(pad - 0.1)` is ultralytics' own rounding, kept so an odd number of
        padding pixels lands on the same side as it would in the torch path.
        """
        height, width = frame_bgr.shape[:2]
        gain = min(self.height / height, self.width / width)
        new_w, new_h = int(round(width * gain)), int(round(height * gain))

        if (new_w, new_h) != (width, height):
            interpolation = cv2.INTER_LINEAR if gain > 1 else cv2.INTER_AREA
            frame_bgr = cv2.resize(frame_bgr, (new_w, new_h), interpolation=interpolation)

        pad_w = (self.width - new_w) / 2
        pad_h = (self.height - new_h) / 2
        left, top = int(round(pad_w - 0.1)), int(round(pad_h - 0.1))

        canvas = np.full((self.height, self.width, 3), _PAD_VALUE, dtype=np.uint8)
        canvas[top : top + new_h, left : left + new_w] = frame_bgr
        return canvas, gain, left, top

    def __call__(self, frame_bgr: np.ndarray) -> list[Detection]:
        height, width = frame_bgr.shape[:2]
        canvas, gain, pad_left, pad_top = self._letterbox(frame_bgr)

        # BGR->RGB, HWC->CHW, 0..1. ascontiguousarray because the channel reverse and
        # transpose leave a non-contiguous view that onnxruntime would have to copy anyway.
        blob = np.ascontiguousarray(
            canvas[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        )
        raw = self.session.run(None, {self.input_name: blob})[0]

        # (1, 4 + nc, anchors) -> (anchors, 4 + nc). YOLOv8 has no objectness channel and the
        # class scores are already activated, so the class score IS the confidence.
        preds = raw[0].T
        scores_per_class = preds[:, 4:]
        class_ids = scores_per_class.argmax(axis=1)
        scores = scores_per_class[np.arange(scores_per_class.shape[0]), class_ids]

        keep = scores >= self.conf
        if not np.any(keep):
            return []
        boxes, scores, class_ids = preds[keep, :4], scores[keep], class_ids[keep]

        # cx, cy, w, h (letterboxed pixels) -> x, y, w, h
        xy = boxes[:, :2] - boxes[:, 2:4] / 2.0
        wh = boxes[:, 2:4]

        # Per-class NMS without cv2.dnn.NMSBoxesBatched (not in every OpenCV build): push
        # each class into its own coordinate region so boxes of different classes can never
        # overlap enough to suppress each other.
        stride = float(max(self.width, self.height) + 1)
        offsets = (class_ids.astype(np.float32) * stride)[:, None]
        nms_input = np.concatenate([xy + offsets, wh], axis=1)

        indices = cv2.dnn.NMSBoxes(
            nms_input.tolist(), scores.astype(np.float32).tolist(), self.conf, self.iou
        )
        if indices is None or len(indices) == 0:
            return []

        detections: list[Detection] = []
        for index in np.asarray(indices).reshape(-1):
            x, y = xy[index]
            box_w, box_h = wh[index]
            # Undo the letterbox: drop the padding, then the resize.
            x1 = (x - pad_left) / gain
            y1 = (y - pad_top) / gain
            x2 = (x + box_w - pad_left) / gain
            y2 = (y + box_h - pad_top) / gain

            label = (
                self.classes[class_ids[index]]
                if class_ids[index] < len(self.classes)
                else "object"
            )
            detections.append(
                Detection(
                    box=(
                        int(max(0, min(round(x1), width - 1))),
                        int(max(0, min(round(y1), height - 1))),
                        int(max(0, min(round(x2), width - 1))),
                        int(max(0, min(round(y2), height - 1))),
                    ),
                    label=label,
                    score=float(scores[index]),
                )
            )
        return detections


# --------------------------------------------------------------------------------------
# torch / ultralytics
# --------------------------------------------------------------------------------------


class TorchDetector:
    name = "yolov8-torch"

    def __init__(self, model_path: str, imgsz: int, conf: float, threads: int = 2):
        import torch
        from ultralytics import YOLO

        torch.set_num_threads(max(1, threads))

        # torch>=2.6 flipped torch.load to weights_only=True, which refuses to unpickle an
        # ultralytics checkpoint. These are our own weights, shipped in this repo.
        original_load = torch.load

        def permissive_load(*args: Any, **kwargs: Any):
            kwargs.setdefault("weights_only", False)
            return original_load(*args, **kwargs)

        torch.load = permissive_load
        try:
            self.model = YOLO(model_path)
        finally:
            torch.load = original_load

        self.imgsz = imgsz
        self.conf = conf
        names = self.model.names or {}
        self.classes = [str(names[key]) for key in sorted(names)]

    def __call__(self, frame_bgr: np.ndarray) -> list[Detection]:
        height, width = frame_bgr.shape[:2]
        results = self.model.predict(
            source=frame_bgr, imgsz=self.imgsz, conf=self.conf, verbose=False
        )
        detections: list[Detection] = []
        for result in results:
            names = result.names or {}
            for box in result.boxes:
                x1, y1, x2, y2 = (int(round(v)) for v in box.xyxy[0].tolist())
                detections.append(
                    Detection(
                        box=(
                            max(0, min(x1, width - 1)),
                            max(0, min(y1, height - 1)),
                            max(0, min(x2, width - 1)),
                            max(0, min(y2, height - 1)),
                        ),
                        label=str(names.get(int(box.cls[0]), "object")),
                        score=float(box.conf[0]),
                    )
                )
        return detections


# --------------------------------------------------------------------------------------
# dlib HOG
# --------------------------------------------------------------------------------------


class HogDetector:
    """Faces only, and it says so: every detection is labelled "face" with no score.

    HOG is slow at full resolution, so frames wider than 480 px are halved first -- roughly
    4x cheaper and plenty for webcam framing.
    """

    name = "dlib-hog"
    classes: list[str] = ["face"]

    def __init__(self) -> None:
        import dlib

        self.detector = dlib.get_frontal_face_detector()

    def __call__(self, frame_bgr: np.ndarray) -> list[Detection]:
        height, width = frame_bgr.shape[:2]
        scale = 2 if width > 480 else 1
        small = frame_bgr if scale == 1 else cv2.resize(frame_bgr, (width // 2, height // 2))

        detections: list[Detection] = []
        for rect in self.detector(cv2.cvtColor(small, cv2.COLOR_BGR2RGB), 0):
            detections.append(
                Detection(
                    box=(
                        max(0, min(rect.left() * scale, width - 1)),
                        max(0, min(rect.top() * scale, height - 1)),
                        max(0, min(rect.right() * scale, width - 1)),
                        max(0, min(rect.bottom() * scale, height - 1)),
                    ),
                    label="face",
                    score=None,
                )
            )
        return detections


# --------------------------------------------------------------------------------------
# Selection
# --------------------------------------------------------------------------------------


def build_detector(
    *,
    onnx_path: str,
    torch_path: str,
    imgsz: int,
    conf: float,
    threads: int,
    preference: str = "auto",
) -> tuple[Detector, list[str]]:
    """Build the best available detector.

    Returns the detector and a list of human-readable warnings about anything that had to be
    skipped, which the app surfaces on the page rather than hiding in a log.
    """
    warnings: list[str] = []
    preference = (preference or "auto").strip().lower()

    if preference == "hog":
        warnings.append("detector forced to dlib HOG: no covering or weapon classes")
        return HogDetector(), warnings

    order = ["onnx", "torch"] if preference in ("auto", "") else [preference]

    for backend in order:
        path = onnx_path if backend == "onnx" else torch_path
        if not os.path.exists(path):
            warnings.append(f"{backend} weights not found at {path}")
            continue
        try:
            if backend == "onnx":
                return OnnxDetector(path, conf=conf, threads=threads), warnings
            return TorchDetector(path, imgsz=imgsz, conf=conf, threads=threads), warnings
        except Exception as exc:  # pragma: no cover - depends on which wheels are installed
            warnings.append(f"{backend} detector unavailable ({exc})")

    warnings.append("falling back to dlib HOG: no covering or weapon classes")
    return HogDetector(), warnings
