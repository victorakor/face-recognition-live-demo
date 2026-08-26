"""Detection + recognition pipeline for the live browser demo.

Stateless with respect to HTTP requests: a single process-wide bundle of models is loaded
once, and `FaceRecognizer.detect()` is called with one frame at a time. Neither dlib nor
the ultralytics predictor is thread-safe, so inference is serialised behind a lock.

Pipeline
    1. Detect -- custom YOLOv8 weights (models/best.pt) locate faces and classify their
       covering (no_mask / mask / other_coverings) plus any weapon in frame. Without the
       YOLO runtime the server falls back to dlib's HOG face detector, which finds faces
       but cannot classify coverings or weapons.
    2. Describe -- dlib's 68-point landmark predictor aligns each face and the ResNet
       encoder maps it to a 128-d embedding.
    3. Match -- nearest neighbour (euclidean) against the gallery in
       models/known_faces.pkl, accepted below the threshold chosen during evaluation
       (0.402, ROC AUC 0.9857).
"""

from __future__ import annotations

import math
import os
import pickle
import threading
import time
from typing import Any

import cv2
import dlib
import numpy as np

# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------

MODELS_DIR = os.getenv("MODELS_DIR", "models")

YOLO_MODEL_PATH = os.getenv("YOLO_MODEL_PATH", os.path.join(MODELS_DIR, "best.pt"))
DLIB_LANDMARK_PATH = os.getenv(
    "DLIB_LANDMARK", os.path.join(MODELS_DIR, "shape_predictor_68_face_landmarks.dat")
)
DLIB_FACE_RECOG_PATH = os.getenv(
    "DLIB_FACE_RECOG", os.path.join(MODELS_DIR, "dlib_face_recognition_resnet_model_v1.dat")
)
KNOWN_FACES_PATH = os.getenv("KNOWN_FACES_PKL", os.path.join(MODELS_DIR, "known_faces.pkl"))

# Distance threshold picked by the ROC sweep in docs/evaluation_report.txt.
DEFAULT_THRESHOLD = float(os.getenv("BEST_THRESHOLD", "0.402"))

# YOLO costs ~400 MB resident once torch is imported. Hosts with a small memory cap can
# set ENABLE_YOLO=0 to run the dlib-only pipeline instead.
ENABLE_YOLO = os.getenv("ENABLE_YOLO", "1").lower() not in ("0", "false", "no")
# 480 is the sweet spot on a 2-vCPU container: ~0.5s per frame with no measurable drop in
# detection confidence versus 640.
YOLO_IMGSZ = int(os.getenv("YOLO_IMGSZ", "480"))
YOLO_CONF = float(os.getenv("YOLO_CONF", "0.35"))

# Guardrails so one busy frame cannot pin the CPU for seconds. Embedding a face is the
# expensive step, so the face cap is tighter than the object cap.
MAX_FACES = int(os.getenv("MAX_FACES", "4"))
MAX_OBJECTS = int(os.getenv("MAX_OBJECTS", "8"))
MAX_FRAME_WIDTH = int(os.getenv("MAX_FRAME_WIDTH", "960"))

# Classes from models/best.pt that are faces (and therefore worth identifying). Anything
# else the detector reports -- currently just "weapon" -- is surfaced as a scene object.
FACE_CLASSES = {"no_mask", "mask", "other_coverings", "face"}

# How each face class is described in the UI, and whether the covering makes identity
# matching unreliable.
COVERAGE = {
    "no_mask": ("Uncovered", False),
    "face": ("Uncovered", False),
    "mask": ("Mask", True),
    "other_coverings": ("Covered", True),
}

# Width of the logistic used to turn a raw embedding distance into a 0..1 score.
_CONFIDENCE_TEMPERATURE = 0.06


def _distance_to_confidence(distance: float, threshold: float) -> float:
    """Squash a raw embedding distance into a 0..1 match score.

    A logistic centred on the decision threshold: exactly at the threshold the score is
    0.5, comfortably below it approaches 1.0, comfortably above it approaches 0.0. This is
    a monotone reparameterisation of the distance for display, not a calibrated
    probability.
    """
    z = (distance - threshold) / _CONFIDENCE_TEMPERATURE
    z = max(-60.0, min(60.0, z))
    return 1.0 / (1.0 + math.exp(z))


def _area(box: tuple[int, int, int, int]) -> int:
    return (box[2] - box[0]) * (box[3] - box[1])


class FaceRecognizer:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.warnings: list[str] = []

        self.detector_backend = "none"
        self.yolo = None
        self.yolo_classes: list[str] = []
        self.hog_detector = None
        self.shape_predictor = None
        self.face_encoder = None

        self.known_embeddings = np.empty((0, 128), dtype=np.float32)
        self.known_labels: list[str] = []
        self.identities: list[str] = []

        self._load_dlib()
        self._load_yolo()
        self._load_gallery()

    # ---------------------------------------------------------------- loading
    def _load_dlib(self) -> None:
        self.hog_detector = dlib.get_frontal_face_detector()

        if os.path.exists(DLIB_LANDMARK_PATH):
            self.shape_predictor = dlib.shape_predictor(DLIB_LANDMARK_PATH)
        else:
            self.warnings.append(f"missing landmark model at {DLIB_LANDMARK_PATH}")

        if os.path.exists(DLIB_FACE_RECOG_PATH):
            self.face_encoder = dlib.face_recognition_model_v1(DLIB_FACE_RECOG_PATH)
        else:
            self.warnings.append(f"missing face encoder at {DLIB_FACE_RECOG_PATH}")

        if not getattr(dlib, "DLIB_USE_BLAS", False):
            # Without BLAS, dlib's ResNet forward pass falls back to a naive matmul and
            # takes ~2s per face instead of ~80ms. The Dockerfile builds dlib against
            # OpenBLAS for exactly this reason.
            self.warnings.append("dlib built without BLAS: face encoding will be slow")

        self.detector_backend = "dlib-hog"

    def _load_yolo(self) -> None:
        if not ENABLE_YOLO:
            self.warnings.append("YOLO disabled via ENABLE_YOLO=0; using dlib HOG detector")
            return
        if not os.path.exists(YOLO_MODEL_PATH):
            self.warnings.append(f"YOLO weights not found at {YOLO_MODEL_PATH}")
            return

        try:
            import torch
            from ultralytics import YOLO

            # Keep CPU inference from oversubscribing the (usually tiny) container.
            torch.set_num_threads(int(os.getenv("TORCH_THREADS", "2")))

            # torch>=2.6 flipped torch.load to weights_only=True, which refuses to unpickle
            # an ultralytics checkpoint. These are our own weights, shipped in this repo,
            # so loading them fully is safe.
            original_load = torch.load

            def _permissive_load(*args: Any, **kwargs: Any):
                kwargs.setdefault("weights_only", False)
                return original_load(*args, **kwargs)

            torch.load = _permissive_load
            try:
                self.yolo = YOLO(YOLO_MODEL_PATH)
                # Warm the graph up so the first visitor does not eat the cold start.
                self.yolo.predict(
                    source=np.zeros((YOLO_IMGSZ, YOLO_IMGSZ, 3), dtype=np.uint8),
                    imgsz=YOLO_IMGSZ,
                    conf=YOLO_CONF,
                    verbose=False,
                )
            finally:
                torch.load = original_load

            names = self.yolo.names or {}
            self.yolo_classes = [str(names[key]) for key in sorted(names)]
            self.detector_backend = "yolov8-custom"
        except Exception as exc:  # pragma: no cover - depends on host wheels
            self.yolo = None
            self.warnings.append(f"YOLO unavailable ({exc}); using dlib HOG detector")

    def _load_gallery(self) -> None:
        if not os.path.exists(KNOWN_FACES_PATH):
            self.warnings.append(f"face gallery not found at {KNOWN_FACES_PATH}")
            return
        with open(KNOWN_FACES_PATH, "rb") as handle:
            data = pickle.load(handle)

        embeddings = np.asarray(data.get("embeddings", []), dtype=np.float32)
        labels = [str(label) for label in data.get("labels", [])]
        if embeddings.size and len(labels) == len(embeddings):
            self.known_embeddings = embeddings
            self.known_labels = labels
            # Preserve first-seen order rather than sorting, so the UI lists identities in
            # the order the gallery was built.
            self.identities = list(dict.fromkeys(labels))
        else:
            self.warnings.append("face gallery is empty or malformed")

    # ---------------------------------------------------------------- metadata
    @property
    def recognition_ready(self) -> bool:
        return (
            self.shape_predictor is not None
            and self.face_encoder is not None
            and len(self.known_labels) > 0
        )

    def info(self) -> dict[str, Any]:
        return {
            "detector": self.detector_backend,
            "detectorClasses": self.yolo_classes,
            "recognitionReady": self.recognition_ready,
            "identities": self.identities,
            "gallerySize": int(len(self.known_labels)),
            "defaultThreshold": DEFAULT_THRESHOLD,
            "embeddingDim": int(self.known_embeddings.shape[1]) if len(self.known_labels) else 0,
            "maxFaces": MAX_FACES,
            "blas": bool(getattr(dlib, "DLIB_USE_BLAS", False)),
            "warnings": self.warnings,
        }

    # ---------------------------------------------------------------- inference
    def detect(self, frame_bgr: np.ndarray, threshold: float | None = None) -> dict[str, Any]:
        """Run detection + recognition on one BGR frame.

        Boxes come back both in pixels (relative to the frame actually analysed) and
        normalised to 0..1, so the browser can scale them onto whatever size it renders the
        video at.
        """
        if threshold is None:
            threshold = DEFAULT_THRESHOLD
        threshold = float(max(0.20, min(0.90, threshold)))

        started = time.perf_counter()
        frame_bgr = self._downscale(frame_bgr)
        height, width = frame_bgr.shape[:2]

        with self._lock:
            detect_started = time.perf_counter()
            face_boxes, objects = self._detect(frame_bgr)
            detect_ms = (time.perf_counter() - detect_started) * 1000.0

            recognise_started = time.perf_counter()
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            faces = [self._describe(frame_rgb, entry, threshold) for entry in face_boxes]
            recognise_ms = (time.perf_counter() - recognise_started) * 1000.0

        for item in (*faces, *objects):
            x1, y1, x2, y2 = item["box"]
            item["boxNorm"] = [x1 / width, y1 / height, x2 / width, y2 / height]

        weapons = sum(1 for obj in objects if obj["label"] == "weapon")
        unknown = sum(1 for face in faces if not face["authorized"])
        if weapons:
            threat = "high"
        elif unknown:
            threat = "elevated"
        else:
            threat = "low"

        return {
            "faces": faces,
            "objects": objects,
            "detector": self.detector_backend,
            "threshold": threshold,
            "threat": threat,
            "frame": {"width": width, "height": height},
            "counts": {
                "total": len(faces),
                "authorized": len(faces) - unknown,
                "unknown": unknown,
                "covered": sum(1 for face in faces if face["covered"]),
                "weapons": weapons,
            },
            "timings": {
                "detectMs": round(detect_ms, 1),
                "recognizeMs": round(recognise_ms, 1),
                "totalMs": round((time.perf_counter() - started) * 1000.0, 1),
            },
        }

    def _downscale(self, frame: np.ndarray) -> np.ndarray:
        height, width = frame.shape[:2]
        if width <= MAX_FRAME_WIDTH:
            return frame
        scale = MAX_FRAME_WIDTH / float(width)
        return cv2.resize(
            frame,
            (MAX_FRAME_WIDTH, max(1, int(round(height * scale)))),
            interpolation=cv2.INTER_AREA,
        )

    def _detect(
        self, frame_bgr: np.ndarray
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Split the detector's output into faces (to identify) and other objects."""
        height, width = frame_bgr.shape[:2]
        faces: list[dict[str, Any]] = []
        objects: list[dict[str, Any]] = []

        if self.yolo is not None:
            try:
                results = self.yolo.predict(
                    source=frame_bgr, imgsz=YOLO_IMGSZ, conf=YOLO_CONF, verbose=False
                )
                for result in results:
                    names = result.names or {}
                    for box in result.boxes:
                        raw = tuple(int(v) for v in box.xyxy[0].tolist())
                        clipped = self._clip(raw, width, height)
                        if clipped is None:
                            continue
                        label = str(names.get(int(box.cls[0]), "object"))
                        score = float(box.conf[0])
                        if label in FACE_CLASSES:
                            faces.append({"box": list(clipped), "cls": label, "detectScore": score})
                        else:
                            objects.append({"box": list(clipped), "label": label, "score": round(score, 3)})
            except Exception:
                faces, objects = [], []

        if not faces and self.hog_detector is not None:
            # HOG is slow at full resolution; a half-size pass is plenty for webcam framing
            # and roughly 4x cheaper.
            scale = 2 if width > 480 else 1
            small = frame_bgr if scale == 1 else cv2.resize(frame_bgr, (width // 2, height // 2))
            for rect in self.hog_detector(cv2.cvtColor(small, cv2.COLOR_BGR2RGB), 0):
                raw = (
                    rect.left() * scale,
                    rect.top() * scale,
                    rect.right() * scale,
                    rect.bottom() * scale,
                )
                clipped = self._clip(raw, width, height)
                if clipped is not None:
                    faces.append({"box": list(clipped), "cls": "face", "detectScore": None})

        # Largest first, so the caps drop background bystanders rather than whoever is
        # actually standing in front of the camera.
        faces.sort(key=lambda item: _area(tuple(item["box"])), reverse=True)
        objects.sort(key=lambda item: item["score"], reverse=True)
        return faces[:MAX_FACES], objects[:MAX_OBJECTS]

    @staticmethod
    def _clip(
        box: tuple[int, ...], width: int, height: int
    ) -> tuple[int, int, int, int] | None:
        x1, y1, x2, y2 = box
        x1, x2 = sorted((max(0, min(x1, width - 1)), max(0, min(x2, width - 1))))
        y1, y2 = sorted((max(0, min(y1, height - 1)), max(0, min(y2, height - 1))))
        if x2 - x1 < 16 or y2 - y1 < 16:
            return None
        return x1, y1, x2, y2

    def _describe(
        self, frame_rgb: np.ndarray, entry: dict[str, Any], threshold: float
    ) -> dict[str, Any]:
        x1, y1, x2, y2 = entry["box"]
        coverage_label, covered = COVERAGE.get(entry["cls"], ("Face", False))

        face: dict[str, Any] = {
            "box": entry["box"],
            "cls": entry["cls"],
            "coverage": coverage_label,
            "covered": covered,
            "detectScore": round(entry["detectScore"], 3) if entry["detectScore"] is not None else None,
            "name": "Unknown",
            "authorized": False,
            "distance": None,
            "confidence": None,
        }

        if not self.recognition_ready:
            return face

        try:
            # Landmarks are predicted on the full frame with the detected rectangle -- the
            # framing dlib's encoder expects, and the framing the gallery was built with.
            shape = self.shape_predictor(frame_rgb, dlib.rectangle(x1, y1, x2, y2))
            embedding = np.asarray(
                self.face_encoder.compute_face_descriptor(frame_rgb, shape), dtype=np.float32
            )
        except Exception:
            return face

        distances = np.linalg.norm(self.known_embeddings - embedding, axis=1)
        best = int(np.argmin(distances))
        distance = float(distances[best])

        face["distance"] = round(distance, 4)
        face["confidence"] = round(_distance_to_confidence(distance, threshold), 3)
        if distance < threshold:
            face["name"] = self.known_labels[best]
            face["authorized"] = True
        return face


_recognizer: FaceRecognizer | None = None
_init_lock = threading.Lock()


def get_recognizer() -> FaceRecognizer:
    """Lazily build the process-wide recognizer (loading the models takes a few seconds)."""
    global _recognizer
    if _recognizer is None:
        with _init_lock:
            if _recognizer is None:
                _recognizer = FaceRecognizer()
    return _recognizer


def is_ready() -> bool:
    """Whether the models are loaded, without triggering a load."""
    return _recognizer is not None


def warmup() -> None:
    """Load the models on a background thread.

    Called at import so the process can bind its port and answer health checks straight
    away, while ~100 MB of dlib weights and the YOLO graph load in parallel. Free-tier
    containers take the better part of a minute to get through this.
    """
    threading.Thread(target=get_recognizer, name="model-warmup", daemon=True).start()

