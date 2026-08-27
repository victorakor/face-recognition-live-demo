"""Detection + recognition pipeline for the live browser demo.

Stateless with respect to HTTP requests: a single process-wide bundle of models is loaded
once, and `FaceRecognizer.detect()` is called with one frame at a time. Neither dlib nor
the ultralytics predictor is thread-safe, so inference is serialised behind a lock.

Pipeline
    1. Detect -- custom YOLOv8 weights locate faces and classify their covering
       (no_mask / mask / other_coverings) plus any weapon in frame. See detector.py for the
       backends; the deployed one is ONNX.
    2. Describe -- dlib's 68-point landmark predictor aligns each face and the ResNet
       encoder maps it to a 128-d embedding.
    3. Match -- nearest neighbour (euclidean) against the gallery in
       models/known_faces.pkl, accepted below the threshold chosen during evaluation
       (0.402, ROC AUC 0.9857).

Steps 2 and 3 only run in `recognition` mode. `detection` mode stops after step 1 and
reports the detector's own classes, which is what makes it worth running the detector at its
full training resolution -- the time saved by skipping the embedding pays for the bigger
input. See MODES below.
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

from detector import TRAIN_IMGSZ, HogDetector, build_detector

# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------

MODELS_DIR = os.getenv("MODELS_DIR", "models")

YOLO_ONNX_PATH = os.getenv("YOLO_ONNX_PATH", os.path.join(MODELS_DIR, "best.onnx"))
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

# "auto" prefers ONNX, then torch, then dlib HOG. Force one with "onnx", "torch" or "hog".
# ENABLE_YOLO=0 is kept as a shorthand for "hog" -- it is the escape hatch for a host too
# small to load the detector at all.
DETECTOR_PREFERENCE = os.getenv("DETECTOR", "auto").strip().lower()
if os.getenv("ENABLE_YOLO", "1").lower() in ("0", "false", "no"):
    DETECTOR_PREFERENCE = "hog"

# The weights were trained at 640 and that is what they are served at. The ONNX export has
# dynamic H/W so both modes share one session at whatever size each asks for.
YOLO_IMGSZ = int(os.getenv("YOLO_IMGSZ", str(TRAIN_IMGSZ)))
YOLO_CONF = float(os.getenv("YOLO_CONF", "0.35"))
TORCH_THREADS = int(os.getenv("TORCH_THREADS", "2"))

# Detection mode exists to show what the detector itself sees, so it runs at the training
# size and drops the confidence floor -- a weak "mask" reading is information here, whereas
# in recognition mode it would just be a box that fails to identify.
DETECTION_IMGSZ = int(os.getenv("DETECTION_IMGSZ", str(YOLO_IMGSZ)))
DETECTION_CONF = float(os.getenv("DETECTION_CONF", "0.25"))

# The two things this demo can do with one camera frame.
#
#   recognition  YOLO locates faces -> dlib embeds each one -> nearest neighbour in the
#                gallery. Answers "who is this?". The HOG fallback is allowed, because a
#                face found by any means is still a face worth identifying.
#   detection    YOLO only, nothing else. Answers "what is in frame?" -- which covering,
#                and is there a weapon. No embedding, no gallery, no HOG: the point is to
#                show the detector's own output, so a miss should read as a miss.
MODE_RECOGNITION = "recognition"
MODE_DETECTION = "detection"
MODES = (MODE_RECOGNITION, MODE_DETECTION)
DEFAULT_MODE = os.getenv("DEFAULT_MODE", MODE_RECOGNITION).strip().lower()
if DEFAULT_MODE not in MODES:
    DEFAULT_MODE = MODE_RECOGNITION

# Guardrails so one busy frame cannot pin the CPU for seconds. Embedding a face is the
# expensive step, so the face cap is tighter than the object cap.
MAX_FACES = int(os.getenv("MAX_FACES", "4"))
MAX_OBJECTS = int(os.getenv("MAX_OBJECTS", "8"))
MAX_FRAME_WIDTH = int(os.getenv("MAX_FRAME_WIDTH", "960"))

# Classes from the detector that are faces (and therefore worth identifying). Anything else
# it reports -- currently just "weapon" -- is surfaced as a scene object.
FACE_CLASSES = {"no_mask", "mask", "other_coverings", "face"}

# How each face class is described in the UI, and whether the covering makes identity
# matching unreliable. "face" is what the HOG fallback emits: it localises a face but cannot
# say anything about a covering, so it must not claim the face is uncovered.
COVERAGE = {
    "no_mask": ("Uncovered", False),
    "mask": ("Mask", True),
    "other_coverings": ("Covered", True),
    "face": ("Coverage unknown", False),
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
        self.detector = None
        self.detector_classes: list[str] = []
        self.hog_detector: HogDetector | None = None
        self.shape_predictor = None
        self.face_encoder = None

        self.known_embeddings = np.empty((0, 128), dtype=np.float32)
        self.known_labels: list[str] = []
        self.identities: list[str] = []

        self._load_dlib()
        self._load_detector()
        self._load_gallery()

    # ---------------------------------------------------------------- loading
    def _load_dlib(self) -> None:
        self.hog_detector = HogDetector()

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
            # takes ~1.5s per face instead of ~80ms. The Dockerfile builds dlib against
            # OpenBLAS for exactly this reason.
            self.warnings.append("dlib built without BLAS: face encoding will be slow")

    def _load_detector(self) -> None:
        self.detector, warnings = build_detector(
            onnx_path=YOLO_ONNX_PATH,
            torch_path=YOLO_MODEL_PATH,
            imgsz=YOLO_IMGSZ,
            conf=YOLO_CONF,
            threads=TORCH_THREADS,
            preference=DETECTOR_PREFERENCE,
        )
        self.warnings.extend(warnings)
        self.detector_backend = self.detector.name
        self.detector_classes = list(self.detector.classes)

        # Warm the graph so the first visitor does not pay for the first allocation. Both
        # sizes, because switching mode mid-session would otherwise pay it again.
        for size in dict.fromkeys((YOLO_IMGSZ, DETECTION_IMGSZ)):
            try:
                self.detector(np.zeros((size, size, 3), dtype=np.uint8), imgsz=size)
            except Exception as exc:  # pragma: no cover - a broken backend is worth reporting
                self.warnings.append(f"detector warm-up failed at {size}px ({exc})")

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
            "detectorClasses": self.detector_classes,
            "faceClasses": [c for c in self.detector_classes if c in FACE_CLASSES],
            "objectClasses": [c for c in self.detector_classes if c not in FACE_CLASSES],
            "recognitionReady": self.recognition_ready,
            "identities": self.identities,
            "gallerySize": int(len(self.known_labels)),
            "defaultThreshold": DEFAULT_THRESHOLD,
            "embeddingDim": int(self.known_embeddings.shape[1]) if len(self.known_labels) else 0,
            "maxFaces": MAX_FACES,
            "blas": bool(getattr(dlib, "DLIB_USE_BLAS", False)),
            "modes": list(MODES),
            "defaultMode": DEFAULT_MODE,
            # Per-mode input size and confidence floor, so the page can explain why the two
            # modes behave differently instead of leaving it to be guessed at.
            "modeConfig": {
                MODE_RECOGNITION: {"imgsz": YOLO_IMGSZ, "conf": YOLO_CONF, "recognises": True},
                MODE_DETECTION: {
                    "imgsz": DETECTION_IMGSZ,
                    "conf": DETECTION_CONF,
                    "recognises": False,
                },
            },
            "resizable": bool(getattr(self.detector, "resizable", False)),
            "trainImgsz": TRAIN_IMGSZ,
            "warnings": self.warnings,
        }

    # ---------------------------------------------------------------- inference
    def detect(
        self,
        frame_bgr: np.ndarray,
        threshold: float | None = None,
        mode: str | None = None,
    ) -> dict[str, Any]:
        """Run one frame through the pipeline selected by `mode`.

        recognition -- detect, then embed and match every face.
        detection   -- detect only, and report the class the detector assigned.

        Boxes come back both in pixels (relative to the frame actually analysed) and
        normalised to 0..1, so the browser can scale them onto whatever size it renders the
        video at.
        """
        if mode not in MODES:
            mode = DEFAULT_MODE
        if threshold is None:
            threshold = DEFAULT_THRESHOLD
        threshold = float(max(0.20, min(0.90, threshold)))

        imgsz = DETECTION_IMGSZ if mode == MODE_DETECTION else YOLO_IMGSZ
        conf = DETECTION_CONF if mode == MODE_DETECTION else YOLO_CONF

        started = time.perf_counter()
        frame_bgr = self._downscale(frame_bgr)
        height, width = frame_bgr.shape[:2]

        with self._lock:
            detect_started = time.perf_counter()
            face_boxes, objects = self._detect(frame_bgr, mode=mode, imgsz=imgsz, conf=conf)
            detect_ms = (time.perf_counter() - detect_started) * 1000.0

            recognise_started = time.perf_counter()
            if mode == MODE_RECOGNITION:
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                faces = [self._describe(frame_rgb, entry, threshold) for entry in face_boxes]
            else:
                # Detection mode never touches dlib -- that is the whole point, and it is
                # what pays for running the detector at its full training size.
                faces = [self._classify(entry) for entry in face_boxes]
            recognise_ms = (time.perf_counter() - recognise_started) * 1000.0

        for item in (*faces, *objects):
            x1, y1, x2, y2 = item["box"]
            item["boxNorm"] = [x1 / width, y1 / height, x2 / width, y2 / height]

        weapons = sum(1 for obj in objects if obj["label"] == "weapon")
        covered = sum(1 for face in faces if face["covered"])
        unknown = sum(1 for face in faces if face["authorized"] is False)

        # A weapon dominates either way. Failing that, the thing each mode is watching for:
        # an unrecognised face when identifying, a covered face when detecting.
        if weapons:
            threat = "high"
        elif mode == MODE_RECOGNITION and unknown:
            threat = "elevated"
        elif mode == MODE_DETECTION and covered:
            threat = "elevated"
        else:
            threat = "low"

        # Per-class tally over everything the detector returned. This is the direct answer to
        # "is the mask class firing at all?", which is otherwise buried in the box list.
        class_counts: dict[str, int] = {name: 0 for name in self.detector_classes}
        for item in (*faces, *objects):
            label = item.get("cls") or item.get("label")
            if label:
                class_counts[label] = class_counts.get(label, 0) + 1

        return {
            "faces": faces,
            "objects": objects,
            "mode": mode,
            "detector": self.detector_backend,
            "threshold": threshold,
            "imgsz": imgsz,
            "conf": conf,
            "threat": threat,
            "frame": {"width": width, "height": height},
            "counts": {
                "total": len(faces),
                "authorized": sum(1 for face in faces if face["authorized"] is True),
                "unknown": unknown,
                "covered": covered,
                "weapons": weapons,
                "objects": len(objects),
            },
            "classCounts": class_counts,
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
        self,
        frame_bgr: np.ndarray,
        *,
        mode: str,
        imgsz: int,
        conf: float,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Split the detector's output into faces (to identify) and other objects."""
        height, width = frame_bgr.shape[:2]
        faces: list[dict[str, Any]] = []
        objects: list[dict[str, Any]] = []

        try:
            detections = self.detector(frame_bgr, imgsz=imgsz, conf=conf)
        except Exception:
            detections = []

        for detection in detections:
            clipped = self._clip(detection.box, width, height)
            if clipped is None:
                continue
            if detection.label in FACE_CLASSES:
                faces.append(
                    {
                        "box": list(clipped),
                        "cls": detection.label,
                        "detectScore": detection.score,
                        "source": self.detector_backend,
                    }
                )
            else:
                objects.append(
                    {
                        "box": list(clipped),
                        "label": detection.label,
                        "score": round(detection.score, 3) if detection.score is not None else None,
                        "source": self.detector_backend,
                    }
                )

        # The weights were trained on covered faces, so a plain uncovered face at an odd
        # angle is the case they most often miss. HOG is cheap enough to be worth a second
        # pass when the detector came back with nothing -- but only when we are identifying.
        # Detection mode deliberately skips it: a HOG box says nothing about coverings, so
        # padding the results with one would misrepresent what the detector actually saw.
        if (
            mode == MODE_RECOGNITION
            and not faces
            and self.hog_detector is not None
            and not isinstance(self.detector, HogDetector)
        ):
            for detection in self.hog_detector(frame_bgr):
                clipped = self._clip(detection.box, width, height)
                if clipped is not None:
                    faces.append(
                        {
                            "box": list(clipped),
                            "cls": "face",
                            "detectScore": None,
                            "source": self.hog_detector.name,
                        }
                    )

        # Largest first, so the caps drop background bystanders rather than whoever is
        # actually standing in front of the camera.
        faces.sort(key=lambda item: _area(tuple(item["box"])), reverse=True)
        objects.sort(key=lambda item: item["score"] or 0.0, reverse=True)
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

    @staticmethod
    def _base_face(entry: dict[str, Any]) -> dict[str, Any]:
        """The part of a face record that needs no embedding: box, class, covering."""
        coverage_label, covered = COVERAGE.get(entry["cls"], ("Face", False))
        return {
            "box": entry["box"],
            "cls": entry["cls"],
            "coverage": coverage_label,
            "covered": covered,
            "detectScore": round(entry["detectScore"], 3)
            if entry["detectScore"] is not None
            else None,
            "source": entry.get("source"),
            "name": None,
            # None means "not attempted", which is what detection mode reports. False means
            # "attempted and no gallery match". The counters rely on the difference.
            "authorized": None,
            "distance": None,
            "confidence": None,
        }

    def _classify(self, entry: dict[str, Any]) -> dict[str, Any]:
        """Detection mode: report the class the detector assigned and stop there."""
        return self._base_face(entry)

    def _describe(
        self, frame_rgb: np.ndarray, entry: dict[str, Any], threshold: float
    ) -> dict[str, Any]:
        x1, y1, x2, y2 = entry["box"]
        face = self._base_face(entry)
        face["name"] = "Unknown"
        face["authorized"] = False

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

