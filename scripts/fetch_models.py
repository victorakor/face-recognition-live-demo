#!/usr/bin/env python3
"""Fetch the two stock dlib models the recogniser needs.

They are ~120 MB uncompressed, so they are downloaded at build time instead of being
committed. The custom pieces of this project -- the YOLO face detector weights
(models/best.pt) and the face gallery (models/known_faces.pkl) -- do live in the repo.

Usage:
    python scripts/fetch_models.py [--models-dir models] [--force]
"""

from __future__ import annotations

import argparse
import bz2
import hashlib
import os
import shutil
import sys
import tempfile
import urllib.error
import urllib.request

CHUNK = 1 << 20
USER_AGENT = "face-recognition-live-demo/1.0"

MODELS = {
    "shape_predictor_68_face_landmarks.dat": {
        "size": 99693937,
        "sha256": "fbdc2cb80eb9aa7a758672cbfdda32ba6300efe9b6e6c7a299ff7e736b11b92f",
        "urls": [
            "https://raw.githubusercontent.com/davisking/dlib-models/master/shape_predictor_68_face_landmarks.dat.bz2",
            "https://github.com/davisking/dlib-models/raw/master/shape_predictor_68_face_landmarks.dat.bz2",
            "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2",
        ],
    },
    "dlib_face_recognition_resnet_model_v1.dat": {
        "size": 22466066,
        "sha256": "55533b28a95800a551ba546ba62fe69625c7e95a7061c338adffead08719da30",
        "urls": [
            "https://raw.githubusercontent.com/davisking/dlib-models/master/dlib_face_recognition_resnet_model_v1.dat.bz2",
            "https://github.com/davisking/dlib-models/raw/master/dlib_face_recognition_resnet_model_v1.dat.bz2",
            "http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2",
        ],
    },
}


def sha256_of(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def already_valid(path: str, spec: dict) -> bool:
    return (
        os.path.exists(path)
        and os.path.getsize(path) == spec["size"]
        and sha256_of(path) == spec["sha256"]
    )


def download_and_decompress(url: str, destination: str) -> None:
    """Stream a .bz2 to a temp file next to the destination, then move it into place."""
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    decompressor = bz2.BZ2Decompressor()
    directory = os.path.dirname(destination) or "."
    handle = tempfile.NamedTemporaryFile(dir=directory, suffix=".part", delete=False)
    try:
        with urllib.request.urlopen(request, timeout=120) as response, handle:
            while True:
                block = response.read(CHUNK)
                if not block:
                    break
                handle.write(decompressor.decompress(block))
        shutil.move(handle.name, destination)
    except BaseException:
        if os.path.exists(handle.name):
            os.unlink(handle.name)
        raise


def fetch(name: str, spec: dict, models_dir: str, force: bool) -> None:
    destination = os.path.join(models_dir, name)

    if not force and already_valid(destination, spec):
        print(f"[skip]  {name} already present and verified")
        return

    for url in spec["urls"]:
        print(f"[fetch] {name} <- {url}")
        try:
            download_and_decompress(url, destination)
        except (urllib.error.URLError, OSError, EOFError) as exc:
            print(f"[warn]  {url} failed: {exc}")
            continue

        actual = os.path.getsize(destination)
        if actual != spec["size"]:
            print(f"[warn]  {name} wrong size: got {actual}, want {spec['size']}")
            os.unlink(destination)
            continue

        digest = sha256_of(destination)
        if digest != spec["sha256"]:
            print(f"[warn]  {name} checksum mismatch: {digest}")
            os.unlink(destination)
            continue

        print(f"[ok]    {name} ({actual} bytes, sha256 verified)")
        return

    raise SystemExit(f"[fatal] could not download {name} from any mirror")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-dir", default=os.getenv("MODELS_DIR", "models"))
    parser.add_argument("--force", action="store_true", help="re-download even if verified")
    args = parser.parse_args()

    os.makedirs(args.models_dir, exist_ok=True)
    for name, spec in MODELS.items():
        fetch(name, spec, args.models_dir, args.force)
    return 0


if __name__ == "__main__":
    sys.exit(main())
