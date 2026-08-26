# Face Recognition & Threat Detection — Live Demo

A browser demo of a two-stage vision pipeline: custom YOLOv8 weights find faces, classify
what is covering them, and flag weapons; dlib's ResNet encoder then turns each face into a
128-d embedding and matches it against an enrolled gallery.

**Live:** [Hugging Face Space](https://huggingface.co/spaces/victorakor/face-recognition-live-demo)
· **Source:** [github.com/victorakor/face-recognition-live-demo](https://github.com/victorakor/face-recognition-live-demo)

No accounts, no login, no database. Open the page, allow the camera, and the model runs on
your own webcam feed. If you would rather not turn a camera on, there is a photo-upload
fallback that runs the identical pipeline on a still image.

---

## How it works

The original version of this project read the camera on the server with
`cv2.VideoCapture(0)`, which only works when the server *is* your machine. A hosted demo has
no camera, so the flow is inverted:

```
browser                                          server
───────                                          ──────
getUserMedia()  ──▶ <video>
       │
       ├─▶ offscreen canvas, 640 px wide
       │        └─▶ JPEG (q=0.72)  ──── POST /api/detect ───▶  decode
       │                                                        │
       │                                                   YOLOv8 (imgsz 480)
       │                                                        ├─ faces ─▶ dlib landmarks
       │                                                        │            └─▶ 128-d embedding
       │                                                        │                 └─▶ nearest neighbour
       │                                                        └─ weapons ─▶ threat level
       │                                                        │
       ◀──────────── JSON: boxes (normalised 0..1), names, distances, timings
       │
       └─▶ overlay <canvas>, drawn on requestAnimationFrame
```

Details worth knowing:

- **Frames are never stored.** Each JPEG is decoded in memory, run through the model, and
  dropped. Nothing is written to disk and nothing is logged beyond an access line.
- **The request loop is sequential**, not on a timer. The next frame is captured only after
  the previous response lands, so a slow server produces a lower frame rate instead of a
  growing queue.
- **Boxes come back normalised to 0..1**, so the client scales them onto whatever size it is
  rendering the video at. The viewport's `aspect-ratio` is pinned to the camera's real
  aspect ratio, which is what keeps the overlay aligned.
- **Mirroring is display-only.** The preview is flipped (`scaleX(-1)`) because that is what
  people expect of a selfie view, and the drawing code flips the x coordinates to match. The
  model always sees the un-mirrored frame.

## The model

| Stage | Model | Notes |
| --- | --- | --- |
| Detection | YOLOv8, custom weights (`models/best.pt`) | Classes: `no_mask`, `mask`, `other_coverings`, `weapon` |
| Alignment | `shape_predictor_68_face_landmarks.dat` | Stock dlib, fetched at build time |
| Embedding | `dlib_face_recognition_resnet_model_v1.dat` | Stock dlib, 128-d output |
| Matching | Nearest neighbour, euclidean | Accept below the evaluated threshold |

Only the face classes are sent through recognition. A `weapon` box is a scene object, not a
face — running an identity match on one would be meaningless, so the pipeline keeps the two
lists separate and lets weapons drive the threat level instead.

### Evaluation

From the threshold sweep in [`docs/evaluation_report.txt`](docs/evaluation_report.txt),
55 verification pairs:

| Metric | Value |
| --- | --- |
| ROC AUC | 0.9857 |
| Distance threshold | 0.4020 |
| Precision | 0.8333 |
| Recall | 1.0000 |
| F1 | 0.9091 |
| Accuracy | 0.9273 |
| Confusion | TP 20 · FN 0 · FP 4 · TN 31 |

The threshold is exposed as a slider in the UI so you can watch precision trade against
recall directly. 55 pairs is a small evaluation set — treat these numbers as indicative of
the pipeline working, not as a production benchmark.

## API

| Endpoint | Purpose |
| --- | --- |
| `GET /` | The demo page |
| `GET /api/health` | `{"ok": true, "ready": bool}` — never blocks on model loading |
| `GET /api/info` | Detector backend, enrolled identities, class list, evaluation metrics |
| `POST /api/detect` | One frame in, detections out |

`POST /api/detect` accepts a raw JPEG/PNG body, a multipart file, or
`{"image": "data:image/jpeg;base64,...", "threshold": 0.402}` as JSON. `threshold` may also
be passed as a query parameter or an `X-Threshold` header, and is clamped to 0.20–0.90.

```bash
curl -s -X POST --data-binary @face.jpg \
     -H 'Content-Type: image/jpeg' \
     'http://localhost:7860/api/detect?threshold=0.402'
```

```json
{
  "faces": [{
    "box": [398, 90, 641, 261],
    "boxNorm": [0.4146, 0.125, 0.6677, 0.3625],
    "cls": "no_mask",
    "coverage": "Uncovered",
    "covered": false,
    "detectScore": 0.882,
    "name": "Barack",
    "authorized": true,
    "distance": 0.3139,
    "confidence": 0.812
  }],
  "objects": [],
  "detector": "yolov8-custom",
  "threat": "low",
  "threshold": 0.402,
  "frame": {"width": 960, "height": 720},
  "counts": {"total": 1, "authorized": 1, "unknown": 0, "covered": 0, "weapons": 0},
  "timings": {"detectMs": 254.8, "recognizeMs": 87.4, "totalMs": 344.1}
}
```

The server caps concurrency at 3 in-flight detections and rate-limits per IP (24 burst,
12/s sustained). Over either limit it returns 429/503 with `"retry": true`, and the client
backs off for a beat rather than piling on.

## Running it locally

Docker is the path that matches production:

```bash
docker build -t face-demo .
docker run --rm -p 7860:7860 face-demo
# http://localhost:7860
```

Without Docker:

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python scripts/fetch_models.py        # downloads the two stock dlib models (~120 MB)
python app.py
```

`getUserMedia` needs a secure context. `localhost` counts as one; a bare LAN IP over plain
HTTP does not, so use the deployed URL (or a tunnel) to test from a phone.

### A note on dlib and BLAS

`requirements.txt` pins `dlib-bin`, a prebuilt wheel, so a local checkout needs no compiler.
That wheel is built **without BLAS**, and dlib's ResNet falls back to a naive matmul —
roughly **1.5 s per face instead of ~80 ms**. The Dockerfile therefore compiles dlib from
source against OpenBLAS. If that build ever fails it falls back to the wheel rather than
failing the deploy, and the app says which one it got under "Model notes" on the page.

### Environment variables

| Variable | Default | Effect |
| --- | --- | --- |
| `PORT` | `7860` | Bind port |
| `BEST_THRESHOLD` | `0.402` | Default match threshold |
| `ENABLE_YOLO` | `1` | `0` drops torch entirely and uses dlib's HOG detector — much less memory, but no covering or weapon classes |
| `YOLO_IMGSZ` | `480` | Detector input size. 640 is ~5x slower here for no measurable confidence gain |
| `YOLO_CONF` | `0.35` | Detection confidence floor |
| `MAX_FACES` | `4` | Faces embedded per frame |
| `MAX_FRAME_WIDTH` | `960` | Frames wider than this are downscaled before inference |
| `MAX_IN_FLIGHT` | `3` | Concurrent detections before 503 |
| `TORCH_THREADS` | `2` | Keeps CPU inference from oversubscribing a small container |

## Enrolling your own faces

`models/known_faces.pkl` is a pickle of `{"embeddings": (N, 128) float32, "labels": [str]}`.
Build embeddings with the same framing the pipeline uses — full frame, real face rectangle:

```python
shape = shape_predictor(frame_rgb, dlib.rectangle(x1, y1, x2, y2))
embedding = face_encoder.compute_face_descriptor(frame_rgb, shape)
```

Encoding a pre-cropped chip with a full-crop rectangle produces embeddings that will not
match the ones this gallery was built from.

## Deployment

Both targets build the same `Dockerfile`:

- **Hugging Face Spaces** — `sdk: docker`, `app_port: 7860`. The generous memory ceiling on
  the free tier is what makes the full torch + dlib + YOLO stack viable.
- **Render** — Docker runtime, see [`render.yaml`](render.yaml). The 512 MB free tier is
  tight for this stack; if it gets OOM-killed, set `ENABLE_YOLO=0` to trade the covering and
  weapon classes for a much smaller resident set.

Cold start is roughly a minute while ~130 MB of weights load. The page polls `/api/info` and
shows "warming up the model…" until they are up, and `/api/health` answers immediately
throughout so platform health checks don't kill the container mid-load.

## Limitations

This is a demonstration, not a security product.

- The gallery is small, and the 55-pair evaluation is smaller. Expect false positives at the
  default threshold — the report shows 4 of them.
- Covered faces are detected and labelled, but their embeddings are unreliable by
  construction; the UI marks them rather than pretending otherwise.
- The weapon class is a detector output, not a judgement. It fires on shapes, and it will be
  wrong.
- Accuracy varies with lighting, pose, and camera quality in ways a 55-sample evaluation
  cannot capture.

## Licence

Code is MIT. The two stock dlib models are redistributed under their own upstream terms; the
face-landmark predictor in particular is licensed for research use, which is why it is
fetched at build time rather than vendored here.
