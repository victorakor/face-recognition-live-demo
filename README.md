# Face Recognition & Threat Detection — Live Demo

A browser demo of a two-stage vision pipeline: custom YOLOv8 weights find faces, classify
what is covering them, and flag weapons; dlib's ResNet encoder then turns each face into a
128-d embedding and matches it against an enrolled gallery.

**Live:** [face-recognition-live-demo.onrender.com](https://face-recognition-live-demo.onrender.com)
· **Source:** [github.com/victorakor/face-recognition-live-demo](https://github.com/victorakor/face-recognition-live-demo)

> Hosted on Render's free tier, which spins the container down after 15 minutes idle. If it's
> been quiet, the first request takes 30–60 s to wake and load the weights — the page shows
> "warming up the model…" while that happens.

No accounts, no login, no database. Open the page, allow the camera, and the model runs on
your own webcam feed. If you would rather not turn a camera on, there is a photo-upload
fallback that runs the identical pipeline on a still image.

A switch on the page picks which of the two pipelines runs on each frame — **Recognise
faces** (who is this?) or **Detect masks & weapons** (what is in frame?). See
[Two modes](#two-modes).

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
       │                                                   YOLOv8 (imgsz 640)
       │                                                        ├─ faces ─┬─ mode=detection ─▶ class + score, done
       │                                                        │         └─ mode=recognition ─▶ dlib landmarks
       │                                                        │                                └─▶ 128-d embedding
       │                                                        │                                     └─▶ nearest neighbour
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

## Two modes

Both modes run the same detector on the same frame. They differ in what happens next, and
that difference buys real accuracy.

| | Recognise faces | Detect masks & weapons |
| --- | --- | --- |
| `mode` | `recognition` (default) | `detection` |
| Answers | *Who is this?* | *What is in frame?* |
| Stages | detect → embed → match | detect only |
| Detector input | 640 | 640 |
| Confidence floor | 0.35 | 0.25 |
| HOG fallback | yes | **no** |
| Latency, local (no BLAS) | 1000–1900 ms | **124–160 ms** |
| Latency, live free tier | 714–1201 ms | **631–788 ms** |
| Threat escalates on | weapon, or unrecognised face | weapon, or covered face |

Recognition mode is *slower* locally than on the free tier because a pip checkout gets the
`dlib-bin` wheel, which is built without BLAS — see
[A note on dlib and BLAS](#a-note-on-dlib-and-blas). Detection mode never touches dlib, so it
is unaffected, and the gap between the two modes is correspondingly enormous locally.

Recognition mode reports a name, a distance and a match score per face. Detection mode
reports the detector's own class and confidence and nothing else — no embedding, no gallery
lookup, no name — plus a per-class tally in `classCounts` so you can see directly whether
the `mask` class is firing.

Two deliberate choices behind that table:

**Detection mode drops the HOG fallback.** In recognition mode, if YOLO finds no face at all,
dlib's HOG detector gets a second pass — a face found by any means is still worth
identifying. But a HOG box says nothing about what is covering the face, so padding detection
results with one would misrepresent what the detector actually saw. In detection mode a miss
reads as a miss. HOG boxes are labelled **"Coverage unknown"**, never "Uncovered": claiming
the latter made masked subjects show up as bare-faced whenever YOLO missed them, which is a
false statement about the frame, not just a UI wording choice.

**Detection mode lowers the confidence floor to 0.25.** A weak `mask` reading is information
when the question is "what is in frame?", whereas in recognition mode it would just be a box
that fails to identify.

### Serving resolution

The weights were trained at `imgsz=640` and are now served at 640. They were previously
exported at 480, which cost real recall on webcam-framed subjects. Measured on synthesised
640×360 frames with the subject scaled down, best score for the true class:

| Subject height | Class | @ 480 | @ 640 |
| --- | --- | --- | --- |
| 180 px | `no_mask` | 0.062 — **missed** | **0.856** |
| 180 px | `mask` | 0.514 | **0.715** |
| 125 px | `mask` | 0.003 — **missed** | 0.234 |
| any | `weapon` | 0.60–0.87 | 0.60–0.87 |

Weapons were never the problem — that class is robust at both sizes. Faces and coverings were,
and a covering the detector misses is exactly the case where the old HOG fallback would step
in and call it "Uncovered". Both halves of that bug are fixed.

`models/best.onnx` is exported with **dynamic height/width**, so one graph serves whatever
size each mode asks for. That is not a compromise: measured, the dynamic graph is *faster*
than a fixed-shape one at the same input size (127 ms vs 144 ms at 640; 71 ms vs 102 ms at
480) and numerically identical. Resident memory at 640 is 223 MB — unchanged from the 480
build, and well inside the 512 MB free instance.

## The model

| Stage | Model | Notes |
| --- | --- | --- |
| Detection | YOLOv8, custom weights (`models/best.onnx`) | Classes: `no_mask`, `mask`, `other_coverings`, `weapon` |
| Alignment | `shape_predictor_68_face_landmarks.dat` | Stock dlib, fetched at build time |
| Embedding | `dlib_face_recognition_resnet_model_v1.dat` | Stock dlib, 128-d output |
| Matching | Nearest neighbour, euclidean | Accept below the evaluated threshold |

Only the face classes are sent through recognition, and only in `recognition` mode. A
`weapon` box is a scene object, not a face — running an identity match on one would be
meaningless, so the pipeline keeps the two lists separate and lets weapons drive the threat
level instead.

The gallery holds 239 embeddings across 5 enrolled identities.

### Why ONNX instead of torch

`models/best.pt` is the trained checkpoint, and it is in the repo. But the server runs
`models/best.onnx` through onnxruntime instead, because importing torch + ultralytics costs
~400 MB resident and the free-tier target has 512 MB total. Measured, whole process:

| | torch + ultralytics | onnxruntime |
| --- | --- | --- |
| Resident memory | 600–800 MB | **223 MB** |
| Detection latency | 230–260 ms | **128–160 ms** |
| Cold start | ~46 s | **~25 s** |

Measured locally, whole process. The onnxruntime column is at `imgsz` 640; the torch column
was measured at 480, so if anything the comparison flatters torch — ONNX is faster while doing
1.8× the work.

Output is equivalent — same classes, boxes within a pixel, same identity decisions:

```
                  ONNX                                torch
barack.jpg  no_mask [378,141,607,415] 0.899     no_mask [379,142,608,414] 0.888
            -> Barack  d=0.3164                 -> Barack  d=0.3154
lena.jpg    face    [234,234,378,378]           face    [234,234,378,378]
            -> Unknown d=0.7663                 -> Unknown d=0.7663
```

The sub-pixel and score differences are onnxruntime's graph fusion reordering float ops, not
a logic difference. The trade is that [`detector.py`](detector.py) has to do the letterboxing
and non-maximum suppression ultralytics would otherwise do — it mirrors ultralytics' own
padding arithmetic, including the `round(pad - 0.1)`, because getting that wrong shifts every
box.

The torch path is still in the code (`DETECTOR=torch`) if you install
[`requirements-export.txt`](requirements-export.txt). Regenerate the ONNX graph after
retraining with `python scripts/export_onnx.py` — it defaults to dynamic axes traced at 640,
which is what the server expects. Exporting with `--no-dynamic` bakes one size into the graph;
`detector.py` detects that and warns, and both modes then run at whatever size was baked in.

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
| `GET /api/info` | Detector backend, enrolled identities, class list, per-mode config, evaluation metrics |
| `POST /api/detect` | One frame in, detections out |

`POST /api/detect` accepts a raw JPEG/PNG body, a multipart file, or
`{"image": "data:image/jpeg;base64,...", "threshold": 0.402, "mode": "detection"}` as JSON.

| Parameter | Where | Values |
| --- | --- | --- |
| `threshold` | query, `X-Threshold` header, or JSON body | clamped to 0.20–0.90 |
| `mode` | query, `X-Mode` header, or JSON body | `recognition` (default) or `detection` |

`mode` is case-insensitive, and anything unrecognised falls back to the default rather than
erroring — a typo in a query string should not break the demo.

```bash
curl -s -X POST --data-binary @face.jpg \
     -H 'Content-Type: image/jpeg' \
     'http://localhost:7860/api/detect?threshold=0.402&mode=recognition'
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
    "source": "yolov8-onnx",
    "name": "Barack",
    "authorized": true,
    "distance": 0.3139,
    "confidence": 0.812
  }],
  "objects": [],
  "mode": "recognition",
  "detector": "yolov8-onnx",
  "threat": "low",
  "threshold": 0.402,
  "imgsz": 640,
  "conf": 0.35,
  "frame": {"width": 960, "height": 720},
  "counts": {"total": 1, "authorized": 1, "unknown": 0, "covered": 0, "weapons": 0, "objects": 0},
  "classCounts": {"mask": 0, "no_mask": 1, "other_coverings": 0, "weapon": 0},
  "timings": {"detectMs": 254.8, "recognizeMs": 87.4, "totalMs": 344.1}
}
```

In `detection` mode the same shape comes back with `name`, `distance`, `confidence` and
`authorized` all `null` — nothing was attempted, which is distinct from "attempted and no
match" (`authorized: false`). `counts.authorized` and `counts.unknown` are therefore always 0
in that mode; read `classCounts` instead.

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
| `DEFAULT_MODE` | `recognition` | Which pipeline the page loads in — `recognition` or `detection` |
| `DETECTOR` | `auto` | `onnx`, `torch` or `hog` to force a backend. `auto` prefers ONNX, then torch, then HOG |
| `ENABLE_YOLO` | `1` | `0` is shorthand for `DETECTOR=hog` — smallest possible footprint, but no covering or weapon classes |
| `YOLO_CONF` | `0.35` | Confidence floor in recognition mode |
| `YOLO_IOU` | `0.7` | NMS IoU threshold (ultralytics' predict default) |
| `YOLO_IMGSZ` | `640` | Detector input size in recognition mode. Rounded to a multiple of 32 and clamped to 320–960 |
| `DETECTION_CONF` | `0.25` | Confidence floor in detection mode |
| `DETECTION_IMGSZ` | `640` | Detector input size in detection mode, same rounding |
| `MAX_FACES` | `4` | Faces embedded per frame |
| `MAX_OBJECTS` | `8` | Non-face detections returned per frame |
| `MAX_FRAME_WIDTH` | `960` | Frames wider than this are downscaled before inference |
| `MAX_IN_FLIGHT` | `3` | Concurrent detections before 503 |
| `TORCH_THREADS` | `2` | Inference thread count (applies to onnxruntime too, despite the name) |

Both `*_IMGSZ` variables only take effect on a backend that can be resized — the torch path,
or an ONNX graph exported with dynamic axes (the default). Against a fixed-shape graph they
are ignored and `/api/info` reports `"resizable": false`.

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

`render.yaml` is a Docker blueprint — point Render at this repo (New → Blueprint) and it
builds and deploys. The whole stack fits the 512 MB free instance because of the ONNX swap
above; nothing needs disabling. The build takes ~6 minutes, almost all of it compiling dlib.

Measured on the live free instance (0.1 CPU), which is the honest performance picture. Three
passes over three images per mode, `imgsz` 640:

| | Recognise faces | Detect masks & weapons |
| --- | --- | --- |
| Detection | 630–886 ms | 631–788 ms |
| Recognition (1 face) | 0–400 ms | — (skipped) |
| Total per frame | 714–1201 ms (≈0.8–1.4 fps) | **631–788 ms (≈1.3–1.6 fps)** |
| `blas` | `true` — the source build worked | `true` |

The spread is real: 0.1 CPU is a shared-core allocation, so the same request is ~40% slower
when a neighbour is busy or the container has just started. Expect the low end warm and the
high end for the first few frames after a wake-up. Add roughly 700–1900 ms of round-trip on
top for the network hop from a home connection.

Detection is slower than it was at `imgsz` 480 (464–592 ms) because 640 is 1.8× the pixels —
that is the cost of the recall the [resolution change](#serving-resolution) bought back.
Detection mode absorbs it by skipping the embedding entirely, which is why it ends up the
faster of the two despite the bigger input.

Recognition at ~200–400 ms rather than ~1000 ms is the entire payoff of compiling dlib against
OpenBLAS. Detection results are identical to local, down to the distance (0.3164).

Cold start is 25–40 s while ~130 MB of weights load, and the free instance spins down after 15
minutes idle, so the first request after a quiet period pays that again. The page polls
`/api/info` and shows "warming up the model…" until the models are up, while `/api/health`
answers immediately throughout so the platform health check can't kill the container mid-load.

Services created through the Render API have no GitHub webhook — that needs the Render GitHub
App and an interactive OAuth grant — so `autoDeploy` does not fire on push.
[`.github/workflows/deploy.yml`](.github/workflows/deploy.yml) covers that by calling the
Render API directly; add a `RENDER_API_KEY` repository secret to enable it. Or deploy by hand:

```bash
curl -X POST -H "Authorization: Bearer $RENDER_API_KEY" \
     -H 'Content-Type: application/json' -d '{"clearCache":"do_not_clear"}' \
     https://api.render.com/v1/services/srv-da7mmmvavr4c73b58ov0/deploys
```

**Hugging Face Spaces no longer works on the free tier** for this app. As of 2026, Docker and
Gradio Spaces on free `cpu-basic` require a PRO subscription — the API refuses the create call
outright:

> Static Spaces are free for everyone, but hosting Gradio and Docker Spaces on free cpu-basic
> requires a PRO subscription.

The Space files are ready if you do have PRO: push this repo to a Docker Space and replace
`README.md` with [`deploy/README.space.md`](deploy/README.space.md), which carries the
`sdk: docker` / `app_port: 7860` frontmatter the platform needs.

## Limitations

This is a demonstration, not a security product.

- The gallery is small, and the 55-pair evaluation is smaller. Expect false positives at the
  default threshold — the report shows 4 of them.
- Covered faces are detected and labelled, but their embeddings are unreliable by
  construction; the UI marks them rather than pretending otherwise.
- The weapon class is a detector output, not a judgement. It fires on shapes, and it will be
  wrong: a surgical mask lying on a table, with no face in frame, scores `weapon` 0.68.
- **The detector has a size floor.** Below roughly 90 px of subject height it stops finding
  faces at all, and coverings degrade before that (see the table under
  [Serving resolution](#serving-resolution)). Sit close enough to fill a reasonable part of
  the frame.
- Accuracy varies with lighting, pose, and camera quality in ways a 55-sample evaluation
  cannot capture.

## Licence

Code is MIT — see [`LICENSE`](LICENSE), which also covers the two stock dlib models (fetched
at build time rather than vendored, and under their own upstream terms) and the AGPL question
that comes with ultralytics-trained weights.
