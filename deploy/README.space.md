---
title: Face Recognition Live Demo
emoji: 👁️
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Live webcam face recognition, mask and weapon detection
---

# Face Recognition & Threat Detection — Live Demo

Allow the camera and the model runs on your own webcam feed, in your browser. No account, no
login. There is also a photo-upload fallback if you would rather not turn a camera on.

Two stages:

1. **Detect** — custom YOLOv8 weights locate faces, classify what is covering them
   (`no_mask` / `mask` / `other_coverings`), and flag weapons in frame.
2. **Identify** — dlib's 68-point landmark predictor aligns each face, the ResNet encoder
   maps it to a 128-d embedding, and a nearest-neighbour search matches it against the
   enrolled gallery.

Frames are captured in your browser, POSTed as JPEGs, decoded in memory, and discarded.
Nothing is stored.

Evaluation on 55 verification pairs: **ROC AUC 0.9857**, threshold 0.4020, precision 0.8333,
recall 1.0000, F1 0.9091, accuracy 0.9273. The threshold is a slider in the UI, so you can
watch precision trade against recall live.

Cold start takes about a minute while ~130 MB of weights load — the page will say "warming up
the model…" until it's ready.

**This is a demonstration, not a security product.** The gallery is small and the evaluation
set is smaller; the report itself shows 4 false positives. Covered faces are labelled but
their embeddings are unreliable by construction, and the weapon class is a detector output
that fires on shapes, not a judgement.

Full write-up, API docs, and source:
[github.com/victorakor/face-recognition-live-demo](https://github.com/victorakor/face-recognition-live-demo)
