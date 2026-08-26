/* Live face-recognition demo.
 *
 * The browser owns the camera. Frames are captured to an offscreen canvas, sent to
 * /api/detect one at a time, and the returned boxes are painted onto an overlay canvas.
 * Sending frames sequentially (rather than on a fixed timer) gives natural backpressure:
 * a slow server means a lower frame rate, never a growing queue.
 */
(() => {
  'use strict';

  const CAPTURE_WIDTH = 640;      // px sent to the model
  const JPEG_QUALITY = 0.72;
  const TARGET_INTERVAL_MS = 150; // ~6.7 fps ceiling
  const BACKOFF_MS = 700;         // after a 429/503/network error
  const STALE_MS = 1200;          // start fading boxes once a result is this old
  const ALERT_COOLDOWN_MS = 3000;
  const LOG_LIMIT = 40;

  const COLORS = {
    known: '#10b981',
    unknown: '#ef4444',
    covered: '#f59e0b',
    object: '#fb7185',
  };

  const el = (id) => document.getElementById(id);
  const video = el('video');
  const overlay = el('overlay');
  const viewport = el('viewport');
  const ctx = overlay.getContext('2d');
  const capture = document.createElement('canvas');
  const captureCtx = capture.getContext('2d');

  const state = {
    stream: null,
    running: false,
    facingMode: 'user',
    mirror: true,
    stillImage: null,
    result: null,
    resultAt: 0,
    threshold: parseFloat(el('threshold').value),
    frameTimes: [],
    lastAlertAt: 0,
    identities: [],
    seen: new Set(),
    audio: null,
    dpr: 1,
  };

  const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

  /* -------------------------------------------------------------- model info */

  async function loadInfo() {
    try {
      const response = await fetch('/api/info');
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const payload = await response.json();

      // A cold free-tier container spends ~a minute loading ~120 MB of weights. The
      // server answers immediately with ready:false while that happens.
      if (!payload.ready) {
        el('backendPill').textContent = 'warming up the model…';
        setTimeout(loadInfo, 2500);
        return;
      }

      const { model, evaluation } = payload;

      const label = {
        'yolov8-custom': 'YOLOv8 (custom) + dlib ResNet',
        'dlib-hog': 'dlib HOG + dlib ResNet',
      }[model.detector] || model.detector;
      const pill = el('backendPill');
      pill.textContent = label;
      pill.classList.toggle('ready', Boolean(model.recognitionReady));

      state.identities = model.identities || [];
      renderIdentities();
      el('galleryCount').textContent = `· ${model.gallerySize} embeddings`;

      renderChips(
        el('classChips'),
        (model.detectorClasses || []).map((name) => ({ text: name, className: 'chip cls' })),
        'Detector class list unavailable (dlib fallback).'
      );

      el('mAuc').textContent = evaluation.rocAuc.toFixed(4);
      el('mRecall').textContent = evaluation.recall.toFixed(2);
      el('mPrecision').textContent = evaluation.precision.toFixed(3);
      el('mF1').textContent = evaluation.f1.toFixed(3);
      el('mAccuracy').textContent = evaluation.accuracy.toFixed(3);
      el('mSamples').textContent = String(evaluation.samples);

      if (model.warnings && model.warnings.length) {
        const node = el('warnings');
        node.textContent = `Model notes: ${model.warnings.join(' · ')}`;
        node.hidden = false;
      }
    } catch (error) {
      el('backendPill').textContent = 'model info unavailable — retrying';
      console.warn('could not load /api/info', error);
      setTimeout(loadInfo, 5000);
    }
  }

  function renderChips(container, items, emptyText) {
    container.innerHTML = '';
    if (!items.length) {
      container.innerHTML = `<span class="muted">${emptyText}</span>`;
      return;
    }
    for (const item of items) {
      const chip = document.createElement('span');
      chip.className = item.className;
      chip.textContent = item.text;
      container.appendChild(chip);
    }
  }

  function renderIdentities() {
    renderChips(
      el('identityChips'),
      state.identities.map((name) => ({
        text: name,
        className: state.seen.has(name) ? 'chip seen' : 'chip',
      })),
      'No identities enrolled.'
    );
  }

  /* ------------------------------------------------------------------ camera */

  async function startCamera() {
    const startBtn = el('startBtn');

    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      showGateError('This browser does not expose getUserMedia. Camera capture needs a modern browser served over HTTPS.');
      return;
    }
    if (!window.isSecureContext) {
      showGateError('Camera access requires a secure context (HTTPS, or localhost during development).');
      return;
    }

    startBtn.disabled = true;
    startBtn.textContent = 'Requesting access…';

    try {
      state.stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: state.facingMode, width: { ideal: 1280 }, height: { ideal: 720 } },
        audio: false,
      });
    } catch (error) {
      startBtn.disabled = false;
      startBtn.textContent = 'Enable camera';
      showGateError(describeCameraError(error));
      return;
    }

    state.mirror = state.facingMode === 'user';
    video.classList.toggle('no-mirror', !state.mirror);
    video.srcObject = state.stream;
    await video.play().catch(() => {});
    await waitForVideoDimensions();

    state.stillImage = null;
    state.result = null;
    el('gate').hidden = true;
    el('gateError').hidden = true;
    el('toggleBtn').disabled = false;
    el('toggleBtn').textContent = 'Pause';
    for (const id of ['liveBadge', 'threatBadge', 'latencyBadge', 'fpsBadge']) el(id).hidden = false;
    ensureAudio();

    // Offer a camera switch only when the device actually has more than one.
    try {
      const devices = await navigator.mediaDevices.enumerateDevices();
      el('switchBtn').hidden = devices.filter((d) => d.kind === 'videoinput').length < 2;
    } catch { /* enumerateDevices can reject on locked-down browsers */ }

    syncViewportAspect(video.videoWidth, video.videoHeight);
    state.running = true;
    detectLoop();
  }

  function showGateError(message) {
    const node = el('gateError');
    node.textContent = message;
    node.hidden = false;
    el('gate').hidden = false;
  }

  function describeCameraError(error) {
    switch (error && error.name) {
      case 'NotAllowedError':
      case 'SecurityError':
        return 'Camera permission was denied. Allow camera access for this site in your browser settings and reload — or analyse a photo instead.';
      case 'NotFoundError':
      case 'OverconstrainedError':
        return 'No camera was found on this device. You can still analyse a photo instead.';
      case 'NotReadableError':
      case 'AbortError':
        return 'The camera is already in use by another application. Close it and try again.';
      default:
        return `Could not start the camera: ${(error && error.message) || error}`;
    }
  }

  function waitForVideoDimensions() {
    if (video.videoWidth && video.videoHeight) return Promise.resolve();
    return new Promise((resolve) => {
      const done = () => {
        video.removeEventListener('loadedmetadata', done);
        resolve();
      };
      video.addEventListener('loadedmetadata', done);
      setTimeout(done, 3000);
    });
  }

  function syncViewportAspect(width, height) {
    // Match the container to the source aspect ratio so normalised box coordinates land
    // exactly on the rendered pixels, whatever resolution the camera picked.
    if (width > 0 && height > 0) viewport.style.aspectRatio = `${width} / ${height}`;
    resizeOverlay();
  }

  async function switchCamera() {
    state.facingMode = state.facingMode === 'user' ? 'environment' : 'user';
    stopCamera();
    el('startBtn').disabled = false;
    el('startBtn').textContent = 'Enable camera';
    await startCamera();
  }

  function stopCamera() {
    state.running = false;
    if (state.stream) {
      state.stream.getTracks().forEach((track) => track.stop());
      state.stream = null;
    }
    video.srcObject = null;
  }

  /* --------------------------------------------------------------- inference */

  function grabFrame() {
    const sourceWidth = video.videoWidth;
    const sourceHeight = video.videoHeight;
    if (!sourceWidth || !sourceHeight) return null;

    const scale = Math.min(1, CAPTURE_WIDTH / sourceWidth);
    capture.width = Math.round(sourceWidth * scale);
    capture.height = Math.round(sourceHeight * scale);
    // Draw un-mirrored: the model should see the real orientation. Mirroring is a
    // display-only concern, undone when the boxes are painted.
    captureCtx.drawImage(video, 0, 0, capture.width, capture.height);
    return new Promise((resolve) => capture.toBlob(resolve, 'image/jpeg', JPEG_QUALITY));
  }

  async function postFrame(blob) {
    const response = await fetch(`/api/detect?threshold=${state.threshold.toFixed(3)}`, {
      method: 'POST',
      headers: { 'Content-Type': 'image/jpeg' },
      body: blob,
    });
    if (response.status === 429 || response.status === 503) return { retry: true };
    if (!response.ok) {
      let detail = `HTTP ${response.status}`;
      try {
        const body = await response.json();
        if (body.error) detail = body.error;
      } catch { /* non-JSON error body */ }
      throw new Error(detail);
    }
    return response.json();
  }

  async function detectLoop() {
    while (state.running) {
      if (document.hidden) {
        await sleep(400);
        continue;
      }

      const started = performance.now();
      try {
        const blob = await grabFrame();
        if (!blob) { await sleep(120); continue; }

        const result = await postFrame(blob);
        if (result.retry) { await sleep(BACKOFF_MS); continue; }

        applyResult(result);
        trackFps();
      } catch (error) {
        console.warn('detect failed', error);
        el('latencyBadge').textContent = 'reconnecting…';
        await sleep(BACKOFF_MS);
        continue;
      }

      const elapsed = performance.now() - started;
      if (elapsed < TARGET_INTERVAL_MS) await sleep(TARGET_INTERVAL_MS - elapsed);
    }
  }

  function trackFps() {
    const now = performance.now();
    state.frameTimes.push(now);
    while (state.frameTimes.length && now - state.frameTimes[0] > 3000) state.frameTimes.shift();
    const fps = state.frameTimes.length > 1
      ? ((state.frameTimes.length - 1) / (now - state.frameTimes[0])) * 1000
      : 0;
    el('fpsBadge').textContent = `${fps.toFixed(1)} fps`;
  }

  function applyResult(result) {
    state.result = result;
    state.resultAt = performance.now();

    const ms = Math.round(result.timings.totalMs);
    el('statFaces').textContent = String(result.counts.total);
    el('statKnown').textContent = String(result.counts.authorized);
    el('statUnknown').textContent = String(result.counts.unknown);
    el('statCovered').textContent = String(result.counts.covered);
    el('statWeapons').textContent = String(result.counts.weapons);
    el('statLatency').textContent = `${ms} ms`;
    el('latencyBadge').textContent = `${ms} ms`;

    const badge = el('threatBadge');
    badge.textContent = `threat: ${result.threat}`;
    badge.className = `badge threat-${result.threat}`;

    let discovered = false;
    for (const face of result.faces) {
      if (face.authorized && !state.seen.has(face.name)) {
        state.seen.add(face.name);
        discovered = true;
      }
    }
    if (discovered) renderIdentities();

    logResult(result);
    if (result.counts.unknown > 0 || result.counts.weapons > 0) maybeAlert();
  }

  /* --------------------------------------------------------------- rendering */

  function resizeOverlay() {
    state.dpr = Math.min(window.devicePixelRatio || 1, 2);
    const rect = overlay.getBoundingClientRect();
    if (!rect.width || !rect.height) return;
    overlay.width = Math.round(rect.width * state.dpr);
    overlay.height = Math.round(rect.height * state.dpr);
    ctx.setTransform(state.dpr, 0, 0, state.dpr, 0, 0);
  }

  function draw() {
    requestAnimationFrame(draw);

    const width = overlay.width / state.dpr;
    const height = overlay.height / state.dpr;
    if (!width || !height) return;
    ctx.clearRect(0, 0, width, height);

    if (state.stillImage) ctx.drawImage(state.stillImage, 0, 0, width, height);

    const result = state.result;
    if (!result) return;

    // Live video moves on while a request is in flight, so let stale boxes fade rather
    // than sit confidently over the wrong pixels.
    const age = performance.now() - state.resultAt;
    ctx.globalAlpha = state.stillImage ? 1 : Math.max(0.25, 1 - Math.max(0, age - STALE_MS) / 2000);

    const lineWidth = Math.max(2, Math.round(width / 320));
    const fontSize = Math.max(12, Math.round(width / 42));

    for (const object of result.objects || []) {
      drawBox(object.boxNorm, width, height, {
        color: COLORS.object,
        lineWidth,
        fontSize,
        dashed: true,
        label: `${object.label} ${(object.score * 100).toFixed(0)}%`,
      });
    }

    for (const face of result.faces) {
      const color = face.authorized ? COLORS.known : (face.covered ? COLORS.covered : COLORS.unknown);
      const parts = [face.authorized ? face.name : 'Unknown'];
      if (face.authorized) parts.push(`${(face.confidence * 100).toFixed(0)}%`);
      if (face.covered) parts.push(face.coverage);

      drawBox(face.boxNorm, width, height, {
        color,
        lineWidth,
        fontSize,
        label: parts.join(' · '),
        sublabel: face.distance === null || face.distance === undefined
          ? null
          : `d=${face.distance.toFixed(3)}`,
      });
    }
    ctx.globalAlpha = 1;
  }

  function drawBox(boxNorm, width, height, options) {
    let [nx1, ny1, nx2, ny2] = boxNorm;
    if (state.mirror && !state.stillImage) [nx1, nx2] = [1 - nx2, 1 - nx1];

    const x = nx1 * width;
    const y = ny1 * height;
    const boxWidth = (nx2 - nx1) * width;
    const boxHeight = (ny2 - ny1) * height;

    ctx.strokeStyle = options.color;
    ctx.lineWidth = options.lineWidth;
    ctx.setLineDash(options.dashed ? [options.lineWidth * 3, options.lineWidth * 2] : []);
    roundRect(x, y, boxWidth, boxHeight, Math.min(12, boxWidth * 0.12));
    ctx.stroke();
    ctx.setLineDash([]);

    const fontSize = options.fontSize;
    ctx.font = `600 ${fontSize}px Inter, system-ui, sans-serif`;
    ctx.textBaseline = 'alphabetic';

    const padding = fontSize * 0.42;
    const chipHeight = fontSize + padding * 1.6;
    const textWidth = ctx.measureText(options.label).width;
    // Flip the label below the box when there is no room above it.
    const chipY = y - chipHeight - options.lineWidth < 0
      ? y + boxHeight + options.lineWidth
      : y - chipHeight - options.lineWidth;

    ctx.fillStyle = options.color;
    roundRect(x, chipY, textWidth + padding * 2, chipHeight, 7);
    ctx.fill();

    ctx.fillStyle = '#04070f';
    ctx.fillText(options.label, x + padding, chipY + chipHeight - padding * 1.15);

    if (options.sublabel) {
      ctx.fillStyle = 'rgba(230, 234, 255, 0.88)';
      ctx.font = `500 ${Math.round(fontSize * 0.78)}px Inter, system-ui, sans-serif`;
      ctx.fillText(options.sublabel, x, chipY + chipHeight + fontSize * 0.95);
    }
  }

  function roundRect(x, y, width, height, radius) {
    const r = Math.max(0, Math.min(radius, Math.abs(width) / 2, Math.abs(height) / 2));
    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.arcTo(x + width, y, x + width, y + height, r);
    ctx.arcTo(x + width, y + height, x, y + height, r);
    ctx.arcTo(x, y + height, x, y, r);
    ctx.arcTo(x, y, x + width, y, r);
    ctx.closePath();
  }

  /* --------------------------------------------------------------------- log */

  function logResult(result) {
    const entries = [
      ...result.faces.map((face) => ({
        who: face.authorized ? face.name : 'Unknown',
        kind: face.authorized ? 'known' : 'unknown',
        detail: face.distance === null || face.distance === undefined
          ? face.coverage
          : `${face.coverage} · d=${face.distance.toFixed(3)}`,
      })),
      ...(result.objects || []).map((object) => ({
        who: object.label,
        kind: 'object',
        detail: `${(object.score * 100).toFixed(0)}% confidence`,
      })),
    ];
    if (!entries.length) return;

    const list = el('log');
    const placeholder = list.querySelector('.muted');
    if (placeholder) placeholder.remove();

    const stamp = new Date().toLocaleTimeString();
    for (const entry of entries) {
      const item = document.createElement('li');

      const who = document.createElement('span');
      who.className = `who ${entry.kind}`;
      who.textContent = entry.who;
      item.appendChild(who);

      const detail = document.createElement('span');
      detail.className = 'dist';
      detail.textContent = entry.detail;
      item.appendChild(detail);

      const when = document.createElement('span');
      when.className = 'when';
      when.textContent = stamp;
      item.appendChild(when);

      list.insertBefore(item, list.firstChild);
    }
    while (list.children.length > LOG_LIMIT) list.removeChild(list.lastChild);
  }

  /* ------------------------------------------------------------------- audio */

  function ensureAudio() {
    if (state.audio) return;
    const AudioCtx = window.AudioContext || window.webkitAudioContext;
    if (AudioCtx) state.audio = new AudioCtx();
  }

  function maybeAlert() {
    if (!el('alertSound').checked) return;
    ensureAudio();
    if (!state.audio) return;

    const now = performance.now();
    if (now - state.lastAlertAt < ALERT_COOLDOWN_MS) return;
    state.lastAlertAt = now;

    // Short two-tone chirp, synthesised so the page ships no audio assets.
    const audio = state.audio;
    if (audio.state === 'suspended') audio.resume();
    const start = audio.currentTime;
    [880, 660].forEach((frequency, index) => {
      const oscillator = audio.createOscillator();
      const gain = audio.createGain();
      oscillator.type = 'triangle';
      oscillator.frequency.value = frequency;
      const at = start + index * 0.18;
      gain.gain.setValueAtTime(0.0001, at);
      gain.gain.exponentialRampToValueAtTime(0.16, at + 0.02);
      gain.gain.exponentialRampToValueAtTime(0.0001, at + 0.16);
      oscillator.connect(gain).connect(audio.destination);
      oscillator.start(at);
      oscillator.stop(at + 0.18);
    });
  }

  /* ------------------------------------------------------- still-image path */

  async function analyseFile(file) {
    stopCamera();
    el('gate').hidden = true;
    el('liveBadge').hidden = true;
    el('fpsBadge').hidden = true;
    el('threatBadge').hidden = false;
    el('latencyBadge').hidden = false;
    el('toggleBtn').disabled = true;
    el('switchBtn').hidden = true;

    const objectUrl = URL.createObjectURL(file);
    try {
      const image = new Image();
      image.src = objectUrl;
      await image.decode();

      state.stillImage = image;
      state.mirror = false;
      state.result = null;
      syncViewportAspect(image.naturalWidth, image.naturalHeight);

      const scale = Math.min(1, 960 / image.naturalWidth);
      capture.width = Math.round(image.naturalWidth * scale);
      capture.height = Math.round(image.naturalHeight * scale);
      captureCtx.drawImage(image, 0, 0, capture.width, capture.height);

      const blob = await new Promise((resolve) => capture.toBlob(resolve, 'image/jpeg', 0.9));
      const result = await postFrame(blob);
      if (result.retry) {
        el('latencyBadge').textContent = 'server busy — try again';
        return;
      }
      applyResult(result);
    } catch (error) {
      el('latencyBadge').textContent = 'analysis failed';
      console.warn('file analysis failed', error);
    } finally {
      URL.revokeObjectURL(objectUrl);
    }
  }

  /* ------------------------------------------------------------------ wiring */

  el('startBtn').addEventListener('click', startCamera);
  el('switchBtn').addEventListener('click', switchCamera);
  el('pickFileBtn').addEventListener('click', () => el('fileInput').click());
  el('photoBtn').addEventListener('click', () => el('fileInput').click());
  el('fileInput').addEventListener('change', (event) => {
    const file = event.target.files && event.target.files[0];
    if (file) analyseFile(file);
    event.target.value = '';
  });

  el('toggleBtn').addEventListener('click', () => {
    state.running = !state.running;
    el('toggleBtn').textContent = state.running ? 'Pause' : 'Resume';
    el('liveBadge').hidden = !state.running;
    if (state.running) detectLoop();
  });

  el('threshold').addEventListener('input', (event) => {
    state.threshold = parseFloat(event.target.value);
    el('thresholdOut').textContent = state.threshold.toFixed(3);
  });

  window.addEventListener('resize', resizeOverlay);
  window.addEventListener('pagehide', stopCamera);
  new ResizeObserver(resizeOverlay).observe(viewport);

  resizeOverlay();
  requestAnimationFrame(draw);
  loadInfo();
})();
