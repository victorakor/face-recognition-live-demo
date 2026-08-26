# Runs on Hugging Face Spaces (sdk: docker, port 7860) and on Render (Docker runtime, with
# $PORT injected). Nothing in here is host-specific beyond that.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# Keep CPU inference from oversubscribing a small container.
ENV OMP_NUM_THREADS=2
ENV OPENBLAS_NUM_THREADS=2
ENV MKL_NUM_THREADS=2
ENV TORCH_THREADS=2

# matplotlib is pulled in transitively and insists on a writable config directory.
ENV MPLCONFIGDIR=/tmp/matplotlib

ENV PORT=7860

# build-essential/cmake are here to compile dlib; libopenblas-dev is the whole point of
# doing so (see the dlib install step). libgl1/libglib2.0-0 keep any stray non-headless
# opencv import from failing at import time.
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential \
      cmake \
      pkg-config \
      libopenblas-dev \
      liblapack-dev \
      libgl1 \
      libglib2.0-0 \
      ca-certificates \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./
RUN pip install -r requirements.txt

# The dlib-bin wheel from requirements.txt ships without BLAS, which makes the ResNet face
# encoder ~20x slower. Compiling dlib here picks up OpenBLAS and takes encoding from ~1.5 s
# down to ~80 ms per face.
#
# CMAKE_BUILD_PARALLEL_LEVEL is not optional. dlib's setup.py sizes its own -j by reading
# SC_PHYS_PAGES, which reports the *host* machine's RAM rather than the build container's
# cgroup limit -- so on a big builder it fans out to more g++ processes than the container is
# allowed and the whole build gets OOM-killed. At ~2 GB per process, 2 is the safe number.
#
# If the build fails anyway we fall back to the wheel rather than failing the deploy; the app
# detects the difference and says so on the page.
ENV CMAKE_BUILD_PARALLEL_LEVEL=2
RUN pip uninstall -y dlib-bin dlib || true; \
    if pip install --no-binary :all: "dlib==19.24.6"; then \
      echo ">>> dlib compiled from source"; \
    else \
      echo ">>> WARNING: dlib source build failed, falling back to dlib-bin" >&2; \
      pip install "dlib-bin==19.24.6"; \
    fi
RUN python -c "import dlib; print('>>> dlib', dlib.__version__, 'BLAS', dlib.DLIB_USE_BLAS, 'LAPACK', dlib.DLIB_USE_LAPACK)"

# The two stock dlib models are ~120 MB uncompressed, so they are fetched here rather than
# committed. The custom pieces (the ONNX detector and the face gallery) come from the repo in
# the COPY below.
COPY scripts/ ./scripts/
RUN python scripts/fetch_models.py --models-dir /app/models

COPY . .

# Load every model once at build time. This turns "the ONNX detector silently fell back to
# dlib HOG in production" into a line in the build log rather than a surprise on the page.
RUN python -c "from recognizer import get_recognizer; print('>>> self-check:', get_recognizer().info())" \
    || echo ">>> WARNING: model self-check failed" >&2

# Hugging Face Spaces runs containers as uid 1000.
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 7860

# One worker: the models are loaded per process and inference is serialised behind a lock
# anyway, so a second worker would double the memory for no throughput. Threads keep static
# files and /api/info responsive while a frame is being processed.
CMD ["sh", "-c", "exec gunicorn --bind 0.0.0.0:${PORT} --workers 1 --threads 8 --worker-class gthread --timeout 120 --graceful-timeout 30 --access-logfile - --error-logfile - app:app"]
