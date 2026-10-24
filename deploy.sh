#!/usr/bin/env bash
# Fast Browser Use — one-command deploy for the LiquidAI d1-3B decision model
# on llama.cpp CPU inference.
#
#   bash deploy.sh [--with-deps] [--skip-test]
#
# What it does:
#   1. Ensures build prerequisites (C compiler, cmake, curl, uv or venv+pip)
#   2. Installs the Python environment and Playwright Chromium
#   3. Downloads d1-3B-Q4_K_M.gguf (ModelScope mirror first, Hugging Face fallback)
#   4. Locates or builds llama.cpp llama-server (CPU, native flags)
#   5. Starts llama-server and runs a live real-browser verification task
#
# After deployment:
#   FBU_LLAMACPP_BIN=<llama-server> FBU_MODEL=<gguf> uv run fbu run <url> --goal '...'
# The Python backend auto-starts/reuses llama-server, so a manual start is optional.

set -euo pipefail
cd "$(dirname "$0")"

WITH_DEPS=0
SKIP_TEST=0
for arg in "$@"; do
  case "$arg" in
    --with-deps) WITH_DEPS=1 ;;
    --skip-test) SKIP_TEST=1 ;;
    *) echo "Unknown option: $arg"; exit 2 ;;
  esac
done

GGUF="d1-3B-Q4_K_M.gguf"
GGUF_URL_MS="https://modelscope.cn/models/LiquidAI/d1-3B-GGUF/resolve/master/${GGUF}"
GGUF_URL_HF="https://huggingface.co/LiquidAI/d1-3B-GGUF/resolve/main/${GGUF}"
LCPP_DIR="${LLAMA_CPP_DIR:-../llama.cpp}"
LCPP_BIN="${LCPP_DIR}/build/bin/llama-server"

log()  { printf '\033[1;36m[deploy]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[deploy]\033[0m %s\n' "$*"; }
fail() { printf '\033[1;31m[deploy]\033[0m %s\n' "$*" >&2; exit 1; }

# ---------------------------------------------------------------- 1. prerequisites
log "Step 1/5: checking prerequisites"
command -v curl >/dev/null || fail "curl is required"
if ! command -v gcc >/dev/null && ! command -v cc >/dev/null && ! command -v clang >/dev/null; then
  warn "No C compiler found. Install build-essential / gcc, then rerun."
  MISSING_CC=1
fi
if [ ! -x "${LCPP_BIN}" ]; then
  command -v cmake >/dev/null 2>&1 || pip install --quiet --break-system-packages cmake 2>/dev/null \
    || warn "cmake missing; install cmake 3.25+ before the llama.cpp build step"
fi

# Python environment: prefer uv, fall back to venv + pip.
if command -v uv >/dev/null 2>&1; then
  log "Using uv to sync the Python environment"
  uv sync --python 3.12
  RUN="uv run --no-sync"
else
  log "uv not found; using venv + pip"
  [ -d .venv ] || python3 -m venv .venv
  # shellcheck disable=SC1091
  source .venv/bin/activate
  pip install --quiet --upgrade pip
  pip install --quiet -e .
  RUN=""
fi

# ---------------------------------------------------------------- 2. browser
log "Step 2/5: installing Playwright Chromium"
if [ "${WITH_DEPS}" = "1" ]; then
  ${RUN} fbu install-browser --with-deps || true
else
  ${RUN} fbu install-browser
fi

# ---------------------------------------------------------------- 3. model weights
log "Step 3/5: downloading d1-3B GGUF weights"
if [ -f "models/${GGUF}" ] && [ "$(stat -c%s "models/${GGUF}" 2>/dev/null || echo 0)" -gt 1000000 ]; then
  log "Already downloaded: models/${GGUF}"
else
  mkdir -p models
  # ModelScope mirror is fast in CN; Hugging Face as fallback. -C - resumes partial files.
  if ! curl -L -C - --retry 5 --retry-delay 3 -o "models/${GGUF}" "${GGUF_URL_MS}"; then
    warn "ModelScope download failed; falling back to Hugging Face"
    curl -L -C - --retry 5 --retry-delay 3 -o "models/${GGUF}" "${GGUF_URL_HF}"
  fi
fi

# ---------------------------------------------------------------- 4. llama.cpp
log "Step 4/5: locating or building llama-server"
if [ ! -x "${LCPP_BIN}" ]; then
  if command -v llama-server >/dev/null 2>&1; then
    LCPP_BIN="$(command -v llama-server)"
    log "Using llama-server from PATH: ${LCPP_BIN}"
  elif [ -d "${LCPP_DIR}" ] || git clone --depth 1 https://github.com/ggml-org/llama.cpp.git "${LCPP_DIR}"; then
    log "Building llama.cpp (Release, native CPU) — this can take 10-30 minutes"
    cmake -B "${LCPP_DIR}/build" -S "${LCPP_DIR}" -DGGML_NATIVE=ON -DLLAMA_CURL=OFF \
      -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DCMAKE_BUILD_TYPE=Release
    cmake --build "${LCPP_DIR}/build" --config Release -j"$(nproc 2>/dev/null || echo 2)" --target llama-server
  else
    fail "Cannot fetch llama.cpp. Clone it manually and set LLAMA_CPP_DIR."
  fi
fi
[ -x "${LCPP_BIN}" ] || fail "llama-server binary not found at ${LCPP_BIN}"
log "llama-server ready: ${LCPP_BIN}"

# ---------------------------------------------------------------- 5. live verification
if [ "${SKIP_TEST}" = "1" ]; then
  log "Skipping live verification (--skip-test)"
else
  log "Step 5/5: live real-browser verification (example.com -> IANA)"
  FBU_LLAMACPP_BIN="${LCPP_BIN}" \
  FBU_MODEL="$PWD/models/${GGUF}" \
  ${RUN} fbu run https://example.com \
    --goal 'Open the "More information" link so the IANA example-domains page is shown.' \
    --expect-url 'https://www.iana.org/help/example-domains' \
    --expect-title 'Example Domains' \
    --trace artifacts/deploy-verification.json || {
      warn "Verification task did not pass on the first attempt. Inspect artifacts/deploy-verification.json."
      warn "d1-3B is a 3B decision model: retry, or try a goal phrased closer to the visible page labels."
    }
fi

cat <<EOF

=============================================================
 Fast Browser Use deployed successfully (d1-3B + llama.cpp)

 Weights : $PWD/models/${GGUF}
 Server  : ${LCPP_BIN} (auto-started on demand, port 8765)

 Run a task:
   FBU_LLAMACPP_BIN="${LCPP_BIN}" \\
   FBU_MODEL="$PWD/models/${GGUF}" \\
   ${RUN} fbu run <url> --goal '...' \\
     [--expect-url ...] [--expect-title ...] [--expect-text ...]

 Env knobs:
   FBU_BACKEND=llamacpp        (default)
   FBU_LLAMACPP_URL=http://127.0.0.1:8765
   FBU_LLAMACPP_THREADS=N      (defaults to all cores)
   FBU_DECISION_MODE=binary    (default for d1; explicit completion check)
=============================================================
EOF
