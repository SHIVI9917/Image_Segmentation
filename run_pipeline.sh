#!/usr/bin/env bash
# Single entrypoint for running the full modular pipeline on a fresh Linux GPU box (e.g. a
# RunPod pod): sets up the uv-managed environment, fetches data/raw/data.pkl from Kaggle if
# needed, then runs prepare_data.py, train_model.py, evaluate_model.py in order.
#
# Usage: ./run_pipeline.sh
#
# Kaggle download needs a token configured on this machine first (kaggle.json/access_token
# under ~/.kaggle/, or KAGGLE_USERNAME/KAGGLE_KEY env vars -- the kaggle CLI resolves whichever
# is present). Falls back to a clear error telling you to transfer data.pkl manually if that
# fails or no token is set up.
set -euo pipefail

cd "$(dirname "$0")"

# .venv, data/, the Hugging Face + uv caches, and models/ are large enough to fill a typical
# 20GB container disk. Redirect them onto a mounted persistent volume instead, if one exists
# (RunPod's default mount point is /workspace; override with RUNPOD_VOLUME_DIR).
VOLUME_DIR="${RUNPOD_VOLUME_DIR:-/workspace}"

relocate_to_volume() {
    # Symlinks $1 (a path relative to the repo root) into $VOLUME_DIR/ImageSegmentation-$1,
    # deleting any existing real directory at $1 first (everything redirected here is either
    # cheaply regenerable or, for models/, not worth preserving as-is).
    local name="$1"
    local target="$VOLUME_DIR/ImageSegmentation-$name"
    if [ -L "$name" ]; then
        return
    fi
    mkdir -p "$target"
    if [ -e "$name" ]; then
        echo "Removing existing $name/ (moving onto persistent volume at $target/)"
        rm -rf "$name"
    fi
    ln -s "$target" "$name"
}

if [ -d "$VOLUME_DIR" ]; then
    echo "== Persistent volume found at $VOLUME_DIR -- redirecting .venv, data, models there =="
    relocate_to_volume .venv
    relocate_to_volume data
    relocate_to_volume models
    export HF_HOME="$VOLUME_DIR/.cache/huggingface"
    export UV_CACHE_DIR="$VOLUME_DIR/.cache/uv"
    mkdir -p "$HF_HOME" "$UV_CACHE_DIR"
else
    echo "== No persistent volume found at $VOLUME_DIR -- using container disk (set RUNPOD_VOLUME_DIR if your volume is mounted elsewhere) =="
fi
# ---------------------------------------------------------------------------------------------

if ! command -v uv >/dev/null 2>&1; then
    echo "== Installing uv =="
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

if [ ! -d .venv ] || [ -z "$(ls -A .venv 2>/dev/null)" ]; then
    echo "== Creating virtual environment (Python 3.11) =="
    uv venv --python 3.11 .venv
fi

echo "== Installing dependencies =="
# UV_HTTP_TIMEOUT raised because the pinned torch wheel is ~2.7GB; uv's 30s default can time
# out on it depending on the pod's network.
UV_HTTP_TIMEOUT=600 uv pip install -r requirements.txt -p .venv

PYTHON=.venv/bin/python

if [ ! -f data/raw/data.pkl ]; then
    echo "== data/raw/data.pkl not found -- attempting to fetch from Kaggle =="
    uv pip install kaggle -p .venv
    mkdir -p data/raw

    if .venv/bin/kaggle datasets download -d shayakbhattacharya/finetune -p data/raw --unzip; then
        # The dataset's internal filename isn't guaranteed to be data.pkl -- find whatever
        # .pkl landed and move it to the path prepare_data.py expects.
        if [ ! -f data/raw/data.pkl ]; then
            found="$(find data/raw -maxdepth 3 -iname '*.pkl' ! -name 'data.pkl' | head -n 1)"
            if [ -n "$found" ]; then
                echo "Found $found -- moving to data/raw/data.pkl"
                mv "$found" data/raw/data.pkl
            fi
        fi
    fi

    if [ ! -f data/raw/data.pkl ]; then
        echo "ERROR: data/raw/data.pkl still not found after attempting Kaggle download." >&2
        echo "Either configure a Kaggle API token on this machine and re-run, or transfer" >&2
        echo "data.pkl here manually (see docs/RUNPOD_GUIDE.md step 4)." >&2
        exit 1
    fi
    echo "== data/raw/data.pkl ready (from Kaggle) =="
fi

echo "== nvidia-smi =="
nvidia-smi || echo "(nvidia-smi not available -- continuing, but training will be very slow/impossible without a GPU)"

echo "== Disk usage =="
df -h "$(pwd)" "$VOLUME_DIR" 2>/dev/null || df -h "$(pwd)"

# prepare_data.py / train_model.py / evaluate_model.py each catch their own exceptions and
# print "Error in <script>.py: ..." rather than raising, so they always exit 0. run_stage greps
# for that pattern so a real failure stops this script instead of silently continuing.
run_stage() {
    local script="$1"
    local log
    log="$(mktemp)"
    set +e
    "$PYTHON" "$script" 2>&1 | tee "$log"
    local exit_code=${PIPESTATUS[0]}
    set -e
    if [ "$exit_code" -ne 0 ] || grep -q "^Error in ${script}:" "$log"; then
        echo "ERROR: $script failed -- stopping pipeline (see output above)." >&2
        rm -f "$log"
        exit 1
    fi
    rm -f "$log"
}

echo "== Stage 1/3: prepare_data.py =="
run_stage prepare_data.py

echo "== Stage 2/3: train_model.py =="
run_stage train_model.py

echo "== Stage 3/3: evaluate_model.py =="
run_stage evaluate_model.py

echo "== Done. Final model: models/finetuned/  |  checkpoints: models/checkpoints/ =="
