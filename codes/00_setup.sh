#!/usr/bin/env bash
# =============================================================================
# 00_setup.sh — one-time setup, run from the repository root:
#
#   conda env create -f environment.yml     # creates the `hydrocat` env
#   conda activate hydrocat
#   bash codes/00_setup.sh
#
# What it does
#   1. Clones Google's hydrology framework (github.com/google-research/flood-forecasting)
#      into external/flood-forecasting at a PINNED commit, so the model code and
#      the pretrained weights used in this study never change underneath us.
#   2. pip-installs it (editable, no deps: the env already holds them) -> gives
#      the `googlehydrology` package and the `run` CLI (run train / run infer).
#   3. Installs this repo's own helper package (lib/hydrocat) in editable mode.
#   4. Checks Google Earth Engine authentication (needed by 02, 04 and the
#      WeatherNext 3 forecast step).
# =============================================================================
set -euo pipefail

FF_URL="https://github.com/google-research/flood-forecasting"
FF_COMMIT="3cd66462b6ce9c5335bfcbc71574aacfbff9bd61"   # 2026-09-26, googlehydrology 1.12.0
FF_DIR="external/flood-forecasting"

if [ ! -d "$FF_DIR/.git" ]; then
  git clone "$FF_URL" "$FF_DIR"
fi
git -C "$FF_DIR" fetch --quiet origin
git -C "$FF_DIR" checkout --quiet "$FF_COMMIT"
echo "flood-forecasting @ $(git -C "$FF_DIR" log -1 --format='%h %cd')"

pip install --quiet --no-deps -e "$FF_DIR"
pip install --quiet --no-deps -e .

python - <<'EOF'
import googlehydrology, hydrocat
print("googlehydrology", googlehydrology.__version__ if hasattr(googlehydrology, "__version__") else "ok")
print("hydrocat", hydrocat.__version__)
EOF

# Earth Engine: authenticate once with the Google account that has
# WeatherNext + Earth Engine access (`earthengine authenticate` or
# `gcloud auth application-default login`). The Cloud project is read from
# config/settings.yaml (ee_project).
python - <<'EOF'
from hydrocat.eeutils import init_ee
init_ee()
print("Earth Engine: OK")
EOF
