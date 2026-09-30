#!/usr/bin/env bash
# CI variant of 00_setup.sh (no Earth Engine check: credentials come from the
# EE_SERVICE_ACCOUNT_KEY secret at run time).
set -euo pipefail
FF_URL="https://github.com/google-research/flood-forecasting"
FF_COMMIT="3cd66462b6ce9c5335bfcbc71574aacfbff9bd61"
FF_DIR="external/flood-forecasting"
[ -d "$FF_DIR/.git" ] || git clone --quiet "$FF_URL" "$FF_DIR"
git -C "$FF_DIR" fetch --quiet origin && git -C "$FF_DIR" checkout --quiet "$FF_COMMIT"
pip install --quiet --no-deps -e "$FF_DIR"
pip install --quiet --no-deps -e .
