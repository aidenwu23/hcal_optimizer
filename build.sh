#!/usr/bin/env bash
# mini_detector: configure, build, *install* (process to ~/.local/bin)
# Workflow: detect the repo root, set cmake options, optionally clean build/, run
#           cmake configure, build, install, then point to helper setup steps.
# This script leaves the shell environment unchanged.
# To expose installed binaries/libs for this session, run:
#   source setup.sh
#
# Usage:
#   ./build.sh                  # build + install
#   # then (optional, for this shell only):
#   source setup.sh
#
# Note: You can still 'source' this script, but doing so has no extra effect.
#       Environment setup lives in setup.sh.

# ---------------- detect if sourced ----------------
if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then _SOURCED=1; else _SOURCED=0; fi

# Save caller's shell option state, enable strict mode only within this script
__OLD_SET_OPTS="$(set +o)"
set -Eeuo pipefail
trap 'eval "$__OLD_SET_OPTS"; if [[ $_SOURCED -eq 1 ]]; then return 1; else exit 1; fi' ERR
if [[ $_SOURCED -eq 1 ]]; then trap 'eval "$__OLD_SET_OPTS"; trap - INT ERR; return 130' INT; fi

_die() { echo "ERROR: $*" >&2; eval "$__OLD_SET_OPTS"; if [[ $_SOURCED -eq 1 ]]; then return 1; else exit 1; fi; }

# ---------------- find repo root ----------------
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[[ -f "$ROOT_DIR/CMakeLists.txt" ]] || _die "missing $ROOT_DIR/CMakeLists.txt"
cd "$ROOT_DIR" || _die "cd to repo failed"
echo "==> repo: $ROOT_DIR"

command -v cmake >/dev/null 2>&1 || _die "cmake not found (activate the build environment)"

# ---------------- settings ----------------
: "${BUILD_TYPE:=RelWithDebInfo}"
: "${INSTALL_PREFIX:=$HOME/.local}"
JOBS="$(command -v nproc >/dev/null 2>&1 && nproc || echo 4)"

# ---------------- prompt: clean build/ ----------------
# Optional clean step keeps the build tree reproducible without hunting for rm flags
read -r -p "Clean build/ directory before building? [y/N] " _ans
lower=$(awk '{print tolower($0)}' <<< "$_ans")
if [[ "${lower}" == y || "${lower}" == yes ]]; then
  echo "==> cleaning build/"
  rm -rf build
fi

# ---------------- configure + build + install ----------------
echo "==> configuring ($BUILD_TYPE, prefix=$INSTALL_PREFIX)"
cmake -S . -B build   -DCMAKE_BUILD_TYPE="$BUILD_TYPE"   -DCMAKE_INSTALL_PREFIX="$INSTALL_PREFIX"   || _die "cmake configure failed"

echo "==> building (jobs=$JOBS)"
cmake --build build -j"$JOBS" || _die "build failed"

echo "==> installing to $INSTALL_PREFIX"
cmake --install build || _die "install failed"

# ---------------- post-install info ----------------
BIN_DIR="$INSTALL_PREFIX/bin"
PROC_BIN="$BIN_DIR/process"
if [[ -x "$PROC_BIN" ]]; then
  echo "==> installed binary: $PROC_BIN"
else
  echo "WARN: process not found at $PROC_BIN (BUILD_PROCESSOR might be OFF)"
fi

# ---------------- guidance ----------------
if [[ $_SOURCED -eq 1 ]]; then
  echo "Note: sourcing build.sh does not export PATH/LD_LIBRARY_PATH."
fi
if [[ -f "$ROOT_DIR/setup.sh" ]]; then
  echo "Tip: to use the installed tools immediately in this shell, run:"
  echo "  source "$ROOT_DIR/setup.sh""
else
  echo "Tip: add $BIN_DIR to PATH and the plugin/lib directories to LD_LIBRARY_PATH for this session."
fi

# restore caller's shell options and clear traps
eval "$__OLD_SET_OPTS"
trap - ERR INT
