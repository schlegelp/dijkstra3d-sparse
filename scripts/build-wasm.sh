#!/usr/bin/env bash
# Build the Pyodide/WebAssembly wheel for dijkstra3d-sparse.
#
# Produces dijkstra3d_sparse-<version>-cp39-abi3-pyemscripten_2026_0_wasm32.whl,
# loadable by Pyodide 314.x (CPython 3.14, PEP 783 ABI pyemscripten_2026_0).
#
# Every toolchain version is read from `pyodide config`, which in turn derives
# them from the pinned cross-build environment below -- nothing is hardcoded, so
# following a new Pyodide release is a matter of bumping PYODIDE_VERSION.
#
# Requirements:
#   * a Python 3.14 environment with `pyodide-build` and `maturin>=1.16`
#     installed, with its bin/ directory on PATH (or set PYODIDE/PYTHON314)
#   * an activated emsdk: source /path/to/emsdk/emsdk_env.sh
#   * rustup, which will install the required toolchain if it is missing
set -euo pipefail

cd "$(dirname "$0")/.."

PYODIDE_VERSION="${PYODIDE_VERSION:-314.0.5}"
PYODIDE="${PYODIDE:-pyodide}"
OUT_DIR="${OUT_DIR:-dist}"

if ! command -v "$PYODIDE" >/dev/null 2>&1; then
    echo "error: '$PYODIDE' not found. Install pyodide-build into a Python 3.14" >&2
    echo "       environment and put it on PATH, or set PYODIDE=/path/to/pyodide." >&2
    exit 1
fi

# maturin must be told which interpreter it is targeting. Left to its own
# devices it picks whatever `python` is on PATH; if that is older than 3.14 it
# silently falls back to the legacy Emscripten code path, which injects the
# nightly-only `-Z link-native-libraries=no` and fails on a stable toolchain.
PYTHON314="${PYTHON314:-$(dirname "$(command -v "$PYODIDE")")/python}"
PY_VERSION="$("$PYTHON314" -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
if [ "$PY_VERSION" != "3.14" ]; then
    echo "error: \$PYTHON314 ($PYTHON314) is Python $PY_VERSION, need 3.14." >&2
    exit 1
fi

"$PYODIDE" xbuildenv install "$PYODIDE_VERSION" >/dev/null

RUST_TOOLCHAIN="$("$PYODIDE" config get rust_toolchain)"
EMSCRIPTEN_VERSION="$("$PYODIDE" config get emscripten_version)"
ABI_VERSION="$("$PYODIDE" config get pyodide_abi_version)"
PYODIDE_RUSTFLAGS="$("$PYODIDE" config get rustflags)"

echo "pyodide          : $PYODIDE_VERSION"
echo "wheel platform   : pyemscripten_${ABI_VERSION}_wasm32"
echo "emscripten       : $EMSCRIPTEN_VERSION"
echo "rust toolchain   : $RUST_TOOLCHAIN"
echo "rustflags        : $PYODIDE_RUSTFLAGS"

if ! command -v emcc >/dev/null 2>&1; then
    echo "error: emcc not found. Activate emsdk $EMSCRIPTEN_VERSION first:" >&2
    echo "       source /path/to/emsdk/emsdk_env.sh" >&2
    exit 1
fi

HAVE_EM="$(emcc -dumpversion)"
if [ "$HAVE_EM" != "$EMSCRIPTEN_VERSION" ]; then
    echo "error: emcc is $HAVE_EM but Pyodide $PYODIDE_VERSION needs $EMSCRIPTEN_VERSION." >&2
    echo "       A mismatched Emscripten yields a wheel Pyodide cannot load." >&2
    exit 1
fi

rustup toolchain install "$RUST_TOOLCHAIN" --profile minimal >/dev/null
rustup target add wasm32-unknown-emscripten --toolchain "$RUST_TOOLCHAIN" >/dev/null
export RUSTUP_TOOLCHAIN="$RUST_TOOLCHAIN"

# SIDE_MODULE=2 leaves the CPython symbols undefined for the Pyodide runtime to
# resolve at load time; together with abi3 that means no cross-compiled
# libpython is needed at build time.
export CARGO_TARGET_WASM32_UNKNOWN_EMSCRIPTEN_RUSTFLAGS="$PYODIDE_RUSTFLAGS"
# Older maturin releases still default to the legacy `pyodide_*` platform tag,
# which PyPI rejects; pin the PEP 783 platform explicitly.
export MATURIN_PYEMSCRIPTEN_PLATFORM_VERSION="$ABI_VERSION"

exec maturin build \
    --release \
    --target wasm32-unknown-emscripten \
    --interpreter "$PYTHON314" \
    --out "$OUT_DIR" \
    "$@"
