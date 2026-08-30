#!/usr/bin/env bash
set -e

# ==============================================================================
# Stockfish (Mini-NN-LMR Branch) Build & Installation Script
# 1. Resolves Stockfish directory via paths.py or defaults to deps/Stockfish.
# 2. Validates C++ build tools (g++/clang++, make).
# 3. Clones the Stockfish repo (mini-NN-lmr branch) if not present.
# 4. Compiles optimized Stockfish binary with AVX2/native SIMD support.
# 5. Runs UCI validation handshake (uci -> uciok / readyok).
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "================================================================================"
echo "               STOCKFISH (MINI-NN) INSTALLATION & BUILD SCRIPT"
echo "================================================================================"

# 1. Resolve Stockfish directory path
SF_DIR=""
if [ -f "$SCRIPT_DIR/paths.py" ] && command -v python >/dev/null 2>&1; then
    SF_BIN=$(python -c "import sys; sys.path.insert(0, '$SCRIPT_DIR'); import paths; print(paths.STOCKFISH_BIN)" 2>/dev/null || echo "")
    if [ -n "$SF_BIN" ] && [ -f "$SF_BIN" ]; then
        SF_DIR="$(cd "$(dirname "$SF_BIN")/.." && pwd)"
    fi
fi

if [ -z "$SF_DIR" ]; then
    if [ -d "$SCRIPT_DIR/../../Stockfish/src" ]; then
        SF_DIR="$(cd "$SCRIPT_DIR/../../Stockfish" && pwd)"
    else
        SF_DIR="$SCRIPT_DIR/deps/Stockfish"
    fi
fi

echo "[1/5] Target Stockfish Directory: $SF_DIR"

# 2. Check System Build Tools
echo "[2/5] Checking system prerequisites..."
for cmd in make git; do
    if ! command -v "$cmd" >/dev/null 2>&1; then
        echo "Error: Required command '$cmd' is not installed."
        echo "Please install it via: sudo apt update && sudo apt install -y build-essential git"
        exit 1
    fi
done

if ! command -v g++ >/dev/null 2>&1 && ! command -v clang++ >/dev/null 2>&1; then
    echo "Error: No C++ compiler found (g++ or clang++ is required)."
    echo "Please install via: sudo apt install -y g++"
    exit 1
fi
echo "      Build prerequisites verified."

# 3. Clone Stockfish repo (mini-NN-lmr branch) if not present
echo "[3/5] Preparing Stockfish source code..."
if [ ! -d "$SF_DIR" ] || [ ! -f "$SF_DIR/src/Makefile" ]; then
    mkdir -p "$(dirname "$SF_DIR")"
    echo "      Cloning Stockfish (branch: mini-NN-lmr)..."
    git clone --branch mini-NN-lmr https://github.com/TonyCongqianWang/Stockfish.git "$SF_DIR"
else
    echo "      Stockfish source directory present."
fi

# 4. Build Stockfish Binary
echo "[4/5] Compiling Stockfish binary..."
cd "$SF_DIR/src"

NUM_CORES=$(nproc 2>/dev/null || echo 4)
# Detect best ARCH support
ARCH="x86-64-avx2"
if ! grep -q "avx2" /proc/cpuinfo 2>/dev/null; then
    if grep -q "sse4_1" /proc/cpuinfo 2>/dev/null; then
        ARCH="x86-64-sse41-popcnt"
    else
        ARCH="x86-64"
    fi
fi

echo "      Building with make -j$NUM_CORES ARCH=$ARCH..."
make -j"$NUM_CORES" build ARCH="$ARCH"

# 5. Sanity check UCI
SF_EXE="$SF_DIR/src/stockfish"
if [ ! -f "$SF_EXE" ]; then
    echo "Error: Build finished but $SF_EXE was not found!"
    exit 1
fi

echo "[5/5] Running UCI validation check..."
UCI_TEST=$(printf "uci\nisready\nquit\n" | "$SF_EXE" 2>/dev/null || true)
if echo "$UCI_TEST" | grep -q "uciok"; then
    echo "      UCI validation SUCCESS: Stockfish is ready."
else
    echo "      Warning: Stockfish compiled, but UCI handshake did not return 'uciok'."
fi

echo "================================================================================"
echo "Stockfish build completed successfully!"
echo "Binary path: $SF_EXE"
echo "================================================================================"
