#!/usr/bin/env bash
set -e

# ==============================================================================
# Monty Build & Installation Script for Mini-NN Toolchain
# 1. Resolves Monty directory via paths.py / paths_config.json or default repo layout.
# 2. Validates system dependencies (make, git, curl).
# 3. Ensures Rust toolchain (cargo, rustc, rustup) is installed and up-to-date.
# 4. Clones Monty repository if not present.
# 5. Compiles Monty native release binary with embedded network weights (make).
# 6. Runs UCI sanity check to verify binary execution.
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Ensure ~/.cargo/bin is in PATH if rust is installed there
if [ -d "$HOME/.cargo/bin" ]; then
    export PATH="$HOME/.cargo/bin:$PATH"
fi

echo "================================================================================"
echo "                   MONTY INSTALLATION & BUILD SCRIPT"
echo "================================================================================"

# 1. Resolve Monty binary and directory path
MONTY_DIR=""
if [ -f "$SCRIPT_DIR/paths.py" ] && command -v python >/dev/null 2>&1; then
    MONTY_BIN=$(python -c "import sys; sys.path.insert(0, '$SCRIPT_DIR'); import paths; print(paths.MONTY_BIN)" 2>/dev/null || echo "")
    if [ -n "$MONTY_BIN" ]; then
        MONTY_DIR="$(dirname "$MONTY_BIN")"
    fi
fi

if [ -z "$MONTY_DIR" ]; then
    MONTY_DIR="$(cd "$SCRIPT_DIR/../Monty" 2>/dev/null && pwd || echo "$SCRIPT_DIR/../Monty")"
fi

echo "[1/5] Target Monty Directory: $MONTY_DIR"

# 2. Check System Tools (make, git, curl)
echo "[2/5] Checking system prerequisites..."
for cmd in make git curl; do
    if ! command -v "$cmd" >/dev/null 2>&1; then
        echo "Error: Required command '$cmd' is not installed."
        echo "Please install it via your package manager (e.g. sudo apt update && sudo apt install -y build-essential git curl)."
        exit 1
    fi
done
echo "      System tools (make, git, curl) verified."

# 3. Check / Install Rust Toolchain
echo "[3/5] Checking Rust / Cargo toolchain..."
if ! command -v cargo >/dev/null 2>&1 || ! command -v rustup >/dev/null 2>&1; then
    echo "Rust/Cargo not found. Installing rustup..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    if [ -f "$HOME/.cargo/env" ]; then
        source "$HOME/.cargo/env"
    fi
fi

if ! command -v cargo >/dev/null 2>&1; then
    echo "Error: Cargo is still not found in PATH after rustup installation."
    exit 1
fi

echo "      Rust toolchain verified: $(rustc --version)"

# 4. Clone Monty if directory does not exist or is empty
echo "[4/5] Preparing Monty source code..."
if [ ! -d "$MONTY_DIR" ] || [ ! -f "$MONTY_DIR/Cargo.toml" ]; then
    mkdir -p "$(dirname "$MONTY_DIR")"
    echo "      Cloning official Monty repository..."
    git clone https://github.com/official-monty/Monty.git "$MONTY_DIR"
fi

# 5. Build Monty Native Release Binary
echo "[5/5] Building Monty (make)..."
cd "$MONTY_DIR"
NUM_CORES=$(nproc 2>/dev/null || echo 4)
export CARGO_BUILD_JOBS="$NUM_CORES"
make -j"$NUM_CORES"

# 6. Sanity check
MONTY_EXE="$MONTY_DIR/monty"
if [ ! -f "$MONTY_EXE" ]; then
    echo "Error: Build finished but $MONTY_EXE was not found!"
    exit 1
fi

echo "      Running UCI validation on $MONTY_EXE..."
UCI_TEST=$(printf "uci\nisready\nquit\n" | "$MONTY_EXE" 2>/dev/null || true)
if echo "$UCI_TEST" | grep -q "readyok"; then
    echo "      UCI validation SUCCESS: Monty is responding to UCI commands."
else
    echo "      Warning: Monty binary compiled, but UCI readyok test did not complete cleanly."
fi

echo "================================================================================"
echo "Monty build completed successfully!"
echo "Binary path: $MONTY_EXE"
echo "================================================================================"
