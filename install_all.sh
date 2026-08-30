#!/usr/bin/env bash
set -e

# ==============================================================================
# Complete Master Installer for Mini-NN Toolchain
# 1. Detects Python 3 and installs dependencies (requirements.txt).
# 2. Clones & builds Stockfish with Mini-NN LMR integration (install_stockfish.sh).
# 3. Sets up Rust toolchain, clones & builds Monty (install_monty.sh).
# 4. Verifies path resolution and tests C++ / Python SIMD inference parity.
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Ensure ~/.cargo/bin is in PATH if rust is installed there
if [ -d "$HOME/.cargo/bin" ]; then
    export PATH="$HOME/.cargo/bin:$PATH"
fi

echo "================================================================================"
echo "          MINI-NN MOVEPICKER & LMR TOOLCHAIN: FULL INSTALLATION"
echo "================================================================================"
echo "Installation Directory: $SCRIPT_DIR"
echo ""

# Determine active Python executable
PYTHON_BIN=""
if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
else
    echo "Error: Python 3 is required but not found in PATH."
    exit 1
fi
echo "Using Python Interpreter: $($PYTHON_BIN -c 'import sys; print(sys.executable)') ($($PYTHON_BIN --version))"

# 1. Install Python Dependencies
echo ""
echo "[Step 1/4] Installing Python dependencies (requirements.txt)..."

if ! $PYTHON_BIN -m pip --version >/dev/null 2>&1; then
    echo "Pip not found for $PYTHON_BIN. Attempting system package installation..."
    if command -v apt-get >/dev/null 2>&1; then
        sudo apt-get update && sudo apt-get install -y python3-pip python3-venv || true
    fi
fi

# Try standard pip install, fallback to --break-system-packages or --user if restricted
$PYTHON_BIN -m pip install -r "$SCRIPT_DIR/requirements.txt" || \
$PYTHON_BIN -m pip install --break-system-packages -r "$SCRIPT_DIR/requirements.txt" || \
$PYTHON_BIN -m pip install --user -r "$SCRIPT_DIR/requirements.txt" || {
    echo "Error: Failed to install Python dependencies. Please ensure pip is installed."
    exit 1
}

# Verify dependencies in python
$PYTHON_BIN -c "import torch, numpy, scipy, chess; print('      Python libraries (torch, numpy, scipy, chess) verified.')"
echo ""

# 2. Build Stockfish
echo "[Step 2/4] Setting up and building Stockfish..."
"$SCRIPT_DIR/install_stockfish.sh"
echo ""

# 3. Build Monty
echo "[Step 3/4] Setting up and building Monty..."
"$SCRIPT_DIR/install_monty.sh"
echo ""

# 4. Verify Path Resolution & Inference Parity
echo "[Step 4/4] Verifying paths configuration and running parity test suite..."
$PYTHON_BIN -c "import paths; print('  [Paths] REPO_ROOT:     ', paths.REPO_ROOT); print('  [Paths] STOCKFISH_BIN: ', paths.STOCKFISH_BIN); print('  [Paths] MONTY_BIN:     ', paths.MONTY_BIN); print('  [Paths] CALIB_CONFIG:  ', paths.CALIB_CONFIG_PATH)"

echo ""
echo "Running Bit-Level C++ / PyTorch SIMD Invariant Parity Test..."
$PYTHON_BIN "$SCRIPT_DIR/test_inference_parity.py"

echo ""
echo "================================================================================"
echo "       INSTALLATION & VERIFICATION COMPLETE! THE TOOLCHAIN IS READY."
echo "================================================================================"
echo "To run on-policy training:"
echo "  $PYTHON_BIN $SCRIPT_DIR/online_trainer.py --grid --iterations 64"
echo "================================================================================"
