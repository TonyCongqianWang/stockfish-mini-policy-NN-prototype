#!/usr/bin/env bash
set -e

# ==============================================================================
# Complete Master Installer for Mini-NN Toolchain
# 1. Installs Python dependencies (requirements.txt).
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

# 1. Install Python Dependencies
echo "[Step 1/4] Installing Python dependencies (requirements.txt)..."
if command -v pip >/dev/null 2>&1; then
    pip install -r "$SCRIPT_DIR/requirements.txt"
elif command -v pip3 >/dev/null 2>&1; then
    pip3 install -r "$SCRIPT_DIR/requirements.txt"
else
    echo "Warning: Neither pip nor pip3 found in PATH. Skipping pip install."
fi
echo "      Python dependencies installed."
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
python -c "import paths; print('  [Paths] REPO_ROOT:     ', paths.REPO_ROOT); print('  [Paths] STOCKFISH_BIN: ', paths.STOCKFISH_BIN); print('  [Paths] MONTY_BIN:     ', paths.MONTY_BIN); print('  [Paths] CALIB_CONFIG:  ', paths.CALIB_CONFIG_PATH)"

echo ""
echo "Running Bit-Level C++ / PyTorch SIMD Invariant Parity Test..."
python "$SCRIPT_DIR/test_inference_parity.py"

echo ""
echo "================================================================================"
echo "       INSTALLATION & VERIFICATION COMPLETE! THE TOOLCHAIN IS READY."
echo "================================================================================"
echo "To run on-policy training:"
echo "  python $SCRIPT_DIR/online_trainer.py --grid --iterations 64"
echo "================================================================================"
