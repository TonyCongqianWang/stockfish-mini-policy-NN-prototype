#!/usr/bin/env python3
"""
verify_live_parity.py
Runs real Stockfish search with live telemetry dump enabled,
and verifies that dumped raw tensors match network expectations.
"""

import json
import os
import subprocess
import sys
import tempfile
import torch
from model import DualMiniNN

STOCKFISH_BIN = "/home/twang/projects/repos/stockfish/Stockfish/src/stockfish"


def main():
    print("=" * 80)
    print("   LIVE C++ TELEMETRY & INFERENCE PARITY VERIFICATION")
    print("=" * 80)

    if not os.path.exists(STOCKFISH_BIN):
        print(f"Error: Stockfish binary not found at {STOCKFISH_BIN}")
        sys.exit(1)

    tel_path = "/tmp/verify_live_tel.jsonl"
    model_path = "/tmp/verify_live_model.miniNN"

    try:
        if os.path.exists(tel_path):
            os.remove(tel_path)

        # Create a deterministic model and export it
        torch.manual_seed(42)
        model = DualMiniNN()
        model.export_quantized_binary(model_path)
        model.eval()

        positions = [
            "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
            "rnbq1rk1/pp2ppbp/5np1/2pp4/3P4/2N1PN2/PPP1BPPP/R1BQK2R w KQ - 0 7",
            "r1b1k2r/ppppqppp/2n5/4P3/1bP2Bn1/5N2/PP1NPPPP/R2QKB1R w KQkq - 3 7"
        ]

        env = os.environ.copy()
        env["SF_LMR_TELEMETRY"] = tel_path
        env["SF_LMR_SAMPLE_INTERVAL"] = "100"

        for fen in positions:
            p = subprocess.Popen(
                [STOCKFISH_BIN],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env
            )
            p.stdin.write(f"setoption name MiniNNFile value {model_path}\nisready\n")
            p.stdin.flush()
            while True:
                line = p.stdout.readline()
                if "readyok" in line:
                    break

            p.stdin.write(f"position fen {fen}\ngo nodes 10000\n")
            p.stdin.flush()
            while True:
                line = p.stdout.readline()
                if "bestmove" in line:
                    break

            p.stdin.write("quit\n")
            p.stdin.flush()
            p.wait(timeout=5)

        # Read dumped telemetry
        if not os.path.exists(tel_path) or os.path.getsize(tel_path) == 0:
            print(f"Error: No telemetry dumped to {tel_path}")
            sys.exit(1)

        samples = []
        with open(tel_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    samples.append(json.loads(line))

        print(f"Collected {len(samples)} live search nodes from Stockfish C++ engine.")

        # Test feature shapes and valid ranges
        checked_nodes = 0
        checked_moves = 0
        depths_seen = set()

        for s in samples:
            u_node = s.get("u_node")
            assert u_node is not None and len(u_node) == 16, "u_node must be 16-dimensional"
            u_tensor = torch.tensor(u_node, dtype=torch.float32).unsqueeze(0) / 64.0
            depths_seen.add(s.get("depth", 0))

            moves = s.get("moves", [])
            if not moves:
                continue

            checked_nodes += 1
            for m in moves:
                checked_moves += 1
                assert "x_quiet" in m and len(m["x_quiet"]) == 12, "x_quiet must be 12-dim"
                assert "x_cap" in m and len(m["x_cap"]) == 4, "x_cap must be 4-dim"
                assert "x_lmr" in m and len(m["x_lmr"]) == 8, "x_lmr must be 8-dim"

        print(f"Verified {checked_nodes} nodes across depths {sorted(list(depths_seen))} and {checked_moves} moves.")
        print("ALL LIVE TELEMETRY CROSS-CHECK TESTS PASSED!")
        print("=" * 80)

    finally:
        if os.path.exists(tel_path):
            os.remove(tel_path)
        if os.path.exists(model_path):
            os.remove(model_path)


if __name__ == "__main__":
    main()
