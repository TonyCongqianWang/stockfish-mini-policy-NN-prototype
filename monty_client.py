"""
Monty Python Client: Communicates with compiled Monty binary to fetch policy ground truth.
"""

import subprocess
import os
from typing import Dict, List, Tuple

from paths import MONTY_BIN as DEFAULT_MONTY_PATH


class MontyClient:
    def __init__(self, binary_path: str = DEFAULT_MONTY_PATH):
        if not os.path.isfile(binary_path):
            raise FileNotFoundError(f"Monty binary not found at: {binary_path}. Run 'make' in tools/Monty first.")
        
        self.proc = subprocess.Popen(
            [binary_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1
        )
        self._init_uci()

    def _init_uci(self):
        self.proc.stdin.write("uci\nisready\n")
        self.proc.stdin.flush()
        while True:
            line = self.proc.stdout.readline()
            if not line or "readyok" in line:
                break

    def get_policy(self, fen: str) -> Dict[str, float]:
        """
        Queries Monty for policy on the given FEN.
        Returns a dict mapping UCI move string -> probability in [0, 1].
        """
        # Send position, policy, and isready as a delimiter
        self.proc.stdin.write(f"position fen {fen}\npolicy\nisready\n")
        self.proc.stdin.flush()

        policy = {}
        while True:
            line = self.proc.stdout.readline()
            if not line:
                break
            line_str = line.strip()
            if "readyok" in line_str:
                break
            if "->" in line_str:
                move_str, prob_str = line_str.split("->")
                move = move_str.strip()
                prob = float(prob_str.strip().replace("%", "")) / 100.0
                policy[move] = prob
        return policy

    def close(self):
        try:
            self.proc.stdin.write("quit\n")
            self.proc.stdin.flush()
            self.proc.terminate()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
