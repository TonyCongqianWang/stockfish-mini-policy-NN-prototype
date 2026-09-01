"""
Dataset & Feature Extractor for Version 4 Dual Mini-NN (MovePicker & LMR).
Extracts 10 handcrafted terms for quiet moves and 8 raw signals for LMR.
"""

import math
from typing import Dict, List, Optional, Tuple
import chess
import numpy as np
import torch

MAX_LEGAL_MOVES = 64


def extract_node_features(
    board: chess.Board,
    ply: int = 16,
    improving: bool = True,
    cut_node: bool = False,
    pv_node: bool = False,
    static_eval: int = 0,
    prev_stat_score: int = 0,
    cutoff_cnt: int = 1,
    tt_hit: bool = False,
    tt_pv: bool = False
) -> torch.Tensor:
    """
    Extracts 16 normalized scalar features for the Node Network in [-1.0, 1.0].
    """
    features = np.zeros(16, dtype=np.float32)

    piece_vals = {chess.PAWN: 100, chess.KNIGHT: 300, chess.BISHOP: 300, chess.ROOK: 500, chess.QUEEN: 900}
    us = board.turn
    them = not us

    npm_us = sum(len(board.pieces(pt, us)) * val for pt, val in piece_vals.items() if pt != chess.PAWN)
    npm_them = sum(len(board.pieces(pt, them)) * val for pt, val in piece_vals.items() if pt != chess.PAWN)
    pawns_us = len(board.pieces(chess.PAWN, us))
    pawns_them = len(board.pieces(chess.PAWN, them))

    features[0] = np.clip((npm_us - 2000.0) / 1000.0, -1.0, 1.0)
    features[1] = np.clip((npm_them - 2000.0) / 1000.0, -1.0, 1.0)
    features[2] = np.clip((pawns_us - 4.0) / 2.0, -1.0, 1.0)
    features[3] = np.clip((pawns_them - 4.0) / 2.0, -1.0, 1.0)

    king_sq = board.king(us)
    num_pinned = board.pin_mask(us, king_sq).bit_count() if king_sq is not None else 0

    features[4] = 1.0 if cut_node else -1.0
    features[5] = np.clip(num_pinned / 4.0, -1.0, 1.0)
    features[6] = 1.0 if pv_node else -1.0
    features[7] = 1.0 if tt_pv else -1.0
    features[8] = np.clip((ply - 24.0) / 16.0, -1.0, 1.0)
    features[9] = 1.0 if improving else -1.0
    features[10] = 1.0 if tt_hit else -1.0
    features[11] = 1.0 if tt_pv else -1.0

    features[12] = np.clip(static_eval / 500.0, -1.0, 1.0)
    features[13] = np.clip(prev_stat_score / 2000.0, -1.0, 1.0)
    features[14] = np.clip((cutoff_cnt - 1.0) / 2.0, -1.0, 1.0)
    features[15] = 1.0 if (npm_us + npm_them < 3000) else -1.0

    return torch.from_numpy(features)


PIECE_VALS = {1: 100, 2: 300, 3: 300, 4: 500, 5: 900, 6: 0}


def extract_node_features_from_data(s: dict) -> Optional[torch.Tensor]:
    if "u_node" in s:
        return torch.from_numpy(np.array(s["u_node"], dtype=np.float32) / 64.0)
    return None


def extract_quiet_terms_from_data(m_data: dict, ply: int = 16) -> torch.Tensor:
    t = np.zeros(8, dtype=np.float32)
    if "t_quiet" in m_data:
        t_arr = np.array(m_data["t_quiet"], dtype=np.float32)
        if len(t_arr) >= 8:
            if len(t_arr) == 8:
                t = t_arr[:8]
            else:
                # Backward compatibility: collapse 10 terms to 8
                t[0] = t_arr[0]
                t[1] = t_arr[1]
                t[2] = t_arr[2]
                t[3] = t_arr[3]
                t[4] = t_arr[4] + t_arr[5] + t_arr[6] # deep even cont
                t[5] = t_arr[7]                       # checks
                t[6] = t_arr[8]                       # threat
                t[7] = t_arr[9]                       # low ply
    elif "stat_score" in m_data:
        t[0] = float(m_data.get("stat_score", 0))
    return torch.from_numpy(t)


def extract_lmr_features_from_data(m_data: dict, tt_pv: bool = False, cut_node: bool = False, improving: bool = True, depth: int = 16, root_delta: int = 200, tt_capture: bool = False) -> torch.Tensor:
    x = np.zeros(8, dtype=np.float32)
    if "x_lmr" in m_data and len(m_data["x_lmr"]) >= 8:
        return torch.from_numpy(np.array(m_data["x_lmr"][:8], dtype=np.float32))
    
    # 0: -delta / rootDelta
    delta = float(m_data.get("delta", 50))
    rd = float(root_delta if root_delta > 0 else 200)
    x[0] = (-delta * 1024.0) / rd

    # 1: !improving * scale / 512
    d_idx = min(depth, 31)
    mc_idx = min(m_data.get("picker_rank", 1), 63)
    red_scale = math.log(max(1, d_idx)) * math.log(max(1, mc_idx)) * 500.0
    x[1] = (0.0 if improving else 1.0) * (red_scale * 1024.0) / 512.0

    # 2: base offset
    x[2] = 1024.0

    # 3: -ttPv
    x[3] = -1024.0 if tt_pv else 0.0

    # 4: -moveCount
    rank = float(m_data.get("picker_rank", 1))
    x[4] = -rank * 1024.0

    # 5: cutNode
    x[5] = 1024.0 if cut_node else 0.0

    # 6: statScore
    stat = float(m_data.get("stat_score", 0))
    x[6] = (-stat * 1024.0) / 4096.0

    # 7: eval margin
    x[7] = 0.0

    return torch.from_numpy(x)

