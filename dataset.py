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
    t = np.zeros(10, dtype=np.float32)
    if "x_quiet" in m_data and len(m_data["x_quiet"]) >= 10:
        xq = np.array(m_data["x_quiet"][:10], dtype=np.float32)
        t[0] = xq[0] * 512.0                  # 2 * mainHistory
        t[1] = xq[1] * 512.0                  # 2 * pawnHistory
        t[2] = xq[2] * 256.0                  # contHistory[0]
        t[3] = xq[3] * 256.0                  # contHistory[1]
        t[4] = xq[4] * 256.0                  # contHistory[2]
        t[5] = xq[5] * 256.0                  # contHistory[3]
        t[6] = xq[6] * 256.0                  # contHistory[5]
        t[7] = 16384.0 if xq[7] > 0 else 0.0  # check bonus
        t[8] = xq[8] * (18000.0 / 64.0)       # threat bonus/penalty
        t[9] = xq[9] * 256.0                  # lowPlyHistory
    elif "stat_score" in m_data:
        t[0] = float(m_data.get("stat_score", 0))
    return torch.from_numpy(t)


def extract_lmr_features_from_data(m_data: dict, tt_pv: bool = False) -> torch.Tensor:
    if "x_lmr" in m_data:
        return torch.from_numpy(np.array(m_data["x_lmr"], dtype=np.float32) / 64.0)
    x = np.zeros(8, dtype=np.float32)
    x[0] = np.clip(m_data.get("stat_score", 0) / 2000.0, -1.0, 1.0)
    rank = m_data.get("picker_rank", 1)
    x[1] = np.clip((rank - 4.0) / 8.0, -1.0, 1.0)
    is_cap = m_data.get("is_capture", False)
    x[2] = 1.0 if is_cap else -1.0
    cap_val = PIECE_VALS.get(m_data.get("captured_pt", 0), 100) if is_cap else 0
    x[3] = np.clip(cap_val / 500.0, 0.0, 1.0) if is_cap else 0.0
    x[4] = 1.0 if m_data.get("gives_check", False) else -1.0
    x[5] = 1.0 if m_data.get("is_promotion", False) else -1.0
    moved_pt = m_data.get("moved_pt", 1)
    x[6] = np.clip((moved_pt - 2.0) / 2.0, -1.0, 1.0)
    x[7] = 1.0 if tt_pv else -1.0
    return torch.from_numpy(x)
