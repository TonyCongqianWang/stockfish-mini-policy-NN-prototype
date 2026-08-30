"""
Dataset & Feature Extractor for Version 3 Dual Mini-NN (MovePicker & LMR).
Standardizes node and move features into [-1.0, 1.0] matching C++ SIMD inference in mininn.cpp.
"""

import math
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


def extract_quiet_features_from_data(m_data: dict, ply: int = 16) -> torch.Tensor:
    x = np.zeros(12, dtype=np.float32)
    x[0] = np.clip(m_data.get("main_hist", 0) / 16384.0, -1.0, 1.0)
    x[1] = np.clip(m_data.get("pawn_hist", 0) / 16384.0, -1.0, 1.0)
    ch = m_data.get("cont_hist", [0, 0, 0, 0, 0])
    for idx, c_val in enumerate(ch[:5]):
        x[2 + idx] = np.clip(c_val / 16384.0, -1.0, 1.0)
    x[7] = 1.0 if m_data.get("gives_check", False) else -1.0
    x[8] = 1.0 if m_data.get("threat_from", False) else -1.0
    x[9] = 1.0 if m_data.get("threat_to", False) else -1.0
    moved_pt = m_data.get("moved_pt", 1)
    x[10] = np.clip((moved_pt - 2.0) / 2.0, -1.0, 1.0)
    low_ply = m_data.get("low_ply_hist", 0)
    x[11] = np.clip((low_ply / (1 + ply)) / 16384.0, -1.0, 1.0)
    return torch.from_numpy(x)


def extract_capture_features_from_data(m_data: dict) -> torch.Tensor:
    x = np.zeros(4, dtype=np.float32)
    x[0] = np.clip(m_data.get("capt_hist", 0) / 16384.0, -1.0, 1.0)
    cap_val = PIECE_VALS.get(m_data.get("captured_pt", 0), 100)
    moved_val = PIECE_VALS.get(m_data.get("moved_pt", 1), 100)
    x[1] = np.clip(cap_val / 500.0, 0.0, 1.0)
    x[2] = np.clip((cap_val - moved_val) / 500.0, -1.0, 1.0)
    x[3] = 1.0 if m_data.get("gives_check", False) else -1.0
    return torch.from_numpy(x)


def extract_lmr_features_from_data(m_data: dict, tt_pv: bool = False) -> torch.Tensor:
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


def extract_quiet_features(
    board: chess.Board,
    move: chess.Move,
    stat_score: int = 0,
    ply: int = 16
) -> torch.Tensor:
    """
    Extracts 12 raw signals for score_quiet in [-1.0, 1.0].
    """
    x = np.zeros(12, dtype=np.float32)
    x[0] = np.clip(stat_score / 16384.0, -1.0, 1.0)
    x[1] = 0.0
    x[2] = 0.0
    x[3] = 0.0
    x[4] = 0.0
    x[5] = 0.0
    x[6] = 0.0

    board.push(move)
    gives_check = board.is_check()
    board.pop()
    x[7] = 1.0 if gives_check else -1.0
    x[8] = 0.0
    x[9] = 0.0

    moved_pt = board.piece_type_at(move.from_square) or 1
    x[10] = np.clip((moved_pt - 2.0) / 2.0, -1.0, 1.0)
    x[11] = np.clip(stat_score / (1 + ply) / 16384.0, -1.0, 1.0)
    return torch.from_numpy(x)


def extract_capture_raw_features(
    board: chess.Board,
    move: chess.Move,
    stat_score: int = 0
) -> torch.Tensor:
    """
    Extracts 4 raw tactical signals for score_capture in [-1.0, 1.0].
    """
    x = np.zeros(4, dtype=np.float32)
    x[0] = np.clip(stat_score / 16384.0, -1.0, 1.0)

    piece_vals = {chess.PAWN: 100, chess.KNIGHT: 300, chess.BISHOP: 300, chess.ROOK: 500, chess.QUEEN: 900}
    captured_pt = board.piece_type_at(move.to_square)
    moved_pt = board.piece_type_at(move.from_square)

    cap_val = piece_vals.get(captured_pt, 100) if captured_pt else 100
    moved_val = piece_vals.get(moved_pt, 100) if moved_pt else 100

    x[1] = np.clip(cap_val / 500.0, 0.0, 1.0)
    x[2] = np.clip((cap_val - moved_val) / 500.0, -1.0, 1.0)

    board.push(move)
    gives_check = board.is_check()
    board.pop()
    x[3] = 1.0 if gives_check else -1.0
    return torch.from_numpy(x)


def extract_lmr_raw_features(
    board: chess.Board,
    move: chess.Move,
    stat_score: int = 0,
    rank: int = 0,
    tt_pv: bool = False
) -> torch.Tensor:
    """
    Extracts 8 raw move features for evaluate_lmr in [-1.0, 1.0].
    """
    x = np.zeros(8, dtype=np.float32)
    x[0] = np.clip(stat_score / 2000.0, -1.0, 1.0)
    x[1] = np.clip((rank - 4.0) / 8.0, -1.0, 1.0)

    is_capture = board.is_capture(move)
    x[2] = 1.0 if is_capture else -1.0

    piece_vals = {chess.PAWN: 100, chess.KNIGHT: 300, chess.BISHOP: 300, chess.ROOK: 500, chess.QUEEN: 900}
    if is_capture:
        captured_pt = board.piece_type_at(move.to_square)
        cap_val = piece_vals.get(captured_pt, 100) if captured_pt else 100
        x[3] = np.clip(cap_val / 500.0, 0.0, 1.0)
    else:
        x[3] = 0.0

    board.push(move)
    gives_check = board.is_check()
    board.pop()
    x[4] = 1.0 if gives_check else -1.0
    x[5] = 1.0 if move.promotion else -1.0

    moved_pt = board.piece_type_at(move.from_square) or 1
    x[6] = np.clip((moved_pt - 2.0) / 2.0, -1.0, 1.0)
    x[7] = 1.0 if tt_pv else -1.0
    return torch.from_numpy(x)
