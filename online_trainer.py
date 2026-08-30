"""
On-Policy Closed-Loop Trainer & Grid Runner for Version 3 Dual Mini-NN (MovePicker & LMR):
- Pure On-Policy RL Architecture (PPO/Actor-Critic style):
    - Uniformly streamed & subsampled from ~500k FENs (eliminating dataset sorting bias).
    - Fresh on-policy rollout buffer D_t collected per iteration with active .miniNN.
    - PPO Multi-Epoch optimization (ppo_epochs) with shuffled mini-batches.
    - Rollout buffer discarded after each iteration (zero stale distribution shift).
    - On-Policy Warmup + Cosine Decay Scheduler with non-zero learning rate floor (preserving policy plasticity).
    - Periodic validation evaluation on fixed 2^15 validation pool.
    - Live closed-loop model re-export every --sync-interval steps.
- Synergistic Dual Policy Architecture:
    - MovePicker: Primary driver trained cleanly on Monty KL + Behavioral Cloning anchor.
    - LMR Search Allocation: Trained with detached MovePicker scores (scores = z_mp.detach() * 1200)
      to eliminate gradient interference on MovePicker policy.
    - Pure Depth-Independent Residual LMR: r_total = r_base(depth, rank) + delta_r_NN(move, pos).
      (Zero depth input in the neural network, guaranteeing infinite-depth generalization).
    - Search Physics Invariants:
        * Move 1 (First searched move): Always searched at FULL DEPTH (physical reduction r = 0.00, effort E = 1.00).
        * Moves 2..4: Individual Rank Log-Effort Profile Anchor (MSE).
        * Moves 5+: Merged Tail Bucket Log-Effort Profile Anchor (MSE).
        * Strong Lean Tree Upward Reduction Push: push_up_coef * E_late (push_up_coef = 0.050).
- Comprehensive 2D Diagnostic Matrix:
    - Physical Move 1 (0.00 plies, E:1.00) and late moves (Moves 2, 3, 4, 5+ bucketed) across Low (d:2-6), Mid (d:7-12), and Deep (d:13+) plies.
"""

import argparse
import json
import math
import os
import random
import sqlite3
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

import chess
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from dataset import (
    MAX_LEGAL_MOVES,
    extract_capture_features_from_data,
    extract_capture_raw_features,
    extract_lmr_features_from_data,
    extract_lmr_raw_features,
    extract_node_features,
    extract_node_features_from_data,
    extract_quiet_features,
    extract_quiet_features_from_data,
)
from model import DualMiniNN
from paths import CACHE_DIR, CALIB_CONFIG_PATH, EPD_FILE, MONTY_BIN, STOCKFISH_BIN


def load_calibration_parameters() -> Tuple[float, float, float, float]:
    if os.path.exists(CALIB_CONFIG_PATH):
        try:
            with open(CALIB_CONFIG_PATH, "r") as f:
                cfg = json.load(f)
            t_lmr = float(cfg.get("t_calib_lmr", 0.8658))
            t_mp = float(cfg.get("t_calib_mp", 0.1154))
            floor_lmr = float(cfg.get("chosen_floor_lmr", 0.010))
            floor_mp = float(cfg.get("chosen_floor_mp", 0.010))
            return t_lmr, t_mp, floor_lmr, floor_mp
        except Exception:
            pass
    return 0.8658, 0.1154, 0.010, 0.010


def load_and_subsample_fens(
    epd_path: str,
    stream_limit: int = 500_000,
    val_count: int = 1_000,
    test_count: int = 1_000,
    train_count: int = 100_000,
    seed: int = 42
) -> Tuple[List[str], List[str], List[str]]:
    print(f"Streaming up to {stream_limit:,} FENs from {epd_path}...", flush=True)
    all_fens = []
    with open(epd_path, "r") as f:
        for line in f:
            if line.startswith("Probing: "):
                fen = line.strip().replace("Probing: ", "").strip()
                if fen:
                    all_fens.append(fen)
            if len(all_fens) >= stream_limit:
                break

    print(f"Streamed {len(all_fens):,} total FENs. Shuffling with seed {seed}...", flush=True)
    rng = random.Random(seed)
    rng.shuffle(all_fens)

    val_fens = all_fens[:val_count]
    test_fens = all_fens[val_count : val_count + test_count]
    train_fens = all_fens[val_count + test_count : val_count + test_count + train_count]
    print(f"Selected {len(val_fens):,} validation FENs, {len(test_fens):,} held-out test FENs, and {len(train_fens):,} training FENs.\n", flush=True)
    return val_fens, test_fens, train_fens


def get_onpolicy_warmup_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup_steps: int,
    floor_ratio: float = 0.30,
    init_ratio: float = 0.10
) -> torch.optim.lr_scheduler.LambdaLR:
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            alpha = float(step) / float(max(1, warmup_steps))
            return init_ratio + (1.0 - init_ratio) * alpha
        else:
            progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            progress = min(1.0, max(0.0, progress))
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            return floor_ratio + (1.0 - floor_ratio) * cosine_decay

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def run_stockfish_search_worker(
    worker_id: int,
    fens_chunk: List[str],
    nodes_per_fen: int,
    sample_interval: int,
    model_path: str,
    output_path: str
) -> int:
    if os.path.exists(output_path):
        os.remove(output_path)

    env = os.environ.copy()
    env["SF_LMR_TELEMETRY"] = output_path
    env["SF_LMR_SAMPLE_INTERVAL"] = str(sample_interval)
    if model_path and os.path.exists(model_path):
        env["SF_MININN_PATH"] = model_path

    proc = subprocess.Popen(
        [STOCKFISH_BIN],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
        bufsize=1
    )

    proc.stdin.write("uci\nsetoption name Threads value 1\nisready\n")
    proc.stdin.flush()
    while True:
        line = proc.stdout.readline()
        if "readyok" in line:
            break

    for fen in fens_chunk:
        proc.stdin.write(f"position fen {fen}\ngo nodes {nodes_per_fen}\n")
        proc.stdin.flush()
        while True:
            line = proc.stdout.readline()
            if "bestmove" in line or not line:
                break

    proc.stdin.write("quit\n")
    proc.stdin.flush()
    proc.terminate()

    sample_count = 0
    if os.path.exists(output_path):
        with open(output_path, "r") as f:
            for line in f:
                if line.strip():
                    sample_count += 1
    return sample_count


def _spawn_monty():
    proc = subprocess.Popen(
        [MONTY_BIN],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1
    )
    proc.stdin.write("uci\nsetoption name Threads value 1\nsetoption name Hash value 16\nisready\n")
    proc.stdin.flush()
    while True:
        line = proc.stdout.readline()
        if "readyok" in line or not line:
            break
    return proc


def query_monty_worker(worker_id: int, fens_chunk: List[str], db_worker_path: str) -> int:
    if os.path.exists(db_worker_path):
        os.remove(db_worker_path)

    conn = sqlite3.connect(db_worker_path)
    conn.execute("PRAGMA synchronous = OFF")
    conn.execute("PRAGMA journal_mode = MEMORY")
    conn.execute("CREATE TABLE IF NOT EXISTS policies (fen TEXT PRIMARY KEY, policy_json TEXT)")

    proc = _spawn_monty()
    batch = []
    total_evaluated = 0

    for fen in fens_chunk:
        policy = {}
        for attempt in range(3):
            try:
                proc.stdin.write(f"position fen {fen}\npolicy\nisready\n")
                proc.stdin.flush()

                while True:
                    line = proc.stdout.readline()
                    if not line or "readyok" in line:
                        break
                    line_str = line.strip()
                    if "->" in line_str:
                        m_str, p_str = line_str.split("->")
                        move = m_str.strip()
                        prob = float(p_str.strip().replace("%", "")) / 100.0
                        policy[move] = prob
                break
            except Exception:
                try:
                    proc.terminate()
                except Exception:
                    pass
                proc = _spawn_monty()

        batch.append((fen, json.dumps(policy)))
        if len(batch) >= 500:
            conn.executemany("INSERT OR REPLACE INTO policies (fen, policy_json) VALUES (?, ?)", batch)
            conn.commit()
            total_evaluated += len(batch)
            batch = []

    if batch:
        conn.executemany("INSERT OR REPLACE INTO policies (fen, policy_json) VALUES (?, ?)", batch)
        conn.commit()
        total_evaluated += len(batch)

    conn.close()
    try:
        proc.stdin.write("quit\n")
        proc.stdin.flush()
        proc.terminate()
    except Exception:
        pass
    return total_evaluated


def merge_worker_dbs(main_db_path: str, worker_db_paths: List[str]):
    conn = sqlite3.connect(main_db_path)
    conn.execute("PRAGMA synchronous = OFF")
    conn.execute("PRAGMA journal_mode = WAL")
    for w_path in worker_db_paths:
        if os.path.exists(w_path):
            w_conn = sqlite3.connect(w_path)
            w_cursor = w_conn.cursor()
            w_cursor.execute("SELECT fen, policy_json FROM policies")
            batch = w_cursor.fetchall()
            conn.executemany("INSERT OR REPLACE INTO policies (fen, policy_json) VALUES (?, ?)", batch)
            conn.commit()
            w_conn.close()
            os.remove(w_path)
    conn.close()


class RolloutDataset(Dataset):
    def __init__(
        self,
        telemetry_path: str,
        monty_db_path: str,
        floor_lmr: float = 0.010,
        floor_mp: float = 0.010,
        t_lmr: float = 0.8658,
        t_mp: float = 0.1154
    ):
        self.floor_lmr = floor_lmr
        self.floor_mp = floor_mp
        self.t_lmr = t_lmr
        self.t_mp = t_mp
        self.items = []

        conn = sqlite3.connect(monty_db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT fen FROM policies")
        cursor = conn.cursor()
        cursor.execute("SELECT fen, policy_json FROM policies")
        policies = {row[0]: json.loads(row[1]) for row in cursor.fetchall()}
        conn.close()

        with open(telemetry_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    s = json.loads(line)
                except Exception:
                    continue

                fen = s["fen"]
                depth = s.get("depth", 8)
                ply = s.get("ply", 16)
                improving = s.get("improving", True)
                static_eval = s.get("static_eval", 0)
                cut_node = s.get("cut_node", False)
                pv_node = s.get("pv_node", False)
                moves_info = s["moves"]

                try:
                    board = chess.Board(fen)
                except Exception:
                    continue

                # Depth-Independent Node Features (with cut_node and pv_node)
                u_node = extract_node_features_from_data(s)
                if u_node is None:
                    prev_stat_score = s.get("prev_stat_score", 0)
                    cutoff_cnt = s.get("cutoff_cnt", 1)
                    u_node = extract_node_features(
                        board,
                        ply=ply,
                        improving=improving,
                        cut_node=cut_node,
                        pv_node=pv_node,
                        static_eval=static_eval,
                        prev_stat_score=prev_stat_score,
                        cutoff_cnt=cutoff_cnt
                    )

                num_moves = min(len(moves_info), MAX_LEGAL_MOVES)
                if num_moves < 3:
                    continue

                x_quiet = torch.zeros(MAX_LEGAL_MOVES, 12, dtype=torch.float32)
                x_cap = torch.zeros(MAX_LEGAL_MOVES, 4, dtype=torch.float32)
                x_lmr = torch.zeros(MAX_LEGAL_MOVES, 8, dtype=torch.float32)
                is_cap_mask = torch.zeros(MAX_LEGAL_MOVES, dtype=torch.bool)
                legal_mask = torch.zeros(MAX_LEGAL_MOVES, dtype=torch.bool)
                r_base = torch.zeros(MAX_LEGAL_MOVES, dtype=torch.float32)
                r_legacy = torch.zeros(MAX_LEGAL_MOVES, dtype=torch.float32)
                z_legacy_mp = torch.zeros(MAX_LEGAL_MOVES, dtype=torch.float32)
                target_p_mp = torch.zeros(MAX_LEGAL_MOVES, dtype=torch.float32)

                m_policy = policies.get(fen, {})

                raw_p_list = []
                for i in range(num_moves):
                    m_data = moves_info[i]
                    uci_str = m_data["move"]
                    stat_score = m_data.get("stat_score", 0)
                    is_capture = m_data.get("is_capture", False)
                    rank = m_data.get("picker_rank", i + 1)

                    try:
                        x_quiet[i] = extract_quiet_features_from_data(m_data, ply=ply)
                        x_cap[i] = extract_capture_features_from_data(m_data)
                        x_lmr[i] = extract_lmr_features_from_data(m_data)
                    except Exception:
                        move_obj = chess.Move.from_uci(uci_str)
                        x_quiet[i] = extract_quiet_features(board, move_obj, stat_score=stat_score, ply=ply)
                        x_cap[i] = extract_capture_raw_features(board, move_obj, stat_score=stat_score)
                        x_lmr[i] = extract_lmr_raw_features(board, move_obj, stat_score=stat_score, rank=rank)

                    is_cap_mask[i] = is_capture
                    legal_mask[i] = True

                    base_red = (math.log(max(1, depth)) * math.log(max(1, rank)) * 500.0) / 1024.0
                    legacy_reduction = (math.log(max(1, depth)) * math.log(max(1, rank)) * 500.0 - stat_score * (439.0 / 4096.0)) / 1024.0
                    r_base[i] = base_red
                    r_legacy[i] = legacy_reduction
                    z_legacy_mp[i] = float(np.clip(stat_score / 16384.0, -1.0, 1.0))

                    raw_p = m_policy.get(uci_str, 0.0)
                    raw_p_list.append(raw_p)

                if raw_p_list:
                    p_arr = np.array(raw_p_list, dtype=np.float64)
                    p_mp_f = np.maximum(p_arr, self.floor_mp)
                    p_mp_norm = p_mp_f / np.sum(p_mp_f)
                    log_mp = np.log(p_mp_norm + 1e-12) / self.t_mp
                    exp_mp = np.exp(log_mp - np.max(log_mp))
                    target_p_mp[:num_moves] = torch.from_numpy(exp_mp / np.sum(exp_mp)).float()

                depth_tensor = torch.tensor(float(depth), dtype=torch.float32)
                self.items.append((u_node, x_quiet, x_cap, x_lmr, is_cap_mask, r_base, r_legacy, z_legacy_mp, target_p_mp, legal_mask, depth_tensor))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


def compute_combined_losses(
    z_quiet: torch.Tensor,
    z_cap: torch.Tensor,
    delta_r_nn: torch.Tensor,
    tau_mp: torch.Tensor,
    tau_lmr: torch.Tensor,
    target_p_mp: torch.Tensor,
    z_legacy_mp: torch.Tensor,
    r_base: torch.Tensor,
    r_legacy: torch.Tensor,
    is_cap_mask: torch.Tensor,
    legal_mask: torch.Tensor,
    depth: torch.Tensor,
    mp_shape_coef: float = 0.20,
    lmr_ord_coef: float = 0.40,
    rank_profile_coef: float = 0.40,
    push_up_coef: float = 0.065
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Computes Direct Physical Search Allocation and Independent Quiet/Capture MovePicker Losses:
    1. MovePicker:
       - Quiets: Monty KL divergence + Shape Loss on quiet moves using z_quiet.
       - Captures: Monty KL divergence + Shape Loss on capture moves using z_cap.
    2. Direct Physical Search Allocation:
       - Move 0: Searched at full depth (Effort = 1.00).
       - Moves 1..M-1: Searched with late reductions r_real = clamp(r_base + delta_r_nn, -2.0, depth - 1.0).
       - Target Allocation Q(i*) = E_eff(i*) / sum(E_eff) directly from physical search sequence.
    3. LMR Profile MSE: Anchored to Master's baseline reduction profile.
    4. Upward Push on Late Moves: push_up_coef * sum(E_late).
    """
    B, M = z_quiet.shape
    i_star = target_p_mp.argmax(dim=-1)

    w_raw = torch.clamp(torch.sqrt(depth.clamp(min=1.0) / 8.0), 0.70, 1.40)
    w_depth = w_raw / w_raw.mean()

    # 1. Quiet Moves KL & Top-1 Match (Filtered by minimum Monty mass threshold >= 5%)
    quiet_mask = legal_mask & (~is_cap_mask)
    masked_zq = z_quiet.masked_fill(~quiet_mask, -1e4)
    log_probs_q = F.log_softmax(masked_zq / tau_mp, dim=-1)
    p_q = target_p_mp * quiet_mask.float()
    w_q_pos = p_q.sum(dim=-1)
    has_quiets = (w_q_pos >= 0.05) & (quiet_mask.sum(dim=-1) > 1)

    loss_mp_kl_q = torch.tensor(0.0, device=z_quiet.device)
    acc_q = torch.tensor(0.0, device=z_quiet.device)
    if has_quiets.sum() > 0:
        loss_q_pos = -(p_q[has_quiets] * log_probs_q[has_quiets]).sum(dim=-1)
        loss_mp_kl_q = (w_depth[has_quiets] * loss_q_pos).mean()
        acc_q = (masked_zq[has_quiets].argmax(dim=-1) == p_q[has_quiets].argmax(dim=-1)).float().mean()

    # 2. Capture Moves KL & Top-1 Match (Filtered by minimum Monty mass threshold >= 5%)
    cap_mask = legal_mask & is_cap_mask
    masked_zc = z_cap.masked_fill(~cap_mask, -1e4)
    log_probs_c = F.log_softmax(masked_zc / tau_mp, dim=-1)
    p_c = target_p_mp * cap_mask.float()
    w_c_pos = p_c.sum(dim=-1)
    has_caps = (w_c_pos >= 0.05) & (cap_mask.sum(dim=-1) > 1)

    loss_mp_kl_c = torch.tensor(0.0, device=z_cap.device)
    acc_c = torch.tensor(0.0, device=z_cap.device)
    if has_caps.sum() > 0:
        loss_c_pos = -(p_c[has_caps] * log_probs_c[has_caps]).sum(dim=-1)
        loss_mp_kl_c = (w_depth[has_caps] * loss_c_pos).mean()
        acc_c = (masked_zc[has_caps].argmax(dim=-1) == p_c[has_caps].argmax(dim=-1)).float().mean()

    # Distributional Shape Matching (Mean + Variance for Quiets and Captures)
    loss_shape_quiet = torch.tensor(0.0, device=z_quiet.device)
    if quiet_mask.sum() > 1:
        z_q_nn = z_quiet[quiet_mask]
        z_q_leg = z_legacy_mp[quiet_mask]
        loss_shape_quiet = (z_q_nn.mean() - z_q_leg.mean()).pow(2) + (z_q_nn.std() - z_q_leg.std()).pow(2)

    loss_shape_cap = torch.tensor(0.0, device=z_cap.device)
    if cap_mask.sum() > 1:
        z_c_nn = z_cap[cap_mask]
        z_c_leg = z_legacy_mp[cap_mask]
        loss_shape_cap = (z_c_nn.mean() - z_c_leg.mean()).pow(2) + (z_c_nn.std() - z_c_leg.std()).pow(2)

    loss_mp_shape = loss_shape_quiet + loss_shape_cap
    loss_mp_total = loss_mp_kl_q + loss_mp_kl_c + mp_shape_coef * loss_mp_shape

    # 3. Residual Reductions on physical moves
    r_total_nn = r_base + delta_r_nn
    max_red = (depth.unsqueeze(1) - 1.0).clamp(min=0.0)
    min_red = torch.tensor(-2.0, device=delta_r_nn.device)
    r_real_nn = torch.minimum(torch.maximum(r_total_nn, min_red), max_red)
    r_real_leg = torch.minimum(torch.maximum(r_legacy, min_red), max_red)

    # 4. Direct Physical Search Effort: Move 0 is searched at full depth (effort = 1.0), Moves 1..M-1 are late moves
    masked_r = r_real_nn.masked_fill(~legal_mask, 1e4)
    E_late = torch.exp(-masked_r / tau_lmr) * legal_mask.float()
    E_eff = E_late.clone()
    E_eff[:, 0] = 1.0  # Move 1 in physical search is always full depth

    D = E_eff.sum(dim=-1, keepdim=True)
    E_star = E_eff.gather(1, i_star.unsqueeze(1))
    Q_star = (E_star / (D + 1e-12)).squeeze(1)
    loss_lmr_order = (w_depth * -torch.log(Q_star + 1e-12)).mean()

    # 5. Rank Profile MSE (Anchored to Master's baseline reduction profile)
    sorted_r_nn, _ = torch.sort(r_real_nn.masked_fill(~legal_mask, 1e4), dim=-1)
    sorted_r_leg, _ = torch.sort(r_real_leg.masked_fill(~legal_mask, 1e4), dim=-1)

    loss_rank_profile = torch.tensor(0.0, device=delta_r_nn.device)
    valid_ranks = 0
    for k in range(1, min(M, 4)):
        mask_k = (legal_mask.sum(dim=-1) > k)
        if mask_k.sum() > 0:
            e_nn_k = torch.exp(-sorted_r_nn[mask_k, k] / tau_lmr).mean()
            e_leg_k = torch.exp(-sorted_r_leg[mask_k, k] / tau_lmr).mean()
            loss_rank_profile = loss_rank_profile + (torch.log(e_nn_k + 1e-6) - torch.log(e_leg_k + 1e-6)).pow(2)
            valid_ranks += 1

    num_legal = legal_mask.sum(dim=-1, keepdim=True)
    rank_indices = torch.arange(M, device=delta_r_nn.device).unsqueeze(0).expand(B, M)
    tail_mask = (rank_indices >= 4) & (rank_indices < num_legal)
    if tail_mask.sum() > 0:
        e_nn_tail = torch.exp(-sorted_r_nn[tail_mask] / tau_lmr).mean()
        e_leg_tail = torch.exp(-sorted_r_leg[tail_mask] / tau_lmr).mean()
        loss_rank_profile = loss_rank_profile + (torch.log(e_nn_tail + 1e-6) - torch.log(e_leg_tail + 1e-6)).pow(2)
        valid_ranks += 1

    loss_rank_profile = loss_rank_profile / max(1, valid_ranks)

    # 6. Upward Push on Late Moves (Moves 1..M-1): Scaled by theoretical capacity log(1 + j) / log(1 + M)
    j_ranks = torch.arange(1, M, device=delta_r_nn.device, dtype=torch.float32).unsqueeze(0)
    push_weight = torch.log(1.0 + j_ranks) / math.log(1.0 + M)
    effort_late = (E_late[:, 1:] * push_weight).sum(dim=-1)
    loss_push_up = (w_depth * effort_late).mean()

    loss_lmr_total = lmr_ord_coef * loss_lmr_order + rank_profile_coef * loss_rank_profile + push_up_coef * loss_push_up
    loss_total = loss_mp_total + loss_lmr_total
    mean_q_star = Q_star.mean()

    return loss_total, loss_mp_kl_q, loss_mp_kl_c, loss_mp_shape, loss_lmr_order, loss_rank_profile, loss_push_up, acc_q, acc_c, mean_q_star


def evaluate_handcrafted_master(
    loader: DataLoader,
    tau_mp: float = 0.1154,
    tau_lmr: float = 0.8658
) -> Dict[str, float]:
    mp_kl_q_sum, mp_kl_c_sum, lmr_ord_sum = 0.0, 0.0, 0.0
    quiet_top1_sum, cap_top1_sum, q_star_sum = 0.0, 0.0, 0.0
    total_count, tot_effort_sum, mean_r_sum = 0, 0.0, 0.0

    with torch.no_grad():
        for u_node, x_quiet, x_cap, x_lmr, is_cap, r_base, r_legacy, z_legacy_mp, target_p_mp, legal_mask, depth in loader:
            B, M = z_legacy_mp.shape
            i_star = target_p_mp.argmax(dim=-1)

            # Master Quiet KL (Filtered by minimum Monty mass threshold >= 5%)
            quiet_mask = legal_mask & (~is_cap)
            masked_zq = z_legacy_mp.masked_fill(~quiet_mask, -1e4)
            log_probs_q = F.log_softmax(masked_zq / tau_mp, dim=-1)
            p_q = target_p_mp * quiet_mask.float()
            w_q_pos = p_q.sum(dim=-1)
            has_quiets = (w_q_pos >= 0.05) & (quiet_mask.sum(dim=-1) > 1)
            loss_mp_kl_q = torch.tensor(0.0, device=z_legacy_mp.device)
            acc_q = torch.tensor(0.0, device=z_legacy_mp.device)
            if has_quiets.sum() > 0:
                loss_mp_kl_q = -(p_q[has_quiets] * log_probs_q[has_quiets]).sum(dim=-1).mean()
                acc_q = (masked_zq[has_quiets].argmax(dim=-1) == p_q[has_quiets].argmax(dim=-1)).float().mean()

            # Master Capture KL (Filtered by minimum Monty mass threshold >= 5%)
            cap_mask = legal_mask & is_cap
            masked_zc = z_legacy_mp.masked_fill(~cap_mask, -1e4)
            log_probs_c = F.log_softmax(masked_zc / tau_mp, dim=-1)
            p_c = target_p_mp * cap_mask.float()
            w_c_pos = p_c.sum(dim=-1)
            has_caps = (w_c_pos >= 0.05) & (cap_mask.sum(dim=-1) > 1)
            loss_mp_kl_c = torch.tensor(0.0, device=z_legacy_mp.device)
            acc_c = torch.tensor(0.0, device=z_legacy_mp.device)
            if has_caps.sum() > 0:
                loss_mp_kl_c = -(p_c[has_caps] * log_probs_c[has_caps]).sum(dim=-1).mean()
                acc_c = (masked_zc[has_caps].argmax(dim=-1) == p_c[has_caps].argmax(dim=-1)).float().mean()

            # Exact physical reductions
            max_red = (depth.unsqueeze(1) - 1.0).clamp(min=0.0)
            min_red = torch.tensor(-2.0, device=r_legacy.device)
            r_real_leg = torch.minimum(torch.maximum(r_legacy, min_red), max_red)

            masked_r = r_real_leg.masked_fill(~legal_mask, 1e4)
            E_late = torch.exp(-masked_r / tau_lmr) * legal_mask.float()
            E_eff = E_late.clone()
            E_eff[:, 0] = 1.0

            D = E_eff.sum(dim=-1, keepdim=True)
            E_star = E_eff.gather(1, i_star.unsqueeze(1))
            Q_star = (E_star / (D + 1e-12)).squeeze(1)
            loss_lmr_order = -(torch.log(Q_star + 1e-12)).mean()

            effort_leg_late = (E_late[:, 1:]).sum(dim=-1).mean()
            mean_r = r_real_leg[legal_mask].mean()

            total_count += B
            mp_kl_q_sum += loss_mp_kl_q.item() * B
            mp_kl_c_sum += loss_mp_kl_c.item() * B
            lmr_ord_sum += loss_lmr_order.item() * B
            quiet_top1_sum += acc_q.item() * B
            cap_top1_sum += acc_c.item() * B
            q_star_sum += Q_star.mean().item() * B
            tot_effort_sum += effort_leg_late.item() * B
            mean_r_sum += mean_r.item() * B

    n = max(1, total_count)
    return {
        "master_quiet_top1": (quiet_top1_sum / n) * 100.0,
        "master_cap_top1": (cap_top1_sum / n) * 100.0,
        "master_mp_kl_q": mp_kl_q_sum / n,
        "master_mp_kl_c": mp_kl_c_sum / n,
        "master_mp_kl": (mp_kl_q_sum + mp_kl_c_sum) / n,
        "master_q_search_star": (q_star_sum / n) * 100.0,
        "master_lmr_ord": lmr_ord_sum / n,
        "master_mean_effort": tot_effort_sum / n,
        "master_mean_reduction": mean_r_sum / n,
    }


def evaluate_validation_rollout(
    model: DualMiniNN,
    loader: DataLoader,
    mp_anchor_coef: float = 0.20,
    lmr_ord_coef: float = 0.40,
    rank_profile_coef: float = 0.40,
    push_up_coef: float = 0.065,
    tau_lmr: float = 0.8658
) -> Dict[str, float]:
    model.eval()
    tot_loss_sum, mp_kl_q_sum, mp_kl_c_sum, mp_anc_sum, lmr_ord_sum = 0.0, 0.0, 0.0, 0.0, 0.0
    rank_prof_loss_sum, push_loss_sum = 0.0, 0.0
    quiet_top1_sum, cap_top1_sum, q_star_sum = 0.0, 0.0, 0.0
    actual_effort_sum, actual_r_sum, total_count = 0.0, 0.0, 0

    with torch.no_grad():
        for u_node, x_quiet, x_cap, x_lmr, is_cap, r_base, r_legacy, z_legacy_mp, target_p_mp, legal_mask, depth in loader:
            w_quiet, z_latents, tau_mp, tau_lmr, quiet_scores, cap_scores, delta_r_nn = model(u_node, x_quiet, x_cap, x_lmr)

            z_quiet = quiet_scores / 32768.0
            z_cap = cap_scores / 32768.0

            loss, loss_mp_kl_q, loss_mp_kl_c, loss_mp_shape, loss_lmr_ord, loss_rank_prof, loss_push, acc_q, acc_c, mean_q_star = compute_combined_losses(
                z_quiet, z_cap, delta_r_nn, tau_mp, tau_lmr, target_p_mp, z_legacy_mp, r_base, r_legacy, is_cap, legal_mask, depth,
                mp_shape_coef=mp_anchor_coef, lmr_ord_coef=lmr_ord_coef, rank_profile_coef=rank_profile_coef, push_up_coef=push_up_coef
            )

            B = u_node.size(0)
            total_count += B
            tot_loss_sum += loss.item() * B
            mp_kl_q_sum += loss_mp_kl_q.item() * B
            mp_kl_c_sum += loss_mp_kl_c.item() * B
            mp_anc_sum += loss_mp_shape.item() * B
            lmr_ord_sum += loss_lmr_ord.item() * B
            rank_prof_loss_sum += loss_rank_prof.item() * B
            push_loss_sum += loss_push.item() * B
            quiet_top1_sum += acc_q.item() * B
            cap_top1_sum += acc_c.item() * B
            q_star_sum += mean_q_star.item() * B

            r_total_nn = r_base + delta_r_nn
            max_red = (depth.unsqueeze(1) - 1.0).clamp(min=0.0)
            min_red = torch.tensor(-2.0, device=delta_r_nn.device)
            r_real_nn = torch.minimum(torch.maximum(r_total_nn, min_red), max_red)

            masked_r = r_real_nn.masked_fill(~legal_mask, 1e4)
            E_late = torch.exp(-masked_r / tau_lmr) * legal_mask.float()
            actual_effort = (E_late[:, 1:]).sum(dim=-1).mean()
            mean_r = (r_real_nn * legal_mask.float()).sum() / legal_mask.sum()

            actual_effort_sum += actual_effort.item() * B
            actual_r_sum += mean_r.item() * B

    model.train()
    n = max(1, total_count)
    return {
        "total_loss": tot_loss_sum / n,
        "mp_kl_q": mp_kl_q_sum / n,
        "mp_kl_c": mp_kl_c_sum / n,
        "mp_kl_loss": (mp_kl_q_sum + mp_kl_c_sum) / n,
        "mp_anchor_loss": mp_anc_sum / n,
        "lmr_order_loss": lmr_ord_sum / n,
        "rank_profile_loss": rank_prof_loss_sum / n,
        "push_loss": push_loss_sum / n,
        "mp_quiet_top1": (quiet_top1_sum / n) * 100.0,
        "mp_cap_top1": (cap_top1_sum / n) * 100.0,
        "mean_q_search_star": (q_star_sum / n) * 100.0,
        "mean_search_effort": actual_effort_sum / n,
        "mean_reduction": actual_r_sum / n,
    }


def evaluate_2d_depth_rank_matrix(
    model: nn.Module,
    loader: DataLoader,
    tau_lmr: float = 0.8658
):
    """
    Computes a comprehensive 2D Matrix of late-move reductions and search efforts:
    - 3 Depth Bands: Low (d: 2-6), Mid (d: 7-12), Deep (d: 13+)
    - 4 Late-Move Buckets: Move 2, Move 3, Move 4, Moves 5+ (Tail)
    """
    model.eval()
    bands = [
        ("Low (d: 2-6)", 2, 6),
        ("Mid (d: 7-12)", 7, 12),
        ("Deep (d: 13+)", 13, 100)
    ]

    stats = {
        name: {
            "nn_red": [0.0] * 4, "nn_eff": [0.0] * 4,
            "leg_red": [0.0] * 4, "leg_eff": [0.0] * 4,
            "counts": [0] * 4
        } for name, _, _ in bands
    }

    with torch.no_grad():
        for u_node, x_quiet, x_cap, x_lmr, is_cap, r_base, r_legacy, z_legacy_mp, target_p_mp, legal_mask, depth in loader:
            w_quiet, z_latents, tau_mp, t_lmr_pred, quiet_scores, cap_scores, delta_r_nn = model(u_node, x_quiet, x_cap, x_lmr)

            r_total_nn = r_base + delta_r_nn
            max_red = (depth.unsqueeze(1) - 1.0).clamp(min=0.0)
            min_red = torch.tensor(-2.0, device=delta_r_nn.device)
            r_real_nn = torch.minimum(torch.maximum(r_total_nn, min_red), max_red)
            r_real_leg = torch.minimum(torch.maximum(r_legacy, min_red), max_red)

            E_nn = torch.exp(-r_real_nn / tau_lmr)
            E_leg = torch.exp(-r_real_leg / tau_lmr)

            sorted_r_nn, _ = torch.sort(r_real_nn.masked_fill(~legal_mask, 1e4), dim=-1)
            sorted_r_leg, _ = torch.sort(r_real_leg.masked_fill(~legal_mask, 1e4), dim=-1)

            sorted_e_nn, _ = torch.sort(E_nn.masked_fill(~legal_mask, 0.0), dim=-1, descending=True)
            sorted_e_leg, _ = torch.sort(E_leg.masked_fill(~legal_mask, 0.0), dim=-1, descending=True)

            B = depth.size(0)
            for b in range(B):
                d_val = depth[b].item()
                num_m = legal_mask[b].sum().item()
                if num_m < 3:
                    continue

                band_name = None
                for name, d_min, d_max in bands:
                    if d_min <= d_val <= d_max:
                        band_name = name
                        break
                if not band_name:
                    continue

                b_stats = stats[band_name]

                # Moves 2, 3, 4 (Late Moves: sorted ranks k = 1, 2, 3 -> index 0, 1, 2)
                for k in range(1, 4):
                    if num_m > k:
                        b_stats["nn_red"][k-1] += sorted_r_nn[b, k].item()
                        b_stats["nn_eff"][k-1] += sorted_e_nn[b, k].item()
                        b_stats["leg_red"][k-1] += sorted_r_leg[b, k].item()
                        b_stats["leg_eff"][k-1] += sorted_e_leg[b, k].item()
                        b_stats["counts"][k-1] += 1

                # Move 5+ (Tail Bucket: sorted ranks k >= 4 -> index 3)
                if num_m > 4:
                    tail_nn_r = sorted_r_nn[b, 4:num_m].mean().item()
                    tail_nn_e = sorted_e_nn[b, 4:num_m].mean().item()
                    tail_leg_r = sorted_r_leg[b, 4:num_m].mean().item()
                    tail_leg_e = sorted_e_leg[b, 4:num_m].mean().item()

                    b_stats["nn_red"][3] += tail_nn_r
                    b_stats["nn_eff"][3] += tail_nn_e
                    b_stats["leg_red"][3] += tail_leg_r
                    b_stats["leg_eff"][3] += tail_leg_e
                    b_stats["counts"][3] += 1

    print("\n" + "=" * 105, flush=True)
    print("                2D LATE-MOVE REDUCTIONS & SEARCH EFFORT MATRIX (DEPTH BANDS x MOVE RANKS)", flush=True)
    print("=" * 105, flush=True)
    print(f"{'Depth Band':<15} | {'Policy':<14} | {'Move 2 (Late)':<18} | {'Move 3 (Late)':<18} | {'Move 4 (Late)':<18} | {'Move 5+ (Tail)':<18}", flush=True)
    print("-" * 105, flush=True)

    for name, _, _ in bands:
        b_stats = stats[name]
        c = [max(1, cnt) for cnt in b_stats["counts"]]

        nn_r_str = [f"{b_stats['nn_red'][i]/c[i]:+5.2f} (E:{b_stats['nn_eff'][i]/c[i]:4.2f})" for i in range(4)]
        leg_r_str = [f"{b_stats['leg_red'][i]/c[i]:+5.2f} (E:{b_stats['leg_eff'][i]/c[i]:4.2f})" for i in range(4)]

        print(f"{name:<15} | {'Neural MiniNN':<14} | {nn_r_str[0]:<18} | {nn_r_str[1]:<18} | {nn_r_str[2]:<18} | {nn_r_str[3]:<18}", flush=True)
        print(f"{'':<15} | {'Legacy Master':<14} | {leg_r_str[0]:<18} | {leg_r_str[1]:<18} | {leg_r_str[2]:<18} | {leg_r_str[3]:<18}", flush=True)
        print("-" * 105, flush=True)
    print("=" * 105 + "\n", flush=True)
    model.train()


def collect_validation_rollout_even(
    val_fens_pool: List[str],
    target_samples: int,
    nodes_per_fen: int,
    workers: int,
    session_tag: str
) -> Tuple[str, str]:
    os.makedirs(CACHE_DIR, exist_ok=True)
    merged_tel_path = os.path.join(CACHE_DIR, f"sf_tel_{session_tag}.jsonl")
    monty_db_path = os.path.join(CACHE_DIR, f"monty_{session_tag}.db")

    conn = sqlite3.connect(monty_db_path)
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("CREATE TABLE IF NOT EXISTS policies (fen TEXT PRIMARY KEY, policy_json TEXT)")
    conn.commit()
    conn.close()

    if os.path.exists(merged_tel_path):
        valid_schema = False
        try:
            with open(merged_tel_path, "r") as f:
                first_line = f.readline()
                if first_line.strip():
                    sample = json.loads(first_line.strip())
                    if sample.get("moves") and "picker_rank" in sample["moves"][0]:
                        valid_schema = True
        except Exception:
            valid_schema = False

        if valid_schema:
            count = sum(1 for _ in open(merged_tel_path))
            if count >= target_samples:
                return merged_tel_path, monty_db_path
        else:
            os.remove(merged_tel_path)

    samples_per_fen = max(1, math.ceil(target_samples / len(val_fens_pool)))
    sample_interval = max(1000, nodes_per_fen // samples_per_fen)

    chunk_size = math.ceil(len(val_fens_pool) / workers)
    chunks = [val_fens_pool[i : i + chunk_size] for i in range(0, len(val_fens_pool), chunk_size)]
    worker_tel_paths = [os.path.join(CACHE_DIR, f"tel_w{w_id}_{session_tag}.jsonl") for w_id in range(len(chunks))]

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(run_stockfish_search_worker, w_id, chunk, nodes_per_fen, sample_interval, "", worker_tel_paths[w_id])
            for w_id, chunk in enumerate(chunks)
        ]
        for f in as_completed(futures):
            f.result()

    all_lines = []
    for p in worker_tel_paths:
        if os.path.exists(p):
            with open(p, "r") as in_f:
                for line in in_f:
                    if line.strip():
                        all_lines.append(line)
            os.remove(p)

    rng = random.Random(42)
    rng.shuffle(all_lines)

    selected_lines = all_lines[:target_samples] if len(all_lines) >= target_samples else all_lines
    if os.path.exists(merged_tel_path):
        os.remove(merged_tel_path)

    with open(merged_tel_path, "w") as out_f:
        for line in selected_lines:
            out_f.write(line)

    conn = sqlite3.connect(monty_db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT fen FROM policies")
    cached = set(row[0] for row in cursor.fetchall())
    conn.close()

    uncached_fens = set()
    with open(merged_tel_path, "r") as f:
        for line in f:
            if line.strip():
                try:
                    fen = json.loads(line.strip()).get("fen")
                    if fen and fen not in cached:
                        uncached_fens.add(fen)
                except Exception:
                    pass

    fens_to_query = list(uncached_fens)
    if fens_to_query:
        m_chunk_size = math.ceil(len(fens_to_query) / workers)
        m_chunks = [fens_to_query[i : i + m_chunk_size] for i in range(0, len(fens_to_query), m_chunk_size)]
        worker_db_paths = [os.path.join(CACHE_DIR, f"m_w{w_id}_{session_tag}.db") for w_id in range(len(m_chunks))]

        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(query_monty_worker, w_id, m_chunk, worker_db_paths[w_id])
                for w_id, m_chunk in enumerate(m_chunks)
            ]
            for f in as_completed(futures):
                f.result()

        merge_worker_dbs(monty_db_path, worker_db_paths)

    return merged_tel_path, monty_db_path


def collect_target_samples(
    fens_pool: List[str],
    fen_offset: int,
    target_samples: int,
    nodes_per_fen: int,
    sample_interval: int,
    model_path: str,
    workers: int,
    session_tag: str
) -> Tuple[str, str, int]:
    os.makedirs(CACHE_DIR, exist_ok=True)
    merged_tel_path = os.path.join(CACHE_DIR, f"sf_tel_{session_tag}.jsonl")
    monty_db_path = os.path.join(CACHE_DIR, f"monty_{session_tag}.db")

    conn = sqlite3.connect(monty_db_path)
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("CREATE TABLE IF NOT EXISTS policies (fen TEXT PRIMARY KEY, policy_json TEXT)")
    conn.commit()
    conn.close()

    curr_ptr = fen_offset
    all_lines = []

    while len(all_lines) < target_samples:
        batch_fens_count = max(30, int(math.ceil((target_samples - len(all_lines)) / 40)))
        fens_slice = [fens_pool[(curr_ptr + i) % len(fens_pool)] for i in range(batch_fens_count)]
        curr_ptr += batch_fens_count

        chunk_size = math.ceil(len(fens_slice) / workers)
        chunks = [fens_slice[i : i + chunk_size] for i in range(0, len(fens_slice), chunk_size)]
        worker_tel_paths = [os.path.join(CACHE_DIR, f"tel_w{w_id}_{session_tag}.jsonl") for w_id in range(len(chunks))]

        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(run_stockfish_search_worker, w_id, chunk, nodes_per_fen, sample_interval, model_path, worker_tel_paths[w_id])
                for w_id, chunk in enumerate(chunks)
            ]
            for f in as_completed(futures):
                f.result()

        for p in worker_tel_paths:
            if os.path.exists(p):
                with open(p, "r") as in_f:
                    for line in in_f:
                        if line.strip():
                            all_lines.append(line)
                os.remove(p)

    random.shuffle(all_lines)

    if os.path.exists(merged_tel_path):
        os.remove(merged_tel_path)

    with open(merged_tel_path, "w") as out_f:
        for line in all_lines[:target_samples]:
            out_f.write(line)

    conn = sqlite3.connect(monty_db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT fen FROM policies")
    cached = set(row[0] for row in cursor.fetchall())
    conn.close()

    uncached_fens = set()
    with open(merged_tel_path, "r") as f:
        for line in f:
            if line.strip():
                try:
                    fen = json.loads(line.strip()).get("fen")
                    if fen and fen not in cached:
                        uncached_fens.add(fen)
                except Exception:
                    pass

    fens_to_query = list(uncached_fens)
    if fens_to_query:
        m_chunk_size = math.ceil(len(fens_to_query) / workers)
        m_chunks = [fens_to_query[i : i + m_chunk_size] for i in range(0, len(fens_to_query), m_chunk_size)]
        worker_db_paths = [os.path.join(CACHE_DIR, f"m_w{w_id}_{session_tag}.db") for w_id in range(len(m_chunks))]

        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(query_monty_worker, w_id, m_chunk, worker_db_paths[w_id])
                for w_id, m_chunk in enumerate(m_chunks)
            ]
            for f in as_completed(futures):
                f.result()

        merge_worker_dbs(monty_db_path, worker_db_paths)

    return merged_tel_path, monty_db_path, curr_ptr


def run_heldout_online_evaluation(
    model: Optional[DualMiniNN],
    test_fens: List[str],
    target_samples: int = 32768,
    nodes_per_fen: int = 50_000,
    workers: int = 4,
    session_tag: str = "heldout_test"
) -> Dict[str, float]:
    tag_name = "NEURAL MININN" if model is not None else "HANDCRAFTED STOCKFISH MASTER"
    print("\n" + "=" * 80, flush=True)
    print(f"   {tag_name} - HELDOUT ON-POLICY EVALUATION ({len(test_fens):,} TEST FENs)", flush=True)
    print("=" * 80, flush=True)

    temp_model_path = ""
    if model is not None:
        temp_model_path = f"/tmp/online_test_{session_tag}.miniNN"
        model.export_quantized_binary(temp_model_path)

    tel_path = os.path.join(CACHE_DIR, f"heldout_tel_{session_tag}.jsonl")
    db_path = os.path.join(CACHE_DIR, f"heldout_monty_{session_tag}.db")
    if os.path.exists(tel_path):
        os.remove(tel_path)

    # 1. Rollout C++ Stockfish searches on heldout test FENs
    samples_per_fen = max(1, math.ceil(target_samples / len(test_fens)))
    sample_interval = max(500, nodes_per_fen // samples_per_fen)

    chunk_size = math.ceil(len(test_fens) / workers)
    chunks = [test_fens[i : i + chunk_size] for i in range(0, len(test_fens), chunk_size)]
    worker_tel_paths = [os.path.join(CACHE_DIR, f"heldout_tel_w{w_id}_{session_tag}.jsonl") for w_id in range(len(chunks))]

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(run_stockfish_search_worker, w_id, chunk, nodes_per_fen, sample_interval, temp_model_path, worker_tel_paths[w_id])
            for w_id, chunk in enumerate(chunks)
        ]
        for f in as_completed(futures):
            f.result()

    all_lines = []
    for p in worker_tel_paths:
        if os.path.exists(p):
            with open(p, "r") as in_f:
                for line in in_f:
                    if line.strip():
                        all_lines.append(line)
            os.remove(p)

    with open(tel_path, "w") as out_f:
        for line in all_lines:
            out_f.write(line)

    # 2. Extract unique FENs from tel_path and query Monty on them
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("CREATE TABLE IF NOT EXISTS policies (fen TEXT PRIMARY KEY, policy_json TEXT)")
    conn.commit()
    conn.close()

    fens_in_tel = set()
    if os.path.exists(tel_path):
        with open(tel_path, "r") as f:
            for line in f:
                if line.strip():
                    try:
                        f_str = json.loads(line.strip()).get("fen")
                        if f_str:
                            fens_in_tel.add(f_str)
                    except Exception:
                        pass

    fens_to_query = list(fens_in_tel)
    if fens_to_query:
        m_chunk_size = math.ceil(len(fens_to_query) / workers)
        m_chunks = [fens_to_query[i : i + m_chunk_size] for i in range(0, len(fens_to_query), m_chunk_size)]
        worker_db_paths = [os.path.join(CACHE_DIR, f"heldout_m_w{w_id}_{session_tag}.db") for w_id in range(len(m_chunks))]

        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(query_monty_worker, w_id, m_chunk, worker_db_paths[w_id])
                for w_id, m_chunk in enumerate(m_chunks)
            ]
            for f in as_completed(futures):
                f.result()

        merge_worker_dbs(db_path, worker_db_paths)

    # 3. Compute On-Policy Physical Move 1 match vs Monty Oracle
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    monty_policies = {}
    for r in cur.execute("SELECT fen, policy_json FROM policies"):
        monty_policies[r[0]] = json.loads(r[1])
    conn.close()

    total_positions = 0
    top1_matches = 0
    quiet1_matches, total_quiet_pos = 0, 0
    cap1_matches, total_cap_pos = 0, 0
    cpp1_in_m3_matches = 0
    m1_in_cpp3_matches = 0
    monty_top1_red_sum = 0.0
    monty_top1_count = 0
    late_moves_red_sum = 0.0
    late_moves_count = 0

    if os.path.exists(tel_path):
        with open(tel_path, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    s = json.loads(line.strip())
                except Exception:
                    continue
                fen = s["fen"]
                moves = s.get("moves", [])
                if not moves or fen not in monty_policies:
                    continue

                top_cpp_move = moves[0]["move"]
                top3_cpp_moves = [m["move"] for m in moves[:min(3, len(moves))]]
                m_poly = monty_policies[fen]
                sorted_monty = sorted(m_poly.keys(), key=lambda m: m_poly[m], reverse=True)

                if sorted_monty:
                    top_monty_move = sorted_monty[0]
                    if top_cpp_move == top_monty_move:
                        top1_matches += 1
                    if top_cpp_move in sorted_monty[:3]:
                        cpp1_in_m3_matches += 1
                    if top_monty_move in top3_cpp_moves:
                        m1_in_cpp3_matches += 1

                    # Quiet moves on-policy match
                    quiets = [m for m in moves if not m.get("is_capture", False)]
                    monty_quiets = [m for m in sorted_monty if any(qm["move"] == m for qm in quiets)]
                    if len(quiets) >= 2 and monty_quiets and sum(m_poly.get(m, 0.0) for m in monty_quiets) >= 0.05:
                        total_quiet_pos += 1
                        if quiets[0]["move"] == monty_quiets[0]:
                            quiet1_matches += 1

                    # Capture moves on-policy match
                    caps = [m for m in moves if m.get("is_capture", False)]
                    monty_caps = [m for m in sorted_monty if any(cm["move"] == m for cm in caps)]
                    if len(caps) >= 2 and monty_caps and sum(m_poly.get(m, 0.0) for m in monty_caps) >= 0.05:
                        total_cap_pos += 1
                        if caps[0]["move"] == monty_caps[0]:
                            cap1_matches += 1

                    depth_val = s.get("depth", 8)
                    for m_idx, m_info in enumerate(moves):
                        rank_val = m_info.get("picker_rank", m_idx + 1)
                        if rank_val == 1:
                            r_val = 0.0
                        else:
                            if model is None:
                                stat_score = m_info.get("stat_score", 0)
                                r_val = (math.log(max(1, depth_val)) * math.log(max(1, rank_val)) * 500.0 - stat_score * (439.0 / 4096.0)) / 1024.0
                            else:
                                base_r = (math.log(max(1, depth_val)) * math.log(max(1, rank_val)) * 500.0) / 1024.0
                                r_val = base_r
                            r_val = max(-2.0, min(depth_val - 1.0, r_val))

                        if m_info["move"] == top_monty_move:
                            monty_top1_red_sum += r_val
                            monty_top1_count += 1
                        elif rank_val >= 2:
                            late_moves_red_sum += r_val
                            late_moves_count += 1

                total_positions += 1

    top1_pct = (top1_matches / max(1, total_positions)) * 100.0
    quiet1_pct = (quiet1_matches / max(1, total_quiet_pos)) * 100.0
    cap1_pct = (cap1_matches / max(1, total_cap_pos)) * 100.0
    cpp1_in_m3_pct = (cpp1_in_m3_matches / max(1, total_positions)) * 100.0
    m1_in_cpp3_pct = (m1_in_cpp3_matches / max(1, total_positions)) * 100.0
    mean_top1_r = monty_top1_red_sum / max(1, monty_top1_count)
    mean_late_r = late_moves_red_sum / max(1, late_moves_count)

    print(f"Heldout Positions Evaluated:                {total_positions:,}", flush=True)
    print(f"Physical C++ Move 1 == Monty Top-1 Match:   {top1_pct:.2f}%", flush=True)
    print(f"  - Physical C++ Quiet 1 Match (within Q):  {quiet1_pct:.2f}%", flush=True)
    print(f"  - Physical C++ Capture 1 Match (within C):{cap1_pct:.2f}%", flush=True)
    print(f"Physical C++ Move 1 in Monty Top-3 Match:   {cpp1_in_m3_pct:.2f}%", flush=True)
    print(f"Monty Top-1 in Physical C++ Top-3 Match:   {m1_in_cpp3_pct:.2f}% (Dual)", flush=True)
    print(f"Mean LMR Reduction on Monty Top-1 Move:     {mean_top1_r:.2f} plies", flush=True)
    print(f"Mean LMR Reduction on Other Late Moves:     {mean_late_r:.2f} plies", flush=True)
    print("=" * 80 + "\n", flush=True)

    if temp_model_path and os.path.exists(temp_model_path):
        os.remove(temp_model_path)

    return {
        "heldout_top1_match": top1_pct,
        "heldout_quiet1_match": quiet1_pct,
        "heldout_cap1_match": cap1_pct,
        "heldout_cpp1_in_m3_match": cpp1_in_m3_pct,
        "heldout_m1_in_cpp3_match": m1_in_cpp3_pct,
        "heldout_top1_reduction": mean_top1_r,
        "heldout_late_reduction": mean_late_r,
        "heldout_total_pos": total_positions
    }


def train_single_run(
    run_name: str,
    lr: float,
    mp_anchor_coef: float,
    lmr_ord_coef: float,
    rank_profile_coef: float,
    push_up_coef: float,
    args,
    val_loader: DataLoader,
    val_dataset: RolloutDataset,
    train_fens_pool: List[str],
    test_fens_pool: List[str],
    t_lmr: float,
    t_mp: float,
    floor_lmr: float,
    floor_mp: float,
    output_path: str
) -> Dict[str, float]:
    print("\n" + "=" * 80, flush=True)
    print(f"   STARTING RUN: {run_name}", flush=True)
    print("=" * 80, flush=True)
    print(f"Iterations:                 {args.iterations:,}", flush=True)
    print(f"Peak Learning Rate:         {lr:.4e}", flush=True)
    print(f"LR Schedule:                Warmup (min 2 iters) -> Cosine Decay (Floor: {0.30 * lr:.4e})", flush=True)
    print(f"Rollout Buffer Size:        {args.rollout_samples}", flush=True)
    print(f"Mini-Batch Size:            {args.minibatch_size}", flush=True)
    print(f"PPO Multi-Epochs / Iter:    {args.ppo_epochs}", flush=True)
    print(f"MovePicker Anchor Coef:     {mp_anchor_coef:.2f}", flush=True)
    print(f"LMR Order Coef (Detached):  {lmr_ord_coef:.2f}", flush=True)
    print(f"Rank-Profile MSE Coef:      {rank_profile_coef:.2f}", flush=True)
    print(f"Lean Tree Upward Push Coef: {push_up_coef:.4f}", flush=True)
    print(f"Output Binary:              {output_path}", flush=True)
    print("=" * 80, flush=True)

    model = DualMiniNN()
    model.export_quantized_binary(output_path)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    steps_per_epoch = max(1, math.ceil(args.rollout_samples / args.minibatch_size))
    total_training_steps = args.iterations * args.ppo_epochs * steps_per_epoch

    warmup_iters = min(5, max(2, args.iterations // 5))
    warmup_steps = warmup_iters * args.ppo_epochs * steps_per_epoch

    scheduler = get_onpolicy_warmup_cosine_scheduler(
        optimizer,
        total_steps=total_training_steps,
        warmup_steps=warmup_steps,
        floor_ratio=0.30
    )

    curr_fen_offset = 0
    total_gradient_steps = 0

    for iteration in range(1, args.iterations + 1):
        t0 = time.time()

        fresh_tel, fresh_db, curr_fen_offset = collect_target_samples(
            fens_pool=train_fens_pool,
            fen_offset=curr_fen_offset,
            target_samples=args.rollout_samples,
            nodes_per_fen=args.nodes,
            sample_interval=args.sample_interval,
            model_path=output_path,
            workers=args.workers,
            session_tag=f"{run_name}_iter{iteration}"
        )

        train_dataset = RolloutDataset(
            telemetry_path=fresh_tel,
            monty_db_path=fresh_db,
            floor_lmr=floor_lmr,
            floor_mp=floor_mp,
            t_lmr=t_lmr,
            t_mp=t_mp
        )
        train_loader = DataLoader(train_dataset, batch_size=args.minibatch_size, shuffle=True)

        iter_steps = 0
        iter_loss, iter_mp_kl_q, iter_mp_kl_c, iter_lmr_ord = 0.0, 0.0, 0.0, 0.0

        for epoch in range(args.ppo_epochs):
            for u_node, x_quiet, x_cap, x_lmr, is_cap, r_base, r_legacy, z_legacy_mp, target_p_mp, legal_mask, depth in train_loader:
                optimizer.zero_grad()
                w_quiet, z_latents, tau_mp, tau_lmr, quiet_scores, cap_scores, delta_r_nn = model(u_node, x_quiet, x_cap, x_lmr)

                z_quiet = quiet_scores / 32768.0
                z_cap = cap_scores / 32768.0

                loss, loss_mp_kl_q, loss_mp_kl_c, loss_mp_shape, loss_lmr_ord, loss_rank_prof, loss_push, _, _, _ = compute_combined_losses(
                    z_quiet, z_cap, delta_r_nn, tau_mp, tau_lmr, target_p_mp, z_legacy_mp, r_base, r_legacy, is_cap, legal_mask, depth,
                    mp_shape_coef=mp_anchor_coef, lmr_ord_coef=lmr_ord_coef, rank_profile_coef=rank_profile_coef, push_up_coef=push_up_coef
                )

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()

                total_gradient_steps += 1
                iter_steps += 1
                iter_loss += loss.item()
                iter_mp_kl_q += loss_mp_kl_q.item()
                iter_mp_kl_c += loss_mp_kl_c.item()
                iter_lmr_ord += loss_lmr_ord.item()

                if total_gradient_steps % args.sync_interval == 0:
                    model.export_quantized_binary(output_path)

        try:
            if os.path.exists(fresh_tel):
                os.remove(fresh_tel)
            if os.path.exists(fresh_db):
                os.remove(fresh_db)
        except Exception:
            pass

        elapsed_iter = time.time() - t0
        curr_lr = scheduler.get_last_lr()[0]
        n_steps = max(1, iter_steps)

        if iteration % args.val_freq == 0 or iteration == 1 or iteration == args.iterations:
            val_stats = evaluate_validation_rollout(
                model, val_loader,
                mp_anchor_coef=mp_anchor_coef, lmr_ord_coef=lmr_ord_coef, rank_profile_coef=rank_profile_coef, push_up_coef=push_up_coef
            )
            print(f"[{run_name} | Iter {iteration:>4d}/{args.iterations}] ({elapsed_iter:4.1f}s | lr: {curr_lr:.4e}) "
                  f"Train Loss: {iter_loss/n_steps:.4f} (MP_Q: {iter_mp_kl_q/n_steps:.3f}, MP_C: {iter_mp_kl_c/n_steps:.3f}, LMR_Ord: {iter_lmr_ord/n_steps:.3f}) | "
                  f"Val MP: (Q:{val_stats['mp_quiet_top1']:5.2f}%, C:{val_stats['mp_cap_top1']:5.2f}%) | Val Alloc Q(i*): {val_stats['mean_q_search_star']:5.2f}% | Effort: {val_stats['mean_search_effort']:.3f} | Val Loss: {val_stats['total_loss']:.4f}", flush=True)
        else:
            print(f"[{run_name} | Iter {iteration:>4d}/{args.iterations}] ({elapsed_iter:4.1f}s | lr: {curr_lr:.4e}) "
                  f"Train Loss: {iter_loss/n_steps:.4f} (MP_Q: {iter_mp_kl_q/n_steps:.3f}, MP_C: {iter_mp_kl_c/n_steps:.3f}, LMR_Ord: {iter_lmr_ord/n_steps:.3f})", flush=True)

    final_stats = evaluate_validation_rollout(
        model, val_loader,
        mp_anchor_coef=mp_anchor_coef, lmr_ord_coef=lmr_ord_coef, rank_profile_coef=rank_profile_coef, push_up_coef=push_up_coef
    )
    model.export_quantized_binary(output_path)

    # Output detailed 2D Depth x Rank Evaluation Matrix
    evaluate_2d_depth_rank_matrix(model, val_loader, tau_lmr=t_lmr)

    # Final Online Testing Step on Heldout FENs (Matching val_samples scale)
    if test_fens_pool:
        heldout_stats = run_heldout_online_evaluation(
            model=model,
            test_fens=test_fens_pool,
            target_samples=args.val_samples,
            nodes_per_fen=args.nodes,
            workers=args.workers,
            session_tag=f"{run_name}_heldout"
        )
        final_stats.update(heldout_stats)

    return final_stats


def main():
    parser = argparse.ArgumentParser(description="On-Policy Dual Mini-NN Closed-Loop Trainer & Grid Runner.")
    parser.add_argument("--iterations", type=int, default=128, help="Total outer iterations per run (default: 128)")
    parser.add_argument("--rollout-samples", type=int, default=512, help="Fresh on-policy rollout buffer size per iteration (default: 512)")
    parser.add_argument("--minibatch-size", type=int, default=64, help="Minibatch size for SGD updates (default: 64)")
    parser.add_argument("--ppo-epochs", type=int, default=4, help="PPO multi-epoch passes over fresh rollout buffer (default: 4)")
    parser.add_argument("--sync-interval", type=int, default=4, help="Gradient steps between model syncs (default: 4)")
    parser.add_argument("--val-freq", type=int, default=8, help="Validation frequency (default: 8)")
    parser.add_argument("--val-samples", type=int, default=32768, help="Fixed validation rollout samples (default: 32768 = 2^15)")
    parser.add_argument("--stream-limit", type=int, default=500000, help="Stream limit for FEN subsampling (default: 500,000)")
    parser.add_argument("--train-fens-pool", type=int, default=100000, help="Training FEN pool (default: 100,000)")
    parser.add_argument("--val-fens-pool", type=int, default=1000, help="Validation FEN pool (default: 1,000)")
    parser.add_argument("--test-fens-pool", type=int, default=1000, help="Held-out Test FEN pool (default: 1,000)")
    parser.add_argument("--nodes", type=int, default=500_000, help="Search budget per FEN (default: 500,000)")
    parser.add_argument("--sample-interval", type=int, default=10_000, help="Subsample interval (default: 10,000)")
    parser.add_argument("--workers", type=int, default=6, help="Parallel worker threads (default: 6)")
    parser.add_argument("--grid", action="store_true", help="Run the 3-experiment hyperparameter grid")
    parser.add_argument("--lr", type=float, default=4e-3, help="Peak learning rate (default: 4e-3)")
    parser.add_argument("--mp-anchor-coef", type=float, default=0.20, help="MovePicker anchor weight (default: 0.20)")
    parser.add_argument("--lmr-ord-coef", type=float, default=0.40, help="LMR order loss weight with detached scores (default: 0.40)")
    parser.add_argument("--rank-profile-coef", type=float, default=0.40, help="Rank profile MSE loss weight (default: 0.40)")
    parser.add_argument("--push-up-coef", type=float, default=0.065, help="Lean tree upward push weight (default: 0.065)")
    parser.add_argument("--output", type=str, default="floored_dual_64it.miniNN", help="Output model binary path")

    args = parser.parse_args()
    t_lmr, t_mp, floor_lmr, floor_mp = load_calibration_parameters()

    print("=" * 80, flush=True)
    print("   ON-POLICY CLOSED-LOOP DUAL MINI-NN TRAINER (RESIDUAL ARCHITECTURE - 6.5E-2 PUSH)", flush=True)
    print("=" * 80, flush=True)
    print(f"Total Iterations:            {args.iterations:,}", flush=True)
    print(f"Validation Frequency:        Every {args.val_freq} iterations", flush=True)
    print(f"Validation Rollout Size:     {args.val_samples:,} samples (2^15)", flush=True)
    print(f"Validation FEN Pool:         {args.val_fens_pool:,} FENs (drawn from 500k stream)", flush=True)
    print(f"Heldout Test FEN Pool:       {args.test_fens_pool:,} FENs (unseen test set)", flush=True)
    print(f"Rollout Buffer Size / Iter:  {args.rollout_samples} transitions", flush=True)
    print(f"Mini-Batch Size:             {args.minibatch_size}", flush=True)
    print(f"PPO Multi-Epochs / Iter:     {args.ppo_epochs} epochs ({args.ppo_epochs * math.ceil(args.rollout_samples / args.minibatch_size)} gradient steps/iter)", flush=True)
    print(f"LMR Calibration Target:      T_calib = {t_lmr:.4f} | Floor = {floor_lmr:.3f}", flush=True)
    print(f"MP Calibration Target:       T_calib = {t_mp:.4f} | Floor = {floor_mp:.3f}", flush=True)
    print(f"Nodes per FEN Search:        {args.nodes:,}", flush=True)
    print("=" * 80, flush=True)

    val_fens_pool, test_fens_pool, train_fens_pool = load_and_subsample_fens(
        epd_path=EPD_FILE,
        stream_limit=args.stream_limit,
        val_count=args.val_fens_pool,
        test_count=args.test_fens_pool,
        train_count=args.train_fens_pool,
        seed=42
    )

    print(f"[1/3] Loading / Generating Fixed {args.val_samples:,} Validation Rollout Samples (evenly from {len(val_fens_pool):,} FENs)...", flush=True)
    t_v0 = time.time()

    val_tel, val_db = collect_validation_rollout_even(
        val_fens_pool=val_fens_pool,
        target_samples=args.val_samples,
        nodes_per_fen=args.nodes,
        workers=args.workers,
        session_tag="val_v3_staged_500k_shared"
    )

    val_dataset = RolloutDataset(
        telemetry_path=val_tel,
        monty_db_path=val_db,
        floor_lmr=floor_lmr,
        floor_mp=floor_mp,
        t_lmr=t_lmr,
        t_mp=t_mp
    )
    val_loader = DataLoader(val_dataset, batch_size=args.minibatch_size, shuffle=False)
    print(f"      Validation set ready: {len(val_dataset):,} samples in {time.time() - t_v0:.1f}s.\n", flush=True)

    # Master Baseline Offline Evaluation
    print("=" * 80, flush=True)
    print("      HANDCRAFTED STOCKFISH MASTER BASELINE (ON UNBIASED 2^15 VALIDATION SET)", flush=True)
    print("=" * 80, flush=True)
    master_stats = evaluate_handcrafted_master(val_loader, tau_mp=t_mp, tau_lmr=t_lmr)
    print(f"{'Metric':<40} | {'Handcrafted Stockfish Master':<25}", flush=True)
    print("-" * 80, flush=True)
    print(f"{'Quiet Moves Top-1 Match (within quiets)':<40} | {master_stats['master_quiet_top1']:<24.2f}%", flush=True)
    print(f"{'Capture Moves Top-1 Match (within caps)':<40} | {master_stats['master_cap_top1']:<24.2f}%", flush=True)
    print(f"{'MovePicker Quiet KL Divergence':<40} | {master_stats['master_mp_kl_q']:<25.4f}", flush=True)
    print(f"{'MovePicker Capture KL Divergence':<40} | {master_stats['master_mp_kl_c']:<25.4f}", flush=True)
    print(f"{'Mean Top Move Search Allocation Q(i*)':<40} | {master_stats['master_q_search_star']:<24.2f}%", flush=True)
    print(f"{'LMR Move-Order Dependent Loss':<40} | {master_stats['master_lmr_ord']:<25.4f}", flush=True)
    print(f"{'Mean Late-Move Search Effort (E_leg)':<40} | {master_stats['master_mean_effort']:<25.4f}", flush=True)
    print(f"{'Mean Reduction (Plies)':<40} | {master_stats['master_mean_reduction']:<25.4f}", flush=True)
    print("=" * 80 + "\n", flush=True)

    # Master Baseline On-Policy Heldout Evaluation
    master_heldout_stats = {}
    if test_fens_pool:
        master_heldout_stats = run_heldout_online_evaluation(
            model=None, # Master Stockfish (no miniNN model)
            test_fens=test_fens_pool,
            target_samples=args.val_samples,
            nodes_per_fen=args.nodes,
            workers=args.workers,
            session_tag="master_heldout_shared"
        )
        master_stats.update({f"master_{k}": v for k, v in master_heldout_stats.items()})

    if args.grid:
        grid_configs = [
            {
                "name": "Run1_Residual_LMR40_Push6e2",
                "lr": 4e-3,
                "mp_anchor": 0.20,
                "lmr_ord": 0.40,
                "rank_profile": 0.40,
                "push_up": 0.065,
                "output": "floored_dual_push6e2_run1.miniNN"
            },
        ]

        results = {}
        for cfg in grid_configs:
            stats = train_single_run(
                run_name=cfg["name"],
                lr=cfg["lr"],
                mp_anchor_coef=cfg["mp_anchor"],
                lmr_ord_coef=cfg["lmr_ord"],
                rank_profile_coef=cfg["rank_profile"],
                push_up_coef=cfg["push_up"],
                args=args,
                val_loader=val_loader,
                val_dataset=val_dataset,
                train_fens_pool=train_fens_pool,
                test_fens_pool=test_fens_pool,
                t_lmr=t_lmr,
                t_mp=t_mp,
                floor_lmr=floor_lmr,
                floor_mp=floor_mp,
                output_path=cfg["output"]
            )
            results[cfg["name"]] = stats

        print("\n" + "=" * 168, flush=True)
        print("                                            FINAL COMPARATIVE BENCHMARK TABLE (GRID EXPERIMENTS)", flush=True)
        print("=" * 168, flush=True)
        header = f"{'Configuration':<30} | {'Heldout Top-1':<13} | {'Heldout Q-1':<11} | {'Heldout C-1':<11} | {'CPP 1 in M3':<11} | {'M1 in CPP 3':<11} | {'Top-1 Red':<9} | {'Late Red':<9} | {'Val Effort':<10} | {'Val Red':<8}"
        print(header, flush=True)
        print("-" * 168, flush=True)
        m_top1_str = f"{master_stats.get('master_heldout_top1_match', 0.0):.2f}%"
        m_q1_str = f"{master_stats.get('master_heldout_quiet1_match', 0.0):.2f}%"
        m_c1_str = f"{master_stats.get('master_heldout_cap1_match', 0.0):.2f}%"
        m_cpp1_in_m3_str = f"{master_stats.get('master_heldout_cpp1_in_m3_match', 0.0):.2f}%"
        m_m1_in_cpp3_str = f"{master_stats.get('master_heldout_m1_in_cpp3_match', 0.0):.2f}%"
        m_r_top1_str = f"{master_stats.get('master_heldout_top1_reduction', 0.0):.2f}"
        m_r_late_str = f"{master_stats.get('master_heldout_late_reduction', 0.0):.2f}"
        print(f"{'Stockfish Master Baseline':<30} | {m_top1_str:>13} | {m_q1_str:>11} | {m_c1_str:>11} | {m_cpp1_in_m3_str:>11} | {m_m1_in_cpp3_str:>11} | {m_r_top1_str:>9} | {m_r_late_str:>9} | {master_stats['master_mean_effort']:>10.4f} | {master_stats['master_mean_reduction']:>8.4f}", flush=True)
        print("-" * 168, flush=True)
        for name, s in results.items():
            h_top1_str = f"{s.get('heldout_top1_match', 0.0):.2f}%"
            h_q1_str = f"{s.get('heldout_quiet1_match', 0.0):.2f}%"
            h_c1_str = f"{s.get('heldout_cap1_match', 0.0):.2f}%"
            h_cpp1_in_m3_str = f"{s.get('heldout_cpp1_in_m3_match', 0.0):.2f}%"
            h_m1_in_cpp3_str = f"{s.get('heldout_m1_in_cpp3_match', 0.0):.2f}%"
            h_r_top1_str = f"{s.get('heldout_top1_reduction', 0.0):.2f}"
            h_r_late_str = f"{s.get('heldout_late_reduction', 0.0):.2f}"
            print(f"{name:<30} | {h_top1_str:>13} | {h_q1_str:>11} | {h_c1_str:>11} | {h_cpp1_in_m3_str:>11} | {h_m1_in_cpp3_str:>11} | {h_r_top1_str:>9} | {h_r_late_str:>9} | {s['mean_search_effort']:>10.4f} | {s['mean_reduction']:>8.4f}", flush=True)
        print("=" * 168 + "\n", flush=True)

    else:
        final_stats = train_single_run(
            run_name="Single_Push6e2_Run",
            lr=args.lr,
            mp_anchor_coef=args.mp_anchor_coef,
            lmr_ord_coef=args.lmr_ord_coef,
            rank_profile_coef=args.rank_profile_coef,
            push_up_coef=args.push_up_coef,
            args=args,
            val_loader=val_loader,
            val_dataset=val_dataset,
            train_fens_pool=train_fens_pool,
            test_fens_pool=test_fens_pool,
            t_lmr=t_lmr,
            t_mp=t_mp,
            floor_lmr=floor_lmr,
            floor_mp=floor_mp,
            output_path=args.output
        )

        print("\n" + "=" * 100, flush=True)
        print("      FINAL BENCHMARK EVALUATION (ON FIXED 2^15 VALIDATION ROLLOUT)", flush=True)
        print("=" * 100, flush=True)
        print(f"{'Metric':<40} | {'Trained Dual Mini-NN (V3)':<25} | {'Handcrafted Master Baseline':<25}", flush=True)
        print("-" * 100, flush=True)
        print(f"{'MovePicker Top-1 Monty Match':<40} | {final_stats['mp_top1_acc']:<24.2f}% | {master_stats['master_mp_top1']:<24.2f}%", flush=True)
        print(f"{'MovePicker KL Divergence Loss':<40} | {final_stats['mp_kl_loss']:<25.4f} | {master_stats['master_mp_kl']:<25.4f}", flush=True)
        print(f"{'Mean Top Move Search Allocation Q(i*)':<40} | {final_stats['mean_q_search_star']:<24.2f}% | {master_stats['master_q_search_star']:<24.2f}%", flush=True)
        print(f"{'LMR Move-Order Dependent Loss':<40} | {final_stats['lmr_order_loss']:<25.4f} | {master_stats['master_lmr_ord']:<25.4f}", flush=True)
        print(f"{'Mean Late-Move Search Effort':<40} | {final_stats['mean_search_effort']:<25.4f} | {master_stats['master_mean_effort']:<25.4f}", flush=True)
        print(f"{'Mean Reduction (Plies)':<40} | {final_stats['mean_reduction']:<25.4f} | {master_stats['master_mean_reduction']:<25.4f}", flush=True)
        print("=" * 100, flush=True)


if __name__ == "__main__":
    main()
