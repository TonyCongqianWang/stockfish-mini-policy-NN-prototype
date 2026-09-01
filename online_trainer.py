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
    extract_lmr_features_from_data,
    extract_node_features,
    extract_node_features_from_data,
    extract_quiet_terms_from_data,
)
from model import DualMiniNN
from paths import CACHE_DIR, CALIB_CONFIG_PATH, EPD_FILE, MONTY_BIN, STOCKFISH_BIN


def load_calibration_parameters() -> Tuple[float, float, float, float, float, float]:
    t_teacher_lmr = 1.00
    t_teacher_mp = 0.50
    tau_student_lmr = 0.8658
    tau_student_mp = 0.1154
    floor_lmr = 0.050
    floor_mp = 0.010
    if os.path.exists(CALIB_CONFIG_PATH):
        try:
            with open(CALIB_CONFIG_PATH, "r") as f:
                cfg = json.load(f)
            t_teacher_lmr = float(cfg.get("t_teacher_lmr", cfg.get("t_calib_lmr", 1.00)))
            t_teacher_mp = float(cfg.get("t_teacher_mp", cfg.get("t_calib_mp", 0.50)))
            tau_student_lmr = float(cfg.get("tau_student_lmr", cfg.get("t_calib_lmr", 0.8658)))
            tau_student_mp = float(cfg.get("tau_student_mp", cfg.get("t_calib_mp", 0.1154)))
            floor_lmr = float(cfg.get("chosen_floor_lmr", 0.050))
            floor_mp = float(cfg.get("chosen_floor_mp", 0.010))
        except Exception:
            pass
    return t_teacher_lmr, t_teacher_mp, tau_student_lmr, tau_student_mp, floor_lmr, floor_mp


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
    output_path: str,
    use_mp: bool = True,
    use_lmr: bool = True
) -> int:
    if os.path.exists(output_path):
        os.remove(output_path)

    env = os.environ.copy()
    env["SF_LMR_TELEMETRY"] = output_path
    env["SF_LMR_SAMPLE_INTERVAL"] = str(sample_interval)
    env["SF_MININN_USE_MP"] = "1" if use_mp else "0"
    env["SF_MININN_USE_LMR"] = "1" if use_lmr else "0"
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

    proc.stdin.write(f"uci\nsetoption name Threads value 1\nsetoption name Use_MiniNN_MP value {str(use_mp).lower()}\nsetoption name Use_MiniNN_LMR value {str(use_lmr).lower()}\nisready\n")
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


def load_calibration_parameters() -> Tuple[float, float, float, float, float, float]:
    t_teacher_lmr = 1.00
    t_teacher_mp = 0.50
    tau_student_lmr = 0.8658
    tau_student_mp = 0.1154
    floor_lmr = 0.050
    floor_mp = 0.010
    if os.path.exists(CALIB_CONFIG_PATH):
        try:
            with open(CALIB_CONFIG_PATH, "r") as f:
                cfg = json.load(f)
            t_teacher_lmr = float(cfg.get("t_teacher_lmr", cfg.get("t_calib_lmr", 1.00)))
            t_teacher_mp = float(cfg.get("t_teacher_mp", cfg.get("t_calib_mp", 0.50)))
            tau_student_lmr = float(cfg.get("tau_student_lmr", cfg.get("t_calib_lmr", 0.8658)))
            tau_student_mp = float(cfg.get("tau_student_mp", cfg.get("t_calib_mp", 0.1154)))
            floor_lmr = float(cfg.get("chosen_floor_lmr", 0.050))
            floor_mp = float(cfg.get("chosen_floor_mp", 0.010))
        except Exception:
            pass
    return t_teacher_lmr, t_teacher_mp, tau_student_lmr, tau_student_mp, floor_lmr, floor_mp


class RolloutDataset(Dataset):
    def __init__(
        self,
        telemetry_path: str,
        monty_db_path: str,
        floor_lmr: float = 0.050,
        floor_mp: float = 0.010,
        t_teacher_lmr: float = 1.00,
        t_teacher_mp: float = 0.50,
        t_lmr: Optional[float] = None,
        t_mp: Optional[float] = None
    ):
        self.floor_lmr = floor_lmr
        self.floor_mp = floor_mp
        self.t_teacher_lmr = t_lmr if t_lmr is not None else t_teacher_lmr
        self.t_teacher_mp = t_mp if t_mp is not None else t_teacher_mp
        self.items = []

        conn = sqlite3.connect(monty_db_path)
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

                t_quiet = torch.zeros(MAX_LEGAL_MOVES, 10, dtype=torch.float32)
                x_lmr = torch.zeros(MAX_LEGAL_MOVES, 8, dtype=torch.float32)
                is_cap_mask = torch.zeros(MAX_LEGAL_MOVES, dtype=torch.bool)
                legal_mask = torch.zeros(MAX_LEGAL_MOVES, dtype=torch.bool)
                r_base = torch.zeros(MAX_LEGAL_MOVES, dtype=torch.float32)
                r_legacy = torch.zeros(MAX_LEGAL_MOVES, dtype=torch.float32)
                z_legacy_mp = torch.zeros(MAX_LEGAL_MOVES, dtype=torch.float32)
                target_p_mp = torch.zeros(MAX_LEGAL_MOVES, dtype=torch.float32)
                target_p_lmr = torch.zeros(MAX_LEGAL_MOVES, dtype=torch.float32)

                m_policy = policies.get(fen, {})

                raw_p_list = []
                for i in range(num_moves):
                    m_data = moves_info[i]
                    uci_str = m_data["move"]
                    stat_score = m_data.get("stat_score", 0)
                    is_capture = m_data.get("is_capture", False)
                    rank = m_data.get("picker_rank", i + 1)

                    t_quiet[i] = extract_quiet_terms_from_data(m_data, ply=ply)
                    x_lmr[i] = extract_lmr_features_from_data(m_data)

                    is_cap_mask[i] = is_capture
                    legal_mask[i] = True

                    if "r_base" in m_data:
                        r_base[i] = float(m_data["r_base"]) / 1024.0
                        r_legacy[i] = float(m_data.get("r_executed", m_data["r_base"])) / 1024.0
                    else:
                        if i == 0 or rank == 1:
                            base_red = 0.0
                            legacy_reduction = 0.0
                        else:
                            base_red = (math.log(max(1, depth)) * math.log(max(1, rank)) * 500.0) / 1024.0
                            legacy_reduction = (math.log(max(1, depth)) * math.log(max(1, rank)) * 500.0 - stat_score * (439.0 / 4096.0)) / 1024.0
                        r_base[i] = base_red
                        r_legacy[i] = legacy_reduction
                    z_legacy_mp[i] = float(np.clip(stat_score / 16384.0, -1.0, 1.0))

                    raw_p = m_policy.get(uci_str, 0.0)
                    raw_p_list.append(raw_p)

                if raw_p_list:
                    p_arr = np.array(raw_p_list, dtype=np.float64)
                    log_mp = np.log(p_arr + 1e-12) / self.t_teacher_mp
                    exp_mp = np.exp(log_mp - np.max(log_mp))
                    p_temp_mp = exp_mp / np.sum(exp_mp)
                    p_target_mp = (1.0 - self.floor_mp) * p_temp_mp + self.floor_mp / float(num_moves)
                    target_p_mp[:num_moves] = torch.from_numpy(p_target_mp).float()

                    log_lmr = np.log(p_arr + 1e-12) / self.t_teacher_lmr
                    exp_lmr = np.exp(log_lmr - np.max(log_lmr))
                    p_temp_lmr = exp_lmr / np.sum(exp_lmr)
                    p_target_lmr = (1.0 - self.floor_lmr) * p_temp_lmr + self.floor_lmr / float(num_moves)
                    target_p_lmr[:num_moves] = torch.from_numpy(p_target_lmr).float()

                depth_tensor = torch.tensor(float(depth), dtype=torch.float32)
                self.items.append((u_node, t_quiet, x_lmr, is_cap_mask, r_base, r_legacy, z_legacy_mp, target_p_mp, target_p_lmr, legal_mask, depth_tensor))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]

    def compute_variance_matched_temperatures(self) -> Tuple[float, float]:
        var_z_legacy_mp = []
        var_log_p_mp = []
        var_r_legacy = []
        var_log_p_lmr = []

        for item in self.items:
            u_node, t_quiet, x_lmr, is_cap, r_base, r_legacy, z_legacy_mp, target_p_mp, target_p_lmr, legal_mask, depth = item
            m_mask = legal_mask.numpy()
            z_m = z_legacy_mp[m_mask].numpy()
            p_mp_m = target_p_mp[m_mask].numpy()
            p_lmr_m = target_p_lmr[m_mask].numpy()
            r_m = r_legacy[m_mask].numpy()

            if len(z_m) > 2:
                log_p_mp_m = np.log(p_mp_m + 1e-12)
                log_p_lmr_m = np.log(p_lmr_m + 1e-12)
                var_z_legacy_mp.append(np.var(z_m))
                var_log_p_mp.append(np.var(log_p_mp_m))
                var_r_legacy.append(np.var(r_m))
                var_log_p_lmr.append(np.var(log_p_lmr_m))

        std_z = np.sqrt(np.mean(var_z_legacy_mp)) if var_z_legacy_mp else 0.4750
        std_logp_mp = np.sqrt(np.mean(var_log_p_mp)) if var_log_p_mp else 1.9911
        matched_tau_mp = float(std_z / max(1e-6, std_logp_mp))

        std_r = np.sqrt(np.mean(var_r_legacy)) if var_r_legacy else 1.4779
        std_logp_lmr = np.sqrt(np.mean(var_log_p_lmr)) if var_log_p_lmr else 0.8259
        matched_tau_lmr = float(std_r / max(1e-6, std_logp_lmr))

        return matched_tau_mp, matched_tau_lmr


def compute_combined_losses(
    z_quiet: torch.Tensor,
    delta_r_nn: torch.Tensor,
    tau_mp: torch.Tensor,
    tau_lmr: torch.Tensor,
    target_p_mp: torch.Tensor,
    target_p_lmr: torch.Tensor,
    z_legacy_mp: torch.Tensor,
    r_base: torch.Tensor,
    r_legacy: torch.Tensor,
    is_cap_mask: torch.Tensor,
    legal_mask: torch.Tensor,
    depth: torch.Tensor,
    w_mp: Optional[torch.Tensor] = None,
    w_lmr: Optional[torch.Tensor] = None,
    reg_coef: float = 0.05,
    lmr_ord_coef: float = 0.50,
    train_mode: str = "both"
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Direct Residual Formulation Losses:
    1. MovePicker: Monty KL divergence on quiet moves.
    2. LMR: Direct Physical Search Effort Allocation (Monty policy-weighted KL).
    3. Output Minimization: L2 regularization on residual weights (w_mp and w_lmr) around 0.
    """
    B, M = z_quiet.shape

    w_raw = torch.clamp(torch.sqrt(depth.clamp(min=1.0) / 8.0), 0.70, 1.40)
    w_depth = w_raw / w_raw.mean()

    # 1. Quiet Moves KL (Filtered by minimum Monty mass threshold >= 25%)
    quiet_mask = legal_mask & (~is_cap_mask)
    masked_zq = z_quiet.masked_fill(~quiet_mask, -1e4)
    log_probs_q = F.log_softmax(masked_zq / tau_mp, dim=-1)
    p_q_raw = target_p_mp * quiet_mask.float()
    w_q_pos = p_q_raw.sum(dim=-1)
    has_quiets = (w_q_pos >= 0.25) & (quiet_mask.sum(dim=-1) > 1)

    loss_mp_kl_q = torch.tensor(0.0, device=z_quiet.device)
    acc_q = torch.tensor(0.0, device=z_quiet.device)
    if has_quiets.sum() > 0:
        p_q_norm = p_q_raw[has_quiets] / w_q_pos[has_quiets].unsqueeze(1)
        loss_q_pos = -(p_q_norm * log_probs_q[has_quiets]).sum(dim=-1)
        loss_mp_kl_q = (w_depth[has_quiets] * loss_q_pos).mean()
        acc_q = (masked_zq[has_quiets].argmax(dim=-1) == p_q_raw[has_quiets].argmax(dim=-1)).float().mean()

    # 2. Residual Reductions on physical moves
    r_total_nn = r_base + delta_r_nn
    max_red = (depth.unsqueeze(1) - 1.0).clamp(min=0.0)
    min_red = torch.tensor(-2.0, device=delta_r_nn.device)
    r_real_nn = torch.minimum(torch.maximum(r_total_nn, min_red), max_red)
    r_real_nn[:, 0] = 0.0  # Move 1 is searched at full depth (r = 0.00)

    # 3. Direct Physical Search Effort
    masked_r = r_real_nn.masked_fill(~legal_mask, 1e4)
    E_late = torch.exp(-masked_r / tau_lmr) * legal_mask.float()
    E_eff = E_late.clone()
    E_eff[:, 0] = 1.0  # Move 1 in physical search is always full depth

    D = E_eff.sum(dim=-1, keepdim=True)
    Q_dist = E_eff / (D + 1e-12)
    loss_lmr_order = (w_depth * -(target_p_lmr * torch.log(Q_dist + 1e-12)).sum(dim=-1)).mean()

    # 4. Pure Output Minimization / Residual Regularization (L2 norm on residual outputs)
    loss_reg = torch.tensor(0.0, device=z_quiet.device)
    if w_mp is not None and w_lmr is not None:
        loss_reg = w_mp.pow(2).mean() + w_lmr.pow(2).mean()
    elif w_mp is not None:
        loss_reg = w_mp.pow(2).mean()
    elif w_lmr is not None:
        loss_reg = w_lmr.pow(2).mean()

    if train_mode == "movepicker":
        loss_total = loss_mp_kl_q + reg_coef * loss_reg
    elif train_mode == "lmr":
        loss_total = lmr_ord_coef * loss_lmr_order + reg_coef * loss_reg
    else:
        loss_total = loss_mp_kl_q + lmr_ord_coef * loss_lmr_order + reg_coef * loss_reg

    return loss_total, loss_mp_kl_q, loss_lmr_order, loss_reg, acc_q


def compute_standardized_rollout_metrics(
    target_p_mp: torch.Tensor,
    target_p_lmr: torch.Tensor,
    r_real: torch.Tensor,
    z_q: torch.Tensor,
    is_cap: torch.Tensor,
    legal_mask: torch.Tensor,
    tau_mp: torch.Tensor,
    tau_lmr: torch.Tensor,
    is_live_rollout: bool = False
) -> Dict[str, float]:
    """
    Standardized, canonical metric calculation across all evaluation streams.
    Adheres strictly to the singular source of truth: the physical order of move evaluations.
    """
    B, M = target_p_mp.shape
    i_star = target_p_mp.argmax(dim=-1)

    # 1. Quiet MovePicker Sub-Distribution (Filtered >= 25% mass)
    quiet_mask = legal_mask & (~is_cap)
    masked_zq = z_q.masked_fill(~quiet_mask, -1e4)
    log_probs_q = F.log_softmax(masked_zq / tau_mp, dim=-1)
    p_q_raw = target_p_mp * quiet_mask.float()
    w_q_pos = p_q_raw.sum(dim=-1)
    has_quiets = (w_q_pos >= 0.25) & (quiet_mask.sum(dim=-1) > 1)

    mp_kl_q = -(p_q_raw[has_quiets] * log_probs_q[has_quiets]).sum(dim=-1).mean() if has_quiets.sum() > 0 else torch.tensor(0.0)

    # Ground Truth Physical Move 1 among Quiets (the first quiet move emitted during search)
    M_range = torch.arange(M, device=target_p_mp.device).unsqueeze(0).expand(B, M)
    masked_ranks = torch.where(quiet_mask, M_range, torch.tensor(M, device=target_p_mp.device))
    phys_first_quiet = masked_ranks.argmin(dim=-1)
    monty_best_quiet = p_q_raw.argmax(dim=-1)
    phys_quiet_match = (phys_first_quiet[has_quiets] == monty_best_quiet[has_quiets]).float().mean() if has_quiets.sum() > 0 else torch.tensor(0.0)

    # Model Scored Quiet Top-1 Match (offline score evaluation)
    model_best_quiet = masked_zq.argmax(dim=-1)
    acc_q = (model_best_quiet[has_quiets] == monty_best_quiet[has_quiets]).float().mean() if has_quiets.sum() > 0 else torch.tensor(0.0)

    # For live search rollouts & Master baseline, the ground truth is physical search emission order
    reported_quiet_top1 = phys_quiet_match if is_live_rollout else acc_q

    # 2. Search Effort & Allocation
    r_real_clean = r_real.clone()
    r_real_clean[:, 0] = 0.0  # Move 1 in physical search is always full depth (r = 0.00)
    masked_r = r_real_clean.masked_fill(~legal_mask, 1e4)
    E_late = torch.exp(-masked_r / tau_lmr) * legal_mask.float()
    E_eff = E_late.clone()
    E_eff[:, 0] = 1.0
    D = E_eff.sum(dim=-1, keepdim=True)
    Q_dist = E_eff / (D + 1e-12)
    lmr_policy_loss = -(target_p_lmr * torch.log(Q_dist + 1e-12)).sum(dim=-1).mean()
    q_star = Q_dist.gather(1, i_star.unsqueeze(1)).squeeze(1).mean()

    # 3. Physical Ordering & Top-1 Metrics
    top1_match = (i_star == 0).float().mean()
    sorted_p = target_p_mp.argsort(dim=-1, descending=True)
    m1_in_top3 = (sorted_p[:, :3] == 0).any(dim=-1).float().mean()
    top1_in_m3 = (i_star < 3).float().mean()

    # 4. Reductions
    top1_red = r_real_clean.gather(1, i_star.unsqueeze(1)).squeeze(1).mean()
    other_late_mask = legal_mask.clone()
    other_late_mask[:, 0] = False
    other_late_mask.scatter_(1, i_star.unsqueeze(1), False)
    late_red = r_real_clean[other_late_mask].mean() if other_late_mask.sum() > 0 else torch.tensor(0.0)

    # 5. Monty-Weighted Late-Move Search Effort
    monty_late_effort = (target_p_lmr[:, 1:] * E_late[:, 1:]).sum(dim=-1).mean()
    mean_red = r_real_clean[legal_mask].mean()

    return {
        "top1_match": top1_match.item() * 100.0,
        "quiet_top1": reported_quiet_top1.item() * 100.0,
        "phys_quiet_top1": phys_quiet_match.item() * 100.0,
        "model_quiet_top1": acc_q.item() * 100.0,
        "cpp1_in_m3_match": m1_in_top3.item() * 100.0,
        "m1_in_cpp3_match": top1_in_m3.item() * 100.0,
        "mp_kl_q": mp_kl_q.item(),
        "mp_kl_loss": mp_kl_q.item(),
        "q_search_star": q_star.item() * 100.0,
        "lmr_policy_loss": lmr_policy_loss.item(),
        "top1_reduction": top1_red.item(),
        "late_reduction": late_red.item(),
        "mean_effort": monty_late_effort.item(),
        "mean_reduction": mean_red.item()
    }


def print_unified_benchmark_report(title: str, stats: Dict[str, float]):
    print("=" * 80, flush=True)
    print(f"      {title.upper()} EVALUATION REPORT", flush=True)
    print("=" * 80, flush=True)
    print(f"{'Physical Move 1 == Monty Top-1 Match:':<45} {stats.get('top1_match', 0.0):.2f}%", flush=True)
    print(f"{'  - Physical Quiet 1 Match (within Q):':<45} {stats.get('quiet_top1', 0.0):.2f}%", flush=True)
    print(f"{'Physical Move 1 in Monty Top-3 Match:':<45} {stats.get('cpp1_in_m3_match', 0.0):.2f}%", flush=True)
    print(f"{'Monty Top-1 in Physical Move 1..3 Match:':<45} {stats.get('m1_in_cpp3_match', 0.0):.2f}% (Dual)", flush=True)
    print(f"{'MovePicker Quiet KL Divergence:':<45} {stats.get('mp_kl_q', 0.0):.4f}", flush=True)
    print(f"{'Mean Top Move Search Allocation Q(i*):':<45} {stats.get('q_search_star', 0.0):.2f}%", flush=True)
    print(f"{'LMR Policy Cross-Entropy Loss:':<45} {stats.get('lmr_policy_loss', 0.0):.4f}", flush=True)
    print(f"{'Mean LMR Reduction on Monty Top-1 Move:':<45} {stats.get('top1_reduction', 0.0):.2f} plies", flush=True)
    print(f"{'Mean LMR Reduction on Other Late Moves:':<45} {stats.get('late_reduction', 0.0):.2f} plies", flush=True)
    print(f"{'Monty-Weighted Late Search Effort (E):':<45} {stats.get('mean_effort', 0.0):.4f}", flush=True)
    print(f"{'Mean Total Reduction:':<45} {stats.get('mean_reduction', 0.0):.4f} plies", flush=True)
    print("=" * 80 + "\n", flush=True)


def evaluate_handcrafted_master(
    loader: DataLoader,
    tau_mp: float = 0.5,
    tau_lmr: float = 1.5
) -> Dict[str, float]:
    stat_sums = {}
    total_count = 0

    with torch.no_grad():
        for u_node, t_quiet, x_lmr, is_cap, r_base, r_legacy, z_legacy_mp, target_p_mp, target_p_lmr, legal_mask, depth in loader:
            B, M = z_legacy_mp.shape
            max_red = (depth.unsqueeze(1) - 1.0).clamp(min=0.0)
            min_red = torch.tensor(-2.0, device=r_legacy.device)
            r_real_leg = torch.minimum(torch.maximum(r_legacy, min_red), max_red)

            tau_mp_t = torch.tensor(tau_mp, device=z_legacy_mp.device)
            tau_lmr_t = torch.tensor(tau_lmr, device=z_legacy_mp.device)
            z_master_q = t_quiet.sum(dim=-1) / 32768.0

            b_stats = compute_standardized_rollout_metrics(
                target_p_mp=target_p_mp,
                target_p_lmr=target_p_lmr,
                r_real=r_real_leg,
                z_q=z_master_q,
                is_cap=is_cap,
                legal_mask=legal_mask,
                tau_mp=tau_mp_t,
                tau_lmr=tau_lmr_t,
                is_live_rollout=True
            )

            total_count += B
            for k, v in b_stats.items():
                stat_sums[k] = stat_sums.get(k, 0.0) + v * B

    n = max(1, total_count)
    return {k: v / n for k, v in stat_sums.items()}


def evaluate_validation_rollout(
    model: DualMiniNN,
    loader: DataLoader,
    mp_anchor_coef: float = 0.05,
    lmr_ord_coef: float = 0.50,
    tau_lmr: float = 1.5,
    train_mode: str = "both",
    is_live_rollout: bool = True
) -> Dict[str, float]:
    model.eval()
    tot_loss_sum, w_anc_sum = 0.0, 0.0
    stat_sums = {}
    total_count = 0

    with torch.no_grad():
        for u_node, t_quiet, x_lmr, is_cap, r_base, r_legacy, z_legacy_mp, target_p_mp, target_p_lmr, legal_mask, depth in loader:
            w_mp, w_lmr, tau_mp, tau_lmr_pred, quiet_scores, delta_r_nn = model(u_node, t_quiet, x_lmr)

            z_quiet = quiet_scores / 32768.0

            loss, loss_mp_kl_q, loss_lmr_ord, loss_reg, acc_q = compute_combined_losses(
                z_quiet, delta_r_nn, tau_mp, tau_lmr_pred, target_p_mp, target_p_lmr, z_legacy_mp, r_base, r_legacy, is_cap, legal_mask, depth,
                w_mp=w_mp, w_lmr=w_lmr, reg_coef=mp_anchor_coef, lmr_ord_coef=lmr_ord_coef, train_mode=train_mode
            )

            max_red = (depth.unsqueeze(1) - 1.0).clamp(min=0.0)
            min_red = torch.tensor(-2.0, device=delta_r_nn.device)

            if train_mode == "movepicker":
                r_real_eff = torch.minimum(torch.maximum(r_base, min_red), max_red)
                eff_tau_lmr = torch.tensor(tau_lmr, device=delta_r_nn.device)
            else:
                r_total_nn = r_base + delta_r_nn
                r_real_eff = torch.minimum(torch.maximum(r_total_nn, min_red), max_red)
                eff_tau_lmr = tau_lmr_pred

            b_stats = compute_standardized_rollout_metrics(
                target_p_mp=target_p_mp,
                target_p_lmr=target_p_lmr,
                r_real=r_real_eff,
                z_q=z_quiet,
                is_cap=is_cap,
                legal_mask=legal_mask,
                tau_mp=tau_mp,
                tau_lmr=eff_tau_lmr,
                is_live_rollout=is_live_rollout
            )

            B = u_node.size(0)
            total_count += B
            tot_loss_sum += loss.item() * B
            w_anc_sum += loss_reg.item() * B

            for k, v in b_stats.items():
                stat_sums[k] = stat_sums.get(k, 0.0) + v * B

    model.train()
    n = max(1, total_count)
    res = {k: v / n for k, v in stat_sums.items()}
    res["total_loss"] = tot_loss_sum / n
    res["w_anchor_loss"] = w_anc_sum / n
    return res


def evaluate_2d_depth_rank_matrix(
    model: Optional[nn.Module],
    loader: DataLoader,
    master_loader: Optional[DataLoader] = None,
    tau_lmr: float = 1.5,
    train_mode: str = "both",
    is_live_rollout: bool = False
):
    if train_mode == "movepicker":
        print("\n" + "=" * 80, flush=True)
        print("  [2D LMR Matrix skipped: MovePicker-only run; LMR is native Stockfish Master]", flush=True)
        print("=" * 80 + "\n", flush=True)
        return

    if model is not None:
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
            "counts": [0] * 4, "leg_counts": [0] * 4
        } for name, _, _ in bands
    }

    with torch.no_grad():
        if is_live_rollout and master_loader is not None:
            # 1. Accumulate Master live search statistics
            for u_node, t_quiet, x_lmr, is_cap, r_base, r_legacy, z_legacy_mp, target_p_mp, target_p_lmr, legal_mask, depth in master_loader:
                max_red = (depth.unsqueeze(1) - 1.0).clamp(min=0.0)
                min_red = torch.tensor(-2.0, device=r_base.device)
                r_real_leg = torch.minimum(torch.maximum(r_base, min_red), max_red)
                r_real_leg[:, 0] = 0.0
                E_leg = torch.exp(-r_real_leg / tau_lmr)

                B = depth.size(0)
                for b in range(B):
                    d_val = depth[b].item()
                    num_m = legal_mask[b].sum().item()
                    if num_m < 3:
                        continue
                    band_name = next((name for name, d_min, d_max in bands if d_min <= d_val <= d_max), None)
                    if not band_name:
                        continue
                    b_stats = stats[band_name]
                    for k in range(1, 4):
                        if num_m > k:
                            b_stats["leg_red"][k-1] += r_real_leg[b, k].item()
                            b_stats["leg_eff"][k-1] += E_leg[b, k].item()
                            b_stats["leg_counts"][k-1] += 1
                    if num_m > 4:
                        b_stats["leg_red"][3] += r_real_leg[b, 4:num_m].mean().item()
                        b_stats["leg_eff"][3] += E_leg[b, 4:num_m].mean().item()
                        b_stats["leg_counts"][3] += 1

            # 2. Accumulate Neural MiniNN live search statistics
            for u_node, t_quiet, x_lmr, is_cap, r_base, r_legacy, z_legacy_mp, target_p_mp, target_p_lmr, legal_mask, depth in loader:
                max_red = (depth.unsqueeze(1) - 1.0).clamp(min=0.0)
                min_red = torch.tensor(-2.0, device=r_base.device)
                r_real_nn = torch.minimum(torch.maximum(r_base, min_red), max_red)
                r_real_nn[:, 0] = 0.0
                E_nn = torch.exp(-r_real_nn / tau_lmr)

                B = depth.size(0)
                for b in range(B):
                    d_val = depth[b].item()
                    num_m = legal_mask[b].sum().item()
                    if num_m < 3:
                        continue
                    band_name = next((name for name, d_min, d_max in bands if d_min <= d_val <= d_max), None)
                    if not band_name:
                        continue
                    b_stats = stats[band_name]
                    for k in range(1, 4):
                        if num_m > k:
                            b_stats["nn_red"][k-1] += r_real_nn[b, k].item()
                            b_stats["nn_eff"][k-1] += E_nn[b, k].item()
                            b_stats["counts"][k-1] += 1
                    if num_m > 4:
                        b_stats["nn_red"][3] += r_real_nn[b, 4:num_m].mean().item()
                        b_stats["nn_eff"][3] += E_nn[b, 4:num_m].mean().item()
                        b_stats["counts"][3] += 1
        else:
            for u_node, t_quiet, x_lmr, is_cap, r_base, r_legacy, z_legacy_mp, target_p_mp, target_p_lmr, legal_mask, depth in loader:
                w_mp, w_lmr, tau_mp, t_lmr_pred, quiet_scores, delta_r_nn = model(u_node, t_quiet, x_lmr)

                r_total_nn = r_base + delta_r_nn
                max_red = (depth.unsqueeze(1) - 1.0).clamp(min=0.0)
                min_red = torch.tensor(-2.0, device=delta_r_nn.device)
                r_real_nn = torch.minimum(torch.maximum(r_total_nn, min_red), max_red)
                r_real_nn[:, 0] = 0.0
                r_real_leg = torch.minimum(torch.maximum(r_base, min_red), max_red)
                r_real_leg[:, 0] = 0.0

                E_nn = torch.exp(-r_real_nn / tau_lmr)
                E_leg = torch.exp(-r_real_leg / tau_lmr)

                B = depth.size(0)
                for b in range(B):
                    d_val = depth[b].item()
                    num_m = legal_mask[b].sum().item()
                    if num_m < 3:
                        continue

                    band_name = next((name for name, d_min, d_max in bands if d_min <= d_val <= d_max), None)
                    if not band_name:
                        continue

                    b_stats = stats[band_name]

                    for k in range(1, 4):
                        if num_m > k:
                            b_stats["nn_red"][k-1] += r_real_nn[b, k].item()
                            b_stats["nn_eff"][k-1] += E_nn[b, k].item()
                            b_stats["leg_red"][k-1] += r_real_leg[b, k].item()
                            b_stats["leg_eff"][k-1] += E_leg[b, k].item()
                            b_stats["counts"][k-1] += 1
                            b_stats["leg_counts"][k-1] += 1

                    if num_m > 4:
                        b_stats["nn_red"][3] += r_real_nn[b, 4:num_m].mean().item()
                        b_stats["nn_eff"][3] += E_nn[b, 4:num_m].mean().item()
                        b_stats["leg_red"][3] += r_real_leg[b, 4:num_m].mean().item()
                        b_stats["leg_eff"][3] += E_leg[b, 4:num_m].mean().item()
                        b_stats["counts"][3] += 1
                        b_stats["leg_counts"][3] += 1

    matrix_title = "LIVE-SEARCH 2D REDUCTIONS & SEARCH EFFORT MATRIX" if is_live_rollout else "2D LATE-MOVE REDUCTIONS & SEARCH EFFORT MATRIX"
    print("\n" + "=" * 105, flush=True)
    print(f"                {matrix_title} (DEPTH BANDS x MOVE RANKS)", flush=True)
    print("=" * 105, flush=True)
    print(f"{'Depth Band':<15} | {'Policy':<14} | {'Move 2 (Late)':<18} | {'Move 3 (Late)':<18} | {'Move 4 (Late)':<18} | {'Move 5+ (Tail)':<18}", flush=True)
    print("-" * 105, flush=True)

    for name, _, _ in bands:
        b_stats = stats[name]
        c_nn = [max(1, cnt) for cnt in b_stats["counts"]]
        c_leg = [max(1, cnt) for cnt in b_stats["leg_counts"]]

        nn_r_str = [f"{b_stats['nn_red'][i]/c_nn[i]:+5.2f} (E:{b_stats['nn_eff'][i]/c_nn[i]:4.2f})" for i in range(4)]
        leg_r_str = [f"{b_stats['leg_red'][i]/c_leg[i]:+5.2f} (E:{b_stats['leg_eff'][i]/c_leg[i]:4.2f})" for i in range(4)]

        print(f"{name:<15} | {'Neural MiniNN':<14} | {nn_r_str[0]:<18} | {nn_r_str[1]:<18} | {nn_r_str[2]:<18} | {nn_r_str[3]:<18}", flush=True)
        print(f"{'':<15} | {'Legacy Master':<14} | {leg_r_str[0]:<18} | {leg_r_str[1]:<18} | {leg_r_str[2]:<18} | {leg_r_str[3]:<18}", flush=True)
        print("-" * 105, flush=True)
    print("=" * 105 + "\n", flush=True)
    if model is not None:
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

    # To guarantee >= target_samples, sample frequently enough from the FEN pool
    samples_per_fen = max(1, math.ceil(target_samples / len(val_fens_pool)))
    # Use high-density sampling rate (at least 3.0x density) to guarantee reaching target_samples
    sample_interval = max(1000, nodes_per_fen // max(1, int(samples_per_fen * 3.0)))

    curr_idx = 0
    all_lines = []

    while len(all_lines) < target_samples:
        batch_fens_count = len(val_fens_pool) if len(all_lines) == 0 else max(50, int(math.ceil((target_samples - len(all_lines)) / 20)))
        fens_slice = [val_fens_pool[(curr_idx + i) % len(val_fens_pool)] for i in range(batch_fens_count)]
        curr_idx += batch_fens_count

        chunk_size = math.ceil(len(fens_slice) / workers)
        chunks = [fens_slice[i : i + chunk_size] for i in range(0, len(fens_slice), chunk_size)]
        worker_tel_paths = [os.path.join(CACHE_DIR, f"tel_w{w_id}_{session_tag}.jsonl") for w_id in range(len(chunks))]

        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(run_stockfish_search_worker, w_id, chunk, nodes_per_fen, sample_interval, "", worker_tel_paths[w_id])
                for w_id, chunk in enumerate(chunks)
            ]
            for f in as_completed(futures):
                f.result()

        for p in worker_tel_paths:
            if os.path.exists(p):
                with open(p, "r") as in_f:
                    for line in in_f:
                        if line.strip():
                            try:
                                obj = json.loads(line.strip())
                                if obj.get("moves") and len(obj["moves"]) >= 3:
                                    all_lines.append(line)
                            except Exception:
                                pass
                os.remove(p)

        if len(all_lines) < target_samples:
            sample_interval = max(1000, sample_interval // 2)

    rng = random.Random(42)
    rng.shuffle(all_lines)

    selected_lines = all_lines[:target_samples]
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


def collect_or_load_offpolicy_buffer(
    train_fens_pool: List[str],
    target_samples: int,
    nodes_per_fen: int,
    sample_interval: int,
    workers: int,
    session_tag: str = "master_offpolicy_shared_v5"
) -> Tuple[str, str]:
    """
    Collects or reuses a large Master off-policy replay buffer:
    - Master Stockfish searches without miniNN (pure handcrafted MovePicker & LMR).
    - Checks cache in CACHE_DIR (sf_tel_{session_tag}.jsonl, monty_{session_tag}.db).
    - If valid cached file with >= target_samples exists, reuses it immediately.
    - Otherwise, searches dynamically only as many FENs as needed to gather target_samples, and queries Monty.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    merged_tel_path = os.path.join(CACHE_DIR, f"sf_tel_{session_tag}.jsonl")
    monty_db_path = os.path.join(CACHE_DIR, f"monty_{session_tag}.db")

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
                print(f"      Reusing cached Master off-policy replay buffer ({count:,} samples) from {merged_tel_path}", flush=True)
                return merged_tel_path, monty_db_path
        else:
            os.remove(merged_tel_path)

    print(f"      Generating Master off-policy replay buffer ({target_samples:,} samples)...", flush=True)
    tel_path, db_path, _ = collect_target_samples(
        fens_pool=train_fens_pool,
        fen_offset=0,
        target_samples=target_samples,
        nodes_per_fen=nodes_per_fen,
        sample_interval=sample_interval,
        model_path="",
        workers=workers,
        session_tag=session_tag
    )
    return tel_path, db_path


def collect_target_samples(
    fens_pool: List[str],
    fen_offset: int,
    target_samples: int,
    nodes_per_fen: int,
    sample_interval: int,
    model_path: str,
    workers: int,
    session_tag: str,
    use_mp: bool = True,
    use_lmr: bool = True
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
    samples_per_fen_est = 40.0

    while len(all_lines) < target_samples:
        needed = target_samples - len(all_lines)
        batch_fens_count = max(workers * 2, int(math.ceil(needed / samples_per_fen_est)))
        fens_slice = [fens_pool[(curr_ptr + i) % len(fens_pool)] for i in range(batch_fens_count)]
        curr_ptr += batch_fens_count

        chunk_size = math.ceil(len(fens_slice) / workers)
        chunks = [fens_slice[i : i + chunk_size] for i in range(0, len(fens_slice), chunk_size)]
        worker_tel_paths = [os.path.join(CACHE_DIR, f"tel_w{w_id}_{session_tag}.jsonl") for w_id in range(len(chunks))]

        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(run_stockfish_search_worker, w_id, chunk, nodes_per_fen, sample_interval, model_path, worker_tel_paths[w_id], use_mp, use_lmr)
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
    session_tag: str = "heldout_test",
    use_mp: bool = True,
    use_lmr: bool = True
) -> Dict[str, float]:
    tag_name = "NEURAL MININN" if model is not None else "HANDCRAFTED STOCKFISH MASTER"
    print("\n" + "=" * 80, flush=True)
    print(f"   {tag_name} - HELDOUT ON-POLICY EVALUATION ({len(test_fens):,} TEST FENs)", flush=True)
    print("=" * 80, flush=True)

    temp_model_path = ""
    if model is not None:
        temp_model_path = f"/tmp/online_test_{session_tag}.miniNN"
        model.export_quantized_binary(temp_model_path)

    if model is None:
        tel_path = os.path.join(CACHE_DIR, "master_heldout_shared_v5.jsonl")
        db_path = os.path.join(CACHE_DIR, "master_heldout_shared_v5.db")
    else:
        tel_path = os.path.join(CACHE_DIR, f"heldout_tel_{session_tag}.jsonl")
        db_path = os.path.join(CACHE_DIR, f"heldout_monty_{session_tag}.db")
        if os.path.exists(tel_path):
            os.remove(tel_path)

    # 1. Rollout C++ Stockfish searches on heldout test FENs (cached for Master)
    if not (model is None and os.path.exists(tel_path) and os.path.getsize(tel_path) > 0 and os.path.exists(db_path)):
        samples_per_fen = max(1, math.ceil(target_samples / len(test_fens)))
        sample_interval = max(500, nodes_per_fen // max(1, int(samples_per_fen * 3.0)))

        curr_ptr = 0
        all_lines = []

        while len(all_lines) < target_samples:
            batch_count = len(test_fens) if len(all_lines) == 0 else max(50, int(math.ceil((target_samples - len(all_lines)) / 20)))
            fens_slice = [test_fens[(curr_ptr + i) % len(test_fens)] for i in range(batch_count)]
            curr_ptr += batch_count

            chunk_size = math.ceil(len(fens_slice) / workers)
            chunks = [fens_slice[i : i + chunk_size] for i in range(0, len(fens_slice), chunk_size)]
            worker_tel_paths = [os.path.join(CACHE_DIR, f"heldout_tel_w{w_id}_{session_tag}.jsonl") for w_id in range(len(chunks))]

            with ProcessPoolExecutor(max_workers=workers) as executor:
                futures = [
                    executor.submit(run_stockfish_search_worker, w_id, chunk, nodes_per_fen, sample_interval, temp_model_path, worker_tel_paths[w_id], use_mp, use_lmr)
                    for w_id, chunk in enumerate(chunks)
                ]
                for f in as_completed(futures):
                    f.result()

            for p in worker_tel_paths:
                if os.path.exists(p):
                    with open(p, "r") as in_f:
                        for line in in_f:
                            if line.strip():
                                try:
                                    obj = json.loads(line.strip())
                                    if obj.get("moves") and len(obj["moves"]) >= 3:
                                        all_lines.append(line)
                                except Exception:
                                    pass
                    os.remove(p)

            if len(all_lines) < target_samples:
                sample_interval = max(500, sample_interval // 2)

        with open(tel_path, "w") as out_f:
            for line in all_lines[:target_samples]:
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

    # 3. Load heldout rollout dataset and evaluate standardized metrics
    t_teacher_lmr_calib, t_teacher_mp_calib, tau_student_lmr_calib, tau_student_mp_calib, floor_lmr_calib, floor_mp_calib = load_calibration_parameters()
    heldout_dataset = RolloutDataset(
        telemetry_path=tel_path,
        monty_db_path=db_path,
        floor_lmr=floor_lmr_calib,
        floor_mp=floor_mp_calib,
        t_teacher_lmr=t_teacher_lmr_calib,
        t_teacher_mp=t_teacher_mp_calib
    )
    heldout_loader = DataLoader(heldout_dataset, batch_size=256, shuffle=False)

    if model is None:
        heldout_stats = evaluate_handcrafted_master(heldout_loader, tau_mp=tau_student_mp_calib, tau_lmr=tau_student_lmr_calib)
    else:
        heldout_train_mode = "movepicker" if (not use_lmr and use_mp) else ("lmr" if (use_lmr and not use_mp) else "both")
        heldout_stats = evaluate_validation_rollout(model, heldout_loader, tau_lmr=tau_student_lmr_calib, train_mode=heldout_train_mode, is_live_rollout=True)

        # Print Live-Search 2D Reductions Matrix comparing Neural vs Master searches
        master_tel = os.path.join(CACHE_DIR, "master_heldout_shared_v5.jsonl")
        master_db = os.path.join(CACHE_DIR, "master_heldout_shared_v5.db")
        if os.path.exists(master_tel) and os.path.exists(master_db):
            master_ds = RolloutDataset(master_tel, master_db, floor_lmr=floor_lmr_calib, floor_mp=floor_mp_calib, t_teacher_lmr=t_teacher_lmr_calib, t_teacher_mp=t_teacher_mp_calib)
            master_ldr = DataLoader(master_ds, batch_size=256, shuffle=False)
            evaluate_2d_depth_rank_matrix(model=None, loader=heldout_loader, master_loader=master_ldr, tau_lmr=tau_student_lmr_calib, train_mode=heldout_train_mode, is_live_rollout=True)

    print_unified_benchmark_report("Neural MiniNN Heldout Test Set (1,000 FENs)", heldout_stats)

    if temp_model_path and os.path.exists(temp_model_path):
        os.remove(temp_model_path)

    return {f"heldout_{k}": v for k, v in heldout_stats.items()}


def train_single_run(
    run_name: str,
    lr: float,
    mp_anchor_coef: float,
    lmr_ord_coef: float,
    rank_profile_coef: float,
    args: argparse.Namespace,
    val_loader: DataLoader,
    val_dataset: RolloutDataset,
    train_fens_pool: List[str],
    test_fens_pool: List[str],
    t_teacher_lmr: float,
    t_teacher_mp: float,
    tau_student_lmr: float,
    tau_student_mp: float,
    floor_lmr: float,
    floor_mp: float,
    output_path: str,
    offpolicy_dataset: Optional[RolloutDataset] = None
) -> Dict[str, float]:
    print("\n" + "=" * 80, flush=True)
    print(f"   STARTING RUN: {run_name}", flush=True)
    print("=" * 80, flush=True)
    print(f"Iterations:                 {args.iterations}", flush=True)
    print(f"Off-Policy Warmup:          {args.offpolicy_iterations} iters (Master replay buffer)", flush=True)
    print(f"Peak Learning Rate:         {lr:.4e}", flush=True)
    print(f"LR Schedule:                Warmup (min 2 iters) -> Cosine Decay (Floor: {lr * 0.30:.4e})", flush=True)
    print(f"Rollout Buffer Size:        {args.rollout_samples:,}", flush=True)
    print(f"Replay Window:              {args.replay_window_iters} iters ({args.replay_window_iters * args.rollout_samples:,} total window samples)", flush=True)
    print(f"Mini-Batch Size:            {args.minibatch_size}", flush=True)
    print(f"PPO Multi-Epochs / Iter:    {args.ppo_epochs}", flush=True)
    print(f"MovePicker Anchor Coef:     {mp_anchor_coef:.2f}", flush=True)
    print(f"LMR Policy KL Coef:         {lmr_ord_coef:.2f}", flush=True)
    print(f"Rank-Profile MSE Coef:      {rank_profile_coef:.2f}", flush=True)
    print(f"Teacher Temperatures:       MP: {t_teacher_mp:.4f} | LMR: {t_teacher_lmr:.4f}", flush=True)
    print(f"Student Temperatures:       MP: {tau_student_mp:.4f} | LMR: {tau_student_lmr:.4f}", flush=True)
    print(f"Output Binary:              {output_path}", flush=True)
    print("=" * 80, flush=True)

    if args.offpolicy_iterations > 0 and offpolicy_dataset is None:
        total_offpolicy_needed = max(16384, args.offpolicy_iterations * args.rollout_samples)
        offpolicy_tel, offpolicy_db = collect_or_load_offpolicy_buffer(
            train_fens_pool=train_fens_pool,
            target_samples=total_offpolicy_needed,
            nodes_per_fen=args.nodes,
            sample_interval=args.sample_interval,
            workers=args.workers,
            session_tag="master_offpolicy_shared_v3"
        )
        offpolicy_dataset = RolloutDataset(
            telemetry_path=offpolicy_tel,
            monty_db_path=offpolicy_db,
            floor_lmr=floor_lmr,
            floor_mp=floor_mp,
            t_teacher_lmr=t_teacher_lmr,
            t_teacher_mp=t_teacher_mp
        )

    train_mode = getattr(args, "train_mode", "both")
    use_mp = (train_mode != "lmr")
    use_lmr = (train_mode != "movepicker")
    mode_str = "MovePicker Only" if train_mode == "movepicker" else ("LMR Only" if train_mode == "lmr" else "Dual MP+LMR")

    model = DualMiniNN(tau_mp_base=tau_student_mp, tau_lmr_base=tau_student_lmr)
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
    onpolicy_datasets_window = []

    for iteration in range(1, args.iterations + 1):
        t0 = time.time()
        is_offpolicy = (iteration <= args.offpolicy_iterations) and (offpolicy_dataset is not None)

        if is_offpolicy:
            phase_tag = "Off-Policy"
            buf_len = len(offpolicy_dataset)
            indices = torch.randperm(buf_len)[:min(buf_len, args.rollout_samples)].tolist()
            iter_subset = torch.utils.data.Subset(offpolicy_dataset, indices)
            train_loader = DataLoader(iter_subset, batch_size=args.minibatch_size, shuffle=True)
            fresh_tel, fresh_db = "", ""

            # Vectorized Live Stats pass on off-policy batch
            fresh_loader = DataLoader(iter_subset, batch_size=len(iter_subset), shuffle=False)
            live_stats = evaluate_validation_rollout(
                model, fresh_loader,
                mp_anchor_coef=mp_anchor_coef, lmr_ord_coef=lmr_ord_coef,
                tau_lmr=tau_student_lmr, train_mode=train_mode
            )
        else:
            if iteration == args.offpolicy_iterations + 1 and args.offpolicy_iterations > 0:
                print("\n" + "=" * 80, flush=True)
                print(f"   >>> SWITCHING TO PHASE 2: ON-POLICY NEURAL SEARCH REFINEMENT ({mode_str.upper()}) <<<", flush=True)
                print("=" * 80 + "\n", flush=True)

            phase_tag = "On-Policy"
            fresh_tel, fresh_db, curr_fen_offset = collect_target_samples(
                fens_pool=train_fens_pool,
                fen_offset=curr_fen_offset,
                target_samples=args.rollout_samples,
                nodes_per_fen=args.nodes,
                sample_interval=args.sample_interval,
                model_path=output_path,
                workers=args.workers,
                session_tag=f"{run_name}_iter{iteration}",
                use_mp=use_mp,
                use_lmr=use_lmr
            )

            curr_dataset = RolloutDataset(
                telemetry_path=fresh_tel,
                monty_db_path=fresh_db,
                floor_lmr=floor_lmr,
                floor_mp=floor_mp,
                t_teacher_lmr=t_teacher_lmr,
                t_teacher_mp=t_teacher_mp
            )

            # Vectorized Live Stats pass on fresh on-policy rollout (Zero Extra Cost)
            fresh_loader = DataLoader(curr_dataset, batch_size=len(curr_dataset), shuffle=False)
            live_stats = evaluate_validation_rollout(
                model, fresh_loader,
                mp_anchor_coef=mp_anchor_coef, lmr_ord_coef=lmr_ord_coef,
                tau_lmr=tau_student_lmr, train_mode=train_mode,
                is_live_rollout=True
            )

            # Maintain sliding replay window across iterations
            onpolicy_datasets_window.append((curr_dataset, fresh_tel, fresh_db))
            if len(onpolicy_datasets_window) > args.replay_window_iters:
                old_ds, old_tel, old_db = onpolicy_datasets_window.pop(0)
                try:
                    if old_tel and os.path.exists(old_tel):
                        os.remove(old_tel)
                    if old_db and os.path.exists(old_db):
                        os.remove(old_db)
                except Exception:
                    pass

            if len(onpolicy_datasets_window) == 1:
                train_dataset = onpolicy_datasets_window[0][0]
            else:
                train_dataset = torch.utils.data.ConcatDataset([d[0] for d in onpolicy_datasets_window])
            train_loader = DataLoader(train_dataset, batch_size=args.minibatch_size, shuffle=True)

        iter_steps = 0
        iter_loss, iter_mp_kl_q, iter_w_anc, iter_lmr_ord = 0.0, 0.0, 0.0, 0.0

        for epoch in range(args.ppo_epochs):
            for u_node, t_quiet, x_lmr, is_cap, r_base, r_legacy, z_legacy_mp, target_p_mp, target_p_lmr, legal_mask, depth in train_loader:
                optimizer.zero_grad()
                w_mp, w_lmr, tau_mp, tau_lmr, quiet_scores, delta_r_nn = model(u_node, t_quiet, x_lmr)

                z_quiet = quiet_scores / 32768.0

                loss, loss_mp_kl_q, loss_lmr_ord, loss_reg, _ = compute_combined_losses(
                    z_quiet, delta_r_nn, tau_mp, tau_lmr, target_p_mp, target_p_lmr, z_legacy_mp, r_base, r_legacy, is_cap, legal_mask, depth,
                    w_mp=w_mp, w_lmr=w_lmr, reg_coef=mp_anchor_coef, lmr_ord_coef=lmr_ord_coef, train_mode=train_mode
                )

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()

                total_gradient_steps += 1
                iter_steps += 1
                iter_loss += loss.item()
                iter_mp_kl_q += loss_mp_kl_q.item()
                iter_w_anc += loss_reg.item()
                iter_lmr_ord += loss_lmr_ord.item()

                if total_gradient_steps % args.sync_interval == 0:
                    model.export_quantized_binary(output_path)

        elapsed_iter = time.time() - t0
        curr_lr = scheduler.get_last_lr()[0]
        n_steps = max(1, iter_steps)

        live_str = f"Live Search: Top1:{live_stats['top1_match']:4.1f}% (Q:{live_stats['quiet_top1']:4.1f}%)"
        if train_mode != "movepicker":
            live_str += f", RedTop1:{live_stats['top1_reduction']:4.2f}p, RedLate:{live_stats['late_reduction']:4.2f}p"

        if iteration % args.val_freq == 0 or iteration == 1 or iteration == args.iterations:
            val_stats = evaluate_validation_rollout(
                model, val_loader,
                mp_anchor_coef=mp_anchor_coef, lmr_ord_coef=lmr_ord_coef,
                tau_lmr=tau_student_lmr, train_mode=train_mode
            )
            print(f"[{run_name} | Iter {iteration:>4d}/{args.iterations} ({phase_tag} | {mode_str})] ({elapsed_iter:4.1f}s | lr: {curr_lr:.4e}) "
                  f"Loss: {iter_loss/n_steps:.4f} (MP_Q: {iter_mp_kl_q/n_steps:.3f}, W_Anc: {iter_w_anc/n_steps:.3f}, LMR: {iter_lmr_ord/n_steps:.3f}) | "
                  f"{live_str} | Val MP: (Q:{val_stats['quiet_top1']:4.1f}%) | Val Loss: {val_stats['total_loss']:.4f}", flush=True)
        else:
            print(f"[{run_name} | Iter {iteration:>4d}/{args.iterations} ({phase_tag} | {mode_str})] ({elapsed_iter:4.1f}s | lr: {curr_lr:.4e}) "
                  f"Loss: {iter_loss/n_steps:.4f} (MP_Q: {iter_mp_kl_q/n_steps:.3f}, W_Anc: {iter_w_anc/n_steps:.3f}, LMR: {iter_lmr_ord/n_steps:.3f}) | {live_str}", flush=True)

    # Clean up any remaining sliding window temp files
    for _, old_tel, old_db in onpolicy_datasets_window:
        try:
            if old_tel and os.path.exists(old_tel):
                os.remove(old_tel)
            if old_db and os.path.exists(old_db):
                os.remove(old_db)
        except Exception:
            pass

    final_stats = evaluate_validation_rollout(
        model, val_loader,
        mp_anchor_coef=mp_anchor_coef, lmr_ord_coef=lmr_ord_coef,
        tau_lmr=tau_student_lmr, train_mode=train_mode
    )
    model.export_quantized_binary(output_path)

    # Output detailed 2D Depth x Rank Evaluation Matrix
    evaluate_2d_depth_rank_matrix(model, val_loader, tau_lmr=tau_student_lmr, train_mode=train_mode)

    # Final Online Testing Step on Heldout FENs (Matching val_samples scale)
    if test_fens_pool and not getattr(args, "skip_heldout_eval", False):
        heldout_stats = run_heldout_online_evaluation(
            model=model,
            test_fens=test_fens_pool,
            target_samples=args.val_samples,
            nodes_per_fen=args.nodes,
            workers=args.workers,
            session_tag=f"{run_name}_heldout_eval",
            use_mp=use_mp,
            use_lmr=use_lmr
        )
        final_stats.update(heldout_stats)

    return final_stats


def main():
    parser = argparse.ArgumentParser(description="On-Policy Dual Mini-NN Closed-Loop Trainer & Grid Runner.")
    parser.add_argument("--iterations", type=int, default=128, help="Total outer iterations per run (default: 128)")
    parser.add_argument("--offpolicy-iterations", type=int, default=16, help="Number of initial off-policy warmup iterations using Master movepicker & LMR before switching to on-policy (default: 16)")
    parser.add_argument("--train-mode", type=str, choices=["both", "movepicker", "lmr"], default="both", help="Training mode: 'movepicker' (MP only, Master LMR), 'lmr' (LMR only, Master MP), 'both' (dual joint MP+LMR) (default: both)")
    parser.add_argument("--movepicker-only", action="store_true", help="Shortcut for --train-mode movepicker")
    parser.add_argument("--lmr-only", action="store_true", help="Shortcut for --train-mode lmr")
    parser.add_argument("--rollout-samples", "--replay-buffer-size", dest="rollout_samples", type=int, default=4096, help="Fresh on-policy rollout buffer size per iteration (default: 4096)")
    parser.add_argument("--replay-window-iters", type=int, default=4, help="Sliding replay window in iterations (accumulates last K iterations for training) (default: 4)")
    parser.add_argument("--minibatch-size", type=int, default=256, help="Minibatch size for SGD updates (default: 256)")
    parser.add_argument("--ppo-epochs", type=int, default=8, help="PPO multi-epoch passes over fresh rollout buffer (default: 8)")
    parser.add_argument("--sync-interval", type=int, default=4, help="Gradient steps between model syncs (default: 4)")
    parser.add_argument("--val-freq", type=int, default=8, help="Validation frequency (default: 8)")
    parser.add_argument("--val-samples", type=int, default=32768, help="Fixed validation rollout samples (default: 32768 = 2^15)")
    parser.add_argument("--stream-limit", type=int, default=500000, help="Stream limit for FEN subsampling (default: 500,000)")
    parser.add_argument("--train-fens-pool", type=int, default=100000, help="Training FEN pool (default: 100,000)")
    parser.add_argument("--val-fens-pool", type=int, default=1000, help="Validation FEN pool (default: 1,000)")
    parser.add_argument("--test-fens-pool", type=int, default=1000, help="Held-out Test FEN pool (default: 1,000)")
    parser.add_argument("--skip-heldout-eval", action="store_true", help="Skip held-out online evaluation at the end of the run")
    parser.add_argument("--nodes", type=int, default=500_000, help="Search budget per FEN (default: 500,000)")
    parser.add_argument("--sample-interval", type=int, default=10_000, help="Subsample interval (default: 10,000)")
    parser.add_argument("--workers", type=int, default=6, help="Parallel worker threads (default: 6)")
    parser.add_argument("--grid", action="store_true", help="Run the 3-experiment hyperparameter grid")
    parser.add_argument("--lr", type=float, default=4e-3, help="Peak learning rate (default: 4e-3)")
    parser.add_argument("--mp-anchor-coef", type=float, default=0.20, help="MovePicker anchor weight (default: 0.20)")
    parser.add_argument("--lmr-ord-coef", type=float, default=0.40, help="LMR policy cross-entropy weight (default: 0.40)")
    parser.add_argument("--rank-profile-coef", type=float, default=0.40, help="Rank profile MSE loss weight (default: 0.40)")
    parser.add_argument("--auto-align-temperatures", action="store_true", default=True, help="Automatically align student logit variance to teacher target log-probability variance on the validation dataset (default: True)")
    parser.add_argument("--no-auto-align-temperatures", dest="auto_align_temperatures", action="store_false", help="Disable automatic student variance matching")
    parser.add_argument("--t-teacher-mp", type=float, default=None, help="Teacher shaping temperature for MovePicker (default from config or 0.50)")
    parser.add_argument("--t-teacher-lmr", type=float, default=None, help="Teacher shaping temperature for LMR (default from config or 1.00)")
    parser.add_argument("--tau-student-mp", type=float, default=None, help="Student logit scale temperature for MovePicker (default: auto variance-matched)")
    parser.add_argument("--tau-student-lmr", type=float, default=None, help="Student reduction scale temperature for LMR (default: auto variance-matched)")
    parser.add_argument("--output", type=str, default="floored_dual_64it.miniNN", help="Output model binary path")

    args = parser.parse_args()
    if args.movepicker_only:
        args.train_mode = "movepicker"
    elif args.lmr_only:
        args.train_mode = "lmr"

    t_teacher_lmr_cfg, t_teacher_mp_cfg, tau_student_lmr_cfg, tau_student_mp_cfg, floor_lmr, floor_mp = load_calibration_parameters()
    t_teacher_lmr = args.t_teacher_lmr if args.t_teacher_lmr is not None else t_teacher_lmr_cfg
    t_teacher_mp = args.t_teacher_mp if args.t_teacher_mp is not None else t_teacher_mp_cfg
    tau_student_lmr = args.tau_student_lmr if args.tau_student_lmr is not None else tau_student_lmr_cfg
    tau_student_mp = args.tau_student_mp if args.tau_student_mp is not None else tau_student_mp_cfg

    mode_display = "MovePicker Only (Master LMR)" if args.train_mode == "movepicker" else ("LMR Only (Master MovePicker)" if args.train_mode == "lmr" else "Dual (MovePicker + LMR)")

    print("=" * 80, flush=True)
    print("   ON-POLICY CLOSED-LOOP DUAL MINI-NN TRAINER (FULL POLICY SEARCH ALLOCATION)", flush=True)
    print("=" * 80, flush=True)
    print(f"Training Mode:               {mode_display}", flush=True)
    print(f"Total Iterations:            {args.iterations:,}", flush=True)
    print(f"Off-Policy Warmup Iterations:{args.offpolicy_iterations:>8d} (Master handcrafted search replay buffer)", flush=True)
    print(f"Validation Frequency:        Every {args.val_freq} iterations", flush=True)
    print(f"Validation Rollout Target:   {args.val_samples:,} samples (2^15)", flush=True)
    print(f"Validation FEN Pool:         {args.val_fens_pool:,} FENs (drawn from 500k stream)", flush=True)
    print(f"Heldout Test FEN Pool:       {args.test_fens_pool:,} FENs (unseen test set)", flush=True)
    print(f"Rollout Buffer Size / Iter:  {args.rollout_samples:,} transitions", flush=True)
    print(f"Replay Window (Phase 2):     {args.replay_window_iters} iterations ({args.replay_window_iters * args.rollout_samples:,} transitions window)", flush=True)
    print(f"Mini-Batch Size:             {args.minibatch_size}", flush=True)
    print(f"PPO Multi-Epochs / Iter:     {args.ppo_epochs} epochs ({args.ppo_epochs * math.ceil(args.rollout_samples / args.minibatch_size)} gradient steps/iter)", flush=True)
    print(f"Teacher Targets Shaping:     MP T = {t_teacher_mp:.4f} (Floor: {floor_mp:.3f}) | LMR T = {t_teacher_lmr:.4f} (Floor: {floor_lmr:.3f})", flush=True)
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

    print(f"[1/3] Loading / Generating Validation Rollout (evenly from {len(val_fens_pool):,} FENs, target max {args.val_samples:,})...", flush=True)
    t_v0 = time.time()

    val_tel, val_db = collect_validation_rollout_even(
        val_fens_pool=val_fens_pool,
        target_samples=args.val_samples,
        nodes_per_fen=args.nodes,
        workers=args.workers,
        session_tag="val_v5_staged_500k_shared"
    )

    val_dataset = RolloutDataset(
        telemetry_path=val_tel,
        monty_db_path=val_db,
        floor_lmr=floor_lmr,
        floor_mp=floor_mp,
        t_teacher_lmr=t_teacher_lmr,
        t_teacher_mp=t_teacher_mp
    )
    val_loader = DataLoader(val_dataset, batch_size=args.minibatch_size, shuffle=False)
    print(f"      Validation set ready: {len(val_dataset):,} transitions (sampled across {len(val_fens_pool):,} FENs) in {time.time() - t_v0:.1f}s.", flush=True)

    matched_tau_mp, matched_tau_lmr = val_dataset.compute_variance_matched_temperatures()
    if args.auto_align_temperatures:
        if args.tau_student_mp is None:
            tau_student_mp = matched_tau_mp
        if args.tau_student_lmr is None:
            tau_student_lmr = matched_tau_lmr
        print(f"      Auto-Aligned Student Scales (Logit Variance Matching): MP tau = {tau_student_mp:.4f} | LMR tau = {tau_student_lmr:.4f}\n", flush=True)
    else:
        print(f"      Configured Student Scales: MP tau = {tau_student_mp:.4f} | LMR tau = {tau_student_lmr:.4f} (Empirical Match: MP={matched_tau_mp:.4f}, LMR={matched_tau_lmr:.4f})\n", flush=True)

    offpolicy_dataset = None
    if args.offpolicy_iterations > 0:
        total_offpolicy_needed = args.offpolicy_iterations * args.rollout_samples
        print(f"[2/3] Loading / Generating Master Off-Policy Replay Buffer ({total_offpolicy_needed:,} samples for {args.offpolicy_iterations} warmup iterations)...", flush=True)
        t_op0 = time.time()
        offpolicy_tel, offpolicy_db = collect_or_load_offpolicy_buffer(
            train_fens_pool=train_fens_pool,
            target_samples=total_offpolicy_needed,
            nodes_per_fen=args.nodes,
            sample_interval=args.sample_interval,
            workers=args.workers,
            session_tag="master_offpolicy_shared_v5"
        )
        offpolicy_dataset = RolloutDataset(
            telemetry_path=offpolicy_tel,
            monty_db_path=offpolicy_db,
            floor_lmr=floor_lmr,
            floor_mp=floor_mp,
            t_teacher_lmr=t_teacher_lmr,
            t_teacher_mp=t_teacher_mp
        )
        print(f"      Master off-policy replay buffer ready: {len(offpolicy_dataset):,} samples in {time.time() - t_op0:.1f}s.\n", flush=True)
    else:
        print(f"[2/3] Skipping Master Off-Policy Replay Buffer (pure on-policy mode).\n", flush=True)

    # Master Baseline Evaluation on Fixed 2^15 Validation Set
    master_stats = evaluate_handcrafted_master(val_loader, tau_mp=tau_student_mp, tau_lmr=tau_student_lmr)
    print_unified_benchmark_report("Handcrafted Stockfish Master Baseline", master_stats)

    if args.grid:
        grid_configs = [
            {
                "name": "Run1_Residual_LMR40",
                "lr": 4e-3,
                "mp_anchor": 0.20,
                "lmr_ord": 0.40,
                "rank_profile": 0.40,
                "output": "floored_dual_64it.miniNN"
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
                args=args,
                val_loader=val_loader,
                val_dataset=val_dataset,
                train_fens_pool=train_fens_pool,
                test_fens_pool=test_fens_pool,
                t_teacher_lmr=t_teacher_lmr,
                t_teacher_mp=t_teacher_mp,
                tau_student_lmr=tau_student_lmr,
                tau_student_mp=tau_student_mp,
                floor_lmr=floor_lmr,
                floor_mp=floor_mp,
                output_path=cfg["output"],
                offpolicy_dataset=offpolicy_dataset
            )
            results[cfg["name"]] = stats

        print("\n" + "=" * 168, flush=True)
        print("                                            FINAL COMPARATIVE BENCHMARK TABLE (GRID EXPERIMENTS)", flush=True)
        print("=" * 168, flush=True)
        header = f"{'Configuration':<30} | {'Move 1 Top-1':<13} | {'Quiet 1':<11} | {'Capture 1':<11} | {'Move1 in M3':<11} | {'M1 in Move3':<11} | {'Top-1 Red':<9} | {'Late Red':<9} | {'Val Effort':<10} | {'Val Red':<8}"
        print(header, flush=True)
        print("-" * 168, flush=True)
        m_top1_str = f"{master_stats.get('top1_match', 0.0):.2f}%"
        m_q1_str = f"{master_stats.get('quiet_top1', 0.0):.2f}%"
        m_c1_str = f"{master_stats.get('cap_top1', 0.0):.2f}%"
        m_cpp1_in_m3_str = f"{master_stats.get('cpp1_in_m3_match', 0.0):.2f}%"
        m_m1_in_cpp3_str = f"{master_stats.get('m1_in_cpp3_match', 0.0):.2f}%"
        m_r_top1_str = f"{master_stats.get('top1_reduction', 0.0):.2f}"
        m_r_late_str = f"{master_stats.get('late_reduction', 0.0):.2f}"
        print(f"{'Stockfish Master Baseline':<30} | {m_top1_str:>13} | {m_q1_str:>11} | {m_c1_str:>11} | {m_cpp1_in_m3_str:>11} | {m_m1_in_cpp3_str:>11} | {m_r_top1_str:>9} | {m_r_late_str:>9} | {master_stats['mean_effort']:>10.4f} | {master_stats['mean_reduction']:>8.4f}", flush=True)
        print("-" * 168, flush=True)
        for name, s in results.items():
            h_top1_str = f"{s.get('heldout_top1_match', s.get('top1_match', 0.0)):.2f}%"
            h_q1_str = f"{s.get('heldout_quiet_top1', s.get('quiet_top1', 0.0)):.2f}%"
            h_c1_str = f"{s.get('heldout_cap_top1', s.get('cap_top1', 0.0)):.2f}%"
            h_cpp1_in_m3_str = f"{s.get('heldout_cpp1_in_m3_match', s.get('cpp1_in_m3_match', 0.0)):.2f}%"
            h_m1_in_cpp3_str = f"{s.get('heldout_m1_in_cpp3_match', s.get('m1_in_cpp3_match', 0.0)):.2f}%"
            h_r_top1_str = f"{s.get('heldout_top1_reduction', s.get('top1_reduction', 0.0)):.2f}"
            h_r_late_str = f"{s.get('heldout_late_reduction', s.get('late_reduction', 0.0)):.2f}"
            print(f"{name:<30} | {h_top1_str:>13} | {h_q1_str:>11} | {h_c1_str:>11} | {h_cpp1_in_m3_str:>11} | {h_m1_in_cpp3_str:>11} | {h_r_top1_str:>9} | {h_r_late_str:>9} | {s['mean_effort']:>10.4f} | {s['mean_reduction']:>8.4f}", flush=True)
        print("=" * 168 + "\n", flush=True)

    else:
        final_stats = train_single_run(
            run_name="Single_PolicyLMR_Run",
            lr=args.lr,
            mp_anchor_coef=args.mp_anchor_coef,
            lmr_ord_coef=args.lmr_ord_coef,
            rank_profile_coef=args.rank_profile_coef,
            args=args,
            val_loader=val_loader,
            val_dataset=val_dataset,
            train_fens_pool=train_fens_pool,
            test_fens_pool=test_fens_pool,
            t_teacher_lmr=t_teacher_lmr,
            t_teacher_mp=t_teacher_mp,
            tau_student_lmr=tau_student_lmr,
            tau_student_mp=tau_student_mp,
            floor_lmr=floor_lmr,
            floor_mp=floor_mp,
            output_path=args.output,
            offpolicy_dataset=offpolicy_dataset
        )

        print("\n" + "=" * 100, flush=True)
        print("      FINAL BENCHMARK EVALUATION (ON FIXED 2^15 VALIDATION ROLLOUT)", flush=True)
        print("=" * 100, flush=True)
        print(f"{'Metric':<40} | {'Trained Dual Mini-NN (V4)':<25} | {'Handcrafted Master Baseline':<25}", flush=True)
        print("-" * 100, flush=True)
        print(f"{'Physical Move 1 == Monty Top-1':<40} | {final_stats['top1_match']:<24.2f}% | {master_stats['top1_match']:<24.2f}%", flush=True)
        print(f"{'Quiet Moves Top-1 Match (within Q)':<40} | {final_stats['quiet_top1']:<24.2f}% | {master_stats['quiet_top1']:<24.2f}%", flush=True)
        print(f"{'Physical Move 1 in Monty Top-3':<40} | {final_stats['cpp1_in_m3_match']:<24.2f}% | {master_stats['cpp1_in_m3_match']:<24.2f}%", flush=True)
        print(f"{'Monty Top-1 in Physical Move 1..3':<40} | {final_stats['m1_in_cpp3_match']:<24.2f}% | {master_stats['m1_in_cpp3_match']:<24.2f}%", flush=True)
        print(f"{'MovePicker Quiet KL Divergence':<40} | {final_stats['mp_kl_q']:<25.4f} | {master_stats['mp_kl_q']:<25.4f}", flush=True)
        print(f"{'Mean Top Move Search Allocation Q(i*)':<40} | {final_stats['q_search_star']:<24.2f}% | {master_stats['q_search_star']:<24.2f}%", flush=True)
        print(f"{'LMR Policy Cross-Entropy Loss':<40} | {final_stats['lmr_policy_loss']:<25.4f} | {master_stats['lmr_policy_loss']:<25.4f}", flush=True)
        print(f"{'Mean LMR Reduction on Monty Top-1':<40} | {final_stats['top1_reduction']:<24.2f} plies | {master_stats['top1_reduction']:<24.2f} plies", flush=True)
        print(f"{'Monty-Weighted Late Search Effort':<40} | {final_stats['mean_effort']:<25.4f} | {master_stats['mean_effort']:<25.4f}", flush=True)
        print(f"{'Mean Reduction (Plies)':<40} | {final_stats['mean_reduction']:<25.4f} | {master_stats['mean_reduction']:<25.4f}", flush=True)
        print("=" * 100, flush=True)


if __name__ == "__main__":
    main()
