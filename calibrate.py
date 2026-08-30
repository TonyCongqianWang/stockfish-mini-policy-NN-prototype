"""
Unified Step 0 Calibration for Dual Mini-NN (MovePicker & LMR):
1. Parallel Stockfish search across FENs (with real-time progress logging).
2. Subsamples internal node states and extracts:
   - Master LMR reduction distribution (r_legacy in plies)
   - Master MovePicker score distribution (s_legacy with native scale 1024)
3. Queries Monty ground-truth policies in parallel with persistent SQLite caching.
4. Computes empirical Shannon Entropies:
   - H_LMR (Master search depth reduction entropy)
   - H_MP  (Master MovePicker move ordering entropy, peaky at ~91% Top-1)
5. Solves for exact calibration temperatures (T_calib_LMR and T_calib_MP) via ENTROPY MATCHING
   with independent probability floor configuration.
6. Exports unified calibration_config.json.
"""

import argparse
import json
import math
import os
import sqlite3
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Tuple

import numpy as np
from scipy.optimize import minimize

from paths import CACHE_DIR, CALIB_CONFIG_PATH as OUTPUT_CONFIG, EPD_FILE, MONTY_BIN, STOCKFISH_BIN


def extract_fens(epd_path: str, count: int = 2000, offset: int = 0) -> List[str]:
    fens = []
    seen = 0
    with open(epd_path, "r") as f:
        for line in f:
            if line.startswith("Probing: "):
                fen = line.strip().replace("Probing: ", "").strip()
                if fen:
                    if seen >= offset:
                        fens.append(fen)
                    seen += 1
            if len(fens) >= count:
                break
    return fens


def run_stockfish_telemetry_worker(
    worker_id: int,
    fens_chunk: List[str],
    nodes_per_fen: int,
    sample_interval: int,
    output_path: str
) -> int:
    if os.path.exists(output_path):
        os.remove(output_path)

    env = os.environ.copy()
    env["SF_LMR_TELEMETRY"] = output_path
    env["SF_LMR_SAMPLE_INTERVAL"] = str(sample_interval)

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

    for idx, fen in enumerate(fens_chunk, 1):
        proc.stdin.write(f"position fen {fen}\ngo nodes {nodes_per_fen}\n")
        proc.stdin.flush()

        while True:
            line = proc.stdout.readline()
            if "bestmove" in line or not line:
                break
        if idx % 100 == 0 or idx == len(fens_chunk):
            print(f"  [Stockfish Worker {worker_id}] Searched {idx:>5d}/{len(fens_chunk)} FENs ({(idx/len(fens_chunk))*100:.0f}%)", flush=True)

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

    for idx, fen in enumerate(fens_chunk, 1):
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
        if len(batch) >= 1000:
            conn.executemany("INSERT OR REPLACE INTO policies (fen, policy_json) VALUES (?, ?)", batch)
            conn.commit()
            total_evaluated += len(batch)
            batch = []

        if idx % 1000 == 0 or idx == len(fens_chunk):
            print(f"  [Monty Worker {worker_id}] Evaluated {idx:>5d}/{len(fens_chunk)} FENs ({(idx/len(fens_chunk))*100:.0f}%)", flush=True)

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


def init_main_cache_db(main_db_path: str):
    conn = sqlite3.connect(main_db_path)
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("CREATE TABLE IF NOT EXISTS policies (fen TEXT PRIMARY KEY, policy_json TEXT)")
    conn.commit()
    conn.close()


def get_cached_fens(main_db_path: str) -> set:
    if not os.path.exists(main_db_path):
        return set()
    conn = sqlite3.connect(main_db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT fen FROM policies")
    cached = set(row[0] for row in cursor.fetchall())
    conn.close()
    return cached


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


def run_unified_calibration(args):
    os.makedirs(CACHE_DIR, exist_ok=True)
    merged_telemetry_path = os.path.join(CACHE_DIR, f"calib_telemetry_{args.fens}fens.jsonl")
    main_db_path = os.path.join(CACHE_DIR, "monty_policy_cache.db")
    init_main_cache_db(main_db_path)

    print("=" * 85, flush=True)
    print("   UNIFIED STEP 0 CALIBRATION: ENTROPY MATCHING FOR MOVEPICKER & LMR", flush=True)
    print("=" * 85, flush=True)
    print(f"Total Calibration FENs:      {args.fens:,}", flush=True)
    print(f"Nodes per FEN:               {args.nodes:,}", flush=True)
    print(f"Subsample Interval:          1 sample per {args.sample_interval:,} nodes", flush=True)
    print(f"Parallel Worker Threads:     {args.workers}", flush=True)
    print(f"Telemetry File Path:         {merged_telemetry_path}", flush=True)
    print(f"Monty Policy Cache:          {main_db_path}", flush=True)
    print("=" * 85, flush=True)

    # 1. Collect Telemetry
    if os.path.exists(merged_telemetry_path) and os.path.getsize(merged_telemetry_path) > 0 and not args.force_search:
        print(f"\n[1/3] Using existing telemetry file on disk: {merged_telemetry_path}", flush=True)
        print(f"      Size: {os.path.getsize(merged_telemetry_path) / (1024*1024):.1f} MB (Skipping search phase).\n", flush=True)
    else:
        fens = extract_fens(EPD_FILE, count=args.fens, offset=0)
        print(f"Extracted {len(fens):,} benchmark FENs from dataset.\n", flush=True)
        print(f"[1/3] Searching {len(fens):,} FENs with Stockfish Master telemetry in parallel...", flush=True)
        t0 = time.time()
        chunk_size = math.ceil(len(fens) / args.workers)
        chunks = [fens[i : i + chunk_size] for i in range(0, len(fens), chunk_size)]

        worker_tel_paths = [os.path.join(CACHE_DIR, f"sf_calib_tel_w{w_id}.jsonl") for w_id in range(len(chunks))]

        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = [
                executor.submit(run_stockfish_telemetry_worker, w_id, chunk, args.nodes, args.sample_interval, worker_tel_paths[w_id])
                for w_id, chunk in enumerate(chunks)
            ]
            for f in as_completed(futures):
                f.result()

        print("      Merging worker telemetry files into single cache...", flush=True)
        total_samples = 0
        with open(merged_telemetry_path, "w") as out_f:
            for p in worker_tel_paths:
                if os.path.exists(p):
                    with open(p, "r") as in_f:
                        for line in in_f:
                            if line.strip():
                                out_f.write(line)
                                total_samples += 1
                    os.remove(p)

        print(f"      Collected & merged {total_samples:,} node telemetry samples in {time.time() - t0:.1f}s.\n", flush=True)

    # 2. Extract Unique FENs for Monty Querying
    print("[2/3] Extracting unique node FENs and checking Monty policy cache...", flush=True)
    cached_fens = get_cached_fens(main_db_path)
    print(f"      Found {len(cached_fens):,} FENs already cached in {main_db_path}.", flush=True)

    unique_fens_needed = set()
    total_node_samples = 0
    with open(merged_telemetry_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total_node_samples += 1
            try:
                data = json.loads(line)
                fen = data.get("fen")
                if fen and fen not in cached_fens:
                    unique_fens_needed.add(fen)
            except Exception:
                pass

    fens_to_query = list(unique_fens_needed)
    print(f"      Total Node Samples: {total_node_samples:,} | Uncached FENs to Query: {len(fens_to_query):,}", flush=True)

    if fens_to_query:
        print(f"      Querying {len(fens_to_query):,} FENs from Monty across {args.workers} workers...", flush=True)
        t_m0 = time.time()
        m_chunk_size = math.ceil(len(fens_to_query) / args.workers)
        m_chunks = [fens_to_query[i : i + m_chunk_size] for i in range(0, len(fens_to_query), m_chunk_size)]
        worker_db_paths = [os.path.join(CACHE_DIR, f"monty_w{w_id}.db") for w_id in range(len(m_chunks))]

        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = [
                executor.submit(query_monty_worker, w_id, m_chunk, worker_db_paths[w_id])
                for w_id, m_chunk in enumerate(m_chunks)
            ]
            for f in as_completed(futures):
                f.result()

        print("      Merging Monty worker databases into main cache...", flush=True)
        merge_worker_dbs(main_db_path, worker_db_paths)
        print(f"      Monty evaluation completed in {time.time() - t_m0:.1f}s.\n", flush=True)
    else:
        print("      All FEN policies already present in cache. Zero queries needed.\n", flush=True)

    # 3. Stream Telemetry and Compute Baseline Distributions
    print("[3/3] Streaming telemetry, calculating master entropy and solving temperature calibration...", flush=True)

    conn = sqlite3.connect(main_db_path)
    cursor = conn.cursor()

    lmr_entropies = []
    lmr_top1_mass = []
    lmr_top3_mass = []

    mp_entropies = []
    mp_top1_mass = []
    mp_top3_mass = []

    aligned_nodes = []
    valid_samples = 0

    print("      Loading cached policy lookup index...", flush=True)
    cursor.execute("SELECT fen, policy_json FROM policies")
    policy_lookup = {}
    for fen, p_json in cursor.fetchall():
        policy_lookup[fen] = json.loads(p_json)
    conn.close()

    print(f"      Loaded {len(policy_lookup):,} policies into memory lookup.", flush=True)

    with open(merged_telemetry_path, "r") as f:
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
            moves_info = s["moves"]
            m_policy = policy_lookup.get(fen, {})

            num_moves = len(moves_info)
            if num_moves < 3:
                continue

            r_legacy = np.zeros(num_moves, dtype=np.float32)
            z_mp = np.zeros(num_moves, dtype=np.float32)
            raw_monty = np.zeros(num_moves, dtype=np.float32)

            for i, m in enumerate(moves_info):
                stat_score = m.get("stat_score", 0)
                # Legacy LMR reduction in plies
                r_val = (math.log(max(1, depth)) * math.log(max(1, i + 1)) * 500.0 - stat_score * (439.0 / 4096.0)) / 1024.0
                r_legacy[i] = r_val
                # MovePicker score in native 1024 scale (ONE_PLY = 1024)
                z_mp[i] = stat_score / 1024.0
                raw_monty[i] = m_policy.get(m["move"], 0.0)

            # Master LMR distribution
            exp_lmr = np.exp(-r_legacy - np.max(-r_legacy))
            p_lmr = exp_lmr / np.sum(exp_lmr)
            h_lmr = -np.sum(p_lmr * np.log(p_lmr + 1e-12))
            lmr_entropies.append(h_lmr)
            lmr_top1_mass.append(np.max(p_lmr))
            lmr_top3_mass.append(np.sum(np.sort(p_lmr)[::-1][:3]))

            # Master MovePicker distribution (peaky)
            exp_mp = np.exp(z_mp - np.max(z_mp))
            p_mp = exp_mp / np.sum(exp_mp)
            h_mp = -np.sum(p_mp * np.log(p_mp + 1e-12))
            mp_entropies.append(h_mp)
            mp_top1_mass.append(np.max(p_mp))
            mp_top3_mass.append(np.sum(np.sort(p_mp)[::-1][:3]))

            aligned_nodes.append((raw_monty, r_legacy, z_mp))
            valid_samples += 1

    del policy_lookup

    mean_h_lmr = float(np.mean(lmr_entropies))
    mean_top1_lmr = float(np.mean(lmr_top1_mass))
    mean_top3_lmr = float(np.mean(lmr_top3_mass))

    mean_h_mp = float(np.mean(mp_entropies))
    mean_top1_mp = float(np.mean(mp_top1_mass))
    mean_top3_mp = float(np.mean(mp_top3_mass))

    print("\n" + "=" * 85, flush=True)
    print("                     MASTER BASELINE MACRO PROFILES", flush=True)
    print("=" * 85, flush=True)
    print(f"{'Metric':<35} | {'Master LMR (Search Policy)':<22} | {'Master MovePicker (Ordering)':<22}", flush=True)
    print("-" * 85, flush=True)
    print(f"{'Mean Shannon Entropy (nats)':<35} | {mean_h_lmr:<22.4f} | {mean_h_mp:<22.4f}", flush=True)
    print(f"{'Mean Top-1 Move Probability':<35} | {mean_top1_lmr * 100:<21.2f}% | {mean_top1_mp * 100:<21.2f}%", flush=True)
    print(f"{'Mean Top-3 Move Probability':<35} | {mean_top3_lmr * 100:<21.2f}% | {mean_top3_mp * 100:<21.2f}%", flush=True)
    print("=" * 85, flush=True)

    # 4. Fit Entropy Parity across Probability Floor Candidates
    candidate_floors = [0.001, 0.005, 0.010, 0.020, 0.050]
    print("\n--- Entropy Matching across Probability Floors ---", flush=True)
    print(f"{'Floor':<8} | {'T_calib_LMR':<14} | {'H_LMR Parity':<14} | {'T_calib_MP':<14} | {'H_MP Parity':<14}", flush=True)
    print("-" * 85, flush=True)

    floor_calibration_results = {}

    for floor_val in candidate_floors:
        floored_nodes = []
        for raw_monty, r_leg, z_m in aligned_nodes:
            f_probs = np.maximum(raw_monty, floor_val)
            f_norm = f_probs / np.sum(f_probs)
            floored_nodes.append((f_norm, np.log(f_norm + 1e-12)))

        # Match LMR Entropy
        def lmr_ent_diff(T):
            if T <= 0.01 or T > 5.0:
                return 1e9
            ents = []
            for _, log_p in floored_nodes:
                scaled = log_p / T
                exp_s = np.exp(scaled - np.max(scaled))
                p = exp_s / np.sum(exp_s)
                ents.append(-np.sum(p * np.log(p + 1e-12)))
            return (np.mean(ents) - mean_h_lmr) ** 2

        res_lmr = minimize(lmr_ent_diff, x0=[0.85], bounds=[(0.01, 3.0)], method="L-BFGS-B")
        opt_t_lmr = float(res_lmr.x[0])

        # Match MovePicker Entropy
        def mp_ent_diff(T):
            if T <= 0.001 or T > 5.0:
                return 1e9
            ents = []
            for _, log_p in floored_nodes:
                scaled = log_p / T
                exp_s = np.exp(scaled - np.max(scaled))
                p = exp_s / np.sum(exp_s)
                ents.append(-np.sum(p * np.log(p + 1e-12)))
            return (np.mean(ents) - mean_h_mp) ** 2

        res_mp = minimize(mp_ent_diff, x0=[0.12], bounds=[(0.001, 2.0)], method="L-BFGS-B")
        opt_t_mp = float(res_mp.x[0])

        floor_calibration_results[str(floor_val)] = {
            "t_calib_lmr": opt_t_lmr,
            "t_calib_mp": opt_t_mp
        }

        print(f"{floor_val:<8.3f} | {opt_t_lmr:<14.4f} | {mean_h_lmr:<14.4f} | {opt_t_mp:<14.4f} | {mean_h_mp:<14.4f}", flush=True)

    # Independent Floor Selection:
    # LMR: 0.005 or 0.010 floor
    # MP: 0.010 or 0.050 floor
    chosen_floor_lmr = args.floor_lmr
    chosen_floor_mp = args.floor_mp
    
    calib_lmr_val = floor_calibration_results[str(chosen_floor_lmr)]["t_calib_lmr"]
    calib_mp_val = floor_calibration_results[str(chosen_floor_mp)]["t_calib_mp"]

    print("\n" + "=" * 85, flush=True)
    print("                     FINAL CALIBRATION SUMMARY", flush=True)
    print("=" * 85, flush=True)
    print(f"LMR Probability Floor (epsilon_floor_lmr):     {chosen_floor_lmr:.3f}", flush=True)
    print(f"LMR Calibration Temperature (T_calib_LMR):    {calib_lmr_val:.4f} (matches Master LMR entropy {mean_h_lmr:.4f} nats)", flush=True)
    print(f"MP Probability Floor (epsilon_floor_mp):       {chosen_floor_mp:.3f}", flush=True)
    print(f"MP Calibration Temperature (T_calib_MP):      {calib_mp_val:.4f} (matches Master MP entropy {mean_h_mp:.4f} nats)", flush=True)
    print(f"Sample Count:                                 {valid_samples:,} node positions", flush=True)
    print("=" * 85, flush=True)

    config_data = {
        "sample_count": valid_samples,
        "fens_searched": args.fens,
        "master_lmr_entropy": mean_h_lmr,
        "master_mp_entropy": mean_h_mp,
        "chosen_floor_lmr": chosen_floor_lmr,
        "chosen_floor_mp": chosen_floor_mp,
        "t_calib_lmr": calib_lmr_val,
        "t_calib_mp": calib_mp_val,
        "all_floor_calibrations": floor_calibration_results
    }

    with open(OUTPUT_CONFIG, "w") as f:
        json.dump(config_data, f, indent=2)
    print(f"Saved unified calibration config to: {OUTPUT_CONFIG}\n", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified Step 0 Calibration via Entropy Matching.")
    parser.add_argument("--fens", type=int, default=2000, help="Total calibration FENs (default: 2,000)")
    parser.add_argument("--nodes", type=int, default=500_000, help="Search budget per FEN (default: 500,000)")
    parser.add_argument("--sample-interval", type=int, default=20_000, help="Subsample interval (default: 20,000 = 1/20k)")
    parser.add_argument("--workers", type=int, default=6, help="Parallel worker processes (default: 6)")
    parser.add_argument("--floor-lmr", type=float, default=0.010, help="Independent probability floor for LMR (default: 0.010)")
    parser.add_argument("--floor-mp", type=float, default=0.010, help="Independent probability floor for MovePicker (default: 0.010)")
    parser.add_argument("--force-search", action="store_true", help="Force re-running Stockfish search even if telemetry cache exists")

    args = parser.parse_args()
    run_unified_calibration(args)
