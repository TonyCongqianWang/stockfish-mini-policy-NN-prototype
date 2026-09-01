"""
Standalone Parity Verification Suite for Version 5 Mini-NN Engine:
- Node Network: 16 -> 32 -> 32 -> 20
- MovePicker Quiet Move Scoring: base_score + ((sum(T_k * delta_w_k) + 128) >> 8)
- LMR Residuals: 8 coefficient tweaks
"""

import os
import struct
import numpy as np
import torch

from model import DualMiniNN


def simulate_cpp_fixed_point_v5(model_path: str, u_node, t_quiet, t_lmr):
    with open(model_path, "rb") as f:
        header = struct.unpack("<8I", f.read(32))
        assert header[0] == 0x4D494E49 and header[1] == 5

        def read_layer(out_dim, in_dim):
            b = np.frombuffer(f.read(out_dim * 4), dtype=np.int32)
            w = np.frombuffer(f.read(out_dim * in_dim), dtype=np.int8).reshape(out_dim, in_dim)
            return b, w

        # Node Net: 16 -> 32 -> 32 -> 20
        node_b0, node_w0 = read_layer(32, 16)
        node_b1, node_w1 = read_layer(32, 32)
        node_b2, node_w2 = read_layer(20, 32)

    # 1. Node Forward Pass
    u_q = np.clip(np.round(u_node * 64.0), -127, 127).astype(np.int32)
    h0 = np.clip((node_b0 + node_w0 @ u_q + 32) >> 6, 0, 127)
    h1 = np.clip((node_b1 + node_w1 @ h0 + 32) >> 6, 0, 127)
    out2 = node_b2 + node_w2 @ h1

    # w_mp (scale 256, range [-512, 512])
    w_mp_cpp = np.clip((out2[0:10] + 8) >> 4, -512, 512)
    # w_lmr (scale 64, range [-128, 128])
    w_lmr_cpp = np.clip((out2[10:18] + 32) >> 6, -128, 128)

    # 2. Quiet Move Forward Pass: base_score + delta_score
    base_score = int(np.sum(np.round(t_quiet)))
    delta_sum = int(np.sum(np.round(t_quiet) * w_mp_cpp))
    score_quiet_cpp = base_score + ((delta_sum + 128) >> 8)

    # 3. LMR Forward Pass: 8 residual terms
    delta_r_int = int(np.floor((np.sum(t_lmr * w_lmr_cpp) + 32.0) / 64.0))
    delta_r_cpp = float(delta_r_int)

    return w_mp_cpp, w_lmr_cpp, score_quiet_cpp, delta_r_cpp


def test_parity():
    print("=" * 80)
    print("   STANDALONE PARITY TEST: PYTORCH VERSION 5 vs C++ FIXED-POINT")
    print("=" * 80)

    torch.manual_seed(42)
    np.random.seed(42)

    model = DualMiniNN()
    for p in model.parameters():
        p.data.uniform_(-1.5, 1.5)

    test_model_path = "/tmp/random_test_v5.miniNN"
    model.export_quantized_binary(test_model_path)
    model.load_quantized_binary(test_model_path)
    print(f"Exported & Loaded Version 5 quantized model binary to {test_model_path}\n")

    # 50 Synthetic Test Cases
    test_vectors = []
    test_vectors.append(("All Zeros", np.zeros(16), np.zeros(10), np.zeros(8)))
    test_vectors.append(("All Max", np.ones(16) * 1.95, np.ones(10) * 10000.0, np.ones(8) * 1.0))
    test_vectors.append(("All Min", -np.ones(16) * 1.95, -np.ones(10) * 10000.0, -np.ones(8) * 1.0))

    for i in range(47):
        test_vectors.append((
            f"Random Case #{i+1:02d}",
            np.random.uniform(-1.95, 1.95, size=16),
            np.random.randint(-16000, 16000, size=10).astype(np.float32),
            np.random.uniform(-1.0, 1.0, size=8)
        ))

    quiet_score_diffs = []
    lmr_reduction_diffs = []
    w_mp_diffs = []
    w_lmr_diffs = []

    for name, u_node, t_quiet, t_lmr in test_vectors:
        w_mp_cpp, w_lmr_cpp, s_quiet_cpp, delta_r_cpp = simulate_cpp_fixed_point_v5(
            test_model_path, u_node, t_quiet, t_lmr
        )

        with torch.no_grad():
            u_t = torch.tensor(u_node, dtype=torch.float32).unsqueeze(0)
            tq_t = torch.tensor(t_quiet, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            tlmr_t = torch.tensor(t_lmr, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

            w_mp_py, w_lmr_py, tau_mp_py, tau_lmr_py, s_quiet_py, delta_r_py = model(u_t, tq_t, tlmr_t)

        w_mp_py_int = np.round(w_mp_py.squeeze(0).numpy() * 256.0)
        w_lmr_py_int = np.round(w_lmr_py.squeeze(0).numpy() * 64.0)
        s_quiet_py_val = s_quiet_py.squeeze(0).squeeze(0).item()
        delta_r_py_val = delta_r_py.squeeze(0).squeeze(0).item()

        w_mp_diffs.append(np.max(np.abs(w_mp_py_int - w_mp_cpp)))
        w_lmr_diffs.append(np.max(np.abs(w_lmr_py_int - w_lmr_cpp)))
        quiet_score_diffs.append(abs(s_quiet_py_val - s_quiet_cpp))
        lmr_reduction_diffs.append(abs(delta_r_py_val - delta_r_cpp))

    max_wmp_diff = max(w_mp_diffs)
    max_wlmr_diff = max(w_lmr_diffs)
    max_quiet_diff = max(quiet_score_diffs)
    max_lmr_diff = max(lmr_reduction_diffs)

    print(f"Max difference in w_mp residuals (fixed-point steps):     {max_wmp_diff:.4f}")
    print(f"Max difference in w_lmr residuals (fixed-point steps):    {max_wlmr_diff:.4f}")
    print(f"Max difference in MovePicker Quiet Score:                 {max_quiet_diff:.4f}")
    print(f"Max difference in LMR Delta Reduction:                    {max_lmr_diff:.6f}\n")

    assert max_wmp_diff == 0, f"w_mp discrepancy detected: {max_wmp_diff}"
    assert max_wlmr_diff == 0, f"w_lmr discrepancy detected: {max_wlmr_diff}"
    assert max_quiet_diff == 0, f"MovePicker score discrepancy detected: {max_quiet_diff}"
    assert max_lmr_diff < 1e-4, f"LMR Delta reduction discrepancy detected: {max_lmr_diff}"

    print(" ALL 50 TESTS PASSED: 100% Exact Bit-Level Parity between PyTorch Version 5 and C++ Fixed Point!\n")


if __name__ == "__main__":
    test_parity()
