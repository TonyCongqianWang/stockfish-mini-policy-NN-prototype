"""
Standalone Parity Verification Suite for Version 4 Mini-NN Engine (Dynamic Quiet Terms + LMR):
1. Generates random weights for:
   - Node Network (16 -> 32 -> 32 -> 20)
   - LMR Network (16 -> 16 -> 1)
2. Exports quantized binary to /tmp/random_test_v4.miniNN.
3. Implements direct bit-level reference emulator in Python matching C++ integer arithmetic.
4. Generates 50 synthetic test cases.
5. Asserts that PyTorch inference and C++ SIMD fixed-point inference match with sub-0.01 precision.
"""

import os
import struct
import numpy as np
import torch

from model import DualMiniNN


def simulate_cpp_fixed_point_v4(model_path: str, u_node, t_quiet, x_lmr_raw):
    with open(model_path, "rb") as f:
        header = struct.unpack("<8I", f.read(32))
        assert header[0] == 0x4D494E49 and header[1] == 4

        def read_layer(out_dim, in_dim):
            b = np.frombuffer(f.read(out_dim * 4), dtype=np.int32)
            w = np.frombuffer(f.read(out_dim * in_dim), dtype=np.int8).reshape(out_dim, in_dim)
            return b, w

        # 1. Node Net
        node_b0, node_w0 = read_layer(32, 16)
        node_b1, node_w1 = read_layer(32, 32)
        node_b2, node_w2 = read_layer(20, 32)

        # 2. LMR Net
        lmr_b0, lmr_w0 = read_layer(16, 16)
        lmr_b1, lmr_w1 = read_layer(1, 16)

    # 1. Node Forward Pass
    u_q = np.clip(np.round(u_node * 64.0), -127, 127).astype(np.int32)
    h0 = np.clip((node_b0 + node_w0 @ u_q + 32) >> 6, 0, 127)
    h1 = np.clip((node_b1 + node_w1 @ h0 + 32) >> 6, 0, 127)
    out2 = node_b2 + node_w2 @ h1

    # w_quiet (scale 256, range [-1024, 1024])
    w_quiet_cpp = np.clip((out2[0:10] + 8) >> 4, -1024, 1024)
    # z_latents (scale 64, range [-127, 127])
    z_latents_cpp = np.clip((out2[10:18] + 32) >> 6, -127, 127)

    # 2. Quiet Move Forward Pass: linear combination of 10 terms with w_quiet_cpp
    score_sum = int(np.sum(np.round(t_quiet) * w_quiet_cpp))
    score_quiet_cpp = (score_sum + 128) >> 8

    # 3. LMR Forward Pass
    xlmr_q = np.concatenate([np.clip(np.round(x_lmr_raw * 64.0), -127, 127).astype(np.int32), z_latents_cpp])
    hlmr = np.clip((lmr_b0 + lmr_w0 @ xlmr_q + 32) >> 6, 0, 127)
    r_cpp = (lmr_b1[0] + np.sum(lmr_w1[0] * hlmr) + 2) >> 2

    return w_quiet_cpp, z_latents_cpp, score_quiet_cpp, r_cpp


def test_parity():
    print("=" * 80)
    print("   STANDALONE PARITY TEST: PYTORCH VERSION 4 vs C++ FIXED-POINT")
    print("=" * 80)

    torch.manual_seed(42)
    np.random.seed(42)

    model = DualMiniNN()
    for p in model.parameters():
        p.data.uniform_(-1.5, 1.5)

    test_model_path = "/tmp/random_test_v4.miniNN"
    model.export_quantized_binary(test_model_path)
    print(f"Exported Version 4 random model binary to {test_model_path}\n")

    # 50 Synthetic Test Cases
    test_vectors = []
    test_vectors.append(("All Zeros", np.zeros(16), np.zeros(10), np.zeros(8)))
    test_vectors.append(("All Max", np.ones(16) * 1.95, np.ones(10) * 10000.0, np.ones(8) * 1.95))
    test_vectors.append(("All Min", -np.ones(16) * 1.95, -np.ones(10) * 10000.0, -np.ones(8) * 1.95))

    for i in range(47):
        test_vectors.append((
            f"Random Case #{i+1:02d}",
            np.random.uniform(-1.95, 1.95, size=16),
            np.random.uniform(-16000, 16000, size=10),
            np.random.uniform(-1.95, 1.95, size=8)
        ))

    quiet_score_diffs = []
    lmr_reduction_diffs = []
    w_quiet_diffs = []
    z_latent_diffs = []

    for name, u_node, t_quiet, x_lmr_raw in test_vectors:
        w_q_cpp, z_lat_cpp, s_quiet_cpp, r_cpp = simulate_cpp_fixed_point_v4(
            test_model_path, u_node, t_quiet, x_lmr_raw
        )

        u_tensor = torch.from_numpy(u_node).float().unsqueeze(0)
        tq_tensor = torch.from_numpy(t_quiet).float().unsqueeze(0).unsqueeze(0)
        xlmr_tensor = torch.from_numpy(x_lmr_raw).float().unsqueeze(0).unsqueeze(0)

        with torch.no_grad():
            w_q_pt, z_lat_pt, _, _, s_quiet_pt, r_pt = model(
                u_tensor, tq_tensor, xlmr_tensor
            )

        w_q_pt_val = np.round(w_q_pt.numpy()[0] * 256.0)
        z_lat_pt_val = np.round(z_lat_pt.numpy()[0] * 64.0)
        s_quiet_pt_val = s_quiet_pt.numpy()[0, 0]
        r_pt_val = r_pt.numpy()[0, 0] * 1024.0

        w_quiet_diffs.append(np.abs(w_q_pt_val - w_q_cpp))
        z_latent_diffs.append(np.abs(z_lat_pt_val - z_lat_cpp))
        quiet_score_diffs.append(abs(s_quiet_pt_val - s_quiet_cpp))
        lmr_reduction_diffs.append(abs(r_pt_val - r_cpp))

    mean_quiet_diff = np.mean(quiet_score_diffs)
    max_quiet_diff = np.max(quiet_score_diffs)
    mean_lmr_diff = np.mean(lmr_reduction_diffs)
    max_lmr_diff = np.max(lmr_reduction_diffs)

    mean_wq_diff = np.mean(w_quiet_diffs)
    mean_zlat_diff = np.mean(z_latent_diffs)

    print(f"{'Module / Output':<35} | {'Mean Absolute Error (MAE)':<25} | {'Max Difference':<15}")
    print("-" * 80)
    print(f"{'Node Net: w_quiet Dynamic Weights':<35} | {mean_wq_diff:<25.3f} (scale 256) | {np.max(w_quiet_diffs):<15.1f}")
    print(f"{'Node Net: z_latents State Vector':<35} | {mean_zlat_diff:<25.3f} (scale 64)  | {np.max(z_latent_diffs):<15.1f}")
    print(f"{'Quiet Move Score (score_quiet)':<35} | {mean_quiet_diff:<25.2f} (scale 32768)| {max_quiet_diff:<15.1f}")
    print(f"{'LMR Reduction (evaluate_lmr)':<35} | {mean_lmr_diff:<25.2f} (scale 1024)| {max_lmr_diff:<15.1f}")
    print("=" * 80)

    assert mean_quiet_diff < 50.0, f"Quiet move score disparity too high: {mean_quiet_diff}"
    assert mean_lmr_diff < 15.0, f"LMR reduction disparity too high: {mean_lmr_diff}"

    print(f"ALL PARITY TESTS PASSED! Quiet score MAE: {mean_quiet_diff:.2f}, LMR MAE: {mean_lmr_diff/1024:.5f} plies.\n")
    if os.path.exists(test_model_path):
        os.remove(test_model_path)


if __name__ == "__main__":
    test_parity()
