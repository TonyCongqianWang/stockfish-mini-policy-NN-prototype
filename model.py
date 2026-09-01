"""
Mini-NN Engine Architecture (Version 5: Dynamic Handcrafted Quiet Terms + LMR Residual Terms):
1. NodeNetwork: 16 -> 32 -> 32 -> 18
   - 10 dynamic weighting multipliers for handcrafted quiet terms (w_quiet, Scale 256: 256 = 1.0x)
   - 6 dynamic residual weights for LMR formula terms (w_lmr, Scale 64: 64 = 1.0 ply)
   - 1 dynamic temperature for MovePicker (tau_mp)
   - 1 dynamic temperature for LMR (tau_lmr)
2. Quiet Move Scoring: Direct linear combination of 10 handcrafted terms (T_0..T_9) with w_quiet in movepick.cpp.
3. LMR Residual Reduction: Direct linear combination of 6 LMR terms with w_lmr in search.cpp.
4. Symmetric Quantization with Straight-Through Estimator (STE).
"""

import json
import math
import os
import struct
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict

from paths import CALIB_CONFIG_PATH


def get_default_calib_temperatures() -> Tuple[float, float]:
    t_mp, t_lmr = 0.11539, 0.86580
    if os.path.exists(CALIB_CONFIG_PATH):
        try:
            with open(CALIB_CONFIG_PATH, "r") as f:
                cfg = json.load(f)
            t_mp = float(cfg.get("t_calib_mp", 0.11539))
            t_lmr = float(cfg.get("t_calib_lmr", 0.86580))
        except Exception:
            pass
    return t_mp, t_lmr


class SymmetricQuantizeSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale=64.0):
        q = torch.clamp(torch.round(x * scale), -127.0, 127.0)
        return q / scale

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


def quantize_ste(x, scale=64.0):
    return SymmetricQuantizeSTE.apply(x, scale)


def quantize_bias_ste(b, scale=4096.0):
    return torch.round(b * scale) / scale


class QuantizedClippedReLU(nn.Module):
    def __init__(self, scale=64.0):
        super().__init__()
        self.scale = scale
        self.max_val = 127.0 / scale

    def forward(self, x):
        clamped = torch.clamp(x, 0.0, self.max_val)
        return quantize_ste(clamped, self.scale)


class NodeNetwork(nn.Module):
    def __init__(self, in_dim=16, hidden_dim=32, out_dim=20, scale=64.0, tau_mp_base=None, tau_lmr_base=None):
        super().__init__()
        self.scale = scale
        if tau_mp_base is None or tau_lmr_base is None:
            def_mp, def_lmr = get_default_calib_temperatures()
            self.tau_mp_base = def_mp if tau_mp_base is None else tau_mp_base
            self.tau_lmr_base = def_lmr if tau_lmr_base is None else tau_lmr_base
        else:
            self.tau_mp_base = tau_mp_base
            self.tau_lmr_base = tau_lmr_base

        self.fc0 = nn.Linear(in_dim, hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)

        self.act0 = QuantizedClippedReLU(scale)
        self.act1 = QuantizedClippedReLU(scale)

        # Pure residual initialization: 0.0 everywhere (Master baseline is exact 0 residual)
        with torch.no_grad():
            self.fc2.bias.data.zero_()
            self.fc2.weight.data.normal_(0.0, 0.01)

    def forward(self, u, detach_lmr_backbone: bool = False):
        u_q = quantize_ste(u, self.scale)
        u_int = torch.clamp(torch.round(u * self.scale), -127.0, 127.0)

        w0_q = quantize_ste(self.fc0.weight, self.scale)
        w0_int = torch.clamp(torch.round(self.fc0.weight * self.scale), -127.0, 127.0)
        b0_int = torch.round(self.fc0.bias * (self.scale * self.scale))
        b0_q = b0_int / (self.scale * self.scale)

        h0_float = self.act0(F.linear(u_q, w0_q, b0_q))
        h0_int = torch.clamp(torch.floor((F.linear(u_int, w0_int, b0_int) + 32.0) / 64.0), 0.0, 127.0)
        h0 = h0_float + (h0_int / 64.0 - h0_float).detach()

        w1_q = quantize_ste(self.fc1.weight, self.scale)
        w1_int = torch.clamp(torch.round(self.fc1.weight * self.scale), -127.0, 127.0)
        b1_int = torch.round(self.fc1.bias * (self.scale * self.scale))
        b1_q = b1_int / (self.scale * self.scale)

        h1_float = self.act1(F.linear(h0, w1_q, b1_q))
        h1_int = torch.clamp(torch.floor((F.linear(h0_int, w1_int, b1_int) + 32.0) / 64.0), 0.0, 127.0)
        h1 = h1_float + (h1_int / 64.0 - h1_float).detach()

        w2_q = quantize_ste(self.fc2.weight, self.scale)
        w2_int = torch.clamp(torch.round(self.fc2.weight * self.scale), -127.0, 127.0)
        b2_int = torch.round(self.fc2.bias * (self.scale * self.scale))
        b2_q = b2_int / (self.scale * self.scale)

        if detach_lmr_backbone:
            # MP head receives full gradient through shared representation h1
            mp_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 18]
            out_mp_float = F.linear(h1, w2_q[mp_indices], b2_q[mp_indices])
            out_mp_int = F.linear(h1_int, w2_int[mp_indices], b2_int[mp_indices])

            # LMR head trains on detached representation h1.detach() (zero gradient to fc0/fc1)
            lmr_indices = [10, 11, 12, 13, 14, 15, 16, 17, 19]
            h1_det = h1.detach()
            h1_int_det = h1_int.detach()
            out_lmr_float = F.linear(h1_det, w2_q[lmr_indices], b2_q[lmr_indices])
            out_lmr_int = F.linear(h1_int_det, w2_int[lmr_indices], b2_int[lmr_indices])

            out_float = torch.cat([out_mp_float[:, 0:10], out_lmr_float[:, 0:8], out_mp_float[:, 10:11], out_lmr_float[:, 8:9]], dim=-1)
            out_int = torch.cat([out_mp_int[:, 0:10], out_lmr_int[:, 0:8], out_mp_int[:, 10:11], out_lmr_int[:, 8:9]], dim=-1)
        else:
            out_float = F.linear(h1, w2_q, b2_q)
            out_int = F.linear(h1_int, w2_int, b2_int)

        # 0..9: 10 dynamic residual weights for handcrafted quiet terms (scale 256, range [-512, 512])
        w_mp_raw = torch.clamp(out_float[:, 0:10], -2.0, 2.0)
        w_mp_q = torch.clamp(torch.floor((out_int[:, 0:10] + 8.0) / 16.0), -512.0, 512.0) / 256.0
        w_mp = w_mp_raw + (w_mp_q - w_mp_raw).detach()

        # 10..17: 8 dynamic residual weights for LMR terms (scale 64, range [-128, 128])
        w_lmr_raw = torch.clamp(out_float[:, 10:18], -2.0, 2.0)
        w_lmr_q = torch.clamp(torch.floor((out_int[:, 10:18] + 32.0) / 64.0), -128.0, 128.0) / 64.0
        w_lmr = w_lmr_raw + (w_lmr_q - w_lmr_raw).detach()

        # 18: log_tau_mp
        log_tau_mp = out_float[:, 18:19]
        tau_mp = self.tau_mp_base * torch.exp(torch.clamp(log_tau_mp, -1.5, 1.5))

        # 19: log_tau_lmr
        log_tau_lmr = out_float[:, 19:20]
        tau_lmr = self.tau_lmr_base * torch.exp(torch.clamp(log_tau_lmr, -1.5, 1.5))

        return w_mp, w_lmr, tau_mp, tau_lmr


class DualMiniNN(nn.Module):
    MAGIC = 0x4D494E49  # 'MINI'
    VERSION = 5         # Version 5: Pure Residual MovePicker (Scale 256) + Residual LMR (Scale 64)
    WEIGHT_SCALE = 64

    def __init__(self, tau_mp_base=None, tau_lmr_base=None):
        super().__init__()
        self.node_net = NodeNetwork(in_dim=16, hidden_dim=32, out_dim=20, scale=self.WEIGHT_SCALE, tau_mp_base=tau_mp_base, tau_lmr_base=tau_lmr_base)

    def forward(self, u_node, t_quiet, t_lmr, detach_lmr_backbone: bool = False):
        w_mp, w_lmr, tau_mp, tau_lmr = self.node_net(u_node, detach_lmr_backbone=detach_lmr_backbone)
        
        # 1. Residual combination of 10 handcrafted quiet terms: Base Master Score + delta_score (Scale 256)
        base_score = t_quiet.sum(dim=-1)
        w_exp_mp = w_mp.unsqueeze(1)
        delta_score_float = (t_quiet * w_exp_mp).sum(dim=-1)
        delta_score_int = torch.floor(((t_quiet * torch.round(w_exp_mp * 256.0)).sum(dim=-1) + 128.0) / 256.0)
        quiet_scores = base_score + delta_score_float + (delta_score_int - delta_score_float).detach()

        # 2. Residual combination of 8 LMR terms (Scale to exact plies: 1024 units = 1.0 ply, 64 units = 1/16 ply)
        w_exp_lmr = w_lmr.unsqueeze(1)
        delta_r_float = (t_lmr * w_exp_lmr).sum(dim=-1) / 16.0
        delta_r_int = (torch.floor(((t_lmr * torch.round(w_exp_lmr * 64.0)).sum(dim=-1) + 32.0) / 64.0)) / 16.0
        delta_r_nn = delta_r_float + (delta_r_int - delta_r_float).detach()

        return w_mp, w_lmr, tau_mp, tau_lmr, quiet_scores, delta_r_nn

    def export_quantized_binary(self, filepath: str):
        with open(filepath, "wb") as f:
            header = [
                self.MAGIC,
                self.VERSION,
                16,
                32,
                20,
                10,
                8,
                self.WEIGHT_SCALE
            ]
            f.write(struct.pack("<8I", *header))

            def write_layer(layer):
                w = layer.weight.detach().cpu().numpy()
                b = layer.bias.detach().cpu().numpy()
                b_q = np.round(b * (self.WEIGHT_SCALE * self.WEIGHT_SCALE)).astype(np.int32)
                f.write(b_q.tobytes())
                w_q = np.clip(np.round(w * self.WEIGHT_SCALE), -127, 127).astype(np.int8)
                f.write(w_q.tobytes())

            # Node Network (16 -> 32 -> 32 -> 20)
            write_layer(self.node_net.fc0)
            write_layer(self.node_net.fc1)
            write_layer(self.node_net.fc2)

    def load_quantized_binary(self, filepath: str):
        with open(filepath, "rb") as f:
            header_bytes = f.read(32)
            header = struct.unpack("<8I", header_bytes)
            assert header[0] == self.MAGIC and header[1] == self.VERSION

            def read_layer(layer, out_dim, in_dim):
                b_bytes = f.read(out_dim * 4)
                b_q = np.frombuffer(b_bytes, dtype=np.int32)
                layer.bias.data.copy_(torch.from_numpy(b_q.astype(np.float32) / (self.WEIGHT_SCALE * self.WEIGHT_SCALE)))

                w_bytes = f.read(out_dim * in_dim)
                w_q = np.frombuffer(w_bytes, dtype=np.int8).reshape(out_dim, in_dim)
                layer.weight.data.copy_(torch.from_numpy(w_q.astype(np.float32) / self.WEIGHT_SCALE))

            # Node Network
            read_layer(self.node_net.fc0, 32, 16)
            read_layer(self.node_net.fc1, 32, 32)
            read_layer(self.node_net.fc2, 20, 32)
