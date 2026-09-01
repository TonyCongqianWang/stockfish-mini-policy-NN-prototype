"""
Mini-NN Engine Architecture (Version 4: Dynamic Handcrafted Quiet Terms Weighting + LMR):
1. NodeNetwork: 16 -> 32 -> 32 -> 20
   - 10 dynamic weighting multipliers for handcrafted quiet terms (w_quiet, Scale 256: 256 = 1.0x)
   - 8 position latents for LMR (z_latents, Scale 64)
   - 1 dynamic temperature for MovePicker (tau_mp)
   - 1 dynamic temperature for LMR (tau_lmr)
2. Quiet Move Scoring: Direct linear combination of 10 handcrafted terms (T_0..T_9) with w_quiet.
3. LMRMoveNetwork: (8 raw + 8 latents) = 16 -> 16 -> 1
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

        # Initialize quiet term weights to 1.0 (Master default)
        with torch.no_grad():
            self.fc2.bias.data.zero_()
            self.fc2.bias.data[0:10] = 1.0
            self.fc2.weight.data.normal_(0.0, 0.01)

    def forward(self, u):
        u_q = quantize_ste(u, self.scale)
        w0 = quantize_ste(self.fc0.weight, self.scale)
        b0 = quantize_bias_ste(self.fc0.bias, self.scale * self.scale)
        h0 = self.act0(F.linear(u_q, w0, b0))

        w1 = quantize_ste(self.fc1.weight, self.scale)
        b1 = quantize_bias_ste(self.fc1.bias, self.scale * self.scale)
        h1 = self.act1(F.linear(h0, w1, b1))

        w2 = quantize_ste(self.fc2.weight, self.scale)
        b2 = quantize_bias_ste(self.fc2.bias, self.scale * self.scale)
        out = F.linear(h1, w2, b2)

        # 0..9: 10 dynamic multipliers for handcrafted quiet terms (scale 256, range [-4.0, 4.0])
        w_quiet_raw = torch.clamp(out[:, 0:10], -4.0, 4.0)
        w_quiet_q = torch.clamp(torch.round(w_quiet_raw * 256.0), -1024.0, 1024.0) / 256.0
        w_quiet = w_quiet_raw + (w_quiet_q - w_quiet_raw).detach()

        # 10..17: 8 position latents for LMR (scale 64, range [-127/64, 127/64])
        z_latents = quantize_ste(torch.clamp(out[:, 10:18], -127.0 / 64.0, 127.0 / 64.0), 64.0)

        # 18: log_tau_mp
        log_tau_mp = out[:, 18:19]
        tau_mp = self.tau_mp_base * torch.exp(torch.clamp(log_tau_mp, -1.5, 1.5))

        # 19: log_tau_lmr
        log_tau_lmr = out[:, 19:20]
        tau_lmr = self.tau_lmr_base * torch.exp(torch.clamp(log_tau_lmr, -1.5, 1.5))

        return w_quiet, z_latents, tau_mp, tau_lmr


class LMRMoveNetwork(nn.Module):
    def __init__(self, in_dim=16, hidden_dim=16, scale=64.0):
        super().__init__()
        self.scale = scale
        self.fc0 = nn.Linear(in_dim, hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, 1)
        self.act0 = QuantizedClippedReLU(scale)

    def forward(self, x_raw_lmr, z_latents):
        B, N, _ = x_raw_lmr.shape
        z_exp = z_latents.unsqueeze(1).expand(B, N, -1)
        x_lmr = torch.cat([quantize_ste(x_raw_lmr, self.scale), z_exp], dim=-1)

        w0 = quantize_ste(self.fc0.weight, self.scale)
        b0 = quantize_bias_ste(self.fc0.bias, self.scale * self.scale)
        h0 = self.act0(F.linear(x_lmr, w0, b0))

        w1 = quantize_ste(self.fc1.weight, self.scale)
        b1 = quantize_bias_ste(self.fc1.bias, self.scale * self.scale)
        r_float = F.linear(h0, w1, b1).squeeze(-1)
        r_int_1024 = torch.floor((r_float * 4096.0 + 2.0) / 4.0)
        r_out = r_float + (r_int_1024 / 1024.0 - r_float).detach()
        return r_out


class DualMiniNN(nn.Module):
    MAGIC = 0x4D494E49  # 'MINI'
    VERSION = 4         # Version 4: Dynamic Quiet Terms (Scale 256) + LMR
    WEIGHT_SCALE = 64

    def __init__(self, tau_mp_base=None, tau_lmr_base=None):
        super().__init__()
        self.node_net = NodeNetwork(in_dim=16, hidden_dim=32, out_dim=20, scale=self.WEIGHT_SCALE, tau_mp_base=tau_mp_base, tau_lmr_base=tau_lmr_base)
        self.lmr_net = LMRMoveNetwork(in_dim=16, hidden_dim=16, scale=self.WEIGHT_SCALE)

    def forward(self, u_node, t_quiet, x_lmr):
        w_quiet, z_latents, tau_mp, tau_lmr = self.node_net(u_node)
        
        # Linear combination of 10 handcrafted terms
        # t_quiet: (B, M, 10), w_quiet: (B, 10)
        w_exp = w_quiet.unsqueeze(1)
        score_float = (t_quiet * w_exp).sum(dim=-1)
        
        # Fixed point integer simulation: (sum * 256 + 128) >> 8
        score_int = torch.floor(((t_quiet * torch.round(w_exp * 256.0)).sum(dim=-1) + 128.0) / 256.0)
        quiet_scores = score_float + (score_int - score_float).detach()
        z_quiet = quiet_scores / 32768.0

        lmr_reductions = self.lmr_net(x_lmr, z_latents)
        return w_quiet, z_latents, tau_mp, tau_lmr, quiet_scores, lmr_reductions

    def export_quantized_binary(self, filepath: str):
        with open(filepath, "wb") as f:
            header = [
                self.MAGIC,
                self.VERSION,
                16,
                32,
                20,
                10,
                16,
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

            # 1. Node Network (16 -> 32 -> 32 -> 20)
            write_layer(self.node_net.fc0)
            write_layer(self.node_net.fc1)
            write_layer(self.node_net.fc2)

            # 2. LMR Network (16 -> 16 -> 1)
            write_layer(self.lmr_net.fc0)
            write_layer(self.lmr_net.fc1)

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

            # 1. Node Network
            read_layer(self.node_net.fc0, 32, 16)
            read_layer(self.node_net.fc1, 32, 32)
            read_layer(self.node_net.fc2, 20, 32)

            # 2. LMR Network
            read_layer(self.lmr_net.fc0, 16, 16)
            read_layer(self.lmr_net.fc1, 1, 16)
