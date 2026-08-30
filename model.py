"""
Mini-NN Engine Architecture (Version 3 with Full Activation QAT):
1. NodeNetwork: 16 -> 32 -> 32 -> 26
   - 16 dynamic meta-weights for quiet moves (w_quiet)
   - 8 position latents for captures & LMR (z_latents)
   - 1 dynamic temperature for MovePicker (tau_mp, base 0.6830)
   - 1 dynamic temperature for LMR (tau_lmr, base 0.1232)
2. QuietMoveNetwork: 12 -> 16 -> 1 (combined via dynamic inner product with w_quiet)
3. CaptureMoveNetwork: (4 raw + 8 latents) = 12 -> 16 -> 1
4. LMRMoveNetwork: (8 raw + 8 latents) = 16 -> 16 -> 1
5. Symmetric Quantization with Straight-Through Estimator (STE) across weights, biases, and activations.
"""

import math
import struct
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict


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
    def __init__(self, in_dim=16, hidden_dim=32, out_dim=26, scale=64.0):
        super().__init__()
        self.scale = scale
        self.fc0 = nn.Linear(in_dim, hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)

        self.act0 = QuantizedClippedReLU(scale)
        self.act1 = QuantizedClippedReLU(scale)

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

        # 0..15: 16 dynamic meta-weights for quiet moves (scale 127, range [-1.0, 1.0])
        w_quiet = quantize_ste(torch.clamp(out[:, 0:16], -1.0, 1.0), 127.0)

        # 16..23: 8 position latents for captures & LMR (scale 64, range [-127/64, 127/64])
        z_latents = quantize_ste(torch.clamp(out[:, 16:24], -127.0 / 64.0, 127.0 / 64.0), 64.0)

        # 24: log_tau_mp (Base calibration ~0.1154)
        log_tau_mp = out[:, 24:25]
        tau_mp = 0.1154 * torch.exp(torch.clamp(log_tau_mp, -1.5, 1.5))

        # 25: log_tau_lmr (Base calibration ~0.8658)
        log_tau_lmr = out[:, 25:26]
        tau_lmr = 0.8658 * torch.exp(torch.clamp(log_tau_lmr, -1.5, 1.5))

        return w_quiet, z_latents, tau_mp, tau_lmr


class QuietMoveNetwork(nn.Module):
    def __init__(self, in_dim=12, hidden_dim=16, scale=64.0):
        super().__init__()
        self.scale = scale
        self.fc0 = nn.Linear(in_dim, hidden_dim)
        self.act0 = QuantizedClippedReLU(scale)

    def forward(self, x_quiet, w_quiet):
        xq_q = quantize_ste(x_quiet, self.scale)
        w0 = quantize_ste(self.fc0.weight, self.scale)
        b0 = quantize_bias_ste(self.fc0.bias, self.scale * self.scale)
        h = self.act0(F.linear(xq_q, w0, b0))

        # Dynamic inner product with w_quiet
        w_exp = w_quiet.unsqueeze(1)
        z_quiet = torch.clamp((h * w_exp).sum(dim=-1), -1.0, 1.0)
        quiet_scores = z_quiet * 32768.0
        return z_quiet, quiet_scores


class CaptureMoveNetwork(nn.Module):
    def __init__(self, in_dim=12, hidden_dim=16, scale=64.0):
        super().__init__()
        self.scale = scale
        self.fc0 = nn.Linear(in_dim, hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, 1)
        self.act0 = QuantizedClippedReLU(scale)

    def forward(self, x_raw_cap, z_latents):
        B, N, _ = x_raw_cap.shape
        z_exp = z_latents.unsqueeze(1).expand(B, N, -1)
        x_cap = torch.cat([quantize_ste(x_raw_cap, self.scale), z_exp], dim=-1)

        w0 = quantize_ste(self.fc0.weight, self.scale)
        b0 = quantize_bias_ste(self.fc0.bias, self.scale * self.scale)
        h0 = self.act0(F.linear(x_cap, w0, b0))

        w1 = quantize_ste(self.fc1.weight, self.scale)
        b1 = quantize_bias_ste(self.fc1.bias, self.scale * self.scale)
        score_out = F.linear(h0, w1, b1).squeeze(-1)
        z_cap = torch.clamp(score_out, -1.0, 1.0)
        cap_scores = z_cap * 32768.0
        return z_cap, cap_scores


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
        r_out = F.linear(h0, w1, b1).squeeze(-1)
        return r_out


class DualMiniNN(nn.Module):
    MAGIC = 0x4D494E49  # 'MINI'
    VERSION = 3         # Version 3: Full Architecture (score_quiet, score_capture, evaluate_lmr)
    WEIGHT_SCALE = 64

    def __init__(self):
        super().__init__()
        self.node_net = NodeNetwork(in_dim=16, hidden_dim=32, out_dim=26, scale=self.WEIGHT_SCALE)
        self.quiet_net = QuietMoveNetwork(in_dim=12, hidden_dim=16, scale=self.WEIGHT_SCALE)
        self.cap_net = CaptureMoveNetwork(in_dim=12, hidden_dim=16, scale=self.WEIGHT_SCALE)
        self.lmr_net = LMRMoveNetwork(in_dim=16, hidden_dim=16, scale=self.WEIGHT_SCALE)

    def forward(self, u_node, x_quiet, x_cap, x_lmr):
        w_quiet, z_latents, tau_mp, tau_lmr = self.node_net(u_node)
        z_quiet, quiet_scores = self.quiet_net(x_quiet, w_quiet)
        z_cap, cap_scores = self.cap_net(x_cap, z_latents)
        lmr_reductions = self.lmr_net(x_lmr, z_latents)
        return w_quiet, z_latents, tau_mp, tau_lmr, quiet_scores, cap_scores, lmr_reductions

    def export_quantized_binary(self, filepath: str):
        with open(filepath, "wb") as f:
            header = [
                self.MAGIC,
                self.VERSION,
                16,
                32,
                26,
                16,
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

            # 1. Node Network (16 -> 32 -> 32 -> 26)
            write_layer(self.node_net.fc0)
            write_layer(self.node_net.fc1)
            write_layer(self.node_net.fc2)

            # 2. Quiet Network (12 -> 16)
            write_layer(self.quiet_net.fc0)

            # 3. Capture Network (12 -> 16 -> 1)
            write_layer(self.cap_net.fc0)
            write_layer(self.cap_net.fc1)

            # 4. LMR Network (16 -> 16 -> 1)
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
            read_layer(self.node_net.fc2, 26, 32)

            # 2. Quiet Network
            read_layer(self.quiet_net.fc0, 16, 12)

            # 3. Capture Network
            read_layer(self.cap_net.fc0, 16, 12)
            read_layer(self.cap_net.fc1, 1, 16)

            # 4. LMR Network
            read_layer(self.lmr_net.fc0, 16, 16)
            read_layer(self.lmr_net.fc1, 1, 16)
