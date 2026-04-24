# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import copy
import os
import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional

from isaaclab_rl.rsl_rl import exporter

def export_visual_policy_as_onnx(
    policy: object, path: str, normalizer: object | None = None, filename="policy.onnx", verbose=False
):
    """Export policy into a Torch ONNX file.

    Args:
        policy: The policy torch module.
        normalizer: The empirical normalizer module. If None, Identity is used.
        path: The path to the saving directory.
        filename: The name of exported ONNX file. Defaults to "policy.onnx".
        verbose: Whether to print the model summary. Defaults to False.
    """
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    policy_exporter = _OnnxVisualPolicyExporter(policy, normalizer, verbose)
    policy_exporter.export(path, filename)


# =========================================================
# user-facing helper
# =========================================================
def export_monolithic_policy_as_onnx(
    policy: nn.Module,
    path: str,
    filename: str = "policy.onnx",
    normalizer: Optional[nn.Module] = None,
    opset_version: int = 17,
    verbose: bool = False,
) -> str:
    os.makedirs(path, exist_ok=True)

    exporter = OnnxMonolithicPolicyExporter(
        policy=policy,
        normalizer=normalizer,
        verbose=verbose,
    )
    out_path = str(Path(path) / filename)
    return exporter.export(out_path=out_path, opset_version=opset_version)

# ====================exporter classes ====================

class _OnnxVisualPolicyExporter(nn.Module):
    """Exporter for VisualActorCritic / VisualActorCriticRecurrent -> ONNX (actor inference only)."""

    def __init__(self, policy, normalizer=None, verbose=False):
        super().__init__()
        self.verbose = verbose

        # ----- recurrent? -----
        self.is_recurrent = bool(getattr(policy, "is_recurrent", False))

        # ----- normalizer -----
        self.normalizer = copy.deepcopy(normalizer) if normalizer else nn.Identity()

        # ----- copy actor (note: for recurrent ActorCritic, actor eats rnn_hidden_dim) -----
        # VisualActorCritic: policy.actor is MLP
        # VisualActorCriticRecurrent: policy.actor is still MLP, but its input is rnn_hidden_dim
        self.actor = copy.deepcopy(policy.actor)

        # ----- visual config -----
        self.use_visual = bool(getattr(policy, "use_visual", False))
        self.visual_dim = int(getattr(policy, "visual_dim", 0))
        self.visual_latent_size = int(getattr(policy, "visual_latent_size", 0))
        self.height = int(getattr(policy, "height", 0))
        self.width = int(getattr(policy, "width", 0))
        self.visual_channels = int(getattr(policy, "visual_channels", 0))

        self.visual_encoder = copy.deepcopy(policy.visual_encoder) if self.use_visual else None

        # ----- full obs dim (export input dim) -----
        if hasattr(policy, "full_num_actor_obs"):
            self.full_num_actor_obs = int(policy.full_num_actor_obs)
        else:
            # fallback: for non-visual this works; for visual it might be embedded dim (not ideal)
            self.full_num_actor_obs = int(getattr(policy, "num_actor_obs", self.actor[0].in_features))

        # ----- setup RNN (if recurrent) -----
        self.rnn = None
        self.rnn_type = None
        if self.is_recurrent:
            # Try common layouts:
            # 1) policy.memory_a.rnn (original rsl-rl)
            # 2) policy.memory_a has attribute rnn (your Memory wrapper)
            if hasattr(policy, "memory_a") and hasattr(policy.memory_a, "rnn"):
                self.rnn = copy.deepcopy(policy.memory_a.rnn)
                self.full_num_actor_obs = int(getattr(self.rnn.input_size, self.full_num_actor_obs))  # override if rnn exposes input_size
            elif hasattr(policy, "memory_a_rnn"):
                self.rnn = copy.deepcopy(policy.memory_a_rnn)
            else:
                raise ValueError("Recurrent policy detected, but cannot find RNN. Expected policy.memory_a.rnn.")

            self.rnn.cpu()
            self.rnn_type = type(self.rnn).__name__.lower()  # 'lstm' or 'gru'
            if "lstm" in self.rnn_type:
                self.rnn_type = "lstm"
                self.forward = self.forward_lstm
            elif "gru" in self.rnn_type:
                self.rnn_type = "gru"
                self.forward = self.forward_gru
            else:
                raise NotImplementedError(f"Unsupported RNN type: {type(self.rnn).__name__}")

            # RNN expects input dim == mem_input_dim_a (visual-embedded)
            # We'll compute it as full_obs - visual_dim + visual_latent (if visual), else full_obs
            if self.use_visual:
                self.rnn_input_size = self.full_num_actor_obs - self.visual_dim + self.visual_latent_size
            else:
                self.rnn_input_size = self.full_num_actor_obs

            # (optional) sanity: some RNN modules expose input_size
            if hasattr(self.rnn, "input_size") and int(self.rnn.input_size) != int(self.rnn_input_size):
                # Don't raise (maybe different implementation), but it's a strong signal for mismatch.
                print(
                    f"[WARN] RNN input_size={int(self.rnn.input_size)} but computed rnn_input_size={int(self.rnn_input_size)}. "
                    "Check visual_dim/latent/full_num_actor_obs."
                )

    # ---------------- visual embedding ----------------
    def _split_visual(self, obs_full: torch.Tensor):
        if (not self.use_visual) or self.visual_dim == 0:
            return obs_full, None
        return obs_full[..., :-self.visual_dim], obs_full[..., -self.visual_dim:]

    def _visual_vec_to_bchw(self, visual_vec: torch.Tensor):
        B = visual_vec.shape[0]
        return visual_vec.view(B, self.visual_channels, self.height, self.width)

    def _embed_visual_latent(self, obs_full: torch.Tensor):
        if not self.use_visual:
            return obs_full
        non_visual, visual_vec = self._split_visual(obs_full)
        bchw = self._visual_vec_to_bchw(visual_vec)
        latent = self.visual_encoder(bchw)
        return torch.cat([non_visual, latent], dim=-1)

    # ---------------- forward paths ----------------
    def forward(self, obs: torch.Tensor):
        # non-recurrent: obs_full -> embed -> actor
        obs = self.normalizer(obs)
        obs = self._embed_visual_latent(obs)
        actions = self.actor(obs)
        return actions

    def forward_lstm(self, obs_full: torch.Tensor, h_in: torch.Tensor, c_in: torch.Tensor):
        # obs_full: (B, full_obs_dim)
        x = self.normalizer(obs_full)
        x = self._embed_visual_latent(x)              # (B, rnn_input_size)
        # rsl-rl exporter uses seq-first with length 1: (1, B, D)
        x, (h_out, c_out) = self.rnn(x.unsqueeze(0), (h_in, c_in))
        x = x.squeeze(0)                              # (B, hidden)
        actions = self.actor(x)
        return actions, h_out, c_out

    def forward_gru(self, obs_full: torch.Tensor, h_in: torch.Tensor):
        x = self.normalizer(obs_full)
        x = self._embed_visual_latent(x)              # (B, rnn_input_size)
        x, h_out = self.rnn(x.unsqueeze(0), h_in)
        x = x.squeeze(0)
        actions = self.actor(x)
        return actions, h_out

    # ---------------- export ----------------
    def export(self, path: str, filename: str, opset_version: int = 17):
        os.makedirs(path, exist_ok=True)
        self.to("cpu")
        self.eval()

        out_file = os.path.join(path, filename)

        if self.is_recurrent:
            # IMPORTANT: exporter input is FULL obs (same as non-recurrent)
            obs = torch.zeros(1, self.full_num_actor_obs, dtype=torch.float32)

            h_in = torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size, dtype=torch.float32)

            if self.rnn_type == "lstm":
                c_in = torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size, dtype=torch.float32)
                torch.onnx.export(
                    self,
                    (obs, h_in, c_in),
                    out_file,
                    export_params=True,
                    opset_version=opset_version,
                    do_constant_folding=True,
                    verbose=self.verbose,
                    input_names=["obs", "h_in", "c_in"],
                    output_names=["actions", "h_out", "c_out"],
                    # 你若要支援 batch 可動，打開 dynamic_axes；hidden state 維度也跟 batch 綁定
                    dynamic_axes={
                        "obs": {0: "batch"},
                        "actions": {0: "batch"},
                        "h_in": {1: "batch"},
                        "c_in": {1: "batch"},
                        "h_out": {1: "batch"},
                        "c_out": {1: "batch"},
                    },
                )
            elif self.rnn_type == "gru":
                torch.onnx.export(
                    self,
                    (obs, h_in),
                    out_file,
                    export_params=True,
                    opset_version=opset_version,
                    do_constant_folding=True,
                    verbose=self.verbose,
                    input_names=["obs", "h_in"],
                    output_names=["actions", "h_out"],
                    dynamic_axes={
                        "obs": {0: "batch"},
                        "actions": {0: "batch"},
                        "h_in": {1: "batch"},
                        "h_out": {1: "batch"},
                    },
                )
            else:
                raise NotImplementedError(f"Unsupported RNN type: {self.rnn_type}")

        else:
            obs = torch.zeros(1, self.full_num_actor_obs, dtype=torch.float32)
            torch.onnx.export(
                self,
                (obs,),
                out_file,
                export_params=True,
                opset_version=opset_version,
                do_constant_folding=True,
                verbose=self.verbose,
                input_names=["obs"],
                output_names=["actions"],
                dynamic_axes={
                    "obs": {0: "batch"},
                    "actions": {0: "batch"},
                },
            )

        return out_file
    
# =========================================================
# ONNX Export Wrapper
#   - p2: export student inference
#   - p3: export actor inference
# =========================================================
class OnnxMonolithicPolicyExporter(nn.Module):
    def __init__(self, policy: nn.Module, normalizer: Optional[nn.Module] = None, verbose: bool = False):
        super().__init__()
        self.verbose = verbose
        self.normalizer = copy.deepcopy(normalizer) if normalizer is not None else nn.Identity()

        # -----------------------------
        # identify model type
        # -----------------------------
        self.is_recurrent = bool(getattr(policy, "is_recurrent", False))

        # p2 has student / memory_s
        self.is_p2 = hasattr(policy, "student") and hasattr(policy, "memory_s")
        if self.is_p2:
            self.obs_dim = int(getattr(policy, "num_student_obs", 0))  # exported obs dim (after visual embedding if any)

        # p3 has actor / memory_a
        self.is_p3 = hasattr(policy, "actor") and hasattr(policy, "memory_a")
        if self.is_p3:
            self.obs_dim = int(getattr(policy, "num_actor_obs", 0))  # exported obs dim (after visual embedding if any)

        if not (self.is_p2 or self.is_p3):
            raise ValueError("Unsupported policy type. Expected monolithicpolicy_p2 or monolithicpolicy_p3.")

        # -----------------------------
        # visual encoder info
        # -----------------------------
        self.height = int(getattr(policy, "height", 0))
        self.width = int(getattr(policy, "width", 0))
        self.visual_channels = int(getattr(policy, "visual_channels", 0))
        self.visual_dim = int(getattr(policy, "visual_dim", self.height * self.width * self.visual_channels))

        self.visual_encoder = copy.deepcopy(policy.visual_encoder).cpu()

        # infer visual latent size from encoder output if possible
        self.visual_latent_size = self._infer_visual_latent_size(policy)

        # -----------------------------
        # core heads / memory
        # -----------------------------
        if self.is_p2:
            self.policy_head = copy.deepcopy(policy.student).cpu()
            self.memory = self._extract_rnn(copy.deepcopy(policy.memory_s)).cpu()
            self.model_name = "p2_student"
        else:
            self.policy_head = copy.deepcopy(policy.actor).cpu()
            self.memory = self._extract_rnn(copy.deepcopy(policy.memory_a)).cpu()
            self.model_name = "p3_actor"

        if self.memory is None:
            raise ValueError("Cannot find inner RNN module from policy memory wrapper.")

        self.rnn_type = type(self.memory).__name__.lower()
        if "lstm" in self.rnn_type:
            self.rnn_type = "lstm"
            self.forward = self.forward_lstm
        elif "gru" in self.rnn_type:
            self.rnn_type = "gru"
            self.forward = self.forward_gru
        else:
            raise NotImplementedError(f"Unsupported RNN type: {type(self.memory).__name__}")

        self.num_layers = int(self.memory.num_layers)
        self.hidden_size = int(self.memory.hidden_size)
        self.rnn_input_size = int(self.memory.input_size)

    # -----------------------------------------------------
    # helpers
    # -----------------------------------------------------
    def _extract_rnn(self, memory_module: nn.Module) -> Optional[nn.Module]:
        """
        Your Memory wrapper often stores the actual RNN in .rnn
        """
        if hasattr(memory_module, "rnn"):
            return memory_module.rnn
        # fallback: maybe the memory module itself is already RNN-like
        if hasattr(memory_module, "hidden_size") and hasattr(memory_module, "input_size"):
            return memory_module
        return None

    def _infer_visual_latent_size(self, policy: nn.Module) -> int:
        # 1) 優先讀 policy 上明確提供的值
        if hasattr(policy, "visual_latent_size"):
            return int(policy.visual_latent_size)

        # 2) 再讀 encoder 自己的 output_size
        if hasattr(self.visual_encoder, "output_size"):
            return int(self.visual_encoder.output_size)

        # 3) 最後才 fallback 用 dummy forward
        if hasattr(self, "visual_encoder"):
            encoder = self.visual_encoder.cpu().eval()
            with torch.no_grad():
                x = torch.zeros(
                    1, self.visual_channels, self.height, self.width,
                    dtype=torch.float32
                )
                y = encoder(x)
                return int(y.shape[-1])

        raise ValueError("Cannot infer visual latent size.")
        
        # if hasattr(policy, "visual_encoder"):
        #     encoder = policy.visual_encoder
        #     # try common attribute
        #     if hasattr(policy, "visual_latent_size"):
        #         return int(policy.visual_latent_size)

        #     # fallback by running one dummy tensor
        #     with torch.no_grad():
        #         x = torch.zeros(1, self.visual_channels, self.height, self.width, dtype=torch.float32)
        #         y = encoder(x)
        #         return int(y.shape[-1])

        # raise ValueError("Cannot infer visual latent size.")

    def _visual_vec_to_bchw(self, latent_obs: torch.Tensor) -> torch.Tensor:
        expected = self.visual_channels * self.height * self.width
        if latent_obs.shape[-1] != expected:
            raise RuntimeError(
                f"latent_obs last dim mismatch: got {latent_obs.shape[-1]}, "
                f"expected {expected} = {self.visual_channels}*{self.height}*{self.width}"
            )
        return latent_obs.view(latent_obs.shape[0], self.visual_channels, self.height, self.width)

    def _encode_visual(self, latent_obs: torch.Tensor) -> torch.Tensor:
        bchw = self._visual_vec_to_bchw(latent_obs)          # (B, C, H, W)
        latent = self.visual_encoder(bchw)                  # (B, visual_latent_size)
        return latent

    def _compose_input(self, obs: torch.Tensor, latent_obs: torch.Tensor) -> torch.Tensor:
        obs = self.normalizer(obs)
        visual_latent = self._encode_visual(latent_obs)
        x = torch.cat([obs, visual_latent], dim=-1)

        if x.shape[-1] != self.rnn_input_size:
            raise RuntimeError(
                f"RNN input mismatch: composed dim={x.shape[-1]}, expected={self.rnn_input_size}"
            )
        return x

    # -----------------------------------------------------
    # forward
    # -----------------------------------------------------
    def forward_lstm(self, obs: torch.Tensor, latent_obs: torch.Tensor, h_in: torch.Tensor, c_in: torch.Tensor):
        """
        obs:       (B, obs_dim)
        latent_obs:(B, visual_dim)
        h_in:      (num_layers, B, hidden_size)
        c_in:      (num_layers, B, hidden_size)
        """
        x = self._compose_input(obs, latent_obs)            # (B, rnn_input_size)
        x = x.unsqueeze(0)                                  # (1, B, D)

        y, (h_out, c_out) = self.memory(x, (h_in, c_in))    # y: (1, B, H)
        y = y.squeeze(0)                                    # (B, H)
        actions = self.policy_head(y)                       # (B, act_dim)
        return actions, h_out, c_out

    def forward_gru(self, obs: torch.Tensor, latent_obs: torch.Tensor, h_in: torch.Tensor):
        x = self._compose_input(obs, latent_obs)
        x = x.unsqueeze(0)

        y, h_out = self.memory(x, h_in)
        y = y.squeeze(0)
        actions = self.policy_head(y)
        return actions, h_out

    # -----------------------------------------------------
    # export
    # -----------------------------------------------------
    def export(
        self,
        out_path: str,
        opset_version: int = 17,
    ) -> str:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

        self.eval()
        self.cpu()

        obs = torch.zeros(1, self.obs_dim, dtype=torch.float32)
        latent_obs = torch.zeros(1, self.visual_dim, dtype=torch.float32)
        h_in = torch.zeros(self.num_layers, 1, self.hidden_size, dtype=torch.float32)

        if self.rnn_type == "lstm":
            c_in = torch.zeros(self.num_layers, 1, self.hidden_size, dtype=torch.float32)

            torch.onnx.export(
                self,
                (obs, latent_obs, h_in, c_in),
                out_path,
                export_params=True,
                opset_version=opset_version,
                do_constant_folding=True,
                verbose=self.verbose,
                input_names=["obs", "latent_obs", "h_in", "c_in"],
                output_names=["actions", "h_out", "c_out"],
                dynamic_axes={
                    "obs": {0: "batch"},
                    "latent_obs": {0: "batch"},
                    "actions": {0: "batch"},
                    "h_in": {1: "batch"},
                    "c_in": {1: "batch"},
                    "h_out": {1: "batch"},
                    "c_out": {1: "batch"},
                },
            )

        elif self.rnn_type == "gru":
            torch.onnx.export(
                self,
                (obs, latent_obs, h_in),
                out_path,
                export_params=True,
                opset_version=opset_version,
                do_constant_folding=True,
                verbose=self.verbose,
                input_names=["obs", "latent_obs", "h_in"],
                output_names=["actions", "h_out"],
                dynamic_axes={
                    "obs": {0: "batch"},
                    "latent_obs": {0: "batch"},
                    "actions": {0: "batch"},
                    "h_in": {1: "batch"},
                    "h_out": {1: "batch"},
                },
            )
        else:
            raise NotImplementedError(f"Unsupported RNN type: {self.rnn_type}")

        return out_path

"""=============DWAQ policy runner export calls================="""
class OnnxDwaqPolicyExporterSimple(nn.Module):
    def __init__(self, policy: nn.Module, normalizer: Optional[nn.Module] = None, verbose: bool = False):
        super().__init__()
        self.verbose = verbose
        self.normalizer = copy.deepcopy(normalizer) if normalizer is not None else nn.Identity()
        self.policy = copy.deepcopy(policy).cpu()

        self.history_len = int(getattr(policy, "history_len", 5))

        # infer dims
        latent_dim = self.policy.proprio_encoder_mu.out_features
        vel_dim = self._infer_last_linear_out_features(self.policy.vel_head)
        actor_in_dim = self._infer_first_linear_in_features(self.policy.actor)
        self.obs_dim = actor_in_dim - latent_dim - vel_dim

    @staticmethod
    def _infer_last_linear_out_features(module: nn.Module) -> int:
        for m in reversed(list(module.modules())):
            if isinstance(m, nn.Linear):
                return int(m.out_features)
        raise ValueError("Cannot infer output dim.")

    @staticmethod
    def _infer_first_linear_in_features(module: nn.Module) -> int:
        for m in module.modules():
            if isinstance(m, nn.Linear):
                return int(m.in_features)
        raise ValueError("Cannot infer input dim.")

    def forward(self, obs: torch.Tensor, obs_history: torch.Tensor):
        obs = self.normalizer(obs)
        action, est_v = self.policy.act_for_onnx_transfer(obs, obs_history, use_mu=True)
        return action, est_v

    def export(self, out_path: str, opset_version: int = 17):
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        self.eval()
        self.cpu()

        dummy_obs = torch.zeros(1, self.obs_dim, dtype=torch.float32)
        dummy_obs_history = torch.zeros(1, self.history_len, self.obs_dim, dtype=torch.float32)

        torch.onnx.export(
            self,
            (dummy_obs, dummy_obs_history),
            out_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            verbose=self.verbose,
            input_names=["obs", "obs_history"],
            output_names=["action", "est_v"],
            dynamic_axes={
                "obs": {0: "batch"},
                "obs_history": {0: "batch"},
                "action": {0: "batch"},
                "est_v": {0: "batch"},
            },
        )
        return out_path


def export_dwaq_policy_as_onnx_simple(
    policy: nn.Module,
    path: str,
    filename: str = "dwaq_policy.onnx",
    normalizer: Optional[nn.Module] = None,
    opset_version: int = 17,
    verbose: bool = False,
):
    os.makedirs(path, exist_ok=True)
    exporter = OnnxDwaqPolicyExporterSimple(policy, normalizer, verbose)
    out_path = str(Path(path) / filename)
    return exporter.export(out_path, opset_version=opset_version)
