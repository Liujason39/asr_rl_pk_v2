# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import warnings
import torch
import torch.nn as nn

from asr_rl_pk.modules import ActorCritic
from asr_rl_pk.networks import Memory
from asr_rl_pk.utils import resolve_nn_activation
from asr_rl_pk.modules.conv2d import Conv2dHeadModel


class VisualActorCriticRecurrent(ActorCritic):
    is_recurrent = True

    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        activation="elu",
        rnn_type="lstm",
        rnn_hidden_dim=256,
        rnn_num_layers=1,
        init_noise_std=1.0,

        # ---------- visual parameters ----------
        use_visual: bool = True,
        visual_dim: int | None = None,          # obs 最後的 visual vector 長度（若 None -> 用 H*W*C 推）
        visual_latent_size: int = 128,
        visual_kwargs=dict(
            channels=[64, 64],
            kernel_sizes=[3, 3],
            strides=[1, 1],
            hidden_sizes=[256],
        ),
        height: int = 8,
        width: int = 6,
        visual_channels: int = 1,               # depth 通常 1
        critic_use_visual: bool = True,         # critic 是否也做 visual embedding
        # --------------------------------------

        **kwargs,
    ):
        print("Initializing VisualActorCriticRecurrent")
        if "rnn_hidden_size" in kwargs:
            warnings.warn(
                "The argument `rnn_hidden_size` is deprecated and will be removed in a future version. "
                "Please use `rnn_hidden_dim` instead.",
                DeprecationWarning,
            )
            if rnn_hidden_dim == 256:
                rnn_hidden_dim = kwargs.pop("rnn_hidden_size")
        if kwargs:
            print(
                "VisualActorCriticRecurrent.__init__ got unexpected arguments, which will be ignored: "
                + str(kwargs.keys()),
            )

        super().__init__(
            num_actor_obs=rnn_hidden_dim,
            num_critic_obs=rnn_hidden_dim,
            num_actions=num_actions,
            actor_hidden_dims=actor_hidden_dims,
            critic_hidden_dims=critic_hidden_dims,
            activation=activation,
            init_noise_std=init_noise_std,
        )

        # update full obs dims for exporter dummy input
        self.full_num_actor_obs = num_actor_obs
        self.full_num_critic_obs = num_critic_obs

        self.activation = resolve_nn_activation(activation)

        # ---- visual setup ----
        self.use_visual = use_visual
        self.critic_use_visual = critic_use_visual
        self.height = height
        self.width = width
        self.visual_channels = visual_channels
        self.visual_latent_size = visual_latent_size

        if self.use_visual:
            if visual_dim is None:
                visual_dim = height * width * visual_channels
            self.visual_dim = visual_dim

            vk = dict(
                channels=[64, 64],
                kernel_sizes=[3, 3],
                strides=[1, 1],
                hidden_sizes=[256],
            )
            vk.update(visual_kwargs)

            self.visual_encoder = Conv2dHeadModel(
                image_shape=(visual_channels, height, width),
                output_size=visual_latent_size,
                **vk,
            )

            # Memory 輸入維度：用 latent 取代 visual_dim
            mem_input_dim_a = num_actor_obs - self.visual_dim + self.visual_latent_size
            mem_input_dim_c = (
                num_critic_obs - self.visual_dim + self.visual_latent_size
                if self.critic_use_visual
                else num_critic_obs
            )
        else:
            self.visual_dim = 0
            self.visual_encoder = None
            mem_input_dim_a = num_actor_obs
            mem_input_dim_c = num_critic_obs

        # ---- RNN memory ----
        self.memory_a = Memory(mem_input_dim_a, type=rnn_type, num_layers=rnn_num_layers, hidden_size=rnn_hidden_dim)
        self.memory_c = Memory(mem_input_dim_c, type=rnn_type, num_layers=rnn_num_layers, hidden_size=rnn_hidden_dim)

        print(f"Actor RNN: {self.memory_a}")
        print(f"Critic RNN: {self.memory_c}")
        if self.use_visual:
            print(f"Visual encoder: {self.visual_encoder}")
            print(f"Visual: dim={self.visual_dim}, latent={self.visual_latent_size}, (C,H,W)=({visual_channels},{height},{width})")
            print(f"Memory actor input dim: {mem_input_dim_a}, critic input dim: {mem_input_dim_c}")

    def reset(self, dones=None):
        self.memory_a.reset(dones)
        self.memory_c.reset(dones)

    # ---------------- visual embedding helpers ----------------
    def _split_visual(self, obs: torch.Tensor):
        # obs: (..., D)
        if self.visual_dim == 0:
            return obs, None
        return obs[..., :-self.visual_dim], obs[..., -self.visual_dim:]

    def _visual_vec_to_bchw(self, visual_vec: torch.Tensor) -> torch.Tensor:
        """
        visual_vec: (N, visual_dim) -> (N, C, H, W)
        """
        expected = self.visual_channels * self.height * self.width
        if visual_vec.shape[-1] != expected:
            raise RuntimeError(
                f"visual_dim mismatch: got {visual_vec.shape[-1]}, expected {expected} = C*H*W "
                f"({self.visual_channels}*{self.height}*{self.width})."
            )
        return visual_vec.view(visual_vec.shape[0], self.visual_channels, self.height, self.width)

    def _embed_visual_latent(self, obs: torch.Tensor, *, for_critic: bool = False) -> torch.Tensor:
        """
        obs: (B, D) 或 (T, B, D)
        回傳: 同 leading dims，但最後 dim 變成 D - visual_dim + latent
        """
        if not self.use_visual:
            return obs
        if for_critic and (not self.critic_use_visual):
            return obs

        leading_dims = obs.shape[:-1]  # (B,) or (T,B)
        non_visual, visual_vec = self._split_visual(obs)

        # 把所有 leading dims 展平當作 batch 丟進 CNN encoder
        visual_vec_flat = visual_vec.reshape(-1, visual_vec.shape[-1])  # (N, visual_dim)
        N = visual_vec_flat.shape[0]

        # 空 batch：直接回傳空 latent，保持 shape 正確
        if N == 0:
            latent = visual_vec_flat.new_zeros((*leading_dims, self.visual_latent_size))
            return torch.cat([non_visual, latent], dim=-1)
        bchw = self._visual_vec_to_bchw(visual_vec_flat)                # (N, C, H, W)

        latent_flat = self.visual_encoder(bchw)                         # (N, latent)
        latent = latent_flat.reshape(*leading_dims, -1)                 # (..., latent)

        return torch.cat([non_visual, latent], dim=-1)

    # ---------------- ActorCritic API ----------------
    def act(self, observations, masks=None, hidden_states=None):
        obs = self._embed_visual_latent(observations, for_critic=False)
        input_a = self.memory_a(obs, masks, hidden_states)
        return super().act(input_a.squeeze(0))

    def act_inference(self, observations):
        obs = self._embed_visual_latent(observations, for_critic=False)
        input_a = self.memory_a(obs)
        return super().act_inference(input_a.squeeze(0))

    def evaluate(self, critic_observations, masks=None, hidden_states=None):
        obs = self._embed_visual_latent(critic_observations, for_critic=True)
        input_c = self.memory_c(obs, masks, hidden_states)
        return super().evaluate(input_c.squeeze(0))

    def get_hidden_states(self):
        return self.memory_a.hidden_states, self.memory_c.hidden_states