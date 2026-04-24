# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal
from asr_rl_pk.modules.conv2d import Conv2dHeadModel
from asr_rl_pk.utils import resolve_nn_activation



class VisualActorCritic(nn.Module):
    is_recurrent = False

    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        activation="elu",
        init_noise_std=1.0,
        noise_std_type: str = "scalar",

        # ---------- visual parameters ----------
        use_visual: bool = True,
        visual_dim: int | None = None,          # <= 最後那段 visual vector 的長度（必填，除非你用 (H,W,1) 直接塞進 obs 不常見）
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
        # 是否讓 critic 也吃 visual latent
        critic_use_visual: bool = True,
        # --------------------------------------

        **kwargs,
    ):
        if kwargs:
            print(
                "VisualActorCritic.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()
        activation = resolve_nn_activation(activation)
        self.full_num_actor_obs = num_actor_obs
        self.full_num_critic_obs = num_critic_obs
        self.use_visual = use_visual
        self.critic_use_visual = critic_use_visual

        self.height = height
        self.width = width
        self.visual_channels = visual_channels
        self.visual_latent_size = visual_latent_size

        # 如果要用 visual，必須知道最後那段的長度（flatten depth 常見是 H*W 或 H*W*1）
        if self.use_visual:
            if visual_dim is None:
                # 預設用 H*W*C 推
                visual_dim = height * width * visual_channels
            self.visual_dim = visual_dim

            # 建立 visual encoder：input image_shape = (C, H, W)
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

            # actor 真正吃到的 obs dim：原本 num_actor_obs 裡面含 visual_dim（在最後）
            # -> 用 latent 取代掉 visual_dim
            mlp_input_dim_a = num_actor_obs - self.visual_dim + self.visual_latent_size

            # critic 如果也用 visual，同理替換；不然維持原樣
            if self.critic_use_visual:
                mlp_input_dim_c = num_critic_obs - self.visual_dim + self.visual_latent_size
            else:
                mlp_input_dim_c = num_critic_obs
        else:
            self.visual_dim = 0
            self.visual_encoder = None
            mlp_input_dim_a = num_actor_obs
            mlp_input_dim_c = num_critic_obs

        # ---------------- Policy (Actor) ----------------
        actor_layers = []
        actor_layers.append(nn.Linear(mlp_input_dim_a, actor_hidden_dims[0]))
        actor_layers.append(activation)
        for layer_index in range(len(actor_hidden_dims)):
            if layer_index == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[layer_index], num_actions))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[layer_index], actor_hidden_dims[layer_index + 1]))
                actor_layers.append(activation)
        self.actor = nn.Sequential(*actor_layers)

        # ---------------- Value (Critic) ----------------
        critic_layers = []
        critic_layers.append(nn.Linear(mlp_input_dim_c, critic_hidden_dims[0]))
        critic_layers.append(activation)
        for layer_index in range(len(critic_hidden_dims)):
            if layer_index == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[layer_index], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dims[layer_index], critic_hidden_dims[layer_index + 1]))
                critic_layers.append(activation)
        self.critic = nn.Sequential(*critic_layers)

        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")
        if self.use_visual:
            print(f"Visual encoder: {self.visual_encoder}")
            print(f"Visual: dim={self.visual_dim}, latent={self.visual_latent_size}, (C,H,W)=({visual_channels},{height},{width})")

        # ---------------- Action noise ----------------
        self.noise_std_type = noise_std_type
        if self.noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif self.noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")

        # Action distribution (populated in update_distribution)
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args(False)

    # ---------- visual embedding helpers ----------
    def _split_visual(self, obs: torch.Tensor):
        """
        obs: (B, D_total)
        回傳: non_visual (B, D_total-visual_dim), visual_vec (B, visual_dim)
        """
        if self.visual_dim == 0:
            return obs, None
        return obs[..., :-self.visual_dim], obs[..., -self.visual_dim:]

    def _visual_vec_to_bchw(self, visual_vec: torch.Tensor) -> torch.Tensor:
        """
        visual_vec: (B, visual_dim) 或 (B, H*W) 或 (B, H*W*1)
        -> (B, C, H, W)
        """
        B = visual_vec.shape[0]
        expected = self.visual_channels * self.height * self.width

        if visual_vec.shape[1] != expected:
            raise RuntimeError(
                f"visual_dim mismatch: got {visual_vec.shape[1]}, expected {expected} = C*H*W "
                f"({self.visual_channels}*{self.height}*{self.width})."
            )
        # (B, C*H*W) -> (B, C, H, W)
        return visual_vec.view(B, self.visual_channels, self.height, self.width)

    def _embed_visual_latent(self, obs: torch.Tensor, *, for_critic: bool = False) -> torch.Tensor:
        """
        obs: (B, D_total) 其中最後 visual_dim 是 visual 向量
        回傳: (B, D_total-visual_dim+latent)
        """
        if not self.use_visual:
            return obs
        if for_critic and (not self.critic_use_visual):
            return obs

        non_visual, visual_vec = self._split_visual(obs)
        # 把所有 leading dims 展平當作 batch 丟進 CNN encoder
        visual_vec_flat = visual_vec.reshape(-1, visual_vec.shape[-1])  # (N, visual_dim)
        N = visual_vec_flat.shape[0]

        # 空 batch：直接回傳空 latent，保持 shape 正確
        if N == 0:
            leading_dims = obs.shape[:-1]  # (B,)
            latent = visual_vec_flat.new_zeros((*leading_dims, self.visual_latent_size))
            return torch.cat([non_visual, latent], dim=-1)
        bchw = self._visual_vec_to_bchw(visual_vec)
        latent = self.visual_encoder(bchw)  # (B, visual_latent_size)
        return torch.cat([non_visual, latent], dim=-1)

    # ---------- API ----------
    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, observations):
        obs = self._embed_visual_latent(observations, for_critic=False)

        mean = self.actor(obs)
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std).expand_as(mean)
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        # create distribution
        self.distribution = Normal(mean, std)

    def act(self, observations, **kwargs):
        self.update_distribution(observations)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations):
        obs = self._embed_visual_latent(observations, for_critic=False)
        return self.actor(obs)

    def evaluate(self, critic_observations, **kwargs):
        obs = self._embed_visual_latent(critic_observations, for_critic=True)
        return self.critic(obs)

    def load_state_dict(self, state_dict, strict=True):
        """Load the parameters of the actor-critic model.

        Args:
            state_dict (dict): State dictionary of the model.
            strict (bool): Whether to strictly enforce that the keys in state_dict match the keys returned by this
                           module's state_dict() function.

        Returns:
            bool: Whether this training resumes a previous training. This flag is used by the `load()` function of
                  `OnPolicyRunner` to determine how to load further parameters (relevant for, e.g., distillation).
        """

        super().load_state_dict(state_dict, strict=strict)
        return True
