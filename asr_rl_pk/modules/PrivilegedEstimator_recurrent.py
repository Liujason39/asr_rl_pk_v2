# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import warnings

import torch
import torch.nn as nn
from torch.distributions import Normal
from asr_rl_pk.modules.conv2d import Conv2dHeadModel, VisualMultiHeadEncoder
from asr_rl_pk.networks import Memory
from asr_rl_pk.utils import resolve_nn_activation


class PrivilegedEstimatorRecurrent(nn.Module):
    """this will load a RL model as student and prepare a encoder to predict specific obs dims of "target" with given priviliged observations.
        Currently, this is only support MLP"""
    is_recurrent = True

    def __init__(
        self,
        num_student_obs,
        num_privileged_obs,
        num_encoder_obs,  # 這個是給 encoder 預測的 obs dims，可在驗證時覆蓋 student 的 obs dims
        num_actions,
        student_hidden_dims=[256, 256, 256],
        activation="elu",
        rnn_type="lstm",
        rnn_hidden_dim=256,
        rnn_num_layers=1,
        init_noise_std=0.1,

        # ---------- visual parameters ----------
        use_visual: bool = True,
        visual_dim: int | None = None,          # <= 最後那段 visual vector 的長度（必填，除非你用 (H,W,1) 直接塞進 obs 不常見）
        visual_latent_size: int = None,
        visual_kwargs=dict(
            channels=[64, 64],
            kernel_sizes=[3, 3],
            strides=[1, 1],
            hidden_sizes=[256],
        ),
        height: int = 8,
        width: int = 6,
        visual_channels: int = 1,               # depth 通常 1

        num_terrain_classes: int=7,
        geom_output_size: int = 8,
        # 是否讓 critic 也吃 visual latent
        critic_use_visual: bool = False,

        encoder_target_obs_indices: int = None,  # 如果不為None,則覆蓋student 此indices開始的student's obs dims
        # --------------------------------------

        **kwargs,
    ):
        if "rnn_hidden_size" in kwargs:
            warnings.warn(
                "The argument `rnn_hidden_size` is deprecated and will be removed in a future version. "
                "Please use `rnn_hidden_dim` instead.",
                DeprecationWarning,
            )
            if rnn_hidden_dim == 256:  # Only override if the new argument is at its default
                rnn_hidden_dim = kwargs.pop("rnn_hidden_size")

        if kwargs:
            print(
                "TeacherEncoder.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()
        activation = resolve_nn_activation(activation)

        self.loaded_student = False  # indicates if student has been loaded

        mlp_input_dim_s = num_student_obs
        mlp_input_dim_p = num_privileged_obs
        mlp_input_dim_e = num_encoder_obs

        self.use_visual = use_visual
        self.critic_use_visual = critic_use_visual

        self.height = height
        self.width = width
        self.visual_channels = visual_channels
        self.visual_latent_size = visual_latent_size

        self.encoder_target_obs_indices = encoder_target_obs_indices

        # 如果要用 visual，必須知道最後那段的長度（flatten depth 常見是 H*W 或 H*W*1）
        if self.use_visual:
            if visual_dim is None:
                # 預設用 H*W*C 推
                visual_dim = height * width * visual_channels
            self.visual_dim = visual_dim
            if (self.visual_dim != mlp_input_dim_e):
                raise ValueError(
                    f"self.visual_dim is not match with mlp_input_dim_e"
                )
            # 建立 visual encoder：input image_shape = (C, H, W)
            vk = dict(
                channels=[64, 64],
                kernel_sizes=[3, 3],
                strides=[1, 1],
                hidden_sizes=[256],
            )
            vk.update(visual_kwargs)
            # encoder for predict mlp_input_dim_t by mlp_input_dim_e
            print(f"mlp_input_dim_p: {mlp_input_dim_p}, mlp_input_dim_e: {mlp_input_dim_e}")
            # self.visual_encoder = Conv2dHeadModel(
            #     image_shape=(visual_channels, height, width),
            #     output_size=mlp_input_dim_p if visual_latent_size is None else visual_latent_size, 
            #     **vk,
            # )

            # get normalization layer for conv encoder
            norm_type = vk.pop("norm_type", None)
            norm_num_groups = vk.pop("norm_num_groups", 4)
            if norm_type == "groupnorm":
                normlayer = lambda oc: torch.nn.GroupNorm(num_groups=norm_num_groups, num_channels=oc)
            elif norm_type == "batchnorm":
                normlayer = "BatchNorm2d"
            else:
                normlayer = None
            # Change to use Multihead encoder
            self.visual_encoder = VisualMultiHeadEncoder(
                image_shape=(visual_channels, height, width),
                channels=vk["channels"],
                kernel_sizes=vk["kernel_sizes"],
                strides=vk["strides"],
                shared_hidden_sizes=vk["shared_hidden_sizes"],
                num_terrain_classes=num_terrain_classes,
                geom_output_size=geom_output_size,
                paddings=vk.get("paddings", None),
                nonlinearity=vk.get("nonlinearity", nn.LeakyReLU),
                use_maxpool=vk.get("use_maxpool", False),
                normlayer=normlayer,
            )
        else:
            # TODO: 目前只支援visual encoder, 對VAE encoder的支援還沒做
            self.visual_dim = 0
            self.visual_encoder = None
        
        if self.use_visual:
            print(f"Visual encoder: {self.visual_encoder}")
            print(f"Visual: dim={self.visual_dim}, latent={self.visual_latent_size}, (C,H,W)=({visual_channels},{height},{width})")
        else:
            raise ValueError(f"No visual encoder")


        # =====students MLP=====

        # student
        self.memory_a = Memory(num_student_obs, type=rnn_type, num_layers=rnn_num_layers, hidden_size=rnn_hidden_dim)
        print(f"Actor RNN: {self.memory_a}")

        student_layers = []
        student_layers.append(nn.Linear(rnn_hidden_dim, student_hidden_dims[0]))
        student_layers.append(activation)
        for layer_index in range(len(student_hidden_dims)):
            if layer_index == len(student_hidden_dims) - 1:
                student_layers.append(nn.Linear(student_hidden_dims[layer_index], num_actions))
            else:
                student_layers.append(nn.Linear(student_hidden_dims[layer_index], student_hidden_dims[layer_index + 1]))
                student_layers.append(activation)
        self.student = nn.Sequential(*student_layers)
        self.student.eval()

        print(f"Student MLP: {self.student}")

        # action noise
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False

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

    def reset(self, dones=None):
        self.memory_a.reset(dones)

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
        mean = self.student(observations)
        std = self.std.expand_as(mean)
        self.distribution = Normal(mean, std)

    def act(self, observations, masks=None, hidden_states=None):
        input_a = self.memory_a(observations, masks, hidden_states)
        self.update_distribution(input_a.squeeze(0))
        
        return self.distribution.sample()

    def act_inference(self, observations, encoder_observations=None):
        with torch.no_grad():
            if encoder_observations is not None and self.encoder_target_obs_indices is not None:
                fused_obs, _ = self.replace_estimated_obs(observations, encoder_observations)
            else:
                fused_obs = observations
            input_a = self.memory_a(fused_obs)
            actions_mean = self.student(input_a.squeeze(0))
        return actions_mean

    # this will run student instead
    def evaluate(self, student_observations):
        with torch.no_grad():
            input_a = self.memory_a(student_observations)
            actions_mean = self.student(input_a.squeeze(0))
        return actions_mean
    
    def _embed_visual_latent(self, visual_obs: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        visual_obs: (B, visual_dim)
        回傳: (B, visual_latent_size)
        """
        if not self.use_visual:
            return visual_obs  # identity, for the case where visual_obs is directly concatenated in obs and fed into MLP student

        # 把所有 leading dims 展平當作 batch 丟進 CNN encoder
        visual_vec_flat = visual_obs.reshape(-1, visual_obs.shape[-1])  # (N, visual_dim)
        N = visual_vec_flat.shape[0]

        # 空 batch：直接回傳錯誤，因為這通常是 code bug（不應該有空 batch），而且 CNN encoder 不支援空 batch
        if N == 0:
            raise ValueError(f"Empty visual_obs")
        bchw = self._visual_vec_to_bchw(visual_vec_flat)
        # using Conv2dHeadModel
        # latent = self.visual_encoder(bchw)  # (B, visual_latent_size)
        # using multihead encoder
        latent :dict[str, torch.Tensor] = self.visual_encoder(bchw)  # (B, visual_latent_size)
        return latent
    
    # this will run visual encoder to predict teacher obs, which will be used for encoder loss calculation
    def encode_encoder(self, encoder_observations) -> dict[str, torch.Tensor]:
        estimated = self._embed_visual_latent(encoder_observations)
        return estimated

    # this will run visual encoder to predict teacher obs, which will be used for encoder loss calculation, and this is for evaluation only (no grad)
    def evaluate_encoder(self, encoder_observations) -> dict[str, torch.Tensor]:
        with torch.no_grad():
            estimated = self._embed_visual_latent(encoder_observations)
        return estimated
    
    

    def load_state_dict(self, state_dict, strict=True):
        """Load the parameters of the student and teacher networks.

        Args:
            state_dict (dict): State dictionary of the model.
            strict (bool): Whether to strictly enforce that the keys in state_dict match the keys returned by this
                           module's state_dict() function.

        Returns:
            bool: Whether this training resumes a previous training. This flag is used by the `load()` function of
                  `OnPolicyRunner` to determine how to load further parameters.
        """

        # check if state_dict contains teacher and student or just teacher parameters
        if any("actor" in key for key in state_dict.keys()):  # loading parameters from rl training
            # rename keys to match teacher and remove critic parameters
            student_state_dict = {}
            for key, value in state_dict.items():
                if "actor." in key:
                    student_state_dict[key.replace("actor.", "")] = value
            self.student.load_state_dict(student_state_dict, strict=strict)
            # also load recurrent memory if teacher is recurrent
            if self.is_recurrent and self.teacher_recurrent:
                raise NotImplementedError("Loading recurrent memory for the teacher is not implemented yet")  # TODO
            # set flag for successfully loading the parameters
            self.loaded_student = True
            self.student.eval()
            return False
        elif any("student" in key for key in state_dict.keys()):  # loading parameters from distillation training
            super().load_state_dict(state_dict, strict=strict)
            # set flag for successfully loading the parameters
            self.loaded_student = True
            self.student.eval()
            return True
        else:
            raise ValueError("state_dict does not contain student or teacher parameters")

    def get_hidden_states(self):
        return self.memory_a.hidden_states, None

    def get_hidden_states(self):
        return self.memory_a.hidden_states, None

    def replace_estimated_obs(self, obs, encoder_obs):
        """
        observations:        (B, obs_dim)
        encoder_observations:(B, encoder_obs_dim)
        return:
            fused_obs:       (B, obs_dim)
            estimated_part:  (B, target_dim)
        """
        estimated_part = self.visual_encoder(encoder_obs)

        if self.encoder_target_obs_indices is None:
            raise ValueError("encoder_target_obs_indices is None, cannot replace obs segment.")
        
        # 把 encoder 的輸出從 multihead encoder 的 dict 轉成 flat vector，準備塞回 obs
        estimated_part = self.visual_encoder.pack_encoder_output(estimated_part)  

        start = self.encoder_target_obs_indices
        end = start + estimated_part.shape[-1]

        if end > obs.shape[-1]:
            raise RuntimeError(
                f"Replacement range [{start}:{end}] exceeds obs dim {obs.shape[-1]}"
            )

        fused_obs = obs.clone()
        fused_obs[..., start:end] = estimated_part
        return fused_obs, estimated_part