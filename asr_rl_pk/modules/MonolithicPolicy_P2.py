# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import warnings

import torch
import torch.nn as nn
from torch.distributions import Normal

from asr_rl_pk.modules import StudentTeacher
from asr_rl_pk.networks import Memory
from asr_rl_pk.utils import resolve_nn_activation

from asr_rl_pk.modules.conv2d import Conv2dHeadModel


class monolithicpolicy_p2(nn.Module):
    is_recurrent = True

    def __init__(
        self,
        num_student_obs,
        num_latent_student_obs,
        num_teacher_obs,
        num_latent_teacher_obs,
        num_actions,
        student_hidden_dims=[256, 256, 256],
        teacher_hidden_dims=[256, 256, 256],
        activation="elu",
        rnn_type="lstm",
        rnn_hidden_dim=256,
        rnn_num_layers=1,
        init_noise_std=1.0,
        mlp_hidden_dims=[256,128],
        # ---------- visual parameters ----------
        visual_latent_size: int = 32,
        visual_kwargs=dict(
            channels=[64, 64],
            kernel_sizes=[3, 3],
            strides=[1, 1],
            hidden_sizes=[256],
        ),
        height: int = 8,
        width: int = 6,
        visual_channels: int = 1,
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
                "monolithicpolicy_p2.__init__ got unexpected arguments, which will be ignored: "
                + str(kwargs.keys()),
            )

        self.teacher_recurrent = True

        super().__init__()

        activation = resolve_nn_activation(activation)

        self.loaded_teacher = False  # indicates if teacher has been loaded


        self.num_student_obs = num_student_obs
        self.num_latent_student_obs = num_latent_student_obs
        self.visual_latent_size = visual_latent_size
        
        # construct monolith teacher student networks
        # =====latent MLP : process privileged observations to latent for actor and critic=====
        latent_mlp_layers_t = []
        latent_mlp_layers_t.append(nn.Linear(num_latent_teacher_obs, mlp_hidden_dims[0]))
        latent_mlp_layers_t.append(activation)
        for layer_index in range(len(mlp_hidden_dims) - 1):
            latent_mlp_layers_t.append(nn.Linear(mlp_hidden_dims[layer_index], mlp_hidden_dims[layer_index + 1]))
            # latent_mlp_layers_t.append(nn.LayerNorm(teacher_hidden_dims[layer_index + 1])) # add layer normalization
            latent_mlp_layers_t.append(activation)
        self.latent_mlp_t = nn.Sequential(*latent_mlp_layers_t)
        self.latent_mlp_t.eval()

        print(f"Latent MLP t: {self.latent_mlp_t}")

        # =====Visual encoder for student=====

        # check if visual_latent_size set as mlp encoder output
        if visual_latent_size != mlp_hidden_dims[-1]:
            raise ValueError("visual_latent_size must match the output dimension of the MLP encoder")

        self.height = height
        self.width = width
        self.visual_channels = visual_channels
        self.visual_dim = height * width * visual_channels

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


        # =====actor & critic will share similar struct of model layers, but have different parameters=====
        self.memory_t = Memory(num_teacher_obs+mlp_hidden_dims[-1], type=rnn_type, num_layers=rnn_num_layers, hidden_size=rnn_hidden_dim)
        self.memory_s = Memory(num_student_obs+mlp_hidden_dims[-1], type=rnn_type, num_layers=rnn_num_layers, hidden_size=rnn_hidden_dim)

        print(f"Teacher RNN: {self.memory_t}")
        print(f"Student RNN: {self.memory_s}")

        mlp_input_dim_t = rnn_hidden_dim
        mlp_input_dim_s = rnn_hidden_dim
        

        # Teacher MLP
        teacher_layers = []
        teacher_layers.append(nn.Linear(mlp_input_dim_t, teacher_hidden_dims[0]))
        teacher_layers.append(activation)
        for layer_index in range(len(teacher_hidden_dims)):
            if layer_index == len(teacher_hidden_dims) - 1:
                teacher_layers.append(nn.Linear(teacher_hidden_dims[layer_index], num_actions))
            else:
                teacher_layers.append(nn.Linear(teacher_hidden_dims[layer_index], teacher_hidden_dims[layer_index + 1]))
                # teacher_layers.append(nn.LayerNorm(teacher_hidden_dims[layer_index + 1])) # add layer normalization
                teacher_layers.append(activation)
        self.teacher = nn.Sequential(*teacher_layers)
        self.teacher.eval()

        # Student MLP
        student_layers = []
        student_layers.append(nn.Linear(mlp_input_dim_s, student_hidden_dims[0]))
        student_layers.append(activation)
        for layer_index in range(len(student_hidden_dims)):
            if layer_index == len(student_hidden_dims) - 1:
                student_layers.append(nn.Linear(student_hidden_dims[layer_index], num_actions))
            else:
                student_layers.append(nn.Linear(student_hidden_dims[layer_index], student_hidden_dims[layer_index + 1]))
                # student_layers.append(nn.LayerNorm(student_hidden_dims[layer_index + 1])) # add layer normalization
                student_layers.append(activation)
        self.student = nn.Sequential(*student_layers)

        print(f"Student MLP: {self.student}")
        print(f"Teacher MLP: {self.teacher}")

        # action noise
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False

    def reset(self, dones=None, hidden_states=None):
        if hidden_states is None:
            hidden_states = (None, None)
        self.memory_s.reset(dones, hidden_states[0])
        if self.teacher_recurrent:
            self.memory_t.reset(dones, hidden_states[1])
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

    def latent_forward_t(self, latent_obs):
        return self.latent_mlp_t(latent_obs)
    
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
    
    def latent_forward_s(self, latent_obs):
        bchw = self._visual_vec_to_bchw(latent_obs) # (B, C, H, W)
        latent_flat = self.visual_encoder(bchw)
        return latent_flat

    def act(self, observations, latent_obs):
        cat_observations = torch.cat([observations, self.latent_forward_s(latent_obs)], dim=-1)
        input_s = self.memory_s(cat_observations)
        self.update_distribution(input_s.squeeze(0))
        return self.distribution.sample()

    def act_inference(self, observations, latent_obs_s):
        cat_observations = torch.cat([observations, self.latent_forward_s(latent_obs_s)], dim=-1)
        input_s = self.memory_s(cat_observations)
        actions_mean = self.student(input_s.squeeze(0))
        return actions_mean

    def evaluate(self, teacher_observations, latent_obs_t):
        with torch.no_grad():
            cat_observations = torch.cat([teacher_observations, self.latent_forward_t(latent_obs_t)], dim=-1)
            input_t = self.memory_t(cat_observations)
            actions = self.teacher(input_t.squeeze(0))
        return actions

    def get_hidden_states(self):
        return self.memory_s.hidden_states, self.memory_t.hidden_states

    def detach_hidden_states(self, dones=None):
        self.memory_s.detach_hidden_states(dones)
        if self.teacher_recurrent:
            self.memory_t.detach_hidden_states(dones)
    
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
            teacher_state_dict = {}
            for key, value in state_dict.items():
                if "actor." in key:
                    teacher_state_dict[key.replace("actor.", "")] = value
            self.teacher.load_state_dict(teacher_state_dict, strict=strict)
            # also load recurrent memory if teacher is recurrent
            if self.is_recurrent and self.teacher_recurrent:
                memory_t_state_dict = {}
                for key, value in state_dict.items():
                    if "memory_a." in key:
                        memory_t_state_dict[key.replace("memory_a.", "")] = value
                self.memory_t.load_state_dict(memory_t_state_dict, strict=strict)
            # set flag for successfully loading the parameters
            self.loaded_teacher = True
            self.teacher.eval()

            # copy paramteters to student as monolithic policy
            student_state_dict = {}
            for key, value in state_dict.items():
                if "actor." in key:
                    student_state_dict[key.replace("actor.", "")] = value
            self.student.load_state_dict(student_state_dict, strict=strict)
            # also load recurrent memory if teacher is recurrent
            if self.is_recurrent and self.teacher_recurrent:
                memory_s_state_dict = {}
                for key, value in state_dict.items():
                    if "memory_a." in key:
                        memory_s_state_dict[key.replace("memory_a.", "")] = value
                self.memory_s.load_state_dict(memory_s_state_dict, strict=strict)
            # set flag for successfully loading the parameters
            self.loaded_teacher = True
            self.loaded_student = True
            return False
        elif any("student" in key for key in state_dict.keys()):  # loading parameters from distillation training
            super().load_state_dict(state_dict, strict=strict)
            # set flag for successfully loading the parameters
            self.loaded_teacher = True
            self.teacher.eval()
            self.loaded_student = True
            return True
        else:
            raise ValueError("state_dict does not contain student or teacher parameters")
