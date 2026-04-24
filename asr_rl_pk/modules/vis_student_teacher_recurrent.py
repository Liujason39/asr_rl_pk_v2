# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import warnings
import torch
import torch.nn as nn

from asr_rl_pk.utils import resolve_nn_activation
from asr_rl_pk.networks import Memory
from asr_rl_pk.modules.visual_actor_critic_recurrent import VisualActorCriticRecurrent  # 依你實際路徑改


class VisStudentTeacherRecurrent(nn.Module):
    is_recurrent = True

    def __init__(
        self,
        num_student_obs,
        num_teacher_obs,
        num_actions,
        student_hidden_dims=[256, 256, 256],
        teacher_hidden_dims=[256, 256, 256],
        activation="elu",
        rnn_type="gru",
        rnn_hidden_dim=256,
        rnn_num_layers=1,
        init_noise_std=0.1,
        teacher_recurrent=False,

        # ---------- visual parameters for student ----------
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
        visual_channels: int = 1,
        critic_use_visual: bool = False,   # distillation student 通常不需要 critic，先關掉也可以
        # ------------------------------------------------------------

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
                "StudentTeacherRecurrent.__init__ got unexpected arguments, which will be ignored: "
                + str(kwargs.keys()),
            )

        super().__init__()

        self.loaded_teacher = False
        self.teacher_recurrent = teacher_recurrent

        # ---------------- student：直接用 VisualActorCriticRecurrent ----------------
        # 注意：num_student_obs 必須是「完整 obs dim（包含最後 visual_dim）」，
        # VisualActorCriticRecurrent 會自己把最後 visual_dim 做 CNN embedding，再進 memory_a/c。
        self.student = VisualActorCriticRecurrent(
            num_actor_obs=num_student_obs,
            num_critic_obs=num_student_obs,   # distillation 通常不需要 critic，但給同 dim 最省事
            num_actions=num_actions,
            actor_hidden_dims=student_hidden_dims,
            critic_hidden_dims=[256, 256, 256],
            activation=activation,
            rnn_type=rnn_type,
            rnn_hidden_dim=rnn_hidden_dim,
            rnn_num_layers=rnn_num_layers,
            init_noise_std=init_noise_std,

            use_visual=use_visual,
            visual_dim=visual_dim,
            visual_latent_size=visual_latent_size,
            visual_kwargs=visual_kwargs,
            height=height,
            width=width,
            visual_channels=visual_channels,
            critic_use_visual=critic_use_visual,
        )

        # 再處理給teacher的activation_type
        activation = resolve_nn_activation(activation)

        # ---------------- teacher：維持原本設計（MLP + 可選 memory_t） ----------------
        # teacher MLP 的輸入維度：
        # - 若 teacher_recurrent=True：先 memory_t -> (.., rnn_hidden_dim) 再餵 teacher MLP
        # - 否則：直接餵 teacher_obs
        teacher_mlp_in_dim = rnn_hidden_dim if teacher_recurrent else num_teacher_obs

        teacher_layers = []
        teacher_layers.append(nn.Linear(teacher_mlp_in_dim, teacher_hidden_dims[0]))
        teacher_layers.append(activation)
        for layer_index in range(len(teacher_hidden_dims)):
            if layer_index == len(teacher_hidden_dims) - 1:
                teacher_layers.append(nn.Linear(teacher_hidden_dims[layer_index], num_actions))
            else:
                teacher_layers.append(nn.Linear(teacher_hidden_dims[layer_index], teacher_hidden_dims[layer_index + 1]))
                teacher_layers.append(activation)
        self.teacher = nn.Sequential(*teacher_layers)
        self.teacher.eval()

        if self.teacher_recurrent:
            self.memory_t = Memory(
                num_teacher_obs, type=rnn_type, num_layers=rnn_num_layers, hidden_size=rnn_hidden_dim
            )
        else:
            self.memory_t = None

        print(f"Student (VisualActorCriticRecurrent): {self.student}")
        print(f"Teacher MLP: {self.teacher}")
        if self.teacher_recurrent:
            print(f"Teacher RNN: {self.memory_t}")

    # ---------------- API：與你原本 StudentTeacherRecurrent 對齊 ----------------
    def reset(self, dones=None, hidden_states=None):
        # student 的 reset：重置 memory_a/c
        # hidden_states 若你要支援「外部指定 hidden」就要改 VisualActorCriticRecurrent.memory_a/c 的 reset API；
        # 目前先採用原本慣例：reset(dones)。
        self.student.reset(dones)

        if self.teacher_recurrent and (self.memory_t is not None):
            # 原本你的 reset 允許 hidden_states=(hs_s, hs_t)，這裡保留
            if hidden_states is None:
                hs_t = None
            else:
                hs_t = hidden_states[1]
            self.memory_t.reset(dones, hs_t)

    def act(self, observations, **kwargs):
        # VisualActorCriticRecurrent.act 支援 masks/hidden_states（kwargs 轉交）
        return self.student.act(observations, **kwargs)

    def act_inference(self, observations):
        return self.student.act_inference(observations)
    
    @property
    def action_mean(self):
        return self.student.action_mean
    
    @property
    def action_std(self):
        return self.student.action_std
    
    @property
    def entropy(self):
        return self.student.entropy()

    def evaluate(self, teacher_observations):
        if self.teacher_recurrent:
            teacher_observations = self.memory_t(teacher_observations)
            teacher_observations = teacher_observations.squeeze(0)
        # teacher 只輸出 action label
        with torch.no_grad():
            return self.teacher(teacher_observations)

    def get_hidden_states(self):
        # student：回傳 actor/critic 兩個 memory hidden（若你外部只需要 actor，也可以只回傳 [0]）
        hs_student = self.student.get_hidden_states()  # (hs_a, hs_c)
        hs_teacher = self.memory_t.hidden_states if self.teacher_recurrent else None
        return hs_student, hs_teacher

    def detach_hidden_states(self, dones=None):
        # student：detach memory_a/c
        self.student.memory_a.detach_hidden_states(dones)
        self.student.memory_c.detach_hidden_states(dones)
        if self.teacher_recurrent:
            self.memory_t.detach_hidden_states(dones)

    def load_state_dict(self, state_dict, strict=True):
        """
        兩種載入：
        A) RL checkpoint：只有 actor.* / (recurrent 時) memory_a.* → 用來載 teacher（以及 teacher 的 memory_t）
        B) distillation checkpoint：含 student.* / teacher.* → 直接整包載入
        """

        # A) RL training checkpoint：只抽 teacher actor + (可選) memory_a
        if any("actor" in k for k in state_dict.keys()):
            teacher_state_dict = {}
            for k, v in state_dict.items():
                if "actor." in k:
                    teacher_state_dict[k.replace("actor.", "")] = v
            self.teacher.load_state_dict(teacher_state_dict, strict=strict)

            if self.teacher_recurrent:
                memory_t_state_dict = {}
                for k, v in state_dict.items():
                    # 你原本就是抓 memory_a.* 當 teacher 的 memory
                    if "memory_a." in k:
                        memory_t_state_dict[k.replace("memory_a.", "")] = v
                self.memory_t.load_state_dict(memory_t_state_dict, strict=strict)

            self.loaded_teacher = True
            self.teacher.eval()
            return False

        # B) distillation checkpoint：student/teacher 都在裡面
        elif any("student" in k for k in state_dict.keys()):
            super().load_state_dict(state_dict, strict=strict)
            # set flag for successfully loading the parameters
            self.loaded_teacher = True
            self.teacher.eval()
            return True

        else:
            raise ValueError("state_dict does not contain student or teacher parameters")