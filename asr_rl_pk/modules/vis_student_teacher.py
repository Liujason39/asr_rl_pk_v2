# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal

from asr_rl_pk.utils import resolve_nn_activation
from asr_rl_pk.modules.visual_actor_critic import VisualActorCritic


class VisStudentTeacher(nn.Module):
    is_recurrent = False

    def __init__(
        self,
        num_student_obs,
        num_teacher_obs,
        num_actions,
        student_hidden_dims=[256, 256, 256],
        teacher_hidden_dims=[256, 256, 256],
        activation="elu",
        init_noise_std=0.1,
        noise_std_type: str = "scalar",
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
        **kwargs,
    ):
        if kwargs:
            print(
                "StudentTeacher.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()
        self.loaded_teacher = False

        # ---------------- student (改成 VisualActorCritic) ----------------
        # 注意：VisualActorCritic 需要 num_actor_obs/num_critic_obs
        # distillation 裡你只用 actor 的話，num_critic_obs 給同樣維度即可（或最小化 critic 結構也行）
        self.student = VisualActorCritic(
            num_actor_obs=num_student_obs,
            num_critic_obs=num_student_obs,   # 不用 critic 也沒差，evaluate 不會叫 student.critic
            num_actions=num_actions,
            actor_hidden_dims=student_hidden_dims,
            critic_hidden_dims=[256, 256, 256],  # 你也可以給 [1] / [64] 之類省參數，但不影響 actor
            activation=activation,
            init_noise_std=init_noise_std,
            noise_std_type=noise_std_type,

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
        # ---------------- teacher (維持 MLP) ----------------
        teacher_layers = []
        teacher_layers.append(nn.Linear(num_teacher_obs, teacher_hidden_dims[0]))
        teacher_layers.append(activation)
        for layer_index in range(len(teacher_hidden_dims)):
            if layer_index == len(teacher_hidden_dims) - 1:
                teacher_layers.append(nn.Linear(teacher_hidden_dims[layer_index], num_actions))
            else:
                teacher_layers.append(nn.Linear(teacher_hidden_dims[layer_index], teacher_hidden_dims[layer_index + 1]))
                teacher_layers.append(activation)
        self.teacher = nn.Sequential(*teacher_layers)
        self.teacher.eval()

        print(f"Student MLP: {self.student}")
        print(f"Teacher MLP: {self.teacher}")

    def reset(self, dones=None, hidden_states=None):
        # student 如果未來支援 recurrent，就在這裡傳 dones；目前 is_recurrent=False 所以先 pass
        pass

    def forward(self):
        raise NotImplementedError

    # --------- distribution 相關：全部 delegate 給 VisualActorCritic ---------
    @property
    def distribution(self):
        return self.student.distribution

    @property
    def action_mean(self):
        return self.student.action_mean

    @property
    def action_std(self):
        return self.student.action_std

    @property
    def entropy(self):
        return self.student.entropy

    def update_distribution(self, observations):
        # 讓 VisualActorCritic 自己做 visual embedding + actor forward + distribution
        self.student.update_distribution(observations)

    def act(self, observations):
        return self.student.act(observations)

    def act_inference(self, observations):
        return self.student.act_inference(observations)

    def evaluate(self, teacher_observations):
        # distillation: teacher 只出 action mean
        with torch.no_grad():
            actions = self.teacher(teacher_observations)
        return actions

    def load_state_dict(self, state_dict, strict=True):
        """
        支援兩種載入：
        A) 從 rl training checkpoint 載 teacher（有 actor. 前綴）
        B) 從 distillation checkpoint 載整個 StudentTeacher（會有 student.xxx / teacher.xxx）
        """
        if any("actor" in key for key in state_dict.keys()):
            # ---- A) loading parameters from rl training: 只抽 actor.* 給 teacher ----
            teacher_state_dict = {}
            for key, value in state_dict.items():
                if "actor." in key:
                    teacher_state_dict[key.replace("actor.", "")] = value
            self.teacher.load_state_dict(teacher_state_dict, strict=strict)
            self.loaded_teacher = True
            self.teacher.eval()
            return False

        elif any("student" in key for key in state_dict.keys()):
            # ---- B) distillation checkpoint: student 是 VisualActorCritic，keys 會長這樣：
            # student.actor..., student.visual_encoder..., student.std/log_std..., teacher....
            super().load_state_dict(state_dict, strict=strict)
            # set flag for successfully loading the parameters
            self.loaded_teacher = True
            self.teacher.eval()
            return True

        else:
            raise ValueError("state_dict does not contain student or teacher parameters")

    def get_hidden_states(self):
        return None

    def detach_hidden_states(self, dones=None):
        pass