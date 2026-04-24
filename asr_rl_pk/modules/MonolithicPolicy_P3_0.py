# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import warnings
import torch
import torch.nn as nn
from torch.distributions import Normal

from asr_rl_pk.networks import Memory
from asr_rl_pk.utils import resolve_nn_activation

from asr_rl_pk.modules.conv2d import Conv2dHeadModel, Conv2dBackboneModel

class monolithicpolicy_p3_0(nn.Module):
    is_recurrent = True

    def __init__(
        self,
        num_actor_obs,
        num_latent_obs,
        num_actions,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        activation="elu",
        rnn_type="lstm",
        rnn_hidden_dim=256,
        rnn_num_layers=1,
        init_noise_std=1.0,
        noise_std_type: str = "scalar",
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
                "monolithicpolicy_p3.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()
        activation = resolve_nn_activation(activation)

        # =====Visual encoder for student=====
        self.height = height
        self.width = width
        self.visual_channels = visual_channels
        self.visual_dim = height * width * visual_channels
        if num_latent_obs != self.visual_dim:
            raise ValueError(f"num_latent_obs should be equal to visual_dim (height*width*visual_channels) for monolithic policy. Got num_latent_obs={num_latent_obs}, visual_dim={self.visual_dim} (height={height}, width={width}, visual_channels={visual_channels}).")

        vk = dict(
            # channels=[64, 64],
            # kernel_sizes=[3, 3],
            # strides=[1, 1],
            # paddings=[1, 1],      # 建議補上，避免 spatial 尺寸掉太快
            # use_maxpool=True,     # 論文比較像 conv + maxpool
            channels=(32, 64, 64),
            kernel_sizes=(5, 3, 3),
            strides=(2, 2, 2),
            paddings=(2, 1, 1),
            nonlinearity=torch.nn.ELU,
            use_maxpool=True,
        )
        vk.update(visual_kwargs)

        self.visual_encoder = Conv2dBackboneModel(
            image_shape=(visual_channels, height, width),
            **vk,
        )
        # =====actor & critic will share similar struct of model layers, but have different parameters=====

        self.num_actor_obs = num_actor_obs
        
        self.memory_a = Memory(num_actor_obs+visual_latent_size, type=rnn_type, num_layers=rnn_num_layers, hidden_size=rnn_hidden_dim)
        self.memory_c = Memory(num_actor_obs+visual_latent_size, type=rnn_type, num_layers=rnn_num_layers, hidden_size=rnn_hidden_dim)

        print(f"Actor RNN: {self.memory_a}")
        print(f"Critic RNN: {self.memory_c}")

        mlp_input_dim_a = rnn_hidden_dim
        mlp_input_dim_c = rnn_hidden_dim

        # Policy
        actor_layers = []
        actor_layers.append(nn.Linear(mlp_input_dim_a, actor_hidden_dims[0]))
        actor_layers.append(activation)
        for layer_index in range(len(actor_hidden_dims)):
            if layer_index == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[layer_index], num_actions))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[layer_index], actor_hidden_dims[layer_index + 1]))
                # actor_layers.append(nn.LayerNorm(actor_hidden_dims[layer_index + 1])) # add layer normalization
                actor_layers.append(activation)
        self.actor = nn.Sequential(*actor_layers)

        # Value function
        critic_layers = []
        critic_layers.append(nn.Linear(mlp_input_dim_c, critic_hidden_dims[0]))
        critic_layers.append(activation)
        for layer_index in range(len(critic_hidden_dims)):
            if layer_index == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[layer_index], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dims[layer_index], critic_hidden_dims[layer_index + 1]))
                # critic_layers.append(nn.LayerNorm(critic_hidden_dims[layer_index + 1])) # add layer normalization
                critic_layers.append(activation)
        self.critic = nn.Sequential(*critic_layers)

        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")

        # Action noise
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


    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [
            torch.nn.init.orthogonal_(module.weight, gain=scales[idx])
            for idx, module in enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))
        ]

    def reset(self, dones=None):
        self.memory_a.reset(dones)
        self.memory_c.reset(dones)

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
        # compute mean
        mean = self.actor(observations)
        # compute standard deviation
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std).expand_as(mean)
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        # create distribution
        self.distribution = Normal(mean, std)

    def _visual_vec_to_bchw(self, visual_vec: torch.Tensor) -> torch.Tensor:
        expected = self.visual_channels * self.height * self.width
        # print("visual_vec.shape =", visual_vec.shape)
        # print("expected visual dim =", expected)

        if visual_vec.shape[-1] != expected:
            raise RuntimeError(
                f"visual_dim mismatch: got {visual_vec.shape[-1]}, expected {expected} = "
                f"{self.visual_channels}*{self.height}*{self.width}"
            )

        if visual_vec.dim() == 2:
            # (B, D) -> (B, C, H, W)
            return visual_vec.view(visual_vec.shape[0], self.visual_channels, self.height, self.width)

        elif visual_vec.dim() == 3:
            # (T, B, D) -> (T*B, C, H, W)
            T, B, _ = visual_vec.shape
            return visual_vec.contiguous().view(T * B, self.visual_channels, self.height, self.width)

        else:
            raise RuntimeError(f"Unsupported visual_vec shape: {tuple(visual_vec.shape)}")

    def latent_forward_a(self, latent_obs):
        if latent_obs.dim() == 2:
            bchw = self._visual_vec_to_bchw(latent_obs)
            latent_flat = self.visual_encoder(bchw)   # (B, F)
            return latent_flat

        elif latent_obs.dim() == 3:
            T, B, _ = latent_obs.shape
            bchw = self._visual_vec_to_bchw(latent_obs)   # (T*B, C, H, W)
            latent_flat = self.visual_encoder(bchw)       # (T*B, F)
            latent_flat = latent_flat.view(T, B, -1)      # back to (T, B, F)
            return latent_flat

        else:
            raise RuntimeError(f"Unsupported latent_obs shape: {tuple(latent_obs.shape)}")

    def latent_forward_c(self, latent_obs):
        if latent_obs.dim() == 2:
            bchw = self._visual_vec_to_bchw(latent_obs)
            latent_flat = self.visual_encoder(bchw)   # (B, F)
            return latent_flat

        elif latent_obs.dim() == 3:
            T, B, _ = latent_obs.shape
            bchw = self._visual_vec_to_bchw(latent_obs)   # (T*B, C, H, W)
            latent_flat = self.visual_encoder(bchw)       # (T*B, F)
            latent_flat = latent_flat.view(T, B, -1)      # back to (T, B, F)
            return latent_flat

        else:
            raise RuntimeError(f"Unsupported latent_obs shape: {tuple(latent_obs.shape)}")

    # def autoencode_a(self, latent_obs):
    #     encode = self.latent_forward_a(latent_obs)
    #     return self.decode_latent_mlp_a(encode)

    # def autoencode_c(self, latent_obs):
    #     encode = self.latent_forward_a(latent_obs)
    #     return self.decode_latent_mlp_a(encode)
    #     # encode = self.latent_forward_c(latent_obs)
    #     # return self.decode_latent_mlp_c(encode)

    def act(self, observations, latent_obs, masks=None, hidden_states=None):
        # print("observations.shape =", observations.shape)
        # print("latent_obs.shape =", latent_obs.shape)
        cat_observations = torch.cat([observations, self.latent_forward_a(latent_obs)], dim=-1)
        input_a = self.memory_a(cat_observations, masks, hidden_states)
        self.update_distribution(input_a.squeeze(0))
        return self.distribution.sample()
    
    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations, latent_obs):
        cat_observations = torch.cat([observations, self.latent_forward_a(latent_obs)], dim=-1)
        input_a = self.memory_a(cat_observations)
        actions_mean = self.actor(input_a.squeeze(0))
        return actions_mean

    def evaluate(self, observations, latent_obs, masks=None, hidden_states=None):
        cat_observations = torch.cat([observations, self.latent_forward_c(latent_obs)], dim=-1)
        input_c = self.memory_c(cat_observations, masks, hidden_states)
        value = self.critic(input_c.squeeze(0))
        return value

    def get_hidden_states(self):
        return self.memory_a.hidden_states, self.memory_c.hidden_states
    
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
        # check if state_dict contains TeacherStudent or RL
        if any("actor" in key for key in state_dict.keys()):  # loading parameters from rl training
            super().load_state_dict(state_dict, strict=strict)
            return True
        elif any("student" in key for key in state_dict.keys()):  # loading parameters from distillation training to finetune
            print("\n===== P3 finetune ... =====\n")
            # rename keys to match actor and remove teacher parameters
            # 1) visual encoder
            visual_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith("visual_encoder."):
                    new_key = key.replace("visual_encoder.", "", 1)
                    visual_state_dict[new_key] = value
            if visual_state_dict:
                self.visual_encoder.load_state_dict(visual_state_dict, strict=strict)
            else:
                raise ValueError("No visual encoder parameters found in state_dict. Cannot load visual encoder weights.")
            # 2) recurrent memory: memory_s -> memory_a
            memory_a_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith("memory_s."):
                    new_key = key.replace("memory_s.", "", 1)
                    memory_a_state_dict[new_key] = value
            if memory_a_state_dict:
                self.memory_a.load_state_dict(memory_a_state_dict, strict=strict)
            else:
                raise ValueError("No recurrent memory parameters found in state_dict. Cannot load recurrent memory weights.")
            # 3) policy mlp: student -> actor
            actor_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith("student."):
                    new_key = key.replace("student.", "", 1)
                    actor_state_dict[new_key] = value
            if actor_state_dict:
                self.actor.load_state_dict(actor_state_dict, strict=strict)
            else:
                raise ValueError("No policy parameters found in state_dict. Cannot load policy weights.")
            #     if "actor." in key:
            #         actor_state_dict[key.replace("student.", "")] = value
            # self.actor.load_state_dict(actor_state_dict, strict=strict)
            # also load recurrent memory if teacher is recurrent
            return False
        else:
            raise ValueError("state_dict does not contain student or teacher parameters")
        
        