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


class monolithicpolicy_p1(nn.Module):
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
        mlp_hidden_dims=[256,128],
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
                "monolithicpolicy_p1.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()
        activation = resolve_nn_activation(activation)

        # =====latent MLP : process privileged observations to latent for actor and critic=====
        latent_mlp_layers_a = []
        latent_mlp_layers_a.append(nn.Linear(num_latent_obs, mlp_hidden_dims[0]))
        latent_mlp_layers_a.append(activation)
        for layer_index in range(len(mlp_hidden_dims) - 1):
            latent_mlp_layers_a.append(nn.Linear(mlp_hidden_dims[layer_index], mlp_hidden_dims[layer_index + 1]))
            # latent_mlp_layers_a.append(nn.LayerNorm(actor_hidden_dims[layer_index + 1])) # add layer normalization
            latent_mlp_layers_a.append(activation)
        self.latent_mlp_a = nn.Sequential(*latent_mlp_layers_a)

        # latent_mlp_layers_c = []
        # latent_mlp_layers_c.append(nn.Linear(num_latent_obs, mlp_hidden_dims[0]))
        # latent_mlp_layers_c.append(activation)
        # for layer_index in range(len(mlp_hidden_dims) - 1):
        #     latent_mlp_layers_c.append(nn.Linear(mlp_hidden_dims[layer_index], mlp_hidden_dims[layer_index + 1]))
        #     # latent_mlp_layers_c.append(nn.LayerNorm(critic_hidden_dims[layer_index + 1])) # add layer normalization
        #     latent_mlp_layers_c.append(activation)
        # self.latent_mlp_c = nn.Sequential(*latent_mlp_layers_c)

        print(f"Latent MLP a: {self.latent_mlp_a}")
        # print(f"Latent MLP c: {self.latent_mlp_c}")

        # =====actor & critic will share similar struct of model layers, but have different parameters=====
        self.memory_a = Memory(num_actor_obs+mlp_hidden_dims[-1], type=rnn_type, num_layers=rnn_num_layers, hidden_size=rnn_hidden_dim)
        self.memory_c = Memory(num_actor_obs+mlp_hidden_dims[-1], type=rnn_type, num_layers=rnn_num_layers, hidden_size=rnn_hidden_dim)

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

        # ===== Autoencoder decode MLP =====
        decode_latent_mlp_layers_a = []

        # hidden dims 反向走
        reversed_dims = mlp_hidden_dims[::-1]   # e.g. [64, 128, 256]

        for layer_index in range(len(reversed_dims) - 1):
            decode_latent_mlp_layers_a.append(
                nn.Linear(reversed_dims[layer_index], reversed_dims[layer_index + 1])
            )
            # decode_latent_mlp_layers_a.append(nn.LayerNorm(reversed_dims[layer_index + 1]))
            decode_latent_mlp_layers_a.append(activation)

        # 最後輸出回原始 observation 維度
        decode_latent_mlp_layers_a.append(nn.Linear(reversed_dims[-1], num_latent_obs))

        self.decode_latent_mlp_a = nn.Sequential(*decode_latent_mlp_layers_a)

        # decode_latent_mlp_layers_c = []

        # # hidden dims 反向走
        # reversed_dims = mlp_hidden_dims[::-1]   # e.g. [64, 128, 256]

        # for layer_index in range(len(reversed_dims) - 1):
        #     decode_latent_mlp_layers_c.append(
        #         nn.Linear(reversed_dims[layer_index], reversed_dims[layer_index + 1])
        #     )
        #     # decode_latent_mlp_layers_c.append(nn.LayerNorm(reversed_dims[layer_index + 1]))
        #     decode_latent_mlp_layers_c.append(activation)

        # # 最後輸出回原始 observation 維度
        # decode_latent_mlp_layers_c.append(nn.Linear(reversed_dims[-1], num_latent_obs))

        # self.decode_latent_mlp_c = nn.Sequential(*decode_latent_mlp_layers_c)

        print(f"Decode MLP a: {self.decode_latent_mlp_a}")
        # print(f"Decode MLP c: {self.decode_latent_mlp_c}")

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

    def latent_forward_a(self, latent_obs):
        return self.latent_mlp_a(latent_obs)

    def latent_forward_c(self, latent_obs):
        return self.latent_mlp_a(latent_obs)
        # return self.latent_mlp_c(latent_obs)

    def autoencode_a(self, latent_obs):
        encode = self.latent_forward_a(latent_obs)
        return self.decode_latent_mlp_a(encode)

    def autoencode_c(self, latent_obs):
        encode = self.latent_forward_a(latent_obs)
        return self.decode_latent_mlp_a(encode)
        # encode = self.latent_forward_c(latent_obs)
        # return self.decode_latent_mlp_c(encode)

    def act(self, observations, latent_obs, masks=None, hidden_states=None):
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
        super().load_state_dict(state_dict, strict=strict)
        return True
        