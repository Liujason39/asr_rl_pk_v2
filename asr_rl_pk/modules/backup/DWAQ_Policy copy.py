# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import warnings
import torch
import torch.nn as nn
from torch.distributions import Normal

from asr_rl_pk.networks import Memory, TemporalBuffer
from asr_rl_pk.utils import resolve_nn_activation

from .MixerMLP import MixerMLP


class dwaq_policy(nn.Module):
    is_recurrent = False

    def __init__(
        self,
        num_actor_obs,
        num_privileged_obs,
        num_hm_obs,
        num_true_vel_obs,
        num_actions,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        proprio_hidden_dims = [256, 256, 32],
        vel_head_hidden_dims = [256, 256, 3],
        est_hidden_dims = [256], # follow with obs_t dims
        activation="elu",
        init_noise_std=1.0,
        noise_std_type: str = "log", #"scalar",
        history_length=5,
        **kwargs,
    ):
        if kwargs:
            print(
                "dwaq_policy.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()
        activation = resolve_nn_activation(activation)

        self.history_len = history_length
        
        # self.zp_hist_buffer = TemporalBuffer(history_length)        # 存 z_p

        # Value function
        critic_layers = []
        critic_layers.append(nn.Linear(num_privileged_obs+num_hm_obs, critic_hidden_dims[0]))
        critic_layers.append(activation)
        for layer_index in range(len(critic_hidden_dims)):
            if layer_index == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[layer_index], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dims[layer_index], critic_hidden_dims[layer_index + 1]))
                # critic_layers.append(nn.LayerNorm(critic_hidden_dims[layer_index + 1])) # add layer normalization
                critic_layers.append(activation)
        self.critic = nn.Sequential(*critic_layers)
        print(f"Critic MLP: {self.critic}")

        # =====proprio_enocder : process prorpio observations to latent for Mixer=====
        self.proprio_hist_buffer = TemporalBuffer(history_length)   # 存 proprio + true_vel/fake_vel
        layers = []
        in_dim = (num_actor_obs) # set H_t = 5 for history stacking
        self.proprio_encoder = MixerMLP(
            num_tokens=in_dim,
            num_channels=history_length,
            hidden_dim=proprio_hidden_dims[0],
            num_layers=1,
            out_dim=proprio_hidden_dims[-1],
        )

        # VAE heads for z_p
        self.proprio_encoder_mu = nn.Linear(proprio_hidden_dims[-1], proprio_hidden_dims[-1])
        self.proprio_encoder_logvar = nn.Linear(proprio_hidden_dims[-1], proprio_hidden_dims[-1])
        print(f"proprio_encoder: {self.proprio_encoder}")
        print(f"proprio_encoder_mu: {self.proprio_encoder_mu}")
        print(f"proprio_encoder_logvar: {self.proprio_encoder_logvar}")

        # =====proprio_est_decoder : process visual observations to latent for Mixer=====
        layers = []
        in_dim = proprio_hidden_dims[-1] # follow with proprio_encoder output dims 
        for i, out_dim in enumerate(est_hidden_dims):
            if i == len(est_hidden_dims) - 1:   # 最後輸出替換成actor_obs
                out_dim = num_actor_obs
            layers.append(nn.Linear(in_dim, out_dim))
            if i != len(est_hidden_dims) - 1:   # 最後一層不加 activation
                layers.append(activation)
            in_dim = out_dim
        self.proprio_est_decoder = nn.Sequential(*layers)

        print(f"proprio_est_decoder: {self.proprio_est_decoder}")

        # =====vel_head : process proprio latent vector to est_vel=====
        layers = []
        in_dim = proprio_hidden_dims[-1] # concat proprio enocder and visual latent as input
        for i, out_dim in enumerate(vel_head_hidden_dims):
            layers.append(nn.Linear(in_dim, out_dim))
            if i != len(vel_head_hidden_dims) - 1:   # 最後一層不加 activation
                layers.append(activation)
            in_dim = out_dim
        self.vel_head = nn.Sequential(*layers)
        # # VAE heads for z_p
        # self.vel_encoder_mu = nn.Linear(vel_head_hidden_dims[-1], vel_head_hidden_dims[-1])
        # self.vel_encoder_logvar = nn.Linear(vel_head_hidden_dims[-1], vel_head_hidden_dims[-1])

        print(f"vel_head: {self.vel_head}")


        # Policy
        actor_layers = []
        actor_layers.append(nn.Linear(num_true_vel_obs+proprio_hidden_dims[-1]+num_actor_obs, actor_hidden_dims[0]))
        actor_layers.append(activation)
        for layer_index in range(len(actor_hidden_dims)):
            if layer_index == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[layer_index], num_actions))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[layer_index], actor_hidden_dims[layer_index + 1]))
                # actor_layers.append(nn.LayerNorm(actor_hidden_dims[layer_index + 1])) # add layer normalization
                actor_layers.append(activation)
        self.actor = nn.Sequential(*actor_layers)

        print(f"Actor MLP: {self.actor}")

        # Action noise
        self.noise_std_type = noise_std_type
        if self.noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif self.noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
            print("Using log scale noise_std_type")
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

    # def kl_divergence_diag_gaussian(self, mu, logvar):
    #     # q(z|x) = N(mu, diag(sigma^2))
    #     # p(z)   = N(0, I)
    #     kl = 0.5 * torch.sum(
    #         torch.exp(logvar) + mu.pow(2) - 1.0 - logvar,
    #         dim=-1
    #     )
    #     return kl.mean()
    
    def forward_proprio_vae(self, proprio_obs_seq):
        """
        proprio_obs_seq: [B, H, proprio_obs_dim]
        """

        proprio_encoder_feat = self.proprio_encoder(proprio_obs_seq)   # [B, H, mixer_latent_dim]

        mu = self.proprio_encoder_mu(proprio_encoder_feat)
        logvar = self.proprio_encoder_logvar(proprio_encoder_feat)
        z_pH, clamp_logvar = self.reparameterize(mu, logvar)

        pred_obs = self.proprio_est_decoder(z_pH)
        return z_pH, mu, clamp_logvar, pred_obs

    # def reparameterize(self, mu, logvar, logvar_max=5):
    #     # 對應論文 constrained reparameterization 的簡化實作
    #     # sigma_max = 5 
    #     clamp_logvar = torch.clamp(logvar, max=logvar_max)
    #     std = torch.exp(0.5 * clamp_logvar)
    #     eps = torch.randn_like(std)
    #     z = mu + eps * std
    #     return z, clamp_logvar
    
    def reparameterize(self, mu, logvar, logvar_min=-10.0, logvar_max=1.609): # log(5) = 1.609
        clamp_logvar = torch.clamp(logvar, min=logvar_min, max=logvar_max)
        std = torch.exp(0.5 * clamp_logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z, clamp_logvar

    def compute_estimated_velocity(self, z_p):
        vel_code = self.vel_head(z_p)
        # vel_encoder_mu = self.vel_encoder_mu(vel_code)
        # vel_encoder_logvar = self.vel_encoder_logvar(vel_code)
        # z, clamp_logvar = self.reparameterize(vel_encoder_mu, vel_encoder_logvar)  # for KL loss computation, not used for action generation
        # return z
        return vel_code

    def reset(self, dones=None):
        self.proprio_hist_buffer.reset(dones)
        # self.zp_hist_buffer.reset(dones)

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
        if not torch.isfinite(mean).all():
            raise RuntimeError("Actor mean became non-finite.")
        # compute standard deviation
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        elif self.noise_std_type == "log":
            # std = torch.exp(self.log_std).expand_as(mean)
            log_std = torch.clamp(self.log_std, min=-5.0, max=2.0)
            std = torch.exp(log_std).expand_as(mean)
            if not torch.isfinite(std).all():
                raise RuntimeError("Action std became non-finite.")
            if (std <= 0).any():
                raise RuntimeError("Action std became non-positive.")

        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        # create distribution
        self.distribution = Normal(mean, std)

    # def latent_forward_a(self, latent_obs):
    #     return self.latent_mlp_a(latent_obs)

    # def latent_forward_c(self, latent_obs):
    #     return self.latent_mlp_a(latent_obs)
    #     # return self.latent_mlp_c(latent_obs)

    # def autoencode_a(self, latent_obs):
    #     encode = self.latent_forward_a(latent_obs)
    #     return self.decode_latent_mlp_a(encode)

    # def autoencode_c(self, latent_obs):
    #     encode = self.latent_forward_a(latent_obs)
    #     return self.decode_latent_mlp_a(encode)
    #     # encode = self.latent_forward_c(latent_obs)
    #     # return self.decode_latent_mlp_c(encode)

    # def act(self, prorpio, true_V, **kwargs):
    #     cat_observations = torch.cat([observations, self.latent_forward_a(latent_obs)], dim=-1)
    #     input_a = self.memory_a(cat_observations, masks, hidden_states)
    #     self.update_distribution(input_a.squeeze(0))
    #     return self.distribution.sample()
    
    def act(self, actor_obs):
        """this will be cover by self.prepare_actor_obs()"""
        self.update_distribution(actor_obs)
        return self.distribution.sample()
    
    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    # def act_inference(self, actor_obs):
    #     """this will be cover by self.prepare_actor_obs()"""
    #     return self.actor(actor_obs)
    @torch.no_grad()
    def act_inference(self, obs, return_aux=True):
        """
        Inference-time action generation using only current observation.
        Internal history buffer will be updated automatically.

        Args:
            obs: Tensor of shape [B, num_actor_obs]
            return_aux: whether to also return estimated auxiliary outputs

        Returns:
            action: Tensor of shape [B, num_actions]
            optionally a dict containing est_vel / mu_p / obs_hist / actor_obs
        """
        actor_obs, obs_history = self.prepare_actor_obs(obs)
        # 2) proprio VAE
        z_p, mu_p, logvar_p, pred_obs = self.forward_proprio_vae(obs_history)

        # 3) velocity estimation
        # est_vel = self.vel_head(z_p)
        est_vel = self.compute_estimated_velocity(z_p)

        # # update temporal history
        # proprio_hist = self.proprio_hist_buffer.append(proprio_obs)

        # # encode history
        # proprio_encoder_feat = self.proprio_encoder(proprio_hist)
        # mu_p = self.proprio_encoder_mu(proprio_encoder_feat)

        # # estimate velocity from mean latent
        # est_vel = self.vel_head(mu_p)

        # # build actor input
        # actor_obs = torch.cat([est_vel, proprio_obs, mu_p], dim=-1)

        # deterministic action output
        action = self.actor(actor_obs)

        if return_aux:
            return action, {
                "est_vel": est_vel,
            }
        return action

    def evaluate(self, privileged_obs, gt_hm=None, **kwargs):
        # hm_latent = self.hm_encoder(gt_hm)
        # privileged_latent = self.privileged_encoder(privileged_obs)
        if gt_hm is None:
            cat_latents = privileged_obs
        else:
            cat_latents = torch.cat([privileged_obs, gt_hm], dim=-1)
        value = self.critic(cat_latents)
        return value

    def get_hidden_states(self):
        raise NotImplementedError
        # return self.memory_a.hidden_states, self.memory_c.hidden_states

    # def prepare_actor_obs(self, actor_obs, true_velocity_obs, p_boot: float = 0.0, inference: bool = False):
    #     """
    #     obs_dict should provide:
    #     - proprio_obs: [B, num_actor_obs]
    #     - true_vel_obs: [B, num_true_vel_obs]
    #     """

    #     proprio_obs = actor_obs
    #     true_vel_obs = true_velocity_obs

    #     # 1) 更新 proprio history
    #     proprio_hist = self.proprio_hist_buffer.append(proprio_obs)

    #     # 2) proprio VAE
    #     z_p, mu_p, logvar_p, pred_obs = self.forward_proprio_vae(proprio_hist)

    #     # 3) velocity estimation
    #     v_est = self.vel_head(mu_p)

    #     # 4) bootstrap mixing
    #     if inference:
    #         vel_for_actor = v_est
    #         boot_mask = torch.ones_like(v_est[:, :1])
    #     else:
    #         B = true_vel_obs.shape[0]
    #         boot_mask = (torch.rand(B, 1, device=true_vel_obs.device) < p_boot).float()
    #         vel_for_actor = boot_mask * v_est + (1.0 - boot_mask) * true_vel_obs

    #     # 8) final actor obs
    #     actor_obs = torch.cat([vel_for_actor, proprio_obs, mu_p], dim=-1)

    #     return actor_obs, proprio_hist, mu_p, vel_for_actor
    def prepare_actor_obs(self, actor_obs):
        """
        obs_dict should provide:
        - proprio_obs: [B, num_actor_obs]
        - true_vel_obs: [B, num_true_vel_obs]
        """

        proprio_obs = actor_obs

        # 1) 更新 proprio history
        proprio_hist = self.proprio_hist_buffer.append(proprio_obs)

        # 2) proprio VAE
        z_p, mu_p, logvar_p, pred_obs = self.forward_proprio_vae(proprio_hist)

        # 3) velocity estimation
        # est_vel = self.vel_head(z_p)
        est_vel = self.compute_estimated_velocity(z_p)

        # 8) final actor obs
        actor_obs = torch.cat([est_vel, proprio_obs, z_p], dim=-1)

        return actor_obs, proprio_hist
    
    # def act_in_update(self, vel_for_actor, obs, obs_hist, **kwargs):
    #     z_pH, mu_p, logvar_p, pred_obs = self.forward_proprio_vae(obs_hist)
    #     actor_obs = torch.cat([vel_for_actor, obs, mu_p], dim=-1)
    #     self.update_distribution(actor_obs)
    #     return self.distribution.sample(), mu_p, z_pH, pred_obs, logvar_p
    # def act_in_update(self, vel_for_actor, obs, mu_p, **kwargs):
    #     actor_obs = torch.cat([vel_for_actor, obs, mu_p], dim=-1)
    #     self.update_distribution(actor_obs)
    #     return self.distribution.sample()
    def act_in_update(self, obs, obs_history, **kwargs):
        z_p, mu_p, logvar_p, pred_obs = self.forward_proprio_vae(obs_history)
        # est_vel = self.vel_head(z_p)
        est_vel = self.compute_estimated_velocity(z_p)
        actor_obs = torch.cat([est_vel, obs, z_p], dim=-1)
        self.update_distribution(actor_obs)
        # return self.distribution.sample()
        return z_p, mu_p, logvar_p, pred_obs, est_vel
    
    # def act_for_onnx_transfer(self, obs, obs_hist, **kwargs):
    #     z_p, mu_p, logvar_p, pred_obs = self.forward_proprio_vae(obs_hist)
    #     est_vel = self.vel_head(mu_p) # use mean for velocity estimation during ONNX transfer
    #     actor_obs = torch.cat([est_vel, obs, mu_p], dim=-1)
        
    #     return self.actor(actor_obs)
    def act_for_onnx_transfer(self, obs, obs_history, **kwargs):
        z_p, mu_p, logvar_p, pred_obs = self.forward_proprio_vae(obs_history)
        # est_vel = self.vel_head(z_p)
        est_vel = self.compute_estimated_velocity(z_p)
        actor_obs = torch.cat([est_vel, obs, z_p], dim=-1)
        
        return self.actor(actor_obs)
    
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
        