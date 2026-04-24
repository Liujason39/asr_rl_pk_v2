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


class dwaqpp_policy(nn.Module):
    is_recurrent = False

    def __init__(
        self,
        num_actor_obs,
        num_privileged_obs,
        num_hm_obs,
        num_true_vel_obs,
        num_visual_obs,
        num_actions,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        hm_hidden_dims = [128,64],
        privileged_hidden_dims = [128,64],
        proprio_hidden_dims = [256, 256, 32],
        vel_head_hidden_dims = [256, 256, 3],
        visual_hidden_dims = [256, 256, 32],
        est_hidden_dims = [256], # follow with obs_t dims
        hm_decoder_hidden_dims = [256, 128], # follow with gt_heightmap dims
        mixer_hidden_dims = [256, 256, 64],
        activation="elu",
        init_noise_std=1.0,
        noise_std_type: str = "scalar",
        **kwargs,
    ):
        if kwargs:
            print(
                "dwaq_policy.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()
        activation = resolve_nn_activation(activation)

        H = 5 # history stacking length
        self.history_len = H
        self.proprio_hist_buffer = TemporalBuffer(H)   # 存 proprio + true_vel
        self.zp_hist_buffer = TemporalBuffer(H)        # 存 z_p
        self.ze_hist_buffer = TemporalBuffer(H)        # 存 z_e

        # =====heighmap_encoder : process privileged observations to latent for critic=====
        layers = []
        in_dim = num_hm_obs
        for i, out_dim in enumerate(hm_hidden_dims):
            layers.append(nn.Linear(in_dim, out_dim))
            if i != len(hm_hidden_dims) - 1:   # 最後一層不加 activation
                layers.append(activation)
            in_dim = out_dim
        self.hm_encoder = nn.Sequential(*layers)

        print(f"hm_encoder: {self.hm_encoder}")

        # =====privileged_enocder : process privileged observations to latent for critic=====
        layers = []
        in_dim = num_privileged_obs
        for i, out_dim in enumerate(privileged_hidden_dims):
            layers.append(nn.Linear(in_dim, out_dim))
            if i != len(privileged_hidden_dims) - 1:   # 最後一層不加 activation
                layers.append(activation)
            in_dim = out_dim
        self.privileged_encoder = nn.Sequential(*layers)

        print(f"privileged_encoder: {self.privileged_encoder}")

        # Value function
        critic_layers = []
        critic_layers.append(nn.Linear(hm_hidden_dims[-1]+privileged_hidden_dims[-1], critic_hidden_dims[0]))
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
        layers = []
        in_dim = (num_actor_obs+num_true_vel_obs) # set H_t = 5 for history stacking
        self.proprio_encoder = MixerMLP(
            num_tokens=in_dim,
            num_channels=H,
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

        # =====visual_enocder : process visual observations to latent for Mixer=====
        layers = []
        in_dim = num_visual_obs 
        for i, out_dim in enumerate(visual_hidden_dims):
            layers.append(nn.Linear(in_dim, out_dim))
            if i != len(visual_hidden_dims) - 1:   # 最後一層不加 activation
                layers.append(activation)
            in_dim = out_dim
        self.visual_encoder = nn.Sequential(*layers)

        print(f"visual_encoder: {self.visual_encoder}")

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

        # =====model_mixer : process 2 latent vector to latent for Actor=====
        layers = []
        H = 5 # history stacking length
        in_dim = proprio_hidden_dims[-1] + visual_hidden_dims[-1] # concat proprio enocder and visual latent as input
        self.model_mixer = MixerMLP(
            num_tokens=in_dim,
            num_channels=H,
            hidden_dim=mixer_hidden_dims[0],
            num_layers=1,
            out_dim=mixer_hidden_dims[-1],
        )

        # VAE heads for z_pe
        self.model_mixer_mu = nn.Linear(mixer_hidden_dims[-1], mixer_hidden_dims[-1])
        self.model_mixer_logvar = nn.Linear(mixer_hidden_dims[-1], mixer_hidden_dims[-1])
        print(f"model_mixer: {self.model_mixer}")
        print(f"model_mixer_mu: {self.model_mixer_mu}")
        print(f"model_mixer_logvar: {self.model_mixer_logvar}")

        # =====hm_decoder : process visual observations to latent for Mixer=====
        layers = []
        in_dim = mixer_hidden_dims[-1] # follow with proprio_encoder output dims 
        for i, out_dim in enumerate(hm_decoder_hidden_dims):
            if i == len(hm_decoder_hidden_dims) - 1:   # 最後輸出替換成num_hm_obs
                out_dim = num_hm_obs
            layers.append(nn.Linear(in_dim, out_dim))
            if i != len(hm_decoder_hidden_dims) - 1:   # 最後一層不加 activation
                layers.append(activation)
            in_dim = out_dim
        self.hm_decoder = nn.Sequential(*layers)

        print(f"hm_decoder: {self.hm_decoder}")
        # =====vel_head : process proprio latent vector to est_vel=====
        layers = []
        in_dim = proprio_hidden_dims[-1] # concat proprio enocder and visual latent as input
        for i, out_dim in enumerate(vel_head_hidden_dims):
            layers.append(nn.Linear(in_dim, out_dim))
            if i != len(vel_head_hidden_dims) - 1:   # 最後一層不加 activation
                layers.append(activation)
            in_dim = out_dim
        self.vel_head = nn.Sequential(*layers)

        print(f"vel_head: {self.vel_head}")


        # Policy
        actor_layers = []
        actor_layers.append(nn.Linear(num_true_vel_obs+num_actor_obs+mixer_hidden_dims[-1], actor_hidden_dims[0]))
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

    def kl_divergence_diag_gaussian(self, mu, logvar):
        # q(z|x) = N(mu, diag(sigma^2))
        # p(z)   = N(0, I)
        kl = 0.5 * torch.sum(
            torch.exp(logvar) + mu.pow(2) - 1.0 - logvar,
            dim=-1
        )
        return kl.mean()
    
    def forward_mixer_vae(self, proprio_latent_seq, visual_latent_seq):
        """
        proprio_latent_seq: [B, H, Dp]
        visual_latent_seq:  [B, H, Dv]
        """
        mixer_input = torch.cat([proprio_latent_seq, visual_latent_seq], dim=-1)  # [B, H, Dp+Dv]

        mixer_feat = self.model_mixer(mixer_input)   # [B, mixer_latent_dim]

        mu = self.model_mixer_mu(mixer_feat)
        logvar = self.model_mixer_logvar(mixer_feat)
        z_pe, clamp_logvar = self.reparameterize(mu, logvar)

        pred_hm = self.hm_decoder(z_pe)
        return z_pe, mu, clamp_logvar, pred_hm
    
    def forward_proprio_vae(self, proprio_obs_seq):
        """
        proprio_obs_seq: [B, H, proprio_obs_dim+vel_obs_dim]
        """

        proprio_encoder_feat = self.proprio_encoder(proprio_obs_seq)   # [B, mixer_latent_dim]

        mu = self.proprio_encoder_mu(proprio_encoder_feat)
        logvar = self.proprio_encoder_logvar(proprio_encoder_feat)
        z_pH, clamp_logvar = self.reparameterize(mu, logvar)

        pred_obs = self.proprio_est_decoder(z_pH)
        return z_pH, mu, clamp_logvar, pred_obs

    def reparameterize(self, mu, logvar, logvar_min=0.0, logvar_max=5):
        # 對應論文 constrained reparameterization 的簡化實作
        # sigma_max = 5 
        clamp_logvar = torch.clamp(logvar, min=logvar_min, max=logvar_max)
        std = torch.exp(0.5 * clamp_logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z, clamp_logvar

    def reset(self, dones=None):
        self.proprio_hist_buffer.reset(dones)
        self.zp_hist_buffer.reset(dones)
        self.ze_hist_buffer.reset(dones)

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
    
    def act(self, actor_obs, **kwargs):
        """this will be cover by self.prepare_actor_obs()"""
        self.update_distribution(actor_obs)
        return self.distribution.sample()
    
    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, actor_obs):
        """this will be cover by self.prepare_actor_obs()"""
        return self.actor(actor_obs)

    def evaluate(self, gt_hm, privileged_obs, **kwargs):
        hm_latent = self.hm_encoder(gt_hm)
        privileged_latent = self.privileged_encoder(privileged_obs)
        cat_latents = torch.cat([hm_latent, privileged_latent], dim=-1)
        value = self.critic(cat_latents)
        return value, hm_latent, privileged_latent

    def get_hidden_states(self):
        raise NotImplementedError
        # return self.memory_a.hidden_states, self.memory_c.hidden_states

    def prepare_actor_obs(self, obs_dict, p_boot: float = 0.0, inference: bool = False):
        """
        obs_dict should provide:
        - proprio_obs: [B, num_actor_obs]
        - true_vel_obs: [B, num_true_vel_obs]
        - visual_obs: [B, num_visual_obs]
        """

        proprio_obs = obs_dict["proprio_obs"]
        true_vel_obs = obs_dict["true_vel_obs"]
        visual_obs = obs_dict["visual_obs"]

        # 1) 更新 proprio history
        proprio_step = torch.cat([proprio_obs, true_vel_obs], dim=-1)
        proprio_hist = self.proprio_hist_buffer.append(proprio_step)

        # 2) proprio VAE
        z_p, mu_p, logvar_p, pred_obs = self.forward_proprio_vae(proprio_hist)

        # 3) velocity estimation
        v_est = self.vel_head(z_p)

        # 4) bootstrap mixing
        if inference:
            vel_for_actor = v_est
            boot_mask = torch.ones_like(v_est[:, :1])
        else:
            B = true_vel_obs.shape[0]
            boot_mask = (torch.rand(B, 1, device=true_vel_obs.device) < p_boot).float()
            vel_for_actor = boot_mask * v_est + (1.0 - boot_mask) * true_vel_obs

        # 5) visual latent
        z_e = self.visual_encoder(visual_obs)

        # 6) latent histories
        zp_hist = self.zp_hist_buffer.append(z_p)
        ze_hist = self.ze_hist_buffer.append(z_e)

        # 7) multimodal mixer
        z_pe, mu_pe, logvar_pe, pred_hm = self.forward_mixer_vae(zp_hist, ze_hist)

        # 8) final actor obs
        actor_obs = torch.cat([proprio_obs, vel_for_actor, z_pe], dim=-1)

        aux = {
            "z_p": z_p,
            "z_e": z_e,
            "z_pe": z_pe,
            "v_est": v_est,
            "vel_for_actor": vel_for_actor,
            "boot_mask": boot_mask,
            "mu_p": mu_p,
            "logvar_p": logvar_p,
            "mu_pe": mu_pe,
            "logvar_pe": logvar_pe,
            "pred_obs": pred_obs,
            "pred_hm": pred_hm,
        }
        return actor_obs, aux
    
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
        