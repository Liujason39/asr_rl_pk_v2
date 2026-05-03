# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import warnings
import torch
import torch.nn as nn
from torch.distributions import Normal

from asr_rl_pk.networks import Memory, TemporalBuffer, TemporalBuffer_v2
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

        self.proprio_hist_buffer = TemporalBuffer_v2(self.history_len)   # 存 proprio
        self.proprio_hist_for_mixer_buffer = TemporalBuffer_v2(history_length)
        self.visual_hist_for_mixer_buffer = TemporalBuffer_v2(history_length)

        # =====heighmap_encoder : process privileged observations to latent for critic=====
        layers = []
        in_dim = num_hm_obs
        for i, out_dim in enumerate(hm_hidden_dims):
            layers.append(nn.Linear(in_dim, out_dim))
            if i != len(hm_hidden_dims) - 1:   # 最後一層不加 activation
                layers.append(activation)
            if i == len(hm_hidden_dims) - 1:
                layers.append(nn.LayerNorm(hm_hidden_dims[i])) # 最後一層加 LayerNorm 為了在mixer跟其他encoder混和
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
            if i == len(privileged_hidden_dims) - 1:
                layers.append(nn.LayerNorm(privileged_hidden_dims[i])) # 最後一層加 LayerNorm 為了在mixer跟其他encoder混和
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

        # =====visual_enocder : process visual observations to latent for Mixer=====
        layers = []
        in_dim = num_visual_obs 
        for i, out_dim in enumerate(visual_hidden_dims):
            layers.append(nn.Linear(in_dim, out_dim))
            if i != len(visual_hidden_dims) - 1:   # 最後一層不加 activation
                layers.append(activation)
            if i == len(visual_hidden_dims) - 1:
                layers.append(nn.LayerNorm(visual_hidden_dims[i])) # 最後一層加 LayerNorm 為了在mixer跟其他encoder混和
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
        in_dim = proprio_hidden_dims[-1] + visual_hidden_dims[-1] # concat proprio enocder and visual latent as input
        self.model_mixer = MixerMLP(
            num_tokens=in_dim,
            num_channels=history_length,
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

    
    def forward_mixer_vae(self, zpze_mix_hist):
        """
        zpze_mix_hist: [B, H, z_p+z_e]
        """
        mixer_input = zpze_mix_hist  # [B, H, z_p+z_e]

        mixer_feat = self.model_mixer(mixer_input)   # [B, mixer_latent_dim]

        mu = self.model_mixer_mu(mixer_feat)
        logvar = self.model_mixer_logvar(mixer_feat)
        z_pe, clamp_logvar = self.reparameterize(mu, logvar)

        pred_hm = self.hm_decoder(z_pe)
        return z_pe, mu, clamp_logvar, pred_hm
    
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
        self.proprio_hist_for_mixer_buffer.reset(dones)
        self.visual_hist_for_mixer_buffer.reset(dones)

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
    
    def act(self, actor_obs, **kwargs):
        """this will be cover by self.prepare_actor_obs()"""
        self.update_distribution(actor_obs)
        return self.distribution.sample()
    
    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    @torch.no_grad()
    def act_inference(self, obs, visual_obs, return_aux=True):
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
        out = self.prepare_actor_obs(
            proprio_obs=obs,
            true_velocity_obs=None,
            visual_obs=visual_obs,
            p_boot=1.0,
            inference=True,
        )

        action = self.actor(out["actor_input"])

        if return_aux:
            return action, {
                "est_vel": out["est_vel"],
                "mu_p": out["mu_p"],
                "z_p": out["z_p"],
                "obs_history": out["obs_history"],
            }
        return action
    
    @torch.no_grad()
    def act_inference_TrueVel(self, obs, true_velocity_obs, return_aux=True):
        """
        Inference-time action generation using only current observation.
        Internal history buffer will be updated automatically.

        Args:
            obs: Tensor of shape [B, num_actor_obs]
            true_velocity_obs: Tensor of shape [B, num_true_vel_obs]
            return_aux: whether to also return estimated auxiliary outputs

        Returns:
            action: Tensor of shape [B, num_actions]
            optionally a dict containing est_vel / mu_p / obs_hist / actor_obs
        """
        out = self.prepare_actor_obs(
            proprio_obs=obs,
            true_velocity_obs=true_velocity_obs,
            visual_obs=None,
            p_boot=0.0,
            inference=False,
        )

        action = self.actor(out["actor_input"])

        if return_aux:
            return action, {
                "est_vel": out["est_vel"],
                "mu_p": out["mu_p"],
                "z_p": out["z_p"],
                "obs_history": out["obs_history"],
            }
        return action

    def evaluate(self, privileged_obs, gt_hm, **kwargs):
        privileged_latent = self.privileged_encoder(privileged_obs)
        hm_latent = self.hm_encoder(gt_hm)
        cat_latents = torch.cat([hm_latent, privileged_latent], dim=-1)
        value = self.critic(cat_latents)
        return value

    def get_hidden_states(self):
        raise NotImplementedError
        # return self.memory_a.hidden_states, self.memory_c.hidden_states

    def prepare_actor_obs(
            self,
            proprio_obs, # [B, num_actor_obs] 
            true_velocity_obs=None,
            visual_obs=None, # [B, visual_obs]
            p_boot: float = 0.0,
            inference: bool = False,
            boot_mask: torch.Tensor | None = None,
        ):
        """
        obs_dict: [B, num_actor_obs] [B, visual_obs]
        true_velocity_obs: [B, num_true_vel_obs]
        p_boot: rollout 時用來抽樣 boot mask 的機率
        inference: True 時永遠使用 estimated velocity
        boot_mask: 若提供，直接使用此外部 mask，不再重新抽樣
        """
    
        # [B, 5, obs_dim]
        proprio_hist = self.proprio_hist_buffer.append(proprio_obs)

        # [B, 5, 5, obs_dim]
        proprio_hist_for_mixer = self.proprio_hist_for_mixer_buffer.append(proprio_hist)

        # [B, 5, visual_dim]
        visual_hist_for_mixer = self.visual_hist_for_mixer_buffer.append(visual_obs)

        out = self.build_actor_obs_from_history(
            actor_obs=proprio_obs,
            proprio_hist=proprio_hist,
            proprio_hist_for_mixer=proprio_hist_for_mixer,
            visual_hist_for_mixer=visual_hist_for_mixer,
            true_velocity_obs=true_velocity_obs,
            inference=inference,
            boot_mask=boot_mask,
            p_boot=p_boot,
        )

        out["obs_history"] = proprio_hist
        out["proprio_hist_for_mixer"] = proprio_hist_for_mixer
        out["visual_hist_for_mixer"] = visual_hist_for_mixer

        return out

        # # 2) VAE encoder
        # z_p, mu_p, logvar_p, pred_next_obs = self.forward_proprio_vae(proprio_hist)

        # # 3) velocity estimation
        # est_vel = self.compute_estimated_velocity(z_p)

        # # 4) bootstrapping
        # if inference:
        #     used_boot_mask = torch.ones_like(est_vel[:, :1])
        #     vel_for_actor = est_vel
        # else:
        #     if true_velocity_obs is None:
        #         raise ValueError("true_velocity_obs is required during training.")

        #     if boot_mask is not None:
        #         # 直接使用 rollout 時記錄下來的 mask
        #         used_boot_mask = boot_mask.float()
        #     else:
        #         # rollout 階段才重新抽樣
        #         B = true_velocity_obs.shape[0]
        #         used_boot_mask = (
        #             torch.rand(B, 1, device=true_velocity_obs.device) < p_boot
        #         ).float()

        #     vel_for_actor = used_boot_mask * est_vel + (1.0 - used_boot_mask) * true_velocity_obs

        # # 5) visual latent
        # z_e = self.visual_encoder(visual_obs)

        # zpze_mix_hist = self.zpze_hist_buffer.append(torch.cat([z_p, z_e], dim=-1))

        # # 6) multimodal mixer
        # z_pe, mu_pe, logvar_pe, pred_hm = self.forward_mixer_vae(zpze_mix_hist)

        # # 7) final actor obs
        # actor_input = torch.cat([vel_for_actor, proprio_obs, z_pe], dim=-1)

        # return {
        #     "actor_input": actor_input,
        #     "obs_history": proprio_hist,
        #     "z_p": z_p,
        #     "mu_p": mu_p,
        #     "logvar_p": logvar_p,
        #     "pred_next_obs": pred_next_obs,
        #     "est_vel": est_vel,
        #     "vel_for_actor": vel_for_actor,
        #     "boot_mask": used_boot_mask,
        #     "z_pe": z_pe,
        #     "mu_pe": mu_pe,
        #     "logvar_pe": logvar_pe,
        #     "pred_hm": pred_hm,
        #     "zpze_mix_hist": zpze_mix_hist,
        # }

    def build_actor_obs_from_history(
        self,
        actor_obs, # [B, num_actor_obs] [B, visual_obs]
        proprio_hist,
        proprio_hist_for_mixer,
        visual_hist_for_mixer,
        true_velocity_obs=None,
        inference: bool = False,
        boot_mask: torch.Tensor | None = None,
        p_boot=0.0,
    ):
        # current z_p[t]
        z_p, mu_p, logvar_p, pred_next_obs = self.forward_proprio_vae(proprio_hist)

        est_vel = self.compute_estimated_velocity(z_p)

        if inference:
            used_boot_mask = torch.ones_like(est_vel[:, :1])
            vel_for_actor = est_vel
        else:
            if boot_mask is None:
                B = true_velocity_obs.shape[0]
                used_boot_mask = (
                    torch.rand(B, 1, device=true_velocity_obs.device) < p_boot
                ).float()
            else:
                used_boot_mask = boot_mask.float()

            vel_for_actor = used_boot_mask * est_vel + (1.0 - used_boot_mask) * true_velocity_obs

        # proprio_hist_for_mixer: [B, 5, 5, obs_dim]
        B, Hmix, Hprop, D = proprio_hist_for_mixer.shape

        prop_flat = proprio_hist_for_mixer.reshape(B * Hmix, Hprop, D)
        z_p_hist, _, _, _ = self.forward_proprio_vae(prop_flat)
        z_p_hist = z_p_hist.reshape(B, Hmix, -1)

        # visual_hist_for_mixer: [B, 5, visual_dim]
        visual_flat = visual_hist_for_mixer.reshape(B * Hmix, -1)
        z_e_hist = self.visual_encoder(visual_flat)
        z_e_hist = z_e_hist.reshape(B, Hmix, -1)

        zpze_mix_hist = torch.cat([z_p_hist, z_e_hist], dim=-1)

        z_pe, mu_pe, logvar_pe, pred_hm = self.forward_mixer_vae(zpze_mix_hist)

        actor_input = torch.cat([vel_for_actor, actor_obs, z_pe], dim=-1)

        return {
            "actor_input": actor_input,
            "z_p": z_p,
            "mu_p": mu_p,
            "logvar_p": logvar_p,
            "pred_next_obs": pred_next_obs,
            "est_vel": est_vel,
            "vel_for_actor": vel_for_actor,
            "boot_mask": used_boot_mask,
            "z_pe": z_pe,
            "mu_pe": mu_pe,
            "logvar_pe": logvar_pe,
            "pred_hm": pred_hm,
            "zpze_mix_hist": zpze_mix_hist,
        }

    def act_in_update(
        self,
        obs,
        proprio_hist,
        proprio_hist_for_mixer,
        visual_hist_for_mixer,
        true_velocity_obs,
        boot_mask,
        **kwargs,
    ):
        out = self.build_actor_obs_from_history(
            actor_obs=obs,
            proprio_hist=proprio_hist,
            proprio_hist_for_mixer=proprio_hist_for_mixer,
            visual_hist_for_mixer=visual_hist_for_mixer,
            true_velocity_obs=true_velocity_obs,
            inference=False,
            boot_mask=boot_mask,
        )

        self.update_distribution(out["actor_input"])
        return out

    def act_for_onnx_transfer(self, 
                              obs, 
                              obs_history, 
                              visual_obs, 
                              zpze_mix_hist, 
                              use_mu=True, **kwargs):
        """
        obs:         [B, obs_dim]
        obs_history: [B, H, obs_dim]
        visual_obs:  [B, visual_obs]
        zpze_mix_hist: [B, H, z_p+z_e] 
        return:
            action: [B, act_dim]
            est_v:  [B, vel_dim]
            zpze_mix: [B, z_p+z_e] 
        """
        proprio_encoder_feat = self.proprio_encoder(obs_history)   # [B, latent_dim]
        mu_p = self.proprio_encoder_mu(proprio_encoder_feat)
        logvar_p = self.proprio_encoder_logvar(proprio_encoder_feat)

        # 5) visual latent
        z_e = self.visual_encoder(visual_obs)

        if use_mu:
            z_p = mu_p
        else:
            z_p, _ = self.reparameterize(mu_p, logvar_p)

        zpze_mix = torch.cat([z_p, z_e], dim=-1)

        est_v = self.compute_estimated_velocity(z_p)

        z_pe, mu_pe, clamp_logvar_pe, pred_hm = self.forward_mixer_vae(zpze_mix_hist)

        actor_obs = torch.cat([est_v, obs, z_pe], dim=-1)
        action = self.actor(actor_obs)

        return action, est_v, zpze_mix

    def act_for_onnx_transfer_with_external_vel(
        self,
        obs: torch.Tensor,
        obs_history: torch.Tensor,
        vel: torch.Tensor,
        visual_obs, 
        zpze_mix_hist, 
        use_mu: bool = True,
        **kwargs,
    ):
        """
        obs:         [B, obs_dim]
        obs_history: [B, H, obs_dim]
        vel:         [B, vel_dim]
        visual_obs:  [B, visual_obs]
        zpze_mix_hist: [B, H, z_p+z_e] 
        return:
            action: [B, act_dim]
            est_v:  [B, vel_dim]
            zpze_mix: [B, z_p+z_e] 
        """

        proprio_encoder_feat = self.proprio_encoder(obs_history)   # [B, latent_dim]
        mu_p = self.proprio_encoder_mu(proprio_encoder_feat)
        logvar_p = self.proprio_encoder_logvar(proprio_encoder_feat)

        # 5) visual latent
        z_e = self.visual_encoder(visual_obs)

        if use_mu:
            z_p = mu_p
        else:
            z_p, _ = self.reparameterize(mu_p, logvar_p)

        zpze_mix = torch.cat([z_p, z_e], dim=-1)

        est_v = self.compute_estimated_velocity(z_p)

        z_pe, mu_pe, clamp_logvar_pe, pred_hm = self.forward_mixer_vae(zpze_mix_hist)

        actor_obs = torch.cat([vel, obs, z_pe], dim=-1)
        action = self.actor(actor_obs)

        return action, est_v, zpze_mix
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
        