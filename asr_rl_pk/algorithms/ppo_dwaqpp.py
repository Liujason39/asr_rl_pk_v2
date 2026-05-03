# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim
from itertools import chain

from asr_rl_pk.modules import ActorCritic, dwaq_policy, dwaqpp_policy
from asr_rl_pk.modules.rnd import RandomNetworkDistillation
from asr_rl_pk.storage import RolloutStorage, RolloutStorageDWAQ, RolloutStorageDWAQPP
from asr_rl_pk.utils import string_to_callable, unpad_trajectories


class PPO_DWAQPP:
    """Proximal Policy Optimization algorithm (https://arxiv.org/abs/1707.06347)."""

    policy: dwaqpp_policy
    """The actor critic module."""

    def __init__(
        self,
        policy,
        num_learning_epochs=1,
        num_mini_batches=1,
        clip_param=0.2,
        gamma=0.998,
        lam=0.95,
        value_loss_coef=1.0,
        entropy_coef=0.0,
        learning_rate=1e-3,
        max_grad_norm=1.0,
        use_clipped_value_loss=True,
        schedule="fixed",
        desired_kl=0.01,
        device="cpu",
        normalize_advantage_per_mini_batch=False,
        # RND parameters
        rnd_cfg: dict | None = None,
        # Symmetry parameters
        symmetry_cfg: dict | None = None,
        # Distributed training parameters
        multi_gpu_cfg: dict | None = None,
    ):
        # device-related parameters
        self.device = device
        self.is_multi_gpu = multi_gpu_cfg is not None
        # Multi-GPU parameters
        if multi_gpu_cfg is not None:
            self.gpu_global_rank = multi_gpu_cfg["global_rank"]
            self.gpu_world_size = multi_gpu_cfg["world_size"]
        else:
            self.gpu_global_rank = 0
            self.gpu_world_size = 1

        # RND components
        if rnd_cfg is not None:
            # Extract learning rate and remove it from the original dict
            learning_rate = rnd_cfg.pop("learning_rate", 1e-3)
            # Create RND module
            self.rnd = RandomNetworkDistillation(device=self.device, **rnd_cfg)
            # Create RND optimizer
            params = self.rnd.predictor.parameters()
            self.rnd_optimizer = optim.Adam(params, lr=learning_rate)
        else:
            self.rnd = None
            self.rnd_optimizer = None

        # Symmetry components
        if symmetry_cfg is not None:
            # Check if symmetry is enabled
            use_symmetry = symmetry_cfg["use_data_augmentation"] or symmetry_cfg["use_mirror_loss"]
            # Print that we are not using symmetry
            if not use_symmetry:
                print("Symmetry not used for learning. We will use it for logging instead.")
            # If function is a string then resolve it to a function
            if isinstance(symmetry_cfg["data_augmentation_func"], str):
                symmetry_cfg["data_augmentation_func"] = string_to_callable(symmetry_cfg["data_augmentation_func"])
            # Check valid configuration
            if symmetry_cfg["use_data_augmentation"] and not callable(symmetry_cfg["data_augmentation_func"]):
                raise ValueError(
                    "Data augmentation enabled but the function is not callable:"
                    f" {symmetry_cfg['data_augmentation_func']}"
                )
            # Store symmetry configuration
            self.symmetry = symmetry_cfg
        else:
            self.symmetry = None

        # PPO components
        self.policy = policy
        self.policy.to(self.device)
        # Create optimizer
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        # Create rollout storage
        self.storage: RolloutStorageDWAQPP = None  # type: ignore
        self.transition = RolloutStorageDWAQPP.Transition()

        # PPO parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate
        self.normalize_advantage_per_mini_batch = normalize_advantage_per_mini_batch

        # for bootstrapping in DWAQ
        self.bootstrap_prob = 0.0
        self.mse_loss_fn = nn.functional.mse_loss
        self.huber_loss_fn = nn.functional.huber_loss
        

        self.p_boot = 0.0

    def set_bootstrap_prob(self, p_boot: float):
        self.bootstrap_prob = float(max(0.0, min(1.0, p_boot)))

    def _repeat_on_batch_dim(self, x: torch.Tensor, num_aug: int, is_recurrent_batch: bool) -> torch.Tensor:
        """Repeat tensor along the batch dimension for symmetry augmentation.
        Non-recurrent (MLP) batches are typically (B, C) or (B,).
        Recurrent batches are typically (T, B, C) or (T, B).
        """
        if num_aug == 1:
            return x

        if not is_recurrent_batch:
            # MLP: batch dim is 0
            if x.dim() == 1:          # (B,)
                return x.repeat(num_aug)
            elif x.dim() == 2:        # (B, C)
                return x.repeat(num_aug, 1)
            else:
                raise RuntimeError(f"Non-recurrent unexpected tensor shape: {tuple(x.shape)}")

        # RNN: batch dim is 1 (assume (T, B, ...) layout)
        if x.dim() == 2:              # (T, B)
            return x.repeat(1, num_aug)
        elif x.dim() == 3:            # (T, B, C)
            return x.repeat(1, num_aug, 1)
        else:
            raise RuntimeError(f"Recurrent unexpected tensor shape: {tuple(x.shape)}")

    def init_storage(
        self, 
        training_type, 
        num_envs, 
        num_transitions_per_env, 
        actor_obs_shape, 
        critic_obs_shape, 
        gt_heightmap_obs_shape, 
        true_velocity_obs_shape, 
        actions_shape, 
        hist_length,
        visual_obs_shape,
    ):
        # create memory for RND as well :)
        if self.rnd:
            rnd_state_shape = [self.rnd.num_states]
        else:
            rnd_state_shape = None
        # create rollout storage
        self.storage = RolloutStorageDWAQPP(
            training_type,
            num_envs,
            num_transitions_per_env,
            actor_obs_shape,
            critic_obs_shape,
            gt_heightmap_obs_shape,
            true_velocity_obs_shape,
            actions_shape,
            hist_length, # specifiable history length for DWAQ
            visual_obs_shape,
            rnd_state_shape,
            self.device,
        )

    
    def act(self, obs, critic_obs, true_velocity_obs, gt_heightmap_obs, visual_obs):
        if self.policy.is_recurrent:
            self.transition.hidden_states = self.policy.get_hidden_states()

        # ---- DreamWaQ++ branch ----
        # actor_obs, obs_history, mu_p, vel_for_actor = self.policy.prepare_actor_obs(
        #     actor_obs=obs,
        #     true_velocity_obs=true_velocity_obs,
        #     p_boot=self.bootstrap_prob,
        #     inference=False,
        # )
        """return_dict = {
            "actor_input": actor_input,
            "obs_history": proprio_hist,
            "z_p": z_p,
            "mu_p": mu_p,
            "logvar_p": logvar_p,
            "pred_next_obs": pred_next_obs,
            "est_vel": est_vel,
            "vel_for_actor": vel_for_actor,
            "boot_mask": boot_mask,
        }"""
        return_dict = self.policy.prepare_actor_obs(
            proprio_obs=obs, 
            true_velocity_obs=true_velocity_obs, 
            visual_obs=visual_obs, 
            p_boot=self.bootstrap_prob, 
            inference=False)

        # compute the actions and values
        self.transition.actions = self.policy.act(return_dict["actor_input"]).detach()
        # self.transition.values = self.policy.evaluate(critic_obs).detach()
        self.transition.values = self.policy.evaluate(critic_obs, gt_heightmap_obs).detach()
        self.transition.actions_log_prob = self.policy.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.policy.action_mean.detach()
        self.transition.action_sigma = self.policy.action_std.detach()
        # need to record obs and critic_obs before env.step()
        self.transition.observations = obs # not actor_obs, it will be prepare again in update for gradient 
        self.transition.visual_obs = visual_obs
        self.transition.privileged_observations = critic_obs
        # dwaq-specific transition data
        self.transition.gt_heightmap_obs = gt_heightmap_obs
        self.transition.true_velocity_obs = true_velocity_obs
        self.transition.obs_history = return_dict["obs_history"].detach()
        self.transition.p_boot_mask = return_dict["boot_mask"].detach()
        self.transition.proprio_hist_for_mixer = return_dict["proprio_hist_for_mixer"].detach()
        self.transition.visual_hist_for_mixer = return_dict["visual_hist_for_mixer"].detach()
        self.transition.visual_obs = visual_obs

        return self.transition.actions

    
    def process_env_step(self, next_obs, rewards, dones, infos):
    # def process_env_step(self, rewards, dones, infos):
        # Record the rewards and dones
        # Note: we clone here because later on we bootstrap the rewards based on timeouts
        # cv = rewards.std() / (rewards.mean().abs() + 1e-6)
        # self.p_boot = 1.0 - torch.tanh(cv)
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones

        # store the next_obs for DWAQ
        self.transition.next_observations = next_obs.clone()
        self.transition.next_obs_valid = (1.0 - dones.float()).unsqueeze(-1)

        # Compute the intrinsic rewards and add to extrinsic rewards
        if self.rnd:
            # Obtain curiosity gates / observations from infos
            rnd_state = infos["observations"]["rnd_state"]
            # Compute the intrinsic rewards
            # note: rnd_state is the gated_state after normalization if normalization is used
            self.intrinsic_rewards, rnd_state = self.rnd.get_intrinsic_reward(rnd_state)
            # Add intrinsic rewards to extrinsic rewards
            self.transition.rewards += self.intrinsic_rewards
            # Record the curiosity gates
            self.transition.rnd_state = rnd_state.clone()

        # Bootstrapping on time outs
        if "time_outs" in infos:
            self.transition.rewards += self.gamma * torch.squeeze(
                self.transition.values * infos["time_outs"].unsqueeze(1).to(self.device), 1
            )

        # record the transition
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.policy.reset(dones)

    def compute_returns(self, last_critic_obs, last_gt_heightmap_obs=None):
        # compute value for the last step
        last_values = self.policy.evaluate(last_critic_obs, last_gt_heightmap_obs).detach()
        self.storage.compute_returns(
            last_values, self.gamma, self.lam, normalize_advantage=not self.normalize_advantage_per_mini_batch
        )

    def update(self):  # noqa: C901
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_entropy = 0
        # for DWAQ auxiliary losses
        mean_EstVel_loss=0
        mean_D_KL_loss=0
        mean_EstObs_loss=0
        mean_HM_loss = 0
        mean_PE_KL_loss = 0

        # -- RND loss
        if self.rnd:
            mean_rnd_loss = 0
        else:
            mean_rnd_loss = None
        # -- Symmetry loss
        if self.symmetry:
            mean_symmetry_loss = 0
        else:
            mean_symmetry_loss = None

        # generator for mini batches
        if self.policy.is_recurrent:
            if self.symmetry:
                generator = self.storage.recurrent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs,augment=True)
            else:
                generator = self.storage.recurrent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs,augment=False)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        # iterate over batches
        for (
            obs_batch,
            critic_obs_batch,
            gt_heightmap_obs_batch,
            true_velocity_obs_batch,
            proprio_hist_batch,
            next_propprio_batch,
            next_obs_valid_batch,
            p_boot_mask_batch,
            visual_obs_batch,
            proprio_hist_for_mixer_batch,
            visual_hist_for_mixer_batch,
            # vel_for_actor_batch,
            # mu_p_batch,
            actions_batch,
            target_values_batch,
            advantages_batch,
            returns_batch,
            old_actions_log_prob_batch,
            old_mu_batch,
            old_sigma_batch,
            hid_states_batch,
            masks_batch,
            rnd_state_batch,
        ) in generator:

            # number of augmentations per sample
            # we start with 1 and increase it if we use symmetry augmentation
            num_aug = 1
            # original batch size
            original_batch_size = obs_batch.shape[0]

            # check if we should normalize advantages per mini batch
            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    advantages_batch = (advantages_batch - advantages_batch.mean()) / (advantages_batch.std() + 1e-8)

            # Perform symmetric augmentation
            if self.symmetry and self.symmetry["use_data_augmentation"]:
                raise NotImplementedError("Symmetry augmentation for recurrent batch is not implemented yet.")
                # augmentation using symmetry
                data_augmentation_func = self.symmetry["data_augmentation_func"]
                # 判斷是不是 recurrent batch
                # recurrent_mini_batch_generator 通常會給 (T, B, D)
                is_recurrent_batch = (obs_batch.dim() == 3)

                # 0) 存原始 B0（augmentation 前）
                B0 = obs_batch.shape[1]  # (T,B0,D)

                # ---- 1) 做 augmentation：一定要擴 batch，不要擴 time ----
                if is_recurrent_batch:
                    raise NotImplementedError("Symmetry augmentation for recurrent batch is not implemented yet.")
                    # # obs_batch: (T, B, D), actions_batch: (T, B, A)
                    # # 這裡要求你的 data_augmentation_func 在 RNN 模式下要 cat 在 dim=1
                    # obs_batch, actions_batch = data_augmentation_func(
                    #     obs=obs_batch, actions=actions_batch, env=self.symmetry["_env"], obs_type="policy"
                    # )
                    # critic_obs_batch, _ = data_augmentation_func(
                    #     obs=critic_obs_batch, actions=None, env=self.symmetry["_env"], obs_type="critic"
                    # )
                    
                    # B_aug = obs_batch.shape[1]
                    # assert B_aug % B0 == 0
                    # num_aug = B_aug // B0

                    # # 2) 先把 masks 擴到 B_aug（一次就好）
                    # # masks_batch: (T,B0) -> (T,B_aug)
                    # if masks_batch is not None:
                    #     assert masks_batch.shape[1] == B0, (masks_batch.shape, B0)
                    #     masks_batch = masks_batch.repeat(1, num_aug)

                    # # 3) repeat hidden_states 到 B_aug（讓 policy.act 不會 hidden mismatch）
                    # def _repeat_hidden(h, k):
                    #     if h is None:
                    #         return None
                    #     if isinstance(h, tuple):
                    #         return tuple(_repeat_hidden(x, k) for x in h)
                    #     return h.repeat(1, k, 1)  # (L,B,H)->(L,kB,H)

                    # if hid_states_batch is not None:
                    #     hid_states_batch = (
                    #         _repeat_hidden(hid_states_batch[0], num_aug),
                    #         _repeat_hidden(hid_states_batch[1], num_aug),
                    #     )

                    # # 4) repeat 這些「padded batch tensor」到 B_aug
                    # old_actions_log_prob_batch = self._repeat_on_batch_dim(old_actions_log_prob_batch, num_aug, True)
                    # target_values_batch       = self._repeat_on_batch_dim(target_values_batch,       num_aug, True)
                    # advantages_batch          = self._repeat_on_batch_dim(advantages_batch,          num_aug, True)
                    # returns_batch             = self._repeat_on_batch_dim(returns_batch,             num_aug, True)
                    # old_mu_batch              = self._repeat_on_batch_dim(old_mu_batch,              num_aug, True)
                    # old_sigma_batch           = self._repeat_on_batch_dim(old_sigma_batch,           num_aug, True)

                    # # 5) ★關鍵：統一 unpad（讓所有 loss tensor 的 batch 都變成同一個 B_unpad）
                    # actions_batch_u            = unpad_trajectories(actions_batch,            masks_batch)
                    # old_actions_log_prob_u     = unpad_trajectories(old_actions_log_prob_batch, masks_batch)
                    # target_values_u            = unpad_trajectories(target_values_batch,       masks_batch)
                    # advantages_u               = unpad_trajectories(advantages_batch,          masks_batch)
                    # returns_u                  = unpad_trajectories(returns_batch,             masks_batch)
                    # old_mu_u                   = unpad_trajectories(old_mu_batch,              masks_batch)
                    # old_sigma_u                = unpad_trajectories(old_sigma_batch,           masks_batch)

                    # # 7) 之後 PPO loss 一律用 *_u（不要再用 padded 版本）
                    # old_actions_log_prob_batch = old_actions_log_prob_u
                    # target_values_batch        = target_values_u
                    # advantages_batch           = advantages_u
                    # returns_batch              = returns_u
                    # old_mu_batch               = old_mu_u
                    # old_sigma_batch            = old_sigma_u
                    # actions_batch              = actions_batch_u

                    # # 5.5) 計算 unpad 後「原始那一半」的 batch 大小（B0_u）
                    # # 取第一份 augmentation (原始) 的 masks，先 unpad 一次得到真正有效的 trajectory 數
                    # masks_orig = masks_batch[:, :B0]  # (T, B0)
                    # # 用任意一個 padded tensor 來推 B0_u（這裡用 returns_batch[:, :B0]）
                    # B0_u = unpad_trajectories(returns_batch[:, :B0], masks_orig).shape[1]

                else:
                    # ---- 非 RNN (MLP) 原本邏輯維持：batch dim=0 ----
                    original_batch_size = obs_batch.shape[0]

                    obs_batch, actions_batch = data_augmentation_func(
                        obs=obs_batch, actions=actions_batch, env=self.symmetry["_env"], obs_type="policy"
                    )
                    critic_obs_batch, _ = data_augmentation_func(
                        obs=critic_obs_batch, actions=None, env=self.symmetry["_env"], obs_type="critic"
                    )

                    num_aug = int(obs_batch.shape[0] / original_batch_size)

                    old_actions_log_prob_batch = self._repeat_on_batch_dim(old_actions_log_prob_batch, num_aug, False)
                    target_values_batch       = self._repeat_on_batch_dim(target_values_batch,       num_aug, False)
                    advantages_batch          = self._repeat_on_batch_dim(advantages_batch,          num_aug, False)
                    returns_batch             = self._repeat_on_batch_dim(returns_batch,             num_aug, False)
            # Recompute actions log prob and entropy for current batch of transitions
            # Note: we need to do this because we updated the policy with the new parameters
            # -- actor

            # 6) 用 policy.act / evaluate 取得 distribution / value（它內部也會 unpad，shape 會對齊上面的 *_u）
            # if masks_batch is not None:
            #     valid_traj = masks_batch.any(dim=0)
            #     print("B=", masks_batch.shape[1], "valid=", valid_traj.sum().item(), "invalid=", (~valid_traj).sum().item())
            # _, mu, z_pH, pred_obs, logvar_p = self.policy.act_in_update(vel_for_actor_batch, obs_batch, proprio_hist_batch, masks=masks_batch, hidden_states=hid_states_batch[0])
            # self.policy.act_in_update(vel_for_actor_batch, obs_batch, mu_p_batch, masks=masks_batch, hidden_states=hid_states_batch[0])
            # z_p, mu_p, logvar_p, pred_obs, est_vel = self.policy.act_in_update(obs_batch, proprio_hist_batch, masks=masks_batch, hidden_states=hid_states_batch[0])
            if not ((p_boot_mask_batch == 0) | (p_boot_mask_batch == 1)).all():
                raise RuntimeError("p_boot_mask_batch contains non-binary values.")
            out = self.policy.act_in_update(
                obs=obs_batch,
                proprio_hist=proprio_hist_batch,
                proprio_hist_for_mixer=proprio_hist_for_mixer_batch,
                visual_hist_for_mixer=visual_hist_for_mixer_batch,
                true_velocity_obs=true_velocity_obs_batch,
                boot_mask=p_boot_mask_batch,
            )
            """
            out = {
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
            """
            pred_obs = out["pred_next_obs"]
            mu_p = out["mu_p"]
            logvar_p = out["logvar_p"]
            est_vel = out["est_vel"]
            pred_hm = out["pred_hm"]
            mu_pe = out["mu_pe"]
            logvar_pe = out["logvar_pe"]
            # print("action_mean:", self.policy.action_mean.shape)
            # print("entropy:", self.policy.entropy.shape)
            # print("masks_batch:", None if masks_batch is None else masks_batch.shape)
            # print("obs_batch:", obs_batch.shape, "actions_batch:", actions_batch.shape, "num_aug:", num_aug)
            actions_log_prob_batch = self.policy.get_actions_log_prob(actions_batch)
            
            # -- critic
            value_batch = self.policy.evaluate(critic_obs_batch, gt_heightmap_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1])

            # -- entropy
            # we only keep the entropy of the first augmentation (the original one)
            if self.policy.is_recurrent and self.symmetry and self.symmetry["use_data_augmentation"]:
                raise NotImplementedError("Symmetry augmentation for recurrent batch is not implemented yet.")
                # # self.policy.action_mean/std/entropy 是 batch_mode + unpad 後的結果：shape (T, B_unpad, ...)
                # mu_batch = self.policy.action_mean[:, :B0_u]
                # sigma_batch = self.policy.action_std[:, :B0_u]
                # entropy_batch = self.policy.entropy[:, :B0_u]
            else:
                mu_batch = self.policy.action_mean[:original_batch_size]
                sigma_batch = self.policy.action_std[:original_batch_size]
                entropy_batch = self.policy.entropy[:original_batch_size]

            # KL
            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    # 確保 old_mu_batch / old_sigma_batch 也只拿「原始那一半」去算 KL
                    if self.policy.is_recurrent and self.symmetry and self.symmetry["use_data_augmentation"]:
                        raise NotImplementedError("Symmetry augmentation for recurrent batch is not implemented yet.")
                        # old_mu_batch = old_mu_batch[:, :B0_u]
                        # old_sigma_batch = old_sigma_batch[:, :B0_u]
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
                        + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch))
                        / (2.0 * torch.square(sigma_batch))
                        - 0.5,
                        axis=-1,
                    )
                    kl_mean = torch.mean(kl)

                    # Reduce the KL divergence across all GPUs
                    if self.is_multi_gpu:
                        torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
                        kl_mean /= self.gpu_world_size

                    # Update the learning rate
                    # Perform this adaptation only on the main process
                    # TODO: Is this needed? If KL-divergence is the "same" across all GPUs,
                    #       then the learning rate should be the same across all GPUs.
                    if self.gpu_global_rank == 0:
                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                    # Update the learning rate for all GPUs
                    if self.is_multi_gpu:
                        lr_tensor = torch.tensor(self.learning_rate, device=self.device)
                        torch.distributed.broadcast(lr_tensor, src=0)
                        self.learning_rate = lr_tensor.item()

                    # Update the learning rate for all parameter groups
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            # Surrogate loss
            # ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            log_ratio = actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch)

            if not torch.isfinite(log_ratio).all():
                raise RuntimeError(
                    f"log_ratio became non-finite. "
                    f"new_logp finite={torch.isfinite(actions_log_prob_batch).all().item()}, "
                    f"old_logp finite={torch.isfinite(old_actions_log_prob_batch).all().item()}"
                )

            log_ratio = torch.clamp(log_ratio, min=-20.0, max=20.0)
            ratio = torch.exp(log_ratio)

            # debug
            print("actions_log_prob finite:", torch.isfinite(actions_log_prob_batch).all().item())
            print("old_actions_log_prob finite:", torch.isfinite(old_actions_log_prob_batch).all().item())
            print("actions_log_prob min/max:",
                torch.nan_to_num(actions_log_prob_batch).min().item(),
                torch.nan_to_num(actions_log_prob_batch).max().item())
            print("old_actions_log_prob min/max:",
                torch.nan_to_num(old_actions_log_prob_batch).min().item(),
                torch.nan_to_num(old_actions_log_prob_batch).max().item())

            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # Value function loss
            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                    -self.clip_param, self.clip_param
                )
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()

            # DWAQ loss: MSE(est_vel, true_vel), MSE(est_obs_t+1, true_obs_t+1), D_KL(mu_proprio, sigma_proprio || N(0,1)=p(z_t))
            # z_pH, mu, logvar_p, pred_obs = self.policy.forward_proprio_vae(proprio_hist_batch)
            # D_KL_loss = self.policy.kl_divergence_diag_gaussian(mu, logvar_p)

            # est_vel_batch = self.policy.compute_estimated_velocity(z_pH)
            est_vel_loss = self.huber_loss_fn(est_vel, true_velocity_obs_batch).mean()
            
            # est_obs_loss = self.mse_loss_fn(pred_obs, obs_batch).mean()
            valid = next_obs_valid_batch.squeeze(-1) > 0
            if valid.any():
                est_obs_loss = self.mse_loss_fn(pred_obs[valid], next_propprio_batch[valid]).mean()
            else:
                est_obs_loss = 0.0 * pred_obs.sum()

            D_kl_loss = (0.5 * (mu_p.pow(2) + logvar_p.exp() - logvar_p - 1)).sum(dim=-1).mean()

            hm_loss = self.mse_loss_fn(pred_hm, gt_heightmap_obs_batch).mean()

            pe_kl_loss = (0.5 * (mu_pe.pow(2) + logvar_pe.exp() - logvar_pe - 1)).sum(dim=-1).mean()
            
            loss = (
                loss
                + 5.0 * est_vel_loss
                + 1.0 * est_obs_loss
                + 0.05 * D_kl_loss
                + 1.0 * hm_loss
                + 0.005 * pe_kl_loss
            )

            # for log
            mean_EstVel_loss += est_vel_loss.item()
            mean_D_KL_loss += D_kl_loss.item()
            mean_EstObs_loss += est_obs_loss.item()
            mean_HM_loss += hm_loss.item()
            mean_PE_KL_loss += pe_kl_loss.item()

            # Symmetry loss
            if self.symmetry:
                raise NotImplementedError("Symmetry loss is not implemented yet.")
                # # obtain the symmetric actions
                # if not self.symmetry["use_data_augmentation"]:
                #     data_augmentation_func = self.symmetry["data_augmentation_func"]
                #     obs_batch, _ = data_augmentation_func(
                #         obs=obs_batch, actions=None, env=self.symmetry["_env"], obs_type="policy"
                #     )
                #     # compute number of augmentations per sample (MLP 假設 batch 在 dim=0)
                #     num_aug = int(obs_batch.shape[0] / original_batch_size)

                # data_augmentation_func = self.symmetry["data_augmentation_func"]
                # mse_loss = torch.nn.MSELoss()

                # if self.policy.is_recurrent:
                #     # -------------------------
                #     # RNN case: obs_batch is (T, B_aug, D)
                #     # -------------------------
                #     # 用 batch-mode 跑序列
                #     # 直接用，不要再 forward
                #     mean_actions_batch = self.policy.action_mean.detach()  # (T, B_aug, A)

                #     # 原始 batch 大小 B0（第一份 augmentation）
                #     B_aug = mean_actions_batch.shape[1]
                #     B0 = B_aug // num_aug

                #     action_mean_orig = mean_actions_batch[:, :B0, :]  # (T, B0, A)

                #     # 對 action 做同樣 augmentation（actions-only call）
                #     _, actions_mean_symm_batch = data_augmentation_func(
                #         obs=None, actions=action_mean_orig, env=self.symmetry["_env"], obs_type="policy"
                #     )
                #     # actions_mean_symm_batch 期望是 (T, B_aug, A)，且第二份是 mirror(orig)
                #     symmetry_loss = mse_loss(
                #         mean_actions_batch[:, B0:, :],              # policy 在 mirrored obs 上的 mean action
                #         actions_mean_symm_batch.detach()[:, B0:, :] # mirror(orig_mean_action)
                #     )

                # else:
                #     # -------------------------
                #     # MLP case: obs_batch is (B_aug, D)
                #     # -------------------------
                #     mean_actions_batch = self.policy.act_inference(obs_batch.detach().clone())
                #     action_mean_orig = mean_actions_batch[:original_batch_size]
                #     _, actions_mean_symm_batch = data_augmentation_func(
                #         obs=None, actions=action_mean_orig, env=self.symmetry["_env"], obs_type="policy"
                #     )
                #     symmetry_loss = mse_loss(
                #         mean_actions_batch[original_batch_size:],
                #         actions_mean_symm_batch.detach()[original_batch_size:],
                #     )

                # # add the loss to the total loss
                # if self.symmetry["use_mirror_loss"]:
                #     loss += self.symmetry["mirror_loss_coeff"] * symmetry_loss
                # else:
                #     symmetry_loss = symmetry_loss.detach()


            # Random Network Distillation loss
            if self.rnd:
                # predict the embedding and the target
                predicted_embedding = self.rnd.predictor(rnd_state_batch)
                target_embedding = self.rnd.target(rnd_state_batch).detach()
                # compute the loss as the mean squared error
                mseloss = torch.nn.MSELoss()
                rnd_loss = mseloss(predicted_embedding, target_embedding)

            # Compute the gradients
            # -- For PPO
            self.optimizer.zero_grad()
            loss.backward()
            if hasattr(self.policy, "log_std"):
                print("log_std data min/max:",
                    self.policy.log_std.data.min().item(),
                    self.policy.log_std.data.max().item())

                if self.policy.log_std.grad is not None:
                    g = self.policy.log_std.grad
                    print("log_std grad finite:", torch.isfinite(g).all().item())
                    print("log_std grad min/max:",
                        torch.nan_to_num(g).min().item(),
                        torch.nan_to_num(g).max().item())
            # -- For RND
            if self.rnd:
                self.rnd_optimizer.zero_grad()  # type: ignore
                rnd_loss.backward()

            # Collect gradients from all GPUs
            if self.is_multi_gpu:
                self.reduce_parameters()

            # Apply the gradients
            # -- For PPO
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            # debug
            for name, p in self.policy.named_parameters():
                if p.grad is not None and not torch.isfinite(p.grad).all():
                    raise RuntimeError(f"Gradient of {name} became non-finite before optimizer.step()")
            self.optimizer.step()
            # -- For RND
            if self.rnd_optimizer:
                self.rnd_optimizer.step()

            # Store the losses
            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy_batch.mean().item()
            # -- RND loss
            if mean_rnd_loss is not None:
                mean_rnd_loss += rnd_loss.item()
            # -- Symmetry loss
            if mean_symmetry_loss is not None:
                raise NotImplementedError("Symmetry loss is not implemented yet.")
                # mean_symmetry_loss += symmetry_loss.item()

        # -- For PPO
        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates
        # -- For RND
        if mean_rnd_loss is not None:
            mean_rnd_loss /= num_updates
        # -- For Symmetry
        if mean_symmetry_loss is not None:
            mean_symmetry_loss /= num_updates
        # -- Clear the storage
        self.storage.clear()

        mean_EstVel_loss /= num_updates
        mean_D_KL_loss /= num_updates
        mean_EstObs_loss /= num_updates
        mean_HM_loss /= num_updates
        mean_PE_KL_loss /= num_updates

        # construct the loss dictionary
        loss_dict = {
            "value_function": mean_value_loss,
            "surrogate": mean_surrogate_loss,
            "entropy": mean_entropy,
            "est_vel": mean_EstVel_loss,
            "D_KL": mean_D_KL_loss,
            "est_obs": mean_EstObs_loss,
            "hm": mean_HM_loss,
            "pe_kl": mean_PE_KL_loss,
        }
        if self.rnd:
            loss_dict["rnd"] = mean_rnd_loss
        if self.symmetry:
            loss_dict["symmetry"] = mean_symmetry_loss

        return loss_dict

    """
    Helper functions
    """

    def broadcast_parameters(self):
        """Broadcast model parameters to all GPUs."""
        # obtain the model parameters on current GPU
        model_params = [self.policy.state_dict()]
        if self.rnd:
            model_params.append(self.rnd.predictor.state_dict())
        # broadcast the model parameters
        torch.distributed.broadcast_object_list(model_params, src=0)
        # load the model parameters on all GPUs from source GPU
        self.policy.load_state_dict(model_params[0])
        if self.rnd:
            self.rnd.predictor.load_state_dict(model_params[1])

    def reduce_parameters(self):
        """Collect gradients from all GPUs and average them.

        This function is called after the backward pass to synchronize the gradients across all GPUs.
        """
        # Create a tensor to store the gradients
        grads = [param.grad.view(-1) for param in self.policy.parameters() if param.grad is not None]
        if self.rnd:
            grads += [param.grad.view(-1) for param in self.rnd.parameters() if param.grad is not None]
        all_grads = torch.cat(grads)

        # Average the gradients across all GPUs
        torch.distributed.all_reduce(all_grads, op=torch.distributed.ReduceOp.SUM)
        all_grads /= self.gpu_world_size

        # Get all parameters
        all_params = self.policy.parameters()
        if self.rnd:
            all_params = chain(all_params, self.rnd.parameters())

        # Update the gradients for all parameters with the reduced gradients
        offset = 0
        for param in all_params:
            if param.grad is not None:
                numel = param.numel()
                # copy data back from shared buffer
                param.grad.data.copy_(all_grads[offset : offset + numel].view_as(param.grad.data))
                # update the offset for the next parameter
                offset += numel
