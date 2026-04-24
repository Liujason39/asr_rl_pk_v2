# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch

from asr_rl_pk.utils import split_and_pad_trajectories


class RolloutStorageDWAQ:
    class Transition:
        def __init__(self):
            self.observations = None
            self.privileged_observations = None
            self.gt_heightmap_obs = None
            self.true_velocity_obs = None
            # self.vel_for_actor = None
            self.obs_history = None
            self.next_observations = None
            self.next_obs_valid = None
            self.actions = None
            self.privileged_actions = None
            self.rewards = None
            self.dones = None
            self.values = None
            self.actions_log_prob = None
            self.action_mean = None
            self.action_sigma = None
            self.hidden_states = None
            self.rnd_state = None
            self.mu_p = None
            self.p_boot_mask = None

        def clear(self):
            self.__init__()

    def __init__(
        self,
        training_type,
        num_envs,
        num_transitions_per_env,
        obs_shape,
        privileged_obs_shape,
        gt_heightmap_obs_shape,
        true_velocity_obs_shape,
        actions_shape,
        hist_length,
        rnd_state_shape=None,
        device="cpu",
    ):
        # store inputs
        self.training_type = training_type
        self.device = device
        self.num_transitions_per_env = num_transitions_per_env
        self.num_envs = num_envs
        self.obs_shape = obs_shape
        self.privileged_obs_shape = privileged_obs_shape
        self.rnd_state_shape = rnd_state_shape
        self.gt_heightmap_obs_shape = gt_heightmap_obs_shape
        self.true_velocity_obs_shape = true_velocity_obs_shape
        self.actions_shape = actions_shape
        self.hist_length = hist_length

        # Core
        self.observations = torch.zeros(num_transitions_per_env, num_envs, *obs_shape, device=self.device)
        if privileged_obs_shape is not None:
            self.privileged_observations = torch.zeros(
                num_transitions_per_env, num_envs, *privileged_obs_shape, device=self.device
            )
        else:
            self.privileged_observations = None
        
        # DWAQ-specific buffers
        self.gt_heightmap_observations = torch.zeros(
                num_transitions_per_env, num_envs, *gt_heightmap_obs_shape, device=self.device
            )
        self.true_velocity_observations = torch.zeros(
                num_transitions_per_env, num_envs, *true_velocity_obs_shape, device=self.device
            )
        self.obs_history = torch.zeros(
                num_transitions_per_env, num_envs, self.hist_length, *obs_shape, device=self.device
            )
        # self.vel_for_actor = torch.zeros(
        #         num_transitions_per_env, num_envs, *true_velocity_obs_shape, device=self.device
        #     )
        self.next_observations = torch.zeros(num_transitions_per_env, num_envs, *obs_shape, device=self.device)
        self.next_obs_valid = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.p_boot_mask = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        # self.mu_p = torch.zeros(num_transitions_per_env, num_envs, 32, device=self.device) # 32 fix latent dimension so far, can be changed later

        self.rewards = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.actions = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)
        self.dones = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device).byte()

        # for distillation
        if training_type == "distillation":
            self.privileged_actions = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)

        # for reinforcement learning
        if training_type == "rl":
            self.values = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
            self.actions_log_prob = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
            self.mu = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)
            self.sigma = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)
            self.returns = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
            self.advantages = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)

        # For RND
        if rnd_state_shape is not None:
            self.rnd_state = torch.zeros(num_transitions_per_env, num_envs, *rnd_state_shape, device=self.device)

        # For RNN networks
        self.saved_hidden_states_a = None
        self.saved_hidden_states_c = None

        # counter for the number of transitions stored
        self.step = 0

    def add_transitions(self, transition: Transition):
        # check if the transition is valid
        if self.step >= self.num_transitions_per_env:
            raise OverflowError("Rollout buffer overflow! You should call clear() before adding new transitions.")

        # Core
        self.observations[self.step].copy_(transition.observations)
        self.actions[self.step].copy_(transition.actions)
        self.rewards[self.step].copy_(transition.rewards.view(-1, 1))
        self.dones[self.step].copy_(transition.dones.view(-1, 1))

        if self.privileged_observations is not None:
            self.privileged_observations[self.step].copy_(transition.privileged_observations)
        # for DWAQ-specific buffers
        if self.gt_heightmap_observations is not None:
            self.gt_heightmap_observations[self.step].copy_(transition.gt_heightmap_obs)
        if self.true_velocity_observations is not None:
            self.true_velocity_observations[self.step].copy_(transition.true_velocity_obs)
        if self.obs_history is not None:
            self.obs_history[self.step].copy_(transition.obs_history)
        # if self.vel_for_actor is not None:
        #     self.vel_for_actor[self.step].copy_(transition.vel_for_actor)
        if self.next_observations is not None:
            self.next_observations[self.step].copy_(transition.next_observations)
        if self.next_obs_valid is not None:
            self.next_obs_valid[self.step].copy_(transition.next_obs_valid)
        if self.p_boot_mask is not None:
            self.p_boot_mask[self.step].copy_(transition.p_boot_mask)   
        # self.mu_p[self.step].copy_(transition.mu_p)

        # for distillation
        if self.training_type == "distillation":
            self.privileged_actions[self.step].copy_(transition.privileged_actions)

        # for reinforcement learning
        if self.training_type == "rl":
            self.values[self.step].copy_(transition.values)
            self.actions_log_prob[self.step].copy_(transition.actions_log_prob.view(-1, 1))
            self.mu[self.step].copy_(transition.action_mean)
            self.sigma[self.step].copy_(transition.action_sigma)

        # For RND
        if self.rnd_state_shape is not None:
            self.rnd_state[self.step].copy_(transition.rnd_state)

        # For RNN networks
        self._save_hidden_states(transition.hidden_states)

        # increment the counter
        self.step += 1

    def _save_hidden_states(self, hidden_states):
        if hidden_states is None or hidden_states == (None, None):
            return
        # make a tuple out of GRU hidden state sto match the LSTM format
        hid_a = hidden_states[0] if isinstance(hidden_states[0], tuple) else (hidden_states[0],)
        hid_c = hidden_states[1] if isinstance(hidden_states[1], tuple) else (hidden_states[1],)
        # initialize if needed
        if self.saved_hidden_states_a is None:
            self.saved_hidden_states_a = [
                torch.zeros(self.observations.shape[0], *hid_a[i].shape, device=self.device) for i in range(len(hid_a))
            ]
            self.saved_hidden_states_c = [
                torch.zeros(self.observations.shape[0], *hid_c[i].shape, device=self.device) for i in range(len(hid_c))
            ]
        # copy the states
        for i in range(len(hid_a)):
            self.saved_hidden_states_a[i][self.step].copy_(hid_a[i])
            self.saved_hidden_states_c[i][self.step].copy_(hid_c[i])

    def clear(self):
        self.step = 0

    def compute_returns(self, last_values, gamma, lam, normalize_advantage: bool = True):
        advantage = 0
        for step in reversed(range(self.num_transitions_per_env)):
            # if we are at the last step, bootstrap the return value
            if step == self.num_transitions_per_env - 1:
                next_values = last_values
            else:
                next_values = self.values[step + 1]
            # 1 if we are not in a terminal state, 0 otherwise
            next_is_not_terminal = 1.0 - self.dones[step].float()
            # TD error: r_t + gamma * V(s_{t+1}) - V(s_t)
            delta = self.rewards[step] + next_is_not_terminal * gamma * next_values - self.values[step]
            # Advantage: A(s_t, a_t) = delta_t + gamma * lambda * A(s_{t+1}, a_{t+1})
            advantage = delta + next_is_not_terminal * gamma * lam * advantage
            # Return: R_t = A(s_t, a_t) + V(s_t)
            self.returns[step] = advantage + self.values[step]

        # Compute the advantages
        self.advantages = self.returns - self.values
        # Normalize the advantages if flag is set
        # This is to prevent double normalization (i.e. if per minibatch normalization is used)
        if normalize_advantage:
            self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-8)

    # for distillation
    def generator(self):
        if self.training_type != "distillation":
            raise ValueError("This function is only available for distillation training.")

        for i in range(self.num_transitions_per_env):
            if self.privileged_observations is not None:
                privileged_observations = self.privileged_observations[i]
            else:
                privileged_observations = self.observations[i]
            yield self.observations[i], privileged_observations, self.actions[i], self.privileged_actions[
                i
            ], self.dones[i]

    # for reinforcement learning with feedforward networks
    def mini_batch_generator(self, num_mini_batches, num_epochs=8):
        if self.training_type != "rl":
            raise ValueError("This function is only available for reinforcement learning training.")
        batch_size = self.num_envs * self.num_transitions_per_env
        mini_batch_size = batch_size // num_mini_batches
        indices = torch.randperm(num_mini_batches * mini_batch_size, requires_grad=False, device=self.device)

        # Core
        observations = self.observations.flatten(0, 1)
        if self.privileged_observations is not None:
            privileged_observations = self.privileged_observations.flatten(0, 1)
        else:
            privileged_observations = observations

        actions = self.actions.flatten(0, 1)
        values = self.values.flatten(0, 1)
        returns = self.returns.flatten(0, 1)

        # For PPO
        old_actions_log_prob = self.actions_log_prob.flatten(0, 1)
        advantages = self.advantages.flatten(0, 1)
        old_mu = self.mu.flatten(0, 1)
        old_sigma = self.sigma.flatten(0, 1)

        # for DWAQ-specific buffers
        gt_heightmap_observations = self.gt_heightmap_observations.flatten(0, 1)
        true_velocity_observations = self.true_velocity_observations.flatten(0, 1)
        obs_history = self.obs_history.flatten(0, 1)
        # vel_for_actor = self.vel_for_actor.flatten(0, 1)
        next_observations = self.next_observations.flatten(0, 1)
        next_obs_valid = self.next_obs_valid.flatten(0, 1)
        p_boot_mask = self.p_boot_mask.flatten(0, 1)
        # mu_p = self.mu_p.flatten(0, 1)

        # For RND
        if self.rnd_state_shape is not None:
            rnd_state = self.rnd_state.flatten(0, 1)

        for epoch in range(num_epochs):
            for i in range(num_mini_batches):
                # Select the indices for the mini-batch
                start = i * mini_batch_size
                end = (i + 1) * mini_batch_size
                batch_idx = indices[start:end]

                # Create the mini-batch
                # -- Core
                obs_batch = observations[batch_idx]
                privileged_observations_batch = privileged_observations[batch_idx]
                actions_batch = actions[batch_idx]

                # -- For PPO
                target_values_batch = values[batch_idx]
                returns_batch = returns[batch_idx]
                old_actions_log_prob_batch = old_actions_log_prob[batch_idx]
                advantages_batch = advantages[batch_idx]
                old_mu_batch = old_mu[batch_idx]
                old_sigma_batch = old_sigma[batch_idx]

                # -- For RND
                if self.rnd_state_shape is not None:
                    rnd_state_batch = rnd_state[batch_idx]
                else:
                    rnd_state_batch = None

                # for DWAQ, we also need to yield the following buffers (if they exist):
                """gt_heightmap_observations_batch,
                true_velocity_obs_batch,
                proprio_hist_batch,
                next_propprio_batch,
                vel_for_acto"""
                gt_heightmap_observations_batch = gt_heightmap_observations[batch_idx]
                true_velocity_observations_batch = true_velocity_observations[batch_idx] 
                obs_history_batch = obs_history[batch_idx]
                # vel_for_actor_batch = vel_for_actor[batch_idx] 
                next_obs_batch = next_observations[batch_idx]
                next_obs_valid_batch = next_obs_valid[batch_idx]
                p_boot_mask_batch = p_boot_mask[batch_idx]
                # mu_p_batch = mu_p[batch_idx]
                # yield the mini-batch
                yield (obs_batch, 
                privileged_observations_batch, 
                gt_heightmap_observations_batch, 
                true_velocity_observations_batch, 
                obs_history_batch, 
                next_obs_batch, 
                next_obs_valid_batch,
                p_boot_mask_batch,
                # vel_for_actor_batch, 
                # mu_p_batch,
                actions_batch, 
                target_values_batch, 
                advantages_batch, 
                returns_batch, 
                old_actions_log_prob_batch, 
                old_mu_batch, 
                old_sigma_batch, (
                    None,
                    None,
                ), None, rnd_state_batch)

    # for reinfrocement learning with recurrent networks
    def recurrent_mini_batch_generator(self, num_mini_batches, num_epochs=8, augment: bool = False):
        if self.training_type != "rl":
            raise ValueError("This function is only available for reinforcement learning training.")

        # --- split & pad everything that must align on trajectories ---
        padded_obs_trajectories, trajectory_masks = split_and_pad_trajectories(self.observations, self.dones)

        if self.privileged_observations is not None:
            padded_privileged_obs_trajectories, _ = split_and_pad_trajectories(self.privileged_observations, self.dones)
        else:
            padded_privileged_obs_trajectories = padded_obs_trajectories

        if augment:
            # actions & PPO buffers must be trajectory-aligned too
            padded_actions_trajectories, _ = split_and_pad_trajectories(self.actions, self.dones)
            padded_mu_trajectories, _      = split_and_pad_trajectories(self.mu, self.dones)
            padded_sigma_trajectories, _   = split_and_pad_trajectories(self.sigma, self.dones)
            padded_returns_trajectories, _ = split_and_pad_trajectories(self.returns, self.dones)
            padded_adv_trajectories, _     = split_and_pad_trajectories(self.advantages, self.dones)
            padded_values_trajectories, _  = split_and_pad_trajectories(self.values, self.dones)
            padded_logp_trajectories, _    = split_and_pad_trajectories(self.actions_log_prob, self.dones)

        if self.rnd_state_shape is not None:
            padded_rnd_state_trajectories, _ = split_and_pad_trajectories(self.rnd_state, self.dones)
        else:
            padded_rnd_state_trajectories = None

        mini_batch_size = self.num_envs // num_mini_batches
        if mini_batch_size == 0:
            raise ValueError(
                f"Invalid recurrent mini-batch setting: num_envs={self.num_envs}, "
                f"num_mini_batches={num_mini_batches}. "
                "For recurrent training, num_mini_batches must be <= num_envs."
            )

        for ep in range(num_epochs):
            first_traj = 0
            for i in range(num_mini_batches):
                start = i * mini_batch_size
                stop  = (i + 1) * mini_batch_size

                dones = self.dones.squeeze(-1)
                last_was_done = torch.zeros_like(dones, dtype=torch.bool)
                last_was_done[1:] = dones[:-1]
                last_was_done[0] = True

                # trajectories_batch_size computed from env slice (start:stop)
                trajectories_batch_size = torch.sum(last_was_done[:, start:stop])
                last_traj = first_traj + trajectories_batch_size

                # ---- trajectory-aligned batches (ALL use first_traj:last_traj) ----
                masks_batch         = trajectory_masks[:, first_traj:last_traj]
                obs_batch           = padded_obs_trajectories[:, first_traj:last_traj]
                privileged_obs_batch= padded_privileged_obs_trajectories[:, first_traj:last_traj]


                if padded_rnd_state_trajectories is not None:
                    rnd_state_batch = padded_rnd_state_trajectories[:, first_traj:last_traj]
                else:
                    rnd_state_batch = None
                if augment:
                    actions_batch       = padded_actions_trajectories[:, first_traj:last_traj]
                    old_mu_batch        = padded_mu_trajectories[:, first_traj:last_traj]
                    old_sigma_batch     = padded_sigma_trajectories[:, first_traj:last_traj]
                    returns_batch       = padded_returns_trajectories[:, first_traj:last_traj]
                    advantages_batch    = padded_adv_trajectories[:, first_traj:last_traj]
                    values_batch        = padded_values_trajectories[:, first_traj:last_traj]
                    old_actions_log_prob_batch = padded_logp_trajectories[:, first_traj:last_traj]
                else:
                    actions_batch = self.actions[:, start:stop]
                    old_mu_batch = self.mu[:, start:stop]
                    old_sigma_batch = self.sigma[:, start:stop]
                    returns_batch = self.returns[:, start:stop]
                    advantages_batch = self.advantages[:, start:stop]
                    values_batch = self.values[:, start:stop]
                    old_actions_log_prob_batch = self.actions_log_prob[:, start:stop]
                # ---- hidden states: keep your original logic ----
                last_was_done_env_time = last_was_done.permute(1, 0)
                hid_a_batch = [
                    saved_hidden_states.permute(2, 0, 1, 3)[last_was_done_env_time][first_traj:last_traj]
                    .transpose(1, 0)
                    .contiguous()
                    for saved_hidden_states in self.saved_hidden_states_a
                ]
                hid_c_batch = [
                    saved_hidden_states.permute(2, 0, 1, 3)[last_was_done_env_time][first_traj:last_traj]
                    .transpose(1, 0)
                    .contiguous()
                    for saved_hidden_states in self.saved_hidden_states_c
                ]
                hid_a_batch = hid_a_batch[0] if len(hid_a_batch) == 1 else hid_a_batch
                hid_c_batch = hid_c_batch[0] if len(hid_c_batch) == 1 else hid_c_batch

                if augment:
                    # sanity check (強烈建議保留，抓 shape mismatch)
                    B = obs_batch.shape[1]
                    assert actions_batch.shape[1] == B, (obs_batch.shape, actions_batch.shape)
                    assert masks_batch.shape[1] == B, (masks_batch.shape, obs_batch.shape)
                    assert old_sigma_batch.shape[1] == B, (old_sigma_batch.shape, obs_batch.shape)

                yield (
                    obs_batch,
                    privileged_obs_batch,
                    actions_batch,
                    values_batch,
                    advantages_batch,
                    returns_batch,
                    old_actions_log_prob_batch,
                    old_mu_batch,
                    old_sigma_batch,
                    (hid_a_batch, hid_c_batch),
                    masks_batch,
                    rnd_state_batch,
                )

                first_traj = last_traj