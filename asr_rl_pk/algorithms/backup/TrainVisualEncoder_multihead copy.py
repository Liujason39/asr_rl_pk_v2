# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# torch
from asr_rl_pk.algorithms.TrainVisualEncoder import convert_onlyfront_target
import torch
import torch.nn as nn
import torch.optim as optim

# rsl-rl
from asr_rl_pk.modules import StudentTeacher, StudentTeacherRecurrent, PrivilegedEstimator
from asr_rl_pk.storage import RolloutStorage, EncoderRolloutStorage


class VisualEncoderbuild_multihead:
    """Distillation algorithm for training a student model to mimic a teacher model."""

    policy: PrivilegedEstimator# | StudentTeacherRecurrent
    """The student teacher model."""

    def __init__(
        self,
        policy,
        num_learning_epochs=1,
        gradient_length=15,
        learning_rate=1e-3,
        max_grad_norm=None,
        loss_type="mse",
        device="cpu",
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

        self.rnd = None  # TODO: remove when runner has a proper base class

        # distillation components
        self.policy : PrivilegedEstimator= policy
        self.policy.to(self.device)
        self.storage = None  # initialized later
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.transition = EncoderRolloutStorage.Transition()
        self.last_hidden_states = None

        # distillation parameters
        self.num_learning_epochs = num_learning_epochs
        self.gradient_length = gradient_length
        self.learning_rate = learning_rate
        self.max_grad_norm = max_grad_norm

        # initialize the loss function
        self.ce_loss_fn = nn.CrossEntropyLoss()
        self.reg_loss_fn = nn.SmoothL1Loss()
        if loss_type == "mse":
            self.loss_fn = nn.functional.mse_loss
        elif loss_type == "huber":
            self.loss_fn = nn.functional.huber_loss
        else:
            raise ValueError(f"Unknown loss type: {loss_type}. Supported types are: mse, huber")

        self.num_updates = 0

    def init_storage(
        self, training_type, num_envs, num_transitions_per_env, student_obs_shape, privileged_obs_shape, encoder_obs_shape, actions_shape
    ):
        # create rollout storage
        self.storage = EncoderRolloutStorage(
            training_type,
            num_envs,
            num_transitions_per_env,
            student_obs_shape,
            privileged_obs_shape,
            encoder_obs_shape,
            actions_shape,
            None,
            self.device,
        )

    def act(self, obs, privileged_obs, encoder_obs):
        # this is for distribution upgate for log, should be clean later
        _ = self.policy.act(obs).detach()
        # compute the actions
        self.transition.actions = self.policy.act_inference(obs, encoder_obs).detach()
        # self.transition.privileged_actions = self.policy.evaluate(teacher_obs).detach()
        # record the observations
        self.transition.observations = obs
        self.transition.privileged_observations = privileged_obs
        self.transition.encoder_observations = encoder_obs
        return self.transition.actions

    def process_env_step(self, rewards, dones, infos):
        # record the rewards and dones
        self.transition.rewards = rewards
        self.transition.dones = dones
        # record the transition
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.policy.reset(dones)

    def convert_onlyfront_target(self, privileged_observations):
        '''for target reshape'''
        # privileged_observations: (B, 13)

        onehot7 = privileged_observations[:, 0:7]      # ring/gap/stair/inv/pit/box
        geom_target = privileged_observations[:, 7:15]
        # convert to classfication target for terrain information

        return onehot7, geom_target

    # amend for encoder loss between encoder output and privileged observations
    def update(self):
        print(f"Updating encoder policy... (update #{self.num_updates + 1})")
        self.num_updates += 1
        mean_encoder_loss = 0.0
        loss = 0.0
        cnt = 0

        for epoch in range(self.num_learning_epochs):
            self.policy.reset(hidden_states=self.last_hidden_states)
            self.policy.detach_hidden_states()
            for obs, privileged_observations, encoder_obs, _, _, dones in self.storage.encoder_generator():

                # # inference the student for gradient computation
                # actions = self.policy.act_inference(obs)

                # output of visual encoder for multihead loss, and divide to different group for different loss 
                estimated_privileged_observations = self.policy.encode_encoder(encoder_obs)
                terrain_logits = estimated_privileged_observations["terrain_logits"]   # (B, 7)
                geom_pred = estimated_privileged_observations["geom"]                  # (B, 8)

                # input of terrain information, will be converted to classfication target, and regression target for geometry information 
                terrain_class, geom_target = self.convert_onlyfront_target(privileged_observations)
                terrain_class = torch.argmax(terrain_class, dim=-1)

                loss_cls = self.ce_loss_fn(terrain_logits, terrain_class)
                loss_reg = self.reg_loss_fn(geom_pred, geom_target)

                encoder_loss = loss_cls + loss_reg

                # total loss
                loss = loss + encoder_loss
                mean_encoder_loss += encoder_loss.item()
                cnt += 1

                # gradient step
                if cnt % self.gradient_length == 0:
                    self.optimizer.zero_grad()
                    loss.backward()
                    if self.is_multi_gpu:
                        self.reduce_parameters()
                    if self.max_grad_norm:
                        # change this to clip the gradients of the visual encoder only
                        nn.utils.clip_grad_norm_(self.policy.visual_encoder.parameters(), self.max_grad_norm)
                    self.optimizer.step()
                    self.policy.detach_hidden_states()
                    loss = 0

                # reset dones
                self.policy.reset(dones.view(-1))
                self.policy.detach_hidden_states(dones.view(-1))

        mean_encoder_loss /= cnt
        self.storage.clear()
        self.last_hidden_states = self.policy.get_hidden_states()
        self.policy.detach_hidden_states()

        # construct the loss dictionary
        loss_dict = {"encoder": mean_encoder_loss}

        return loss_dict

    """
    Helper functions
    """

    def broadcast_parameters(self):
        """Broadcast model parameters to all GPUs."""
        # obtain the model parameters on current GPU
        model_params = [self.policy.state_dict()]
        # broadcast the model parameters
        torch.distributed.broadcast_object_list(model_params, src=0)
        # load the model parameters on all GPUs from source GPU
        self.policy.load_state_dict(model_params[0])

    def reduce_parameters(self):
        """Collect gradients from all GPUs and average them.

        This function is called after the backward pass to synchronize the gradients across all GPUs.
        """
        # Create a tensor to store the gradients
        grads = [param.grad.view(-1) for param in self.policy.parameters() if param.grad is not None]
        all_grads = torch.cat(grads)
        # Average the gradients across all GPUs
        torch.distributed.all_reduce(all_grads, op=torch.distributed.ReduceOp.SUM)
        all_grads /= self.gpu_world_size
        # Update the gradients for all parameters with the reduced gradients
        offset = 0
        for param in self.policy.parameters():
            if param.grad is not None:
                numel = param.numel()
                # copy data back from shared buffer
                param.grad.data.copy_(all_grads[offset : offset + numel].view_as(param.grad.data))
                # update the offset for the next parameter
                offset += numel


