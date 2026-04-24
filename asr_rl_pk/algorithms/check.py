# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# SPDX-License-Identifier: BSD-3-Clause

import os
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim

from asr_rl_pk.modules import PrivilegedEstimator
from asr_rl_pk.storage import EncoderRolloutStorage


class VisualEncoderbuild_multihead:
    policy: PrivilegedEstimator

    def __init__(
        self,
        policy,
        num_learning_epochs=1,
        gradient_length=15,
        learning_rate=1e-3,
        max_grad_norm=None,
        device="cpu",
        multi_gpu_cfg: dict | None = None,
        save_debug_data: bool = False,
        debug_save_dir: str = "./outputs/visual_encoder_debug",
        train_val_split: float = 0.9,
        val_batches: int = 4,
    ):
        self.device = device
        self.is_multi_gpu = multi_gpu_cfg is not None
        if multi_gpu_cfg is not None:
            self.gpu_global_rank = multi_gpu_cfg["global_rank"]
            self.gpu_world_size = multi_gpu_cfg["world_size"]
        else:
            self.gpu_global_rank = 0
            self.gpu_world_size = 1

        self.policy: PrivilegedEstimator = policy
        self.policy.to(self.device)

        self.storage = None
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.transition = EncoderRolloutStorage.Transition()
        self.last_hidden_states = None

        self.num_learning_epochs = num_learning_epochs
        self.gradient_length = gradient_length
        self.learning_rate = learning_rate
        self.max_grad_norm = max_grad_norm

        self.ce_loss_fn = nn.CrossEntropyLoss()
        self.num_updates = 0

        # debug / validation
        self.save_debug_data = save_debug_data
        self.debug_save_dir = debug_save_dir
        self.train_val_split = train_val_split
        self.val_batches = val_batches
        os.makedirs(self.debug_save_dir, exist_ok=True)

    def init_storage(
        self,
        training_type,
        num_envs,
        num_transitions_per_env,
        student_obs_shape,
        privileged_obs_shape,
        encoder_obs_shape,
        actions_shape,
    ):
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
        _ = self.policy.act(obs).detach()
        self.transition.actions = self.policy.act_inference(obs, encoder_obs).detach()
        self.transition.observations = obs
        self.transition.privileged_observations = privileged_obs
        self.transition.encoder_observations = encoder_obs
        return self.transition.actions

    def process_env_step(self, rewards, dones, infos):
        self.transition.rewards = rewards
        self.transition.dones = dones
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.policy.reset(dones)

    # --------------------------------------------------
    # target reshape
    # privileged_observations shape: (B, 15)
    # [0:7]  : onehot7
    # [7:15] : [cos, sin, front, back, left, right, width, height]
    # --------------------------------------------------
    def convert_onlyfront_target(self, privileged_observations):
        onehot7 = privileged_observations[:, 0:7]              # (B, 7)
        geom8 = privileged_observations[:, 7:15]               # (B, 8)

        # one-hot -> class index for CE loss
        terrain_class = torch.argmax(onehot7, dim=-1).long()   # (B,)

        # onlyfront version: [cos, sin, front, width, height]
        geom5 = torch.stack(
            [
                geom8[:, 0],   # cos
                geom8[:, 1],   # sin
                geom8[:, 2],   # dist_front
                geom8[:, 6],   # width
                geom8[:, 7],   # height
            ],
            dim=-1,
        )  # (B, 5)

        # flat class = 0 -> do not apply geom loss
        geom_mask = (terrain_class != 0).float().unsqueeze(-1)  # (B, 1)

        return terrain_class, geom5, geom_mask

    def _compute_geom_loss_terms(self, geom_pred, geom_target, geom_mask):
        """
        geom_pred/target: (B, 5)
        order: [cos, sin, front, width, height]
        geom_mask: (B, 1), flat=0 else 1
        """
        diff = nn.functional.smooth_l1_loss(
            geom_pred, geom_target, reduction="none"
        )  # (B, 5)

        masked_diff = diff * geom_mask  # broadcast to (B, 5)
        denom = geom_mask.sum().clamp_min(1.0)

        loss_cos = masked_diff[:, 0].sum() / denom
        loss_sin = masked_diff[:, 1].sum() / denom
        loss_front = masked_diff[:, 2].sum() / denom
        loss_width = masked_diff[:, 3].sum() / denom
        loss_height = masked_diff[:, 4].sum() / denom

        loss_geom = (
            loss_cos
            + loss_sin
            + loss_front
            + loss_width
            + loss_height
        )

        return loss_geom, {
            "geom_cos": loss_cos,
            "geom_sin": loss_sin,
            "geom_front": loss_front,
            "geom_width": loss_width,
            "geom_height": loss_height,
        }

    def _save_batch_snapshot(
        self,
        split_name: str,
        update_idx: int,
        batch_idx: int,
        obs: torch.Tensor,
        privileged_observations: torch.Tensor,
        encoder_obs: torch.Tensor,
        terrain_class: torch.Tensor,
        geom_target: torch.Tensor,
        geom_mask: torch.Tensor,
    ):
        if not self.save_debug_data:
            return

        save_path = os.path.join(
            self.debug_save_dir,
            f"{split_name}_update{update_idx:06d}_batch{batch_idx:04d}.pt",
        )
        torch.save(
            {
                "obs": obs.detach().cpu(),
                "privileged_observations": privileged_observations.detach().cpu(),
                "encoder_obs": encoder_obs.detach().cpu(),
                "terrain_class": terrain_class.detach().cpu(),
                "geom_target": geom_target.detach().cpu(),
                "geom_mask": geom_mask.detach().cpu(),
            },
            save_path,
        )

    @torch.no_grad()
    def evaluate_validation(self, max_batches: Optional[int] = None):
        self.policy.eval()

        val_total = 0.0
        val_cls = 0.0
        val_geom = 0.0
        val_acc = 0.0
        val_cos = 0.0
        val_sin = 0.0
        val_front = 0.0
        val_width = 0.0
        val_height = 0.0
        cnt = 0

        max_batches = self.val_batches if max_batches is None else max_batches

        for batch_idx, (obs, privileged_observations, encoder_obs, _, _, dones) in enumerate(
            self.storage.encoder_generator()
        ):
            if batch_idx >= max_batches:
                break

            out = self.policy.encode_encoder(encoder_obs)
            terrain_logits = out["terrain_logits"]   # (B, 7)
            geom_pred = out["geom"]                  # (B, 5)

            terrain_class, geom_target, geom_mask = self.convert_onlyfront_target(privileged_observations)

            loss_cls = self.ce_loss_fn(terrain_logits, terrain_class)
            loss_geom, geom_dict = self._compute_geom_loss_terms(geom_pred, geom_target, geom_mask)
            loss_total = loss_cls + loss_geom

            pred_class = torch.argmax(terrain_logits, dim=-1)
            acc = (pred_class == terrain_class).float().mean()

            val_total += loss_total.item()
            val_cls += loss_cls.item()
            val_geom += loss_geom.item()
            val_acc += acc.item()
            val_cos += geom_dict["geom_cos"].item()
            val_sin += geom_dict["geom_sin"].item()
            val_front += geom_dict["geom_front"].item()
            val_width += geom_dict["geom_width"].item()
            val_height += geom_dict["geom_height"].item()
            cnt += 1

            if batch_idx == 0:
                self._save_batch_snapshot(
                    split_name="val",
                    update_idx=self.num_updates,
                    batch_idx=batch_idx,
                    obs=obs,
                    privileged_observations=privileged_observations,
                    encoder_obs=encoder_obs,
                    terrain_class=terrain_class,
                    geom_target=geom_target,
                    geom_mask=geom_mask,
                )

        self.policy.train()

        if cnt == 0:
            return {}

        return {
            "val_encoder": val_total / cnt,
            "val_cls": val_cls / cnt,
            "val_geom": val_geom / cnt,
            "val_acc": val_acc / cnt,
            "val_geom_cos": val_cos / cnt,
            "val_geom_sin": val_sin / cnt,
            "val_geom_front": val_front / cnt,
            "val_geom_width": val_width / cnt,
            "val_geom_height": val_height / cnt,
        }

    def update(self):
        print(f"Updating encoder policy... (update #{self.num_updates + 1})")
        self.num_updates += 1
        self.policy.train()

        mean_encoder_loss = 0.0
        mean_cls_loss = 0.0
        mean_geom_loss = 0.0
        mean_geom_cos = 0.0
        mean_geom_sin = 0.0
        mean_geom_front = 0.0
        mean_geom_width = 0.0
        mean_geom_height = 0.0
        mean_cls_acc = 0.0
        mean_flat_ratio = 0.0

        accum_loss = 0.0
        cnt = 0
        batch_idx_global = 0

        for epoch in range(self.num_learning_epochs):
            self.policy.reset(hidden_states=self.last_hidden_states)
            self.policy.detach_hidden_states()

            for obs, privileged_observations, encoder_obs, _, _, dones in self.storage.encoder_generator():
                out = self.policy.encode_encoder(encoder_obs)
                terrain_logits = out["terrain_logits"]   # (B, 7)
                geom_pred = out["geom"]                  # (B, 5)

                terrain_class, geom_target, geom_mask = self.convert_onlyfront_target(privileged_observations)

                # 1) classification loss
                loss_cls = self.ce_loss_fn(terrain_logits, terrain_class)

                # 2) geom loss with flat mask
                loss_geom, geom_dict = self._compute_geom_loss_terms(
                    geom_pred, geom_target, geom_mask
                )

                # total
                encoder_loss = loss_cls + loss_geom
                accum_loss = accum_loss + encoder_loss

                # metrics
                with torch.no_grad():
                    pred_class = torch.argmax(terrain_logits, dim=-1)
                    cls_acc = (pred_class == terrain_class).float().mean()
                    flat_ratio = (terrain_class == 0).float().mean()

                mean_encoder_loss += encoder_loss.item()
                mean_cls_loss += loss_cls.item()
                mean_geom_loss += loss_geom.item()
                mean_geom_cos += geom_dict["geom_cos"].item()
                mean_geom_sin += geom_dict["geom_sin"].item()
                mean_geom_front += geom_dict["geom_front"].item()
                mean_geom_width += geom_dict["geom_width"].item()
                mean_geom_height += geom_dict["geom_height"].item()
                mean_cls_acc += cls_acc.item()
                mean_flat_ratio += flat_ratio.item()
                cnt += 1

                if self.save_debug_data and batch_idx_global == 0:
                    self._save_batch_snapshot(
                        split_name="train",
                        update_idx=self.num_updates,
                        batch_idx=batch_idx_global,
                        obs=obs,
                        privileged_observations=privileged_observations,
                        encoder_obs=encoder_obs,
                        terrain_class=terrain_class,
                        geom_target=geom_target,
                        geom_mask=geom_mask,
                    )

                # gradient step
                if cnt % self.gradient_length == 0:
                    self.optimizer.zero_grad()
                    accum_loss.backward()

                    if self.is_multi_gpu:
                        self.reduce_parameters()

                    if self.max_grad_norm:
                        nn.utils.clip_grad_norm_(
                            self.policy.visual_encoder.parameters(),
                            self.max_grad_norm,
                        )

                    self.optimizer.step()
                    self.policy.detach_hidden_states()
                    accum_loss = 0.0

                self.policy.reset(dones.view(-1))
                self.policy.detach_hidden_states(dones.view(-1))
                batch_idx_global += 1

        # flush remainder
        if isinstance(accum_loss, torch.Tensor):
            if accum_loss.requires_grad:
                self.optimizer.zero_grad()
                accum_loss.backward()
                if self.is_multi_gpu:
                    self.reduce_parameters()
                if self.max_grad_norm:
                    nn.utils.clip_grad_norm_(
                        self.policy.visual_encoder.parameters(),
                        self.max_grad_norm,
                    )
                self.optimizer.step()

        mean_encoder_loss /= max(cnt, 1)
        mean_cls_loss /= max(cnt, 1)
        mean_geom_loss /= max(cnt, 1)
        mean_geom_cos /= max(cnt, 1)
        mean_geom_sin /= max(cnt, 1)
        mean_geom_front /= max(cnt, 1)
        mean_geom_width /= max(cnt, 1)
        mean_geom_height /= max(cnt, 1)
        mean_cls_acc /= max(cnt, 1)
        mean_flat_ratio /= max(cnt, 1)

        self.storage.clear()
        self.last_hidden_states = self.policy.get_hidden_states()
        self.policy.detach_hidden_states()

        loss_dict = {
            "encoder": mean_encoder_loss,
            "cls": mean_cls_loss,
            "geom": mean_geom_loss,
            "geom_cos": mean_geom_cos,
            "geom_sin": mean_geom_sin,
            "geom_front": mean_geom_front,
            "geom_width": mean_geom_width,
            "geom_height": mean_geom_height,
            "cls_acc": mean_cls_acc,
            "flat_ratio": mean_flat_ratio,
        }

        # optional validation
        val_dict = self.evaluate_validation()
        loss_dict.update(val_dict)

        return loss_dict

    def broadcast_parameters(self):
        model_params = [self.policy.state_dict()]
        torch.distributed.broadcast_object_list(model_params, src=0)
        self.policy.load_state_dict(model_params[0])

    def reduce_parameters(self):
        grads = [param.grad.view(-1) for param in self.policy.parameters() if param.grad is not None]
        all_grads = torch.cat(grads)
        torch.distributed.all_reduce(all_grads, op=torch.distributed.ReduceOp.SUM)
        all_grads /= self.gpu_world_size

        offset = 0
        for param in self.policy.parameters():
            if param.grad is not None:
                numel = param.numel()
                param.grad.data.copy_(all_grads[offset: offset + numel].view_as(param.grad.data))
                offset += numel