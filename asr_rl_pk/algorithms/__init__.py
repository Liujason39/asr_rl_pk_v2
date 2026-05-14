# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Implementation of different RL agents."""

from .distillation import Distillation
from .ppo import PPO
from .TrainVisualEncoder import VisualEncoderbuild
from .TrainVisualEncoder_multihead import VisualEncoderbuild_multihead
from .ppo_monolith import PPO_Monolith
from .distillation_monolith import Distillation_Monolith
from .ppo_dwaq import PPO_DWAQ
from .ppo_dwaqpp import PPO_DWAQPP
from .ppo_dwaqae import PPO_DWAQAE

__all__ = ["PPO", "Distillation", "VisualEncoderbuild", "VisualEncoderbuild_multihead", "PPO_Monolith", "Distillation_Monolith", "PPO_DWAQ", "PPO_DWAQPP", "PPO_DWAQAE"]
