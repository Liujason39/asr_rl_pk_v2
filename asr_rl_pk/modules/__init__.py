# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Definitions for neural-network components for RL-agents."""

from .actor_critic import ActorCritic
from .actor_critic_recurrent import ActorCriticRecurrent
from .normalizer import EmpiricalNormalization
from .rnd import RandomNetworkDistillation
from .student_teacher import StudentTeacher
from .student_teacher_recurrent import StudentTeacherRecurrent

from .visual_actor_critic import VisualActorCritic
from .visual_actor_critic_recurrent import VisualActorCriticRecurrent

from .vis_student_teacher import VisStudentTeacher
from .vis_student_teacher_recurrent import VisStudentTeacherRecurrent

from .PrivilegedEstimator import PrivilegedEstimator, VisualMultiHeadEncoder

from .MonolithicPolicy_P1 import monolithicpolicy_p1
from .MonolithicPolicy_P2 import monolithicpolicy_p2
from .MonolithicPolicy_P2_0 import monolithicpolicy_p2_0
from .MonolithicPolicy_P3 import monolithicpolicy_p3
from .MonolithicPolicy_P3_0 import monolithicpolicy_p3_0

from .DWAQ_Policy import dwaq_policy
from .DWAQPP_Policy import dwaqpp_policy
from .DWAQAE_Policy import dwaqae_policy

__all__ = [
    "ActorCritic",
    "ActorCriticRecurrent",
    "EmpiricalNormalization",
    "RandomNetworkDistillation",
    "StudentTeacher",
    "StudentTeacherRecurrent",
    "VisualActorCritic",
    "VisualActorCriticRecurrent",
    "VisStudentTeacher",
    "VisStudentTeacherRecurrent",
    "PrivilegedEstimator",
    "monolithicpolicy_p1",
    "monolithicpolicy_p2",
    "monolithicpolicy_p2_0",
    "monolithicpolicy_p3",
    "monolithicpolicy_p3_0",
    "dwaq_policy",
    "dwaqpp_policy",
    "dwaqae_policy"
]
