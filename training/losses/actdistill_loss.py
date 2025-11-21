"""
actdistill_loss.py

Action-Guided Knowledge Distillation (ActDistill) Loss
Implements the distillation loss for transferring knowledge from teacher to student VLA models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple


class ActDistillLoss(nn.Module):
    """
    Action-Guided Knowledge Distillation Loss

    Loss Components:
    - L_align: Instance-level alignment (cosine similarity)
    - L_struct: Relational structure preservation (pairwise similarity)
    - L_act: Action consistency (triple MSE)
    """

    def __init__(
        self,
        num_layers: int,
        alpha: float = 1.0,
        beta: float = 1.0,
        eta: float = 0.5,
        gamma: float = 2.0,
        device: str = 'cuda'
    ):
        """
        Args:
            num_layers: Number of transformer layers
            alpha: Weight for semantic loss
            beta: Weight for action loss
            eta: Weight for structural loss
            gamma: Exponent for layer-wise weighting
            device: Device for computation
        """
        super().__init__()
        self.num_layers = num_layers
        self.alpha = alpha
        self.beta = beta
        self.eta = eta
        self.gamma = gamma

        self.register_buffer(
            'layer_weights',
            torch.tensor([
                (l / (num_layers - 1)) ** gamma if num_layers > 1 else 1.0
                for l in range(num_layers)
            ], device=device)
        )
        if num_layers > 1:
            self.layer_weights = self.layer_weights / self.layer_weights.mean()

    def compute_align_loss(self, s_stu: torch.Tensor, s_tea: torch.Tensor) -> torch.Tensor:
        """Instance-level alignment using cosine similarity."""
        cos_sim = F.cosine_similarity(s_stu, s_tea, dim=-1)
        return (1 - cos_sim).mean()

    def compute_struct_loss(self, S_stu: torch.Tensor, S_tea: torch.Tensor) -> torch.Tensor:
        """Relational structure preservation using pairwise similarity."""
        M_stu = F.cosine_similarity(
            S_stu.unsqueeze(1),
            S_stu.unsqueeze(0),
            dim=-1
        )

        M_tea = F.cosine_similarity(
            S_tea.unsqueeze(1),
            S_tea.unsqueeze(0),
            dim=-1
        )

        return F.mse_loss(M_stu, M_tea)

    def compute_action_loss(
        self,
        a_stu: torch.Tensor,
        a_tea: torch.Tensor,
        a_gt: torch.Tensor,
        a_prev: torch.Tensor
    ) -> torch.Tensor:
        """Action consistency loss (Triple MSE)."""
        loss_gt = F.mse_loss(a_stu, a_gt)
        loss_teacher = F.mse_loss(a_stu, a_tea)
        loss_consistency = F.mse_loss(a_stu, a_prev.detach())
        return loss_gt + loss_teacher + loss_consistency

    def forward(
        self,
        student_outputs: Tuple[List[torch.Tensor], List[torch.Tensor]],
        teacher_outputs: Tuple[List[torch.Tensor], List[torch.Tensor]],
        action_gt: torch.Tensor
    ) -> torch.Tensor:
        """Compute total ActDistill loss."""
        stu_semantics, stu_actions = student_outputs
        tea_semantics, tea_actions = teacher_outputs

        assert len(stu_semantics) == self.num_layers
        assert len(stu_actions) == self.num_layers

        L_sem_total = 0.0
        L_act_total = 0.0

        for l in range(self.num_layers):
            L_align = self.compute_align_loss(stu_semantics[l], tea_semantics[l])
            L_struct = self.compute_struct_loss(stu_semantics[l], tea_semantics[l])
            L_sem = L_align + self.eta * L_struct

            if l > 0:
                a_prev = stu_actions[l - 1]
            else:
                a_prev = torch.zeros_like(stu_actions[0])

            L_act = self.compute_action_loss(
                stu_actions[l],
                tea_actions[l],
                action_gt,
                a_prev
            )

            weight = self.layer_weights[l]
            L_sem_total += weight * self.alpha * L_sem
            L_act_total += weight * self.beta * L_act

        return L_sem_total + L_act_total
