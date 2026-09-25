from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .model import TRACNetOutput


def pearson_loss(prediction: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    prediction = prediction - prediction.mean()
    target = target - target.mean()
    numerator = (prediction * target).sum()
    denominator = torch.sqrt(prediction.square().sum() * target.square().sum()).clamp_min(eps)
    return 1.0 - numerator / denominator


@dataclass
class LossWeights:
    regression: float = 1.0
    pearson: float = 0.2
    consistency: float = 0.5
    consistency_regression: float = 1.0
    reconstruction: float = 0.2
    unimodal: float = 0.1
    reliability: float = 0.05
    temperature: float = 2.0


class TRACNetCriterion(nn.Module):
    def __init__(self, weights: LossWeights | None = None) -> None:
        super().__init__()
        self.weights = weights or LossWeights()

    def task_loss(
        self,
        output: TRACNetOutput,
        class_target: torch.Tensor,
        regression_target: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        classification = F.cross_entropy(output.logits, class_target)
        regression = F.smooth_l1_loss(output.regression, regression_target)
        correlation = pearson_loss(output.regression, regression_target)
        total = classification + self.weights.regression * (
            regression + self.weights.pearson * correlation
        )
        return total, {
            "classification": classification,
            "regression": regression,
            "pearson": correlation,
        }

    def forward(
        self,
        reference: TRACNetOutput,
        corrupted: TRACNetOutput,
        batch: dict[str, torch.Tensor],
        artificial_missing: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        class_target = batch["classification_label"]
        regression_target = batch["regression_label"]
        ref_task, ref_parts = self.task_loss(reference, class_target, regression_target)
        cor_task, cor_parts = self.task_loss(corrupted, class_target, regression_target)
        task = 0.5 * (ref_task + cor_task)

        temperature = self.weights.temperature
        teacher = torch.softmax(reference.logits.detach() / temperature, dim=-1)
        consistency_class = F.kl_div(
            torch.log_softmax(corrupted.logits / temperature, dim=-1),
            teacher,
            reduction="batchmean",
        ) * (temperature**2)
        consistency_reg = F.smooth_l1_loss(
            corrupted.regression, reference.regression.detach()
        )
        consistency = consistency_class + self.weights.consistency_regression * consistency_reg

        missing = artificial_missing.bool()
        if missing.any():
            reconstruction = F.smooth_l1_loss(
                corrupted.compensated[missing], reference.encoded.detach()[missing]
            )
        else:
            reconstruction = corrupted.compensated.sum() * 0.0

        class_targets = class_target[:, None].expand(-1, 3).reshape(-1)
        regression_targets = regression_target[:, None].expand(-1, 3)
        unimodal = F.cross_entropy(
            reference.unimodal_logits.reshape(-1, reference.unimodal_logits.shape[-1]),
            class_targets,
        ) + self.weights.regression * F.smooth_l1_loss(
            reference.unimodal_regression, regression_targets
        )

        valid = batch["padding_mask"][:, None, :].expand_as(corrupted.observed_reliability)
        reliability_target = corrupted.effective_reliability.new_zeros(
            corrupted.observed_reliability.shape
        )
        reliability_target.copy_(
            (batch["observed_mask"] & ~artificial_missing).to(reliability_target.dtype)
        )
        reliability = F.binary_cross_entropy_with_logits(
            corrupted.observed_reliability_logits[valid], reliability_target[valid]
        )

        total = (
            task
            + self.weights.consistency * consistency
            + self.weights.reconstruction * reconstruction
            + self.weights.unimodal * unimodal
            + self.weights.reliability * reliability
        )
        parts = {
            "total": total.detach(),
            "task": task.detach(),
            "classification_ref": ref_parts["classification"].detach(),
            "classification_cor": cor_parts["classification"].detach(),
            "regression_ref": ref_parts["regression"].detach(),
            "regression_cor": cor_parts["regression"].detach(),
            "pearson_ref": ref_parts["pearson"].detach(),
            "pearson_cor": cor_parts["pearson"].detach(),
            "consistency": consistency.detach(),
            "reconstruction": reconstruction.detach(),
            "unimodal": unimodal.detach(),
            "reliability": reliability.detach(),
        }
        return total, parts
