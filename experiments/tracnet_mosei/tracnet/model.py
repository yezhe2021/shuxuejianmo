from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .config import TRACNetConfig


@dataclass
class TRACNetOutput:
    logits: torch.Tensor
    regression: torch.Tensor
    evidence: torch.Tensor
    modality_attention: torch.Tensor
    temporal_attention: torch.Tensor
    observed_reliability: torch.Tensor
    observed_reliability_logits: torch.Tensor
    compensation_confidence: torch.Tensor
    effective_reliability: torch.Tensor
    encoded: torch.Tensor
    compensated: torch.Tensor
    unimodal_logits: torch.Tensor
    unimodal_regression: torch.Tensor


def masked_softmax(
    logits: torch.Tensor, mask: torch.Tensor, dim: int, eps: float = 1e-8
) -> torch.Tensor:
    masked = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
    weights = torch.softmax(masked, dim=dim)
    weights = weights * mask.to(weights.dtype)
    return weights / weights.sum(dim=dim, keepdim=True).clamp_min(eps)


def masked_mean(values: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    weights = mask.to(values.dtype).unsqueeze(-1)
    return (values * weights).sum(dim=dim) / weights.sum(dim=dim).clamp_min(1.0)


class LocalCrossModalCompensator(nn.Module):
    def __init__(self, config: TRACNetConfig) -> None:
        super().__init__()
        self.window = config.local_window
        self.d_model = config.d_model
        self.attention = nn.MultiheadAttention(
            config.d_model,
            config.num_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(config.d_model)
        self.null_context = nn.Parameter(torch.zeros(1, 1, config.d_model))
        self.confidence = nn.Sequential(
            nn.Linear(config.d_model + 1, config.d_model // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model // 2, 1),
        )

    def _windows(self, values: torch.Tensor) -> torch.Tensor:
        # [B,T,D] -> [B,T,2w+1,D]
        padded = F.pad(values, (0, 0, self.window, self.window))
        return torch.stack(
            [padded[:, offset : offset + values.shape[1]] for offset in range(2 * self.window + 1)],
            dim=2,
        )

    def _mask_windows(self, mask: torch.Tensor) -> torch.Tensor:
        padded = F.pad(mask, (self.window, self.window), value=False)
        return torch.stack(
            [padded[:, offset : offset + mask.shape[1]] for offset in range(2 * self.window + 1)],
            dim=2,
        )

    def forward(
        self,
        query: torch.Tensor,
        source_values: list[torch.Tensor],
        source_masks: list[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, seq_len, dim = query.shape
        windows = torch.cat([self._windows(value) for value in source_values], dim=2)
        masks = torch.cat([self._mask_windows(mask) for mask in source_masks], dim=2)
        coverage = masks.float().mean(dim=2, keepdim=True)

        flat_values = windows.reshape(batch * seq_len, -1, dim)
        flat_masks = masks.reshape(batch * seq_len, -1)
        null = self.null_context.expand(batch * seq_len, -1, -1)
        flat_values = torch.cat([flat_values, null], dim=1)
        flat_masks = torch.cat(
            [flat_masks, torch.ones(batch * seq_len, 1, dtype=torch.bool, device=query.device)],
            dim=1,
        )
        flat_query = query.reshape(batch * seq_len, 1, dim)
        compensated, _ = self.attention(
            flat_query,
            flat_values,
            flat_values,
            key_padding_mask=~flat_masks,
            need_weights=False,
        )
        compensated = self.norm(compensated.reshape(batch, seq_len, dim) + query)
        confidence = torch.sigmoid(
            self.confidence(torch.cat([compensated, coverage], dim=-1)).squeeze(-1)
        )
        confidence = confidence * (coverage.squeeze(-1) > 0).to(confidence.dtype)
        return compensated, confidence


class TRACNet(nn.Module):
    modality_names = ("text", "audio", "vision")

    def __init__(self, config: TRACNetConfig) -> None:
        super().__init__()
        self.config = config
        self.bert: nn.Module | None = None
        if config.text_mode == "bert":
            from transformers import BertModel

            bert_path = Path(config.bert_path).expanduser().resolve()
            self.bert = BertModel.from_pretrained(bert_path, local_files_only=True)
            self._configure_bert_trainability()

        self.projections = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(config.text_dim, config.d_model), nn.LayerNorm(config.d_model)),
                nn.Sequential(nn.Linear(config.audio_dim, config.d_model), nn.LayerNorm(config.d_model)),
                nn.Sequential(nn.Linear(config.vision_dim, config.d_model), nn.LayerNorm(config.d_model)),
            ]
        )
        self.position_embedding = nn.Embedding(config.max_seq_len, config.d_model)
        self.missing_tokens = nn.Parameter(torch.empty(3, config.d_model))
        nn.init.normal_(self.missing_tokens, std=0.02)

        def temporal_encoder() -> nn.TransformerEncoder:
            layer = nn.TransformerEncoderLayer(
                d_model=config.d_model,
                nhead=config.num_heads,
                dim_feedforward=config.feedforward_dim,
                dropout=config.dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            return nn.TransformerEncoder(layer, num_layers=config.temporal_layers)

        self.temporal_encoders = nn.ModuleList([temporal_encoder() for _ in range(3)])
        # Reliability measures observation quality. Cross-modal semantic
        # disagreement is deliberately kept out of this estimator because
        # sarcasm and emotional incongruity can be valid evidence.
        reliability_in = config.d_model + 1
        self.observed_reliability = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(reliability_in, config.d_model),
                    nn.GELU(),
                    nn.Dropout(config.dropout),
                    nn.Linear(config.d_model, 1),
                )
                for _ in range(3)
            ]
        )
        self.compensators = nn.ModuleList(
            [LocalCrossModalCompensator(config) for _ in range(3)]
        )
        self.fusion_scores = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(config.d_model * 3, config.d_model // 2),
                    nn.GELU(),
                    nn.Linear(config.d_model // 2, 1),
                )
                for _ in range(3)
            ]
        )
        self.temporal_score = nn.Sequential(
            nn.Linear(config.d_model, config.d_model // 2),
            nn.Tanh(),
            nn.Linear(config.d_model // 2, 1),
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(config.d_model),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model, config.num_classes),
        )
        self.regressor = nn.Sequential(
            nn.LayerNorm(config.d_model),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model, 1),
        )
        self.unimodal_classifiers = nn.ModuleList(
            [nn.Linear(config.d_model, config.num_classes) for _ in range(3)]
        )
        self.unimodal_regressors = nn.ModuleList(
            [nn.Linear(config.d_model, 1) for _ in range(3)]
        )

    def _configure_bert_trainability(self) -> None:
        assert self.bert is not None
        for parameter in self.bert.parameters():
            parameter.requires_grad = not self.config.freeze_bert
        if self.config.freeze_bert and self.config.bert_trainable_layers > 0:
            layers = self.bert.encoder.layer[-self.config.bert_trainable_layers :]
            for layer in layers:
                for parameter in layer.parameters():
                    parameter.requires_grad = True

    def _text_features(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        if self.config.text_mode == "features":
            return batch["text"]
        assert self.bert is not None
        bert = batch["text_bert"].long()
        input_ids = bert[:, 0]
        attention_mask = bert[:, 1]
        token_type_ids = bert[:, 2]
        # Avoid an all-masked row in pathological fully missing text samples.
        no_tokens = attention_mask.sum(dim=1) == 0
        if no_tokens.any():
            attention_mask = attention_mask.clone()
            attention_mask[no_tokens, 0] = 1
        frozen = not any(p.requires_grad for p in self.bert.parameters())
        if frozen:
            self.bert.eval()
        context = torch.no_grad() if frozen else torch.enable_grad()
        with context:
            return self.bert(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                return_dict=True,
            ).last_hidden_state

    def _encode(
        self,
        raw: torch.Tensor,
        modality: int,
        padding_mask: torch.Tensor,
        observed_mask: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        projected = self.projections[modality](raw)
        pos = self.position_embedding(positions)[None, :, :]
        missing = self.missing_tokens[modality][None, None, :] + pos
        values = torch.where(observed_mask.unsqueeze(-1), projected + pos, missing)
        values = values * padding_mask.unsqueeze(-1).to(values.dtype)
        return self.temporal_encoders[modality](
            values, src_key_padding_mask=~padding_mask
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> TRACNetOutput:
        padding = batch["padding_mask"].bool()
        observed = batch["observed_mask"].bool() & padding[:, None, :]
        seq_len = padding.shape[1]
        if seq_len > self.config.max_seq_len:
            raise ValueError(f"Sequence length {seq_len} exceeds max_seq_len")
        positions = torch.arange(seq_len, device=padding.device)
        raw_modalities = [self._text_features(batch), batch["audio"], batch["vision"]]
        encoded = [
            self._encode(raw, m, padding, observed[:, m], positions)
            for m, raw in enumerate(raw_modalities)
        ]

        observed_reliability: list[torch.Tensor] = []
        observed_reliability_logits: list[torch.Tensor] = []
        compensated: list[torch.Tensor] = []
        compensation_confidence: list[torch.Tensor] = []
        pos = self.position_embedding(positions)[None, :, :]
        for m in range(3):
            others = [index for index in range(3) if index != m]
            rel_input = torch.cat(
                [
                    encoded[m],
                    observed[:, m].unsqueeze(-1).float(),
                ],
                dim=-1,
            )
            q_obs_logits = self.observed_reliability[m](rel_input).squeeze(-1)
            q_obs = torch.sigmoid(q_obs_logits)
            query = (self.missing_tokens[m][None, None, :] + pos).expand(
                padding.shape[0], -1, -1
            )
            comp, q_comp = self.compensators[m](
                query,
                [encoded[index] for index in others],
                [observed[:, index] for index in others],
            )
            observed_reliability.append(q_obs)
            observed_reliability_logits.append(q_obs_logits)
            compensated.append(comp)
            compensation_confidence.append(q_comp)

        h = torch.stack(encoded, dim=1)
        h_comp = torch.stack(compensated, dim=1)
        q_obs = torch.stack(observed_reliability, dim=1)
        q_obs_logits = torch.stack(observed_reliability_logits, dim=1)
        q_comp = torch.stack(compensation_confidence, dim=1)
        obs_float = observed.float()
        # Observed low-confidence features can be softly repaired; missing features
        # use compensation directly. This is distinct from fusion confidence.
        h_bar = obs_float.unsqueeze(-1) * (
            q_obs.unsqueeze(-1) * h + (1.0 - q_obs).unsqueeze(-1) * h_comp
        ) + (1.0 - obs_float).unsqueeze(-1) * h_comp
        effective_reliability = padding[:, None, :].float() * (
            obs_float * (q_obs + (1.0 - q_obs) * q_comp)
            + (1.0 - obs_float) * q_comp
        )

        fusion_scores = []
        for m in range(3):
            others = [index for index in range(3) if index != m]
            consensus = 0.5 * (h_bar[:, others[0]] + h_bar[:, others[1]])
            fusion_input = torch.cat(
                [h_bar[:, m], consensus, torch.abs(h_bar[:, m] - consensus)], dim=-1
            )
            fusion_scores.append(self.fusion_scores[m](fusion_input).squeeze(-1))
        score = torch.stack(fusion_scores, dim=1)
        score = score + torch.log(effective_reliability.clamp_min(self.config.eps))
        modality_mask = padding[:, None, :].expand_as(score)
        alpha = masked_softmax(score, modality_mask, dim=1, eps=self.config.eps)
        fused = (alpha.unsqueeze(-1) * h_bar).sum(dim=1)
        time_logits = self.temporal_score(fused).squeeze(-1)
        beta = masked_softmax(time_logits, padding, dim=1, eps=self.config.eps)
        global_state = (beta.unsqueeze(-1) * fused).sum(dim=1)
        logits = self.classifier(global_state)
        regression = self.regressor(global_state).squeeze(-1).clamp(-3.0, 3.0)

        unimodal_states = [
            masked_mean(encoded[m], observed[:, m], dim=1) for m in range(3)
        ]
        unimodal_logits = torch.stack(
            [self.unimodal_classifiers[m](unimodal_states[m]) for m in range(3)], dim=1
        )
        unimodal_regression = torch.stack(
            [self.unimodal_regressors[m](unimodal_states[m]).squeeze(-1) for m in range(3)],
            dim=1,
        ).clamp(-3.0, 3.0)
        evidence = alpha * beta[:, None, :]
        return TRACNetOutput(
            logits=logits,
            regression=regression,
            evidence=evidence,
            modality_attention=alpha,
            temporal_attention=beta,
            observed_reliability=q_obs,
            observed_reliability_logits=q_obs_logits,
            compensation_confidence=q_comp,
            effective_reliability=effective_reliability,
            encoded=h,
            compensated=h_comp,
            unimodal_logits=unimodal_logits,
            unimodal_regression=unimodal_regression,
        )
