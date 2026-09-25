from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from .config import TRACNetConfig
from .model import masked_softmax


@dataclass
class BaselineOutput:
    logits: torch.Tensor
    regression: torch.Tensor
    neutral_logit: torch.Tensor | None = None
    polarity_logit: torch.Tensor | None = None


class FrozenBertEncoder(nn.Module):
    def __init__(self, bert_path: str) -> None:
        super().__init__()
        from transformers import BertModel

        self.bert = BertModel.from_pretrained(
            Path(bert_path).expanduser().resolve(), local_files_only=True
        )
        for parameter in self.bert.parameters():
            parameter.requires_grad = False

    def forward(self, text_bert: torch.Tensor) -> torch.Tensor:
        self.bert.eval()
        values = text_bert.long()
        input_ids = values[:, 0]
        attention_mask = values[:, 1]
        token_type_ids = values[:, 2]
        no_tokens = attention_mask.sum(dim=1) == 0
        if no_tokens.any():
            attention_mask = attention_mask.clone()
            attention_mask[no_tokens, 0] = 1
        with torch.no_grad():
            return self.bert(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                return_dict=True,
            ).last_hidden_state


class PredictionHeads(nn.Module):
    def __init__(self, d_model: int, num_classes: int, dropout: float) -> None:
        super().__init__()
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )
        self.regressor = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(self, state: torch.Tensor) -> BaselineOutput:
        return BaselineOutput(
            logits=self.classifier(state),
            regression=self.regressor(state).squeeze(-1).clamp(-3.0, 3.0),
        )


class HierarchicalPredictionHeads(nn.Module):
    """Factor three classes into neutral detection and non-neutral polarity."""

    def __init__(self, d_model: int, dropout: float) -> None:
        super().__init__()
        self.shared = nn.Sequential(nn.LayerNorm(d_model), nn.Dropout(dropout))
        self.neutral = nn.Linear(d_model, 1)
        self.polarity = nn.Linear(d_model, 1)
        self.regressor = nn.Linear(d_model, 1)

    def forward(self, state: torch.Tensor) -> BaselineOutput:
        hidden = self.shared(state)
        neutral_logit = self.neutral(hidden).squeeze(-1)
        polarity_logit = self.polarity(hidden).squeeze(-1)
        neutral = torch.sigmoid(neutral_logit)
        positive_given_emotional = torch.sigmoid(polarity_logit)
        emotional = 1.0 - neutral
        probabilities = torch.stack(
            [
                emotional * (1.0 - positive_given_emotional),
                neutral,
                emotional * positive_given_emotional,
            ],
            dim=-1,
        )
        return BaselineOutput(
            logits=torch.log(probabilities.clamp_min(1e-7)),
            regression=self.regressor(hidden).squeeze(-1).clamp(-3.0, 3.0),
            neutral_logit=neutral_logit,
            polarity_logit=polarity_logit,
        )


class TextOnlyBaseline(nn.Module):
    """Frozen BERT plus projection and masked attention pooling."""

    def __init__(self, config: TRACNetConfig) -> None:
        super().__init__()
        self.bert = FrozenBertEncoder(config.bert_path)
        self.projection = nn.Sequential(
            nn.Linear(config.text_dim, config.d_model),
            nn.GELU(),
            nn.LayerNorm(config.d_model),
            nn.Dropout(config.dropout),
        )
        self.attention = nn.Sequential(
            nn.Linear(config.d_model, config.d_model // 2),
            nn.Tanh(),
            nn.Linear(config.d_model // 2, 1),
        )
        self.heads = PredictionHeads(
            config.d_model, config.num_classes, config.dropout
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> BaselineOutput:
        text = self.projection(self.bert(batch["text_bert"]))
        mask = batch["observed_mask"][:, 0].bool() & batch["padding_mask"].bool()
        scores = self.attention(text).squeeze(-1)
        weights = masked_softmax(scores, mask, dim=1)
        state = (weights.unsqueeze(-1) * text).sum(dim=1)
        return self.heads(state)


class SimpleMultimodalBaseline(nn.Module):
    """Masked early fusion without reliability estimation or compensation."""

    def __init__(self, config: TRACNetConfig) -> None:
        super().__init__()
        self.config = config
        self.bert = FrozenBertEncoder(config.bert_path)
        self.projections = nn.ModuleList(
            [
                nn.Linear(config.text_dim, config.d_model),
                nn.Linear(config.audio_dim, config.d_model),
                nn.Linear(config.vision_dim, config.d_model),
            ]
        )
        self.fusion = nn.Sequential(
            nn.Linear(config.d_model * 3 + 3, config.d_model),
            nn.GELU(),
            nn.LayerNorm(config.d_model),
            nn.Dropout(config.dropout),
        )
        self.position_embedding = nn.Embedding(config.max_seq_len, config.d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.num_heads,
            dim_feedforward=config.feedforward_dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(layer, num_layers=1)
        self.attention = nn.Sequential(
            nn.Linear(config.d_model, config.d_model // 2),
            nn.Tanh(),
            nn.Linear(config.d_model // 2, 1),
        )
        self.heads = PredictionHeads(
            config.d_model, config.num_classes, config.dropout
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> BaselineOutput:
        padding = batch["padding_mask"].bool()
        observed = batch["observed_mask"].bool() & padding[:, None, :]
        raw = [self.bert(batch["text_bert"]), batch["audio"], batch["vision"]]
        projected = [
            self.projections[index](values)
            * observed[:, index].unsqueeze(-1).to(values.dtype)
            for index, values in enumerate(raw)
        ]
        fused = self.fusion(
            torch.cat(projected + [observed.permute(0, 2, 1).float()], dim=-1)
        )
        positions = torch.arange(fused.shape[1], device=fused.device)
        fused = fused + self.position_embedding(positions)[None, :, :]
        fused = self.temporal_encoder(fused, src_key_padding_mask=~padding)
        scores = self.attention(fused).squeeze(-1)
        weights = masked_softmax(scores, padding, dim=1)
        state = (weights.unsqueeze(-1) * fused).sum(dim=1)
        return self.heads(state)


class TextAnchorHierarchicalBaseline(nn.Module):
    """Use text as the prediction anchor and other modalities as gated corrections."""

    def __init__(self, config: TRACNetConfig) -> None:
        super().__init__()
        self.bert = FrozenBertEncoder(config.bert_path)
        self.text_projection = nn.Sequential(
            nn.Linear(config.text_dim, config.d_model),
            nn.GELU(),
            nn.LayerNorm(config.d_model),
            nn.Dropout(config.dropout),
        )
        self.other_projections = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(input_dim, config.d_model),
                    nn.GELU(),
                    nn.LayerNorm(config.d_model),
                    nn.Dropout(config.dropout),
                )
                for input_dim in (config.audio_dim, config.vision_dim)
            ]
        )
        self.position_embedding = nn.Embedding(config.max_seq_len, config.d_model)
        self.pooling = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(config.d_model, config.d_model // 2),
                    nn.Tanh(),
                    nn.Linear(config.d_model // 2, 1),
                )
                for _ in range(3)
            ]
        )
        comparison_dim = config.d_model * 3
        self.corrections = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(comparison_dim, config.d_model),
                    nn.Tanh(),
                    nn.Dropout(config.dropout),
                )
                for _ in range(2)
            ]
        )
        self.gates = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(comparison_dim, config.d_model // 2),
                    nn.GELU(),
                    nn.Linear(config.d_model // 2, 1),
                    nn.Sigmoid(),
                )
                for _ in range(2)
            ]
        )
        self.output_norm = nn.LayerNorm(config.d_model)
        self.heads = HierarchicalPredictionHeads(config.d_model, config.dropout)

    def _pool(
        self, values: torch.Tensor, mask: torch.Tensor, index: int
    ) -> torch.Tensor:
        scores = self.pooling[index](values).squeeze(-1)
        weights = masked_softmax(scores, mask, dim=1)
        return (weights.unsqueeze(-1) * values).sum(dim=1)

    def forward(self, batch: dict[str, torch.Tensor]) -> BaselineOutput:
        padding = batch["padding_mask"].bool()
        observed = batch["observed_mask"].bool() & padding[:, None, :]
        positions = torch.arange(padding.shape[1], device=padding.device)
        position = self.position_embedding(positions)[None, :, :]

        text_sequence = self.text_projection(self.bert(batch["text_bert"])) + position
        audio_sequence = self.other_projections[0](batch["audio"]) + position
        vision_sequence = self.other_projections[1](batch["vision"]) + position
        sequences = (text_sequence, audio_sequence, vision_sequence)
        states = [
            self._pool(sequence, observed[:, index], index)
            for index, sequence in enumerate(sequences)
        ]
        text_state = states[0]
        fused = text_state
        for index, other_state in enumerate(states[1:]):
            comparison = torch.cat(
                [text_state, other_state, torch.abs(text_state - other_state)], dim=-1
            )
            fused = fused + self.gates[index](comparison) * self.corrections[index](
                comparison
            )
        return self.heads(self.output_norm(fused))


def build_baseline(name: str, config: TRACNetConfig) -> nn.Module:
    if name == "text_only":
        return TextOnlyBaseline(config)
    if name == "simple_multimodal":
        return SimpleMultimodalBaseline(config)
    if name == "hierarchical_multimodal":
        model = SimpleMultimodalBaseline(config)
        model.heads = HierarchicalPredictionHeads(config.d_model, config.dropout)
        return model
    if name == "text_anchor_hierarchical":
        return TextAnchorHierarchicalBaseline(config)
    raise ValueError(f"Unknown baseline: {name}")
