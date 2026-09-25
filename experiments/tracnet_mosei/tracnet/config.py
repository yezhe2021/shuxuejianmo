from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class TRACNetConfig:
    text_mode: str = "bert"
    bert_path: str = "E题数据/model"
    freeze_bert: bool = True
    bert_trainable_layers: int = 0
    text_dim: int = 768
    audio_dim: int = 74
    vision_dim: int = 35
    d_model: int = 128
    num_heads: int = 4
    temporal_layers: int = 2
    feedforward_dim: int = 256
    dropout: float = 0.15
    max_seq_len: int = 50
    local_window: int = 2
    num_classes: int = 3
    eps: float = 1e-6

    def __post_init__(self) -> None:
        if self.text_mode not in {"bert", "features"}:
            raise ValueError("text_mode must be 'bert' or 'features'")
        if self.d_model % self.num_heads:
            raise ValueError("d_model must be divisible by num_heads")
        if self.local_window < 0:
            raise ValueError("local_window must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "TRACNetConfig":
        known = cls.__dataclass_fields__
        return cls(**{key: value for key, value in values.items() if key in known})
