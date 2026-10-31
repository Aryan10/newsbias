"""Article-level hierarchical document model architecture for political leaning.

2-Level Document Hierarchy:
- Passage Level: Encodes overlapping 512-token windows using shared backbone.
- Document Level: Projects passage vectors, adds passage-position embeddings,
  applies 2-layer TransformerEncoder blocks, and aggregates with gated attention.
- Residual Formulation: Stable mean-probability base + document-level residual head.
"""

from typing import Any

import torch
from torch import nn
from transformers import AutoConfig, AutoModel, PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput


class ArticleConfig(PretrainedConfig):
    model_type = "newsbias_article"

    def __init__(
        self,
        backbone_config: dict[str, Any] | None = None,
        document_width: int = 256,
        document_layers: int = 2,
        document_heads: int = 4,
        position_count: int = 512,
        dropout: float = 0.1,
        passage_microbatch: int = 2,
        **kwargs: Any,
    ) -> None:
        kwargs.setdefault("num_labels", 3)
        super().__init__(**kwargs)
        self.backbone_config = backbone_config or {}
        self.document_width = document_width
        self.document_layers = document_layers
        self.document_heads = document_heads
        self.position_count = position_count
        self.dropout = dropout
        self.passage_microbatch = passage_microbatch


class ArticleModel(PreTrainedModel):
    config_class = ArticleConfig
    base_model_prefix = "encoder"

    def __init__(self, config: ArticleConfig) -> None:
        super().__init__(config)
        backbone = dict(config.backbone_config)
        model_type = backbone.pop("model_type")
        self.encoder = AutoModel.from_config(AutoConfig.for_model(model_type, **backbone))
        hidden = self.encoder.config.hidden_size
        width = config.document_width
        self.passage_head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(hidden, config.num_labels),
        )
        self.projection = nn.Linear(hidden, width)
        self.positions = nn.Embedding(config.position_count, width)
        layer = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=config.document_heads,
            dim_feedforward=4 * width,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.document_encoder = nn.TransformerEncoder(
            layer,
            num_layers=config.document_layers,
            enable_nested_tensor=False,
        )
        self.attention_tanh = nn.Linear(width, 128)
        self.attention_gate = nn.Linear(width, 128)
        self.attention_score = nn.Linear(128, 1, bias=False)
        self.residual = nn.Sequential(
            nn.Linear(width, 128),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(128, config.num_labels),
        )
        self.post_init()
        # Preserve the mean-probability path until the document correction learns.
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)

    @classmethod
    def from_backbone(cls, name: str, **kwargs: Any) -> "ArticleModel":
        revision = kwargs.pop("revision", None)
        encoder = AutoModel.from_pretrained(name, revision=revision)
        model = cls(ArticleConfig(backbone_config=encoder.config.to_dict(), **kwargs))
        model.encoder = encoder
        return model

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        passage_positions: torch.Tensor,
        **_: Any,
    ) -> SequenceClassifierOutput:
        batch, passages, length = input_ids.shape
        valid = attention_mask.any(dim=-1)
        indices = valid.flatten().nonzero(as_tuple=True)[0]
        ids = input_ids.reshape(-1, length)[indices]
        masks = attention_mask.reshape(-1, length)[indices]
        vectors = []
        for start in range(0, len(ids), self.config.passage_microbatch):
            chunk_mask = masks[start:start + self.config.passage_microbatch]
            hidden = self.encoder(
                input_ids=ids[start:start + self.config.passage_microbatch],
                attention_mask=chunk_mask,
            ).last_hidden_state
            weights = chunk_mask.unsqueeze(-1).to(hidden.dtype)
            vectors.append((hidden * weights).sum(1) / weights.sum(1).clamp_min(1))
        encoded = torch.cat(vectors)
        h = encoded.new_zeros(batch * passages, encoded.shape[-1])
        h = h.index_copy(0, indices, encoded).reshape(batch, passages, -1)
        # Float32 softmax/log keeps the residual probability path stable under AMP.
        q = self.passage_head(h).float().softmax(-1)
        mean = (q * valid.unsqueeze(-1)).sum(1) / valid.sum(1, keepdim=True)
        u = self.projection(h) + self.positions(
            passage_positions.clamp(0, self.config.position_count - 1)
        )
        u = self.document_encoder(u, src_key_padding_mask=~valid)
        gated = torch.tanh(self.attention_tanh(u)) * torch.sigmoid(self.attention_gate(u))
        scores = self.attention_score(gated).squeeze(-1).float()
        attention = scores.masked_fill(~valid, float("-inf")).softmax(-1)
        document = (u * attention.unsqueeze(-1).to(u.dtype)).sum(1)
        logits = mean.clamp_min(1e-8).log() + self.residual(document).float()
        return SequenceClassifierOutput(logits=logits)
