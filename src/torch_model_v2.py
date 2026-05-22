"""
Improved hybrid survival model for resume career sequences.
Bidirectional LSTM + Attention + DeepSurv head with BatchNorm.
"""
from __future__ import annotations
from typing import Tuple

import torch
from torch import nn


class CareerLSTMAttentionDeepSurv(nn.Module):
    def __init__(
        self,
        *,
        seq_input_dim: int,
        numeric_dim: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        mlp_hidden: int = 128,
        bidirectional: bool = True,
    ) -> None:
        super().__init__()
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.bidirectional = bidirectional
        self.lstm = nn.LSTM(
            input_size=seq_input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
            bidirectional=bidirectional,
        )
        lstm_out_dim = hidden_size * (2 if bidirectional else 1)
        self.attention = nn.Sequential(
            nn.Linear(lstm_out_dim, lstm_out_dim),
            nn.Tanh(),
            nn.Linear(lstm_out_dim, 1),
        )
        # Numeric branch with BN + Dropout
        self.numeric_bn = nn.BatchNorm1d(numeric_dim) if numeric_dim > 0 else nn.Identity()
        merged_dim = lstm_out_dim + numeric_dim
        self.head = nn.Sequential(
            nn.Linear(merged_dim, mlp_hidden),
            nn.BatchNorm1d(mlp_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, mlp_hidden // 2),
            nn.BatchNorm1d(mlp_hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(mlp_hidden // 2, 1),
        )

    def forward(
        self,
        sequence: torch.Tensor,
        lengths: torch.Tensor,
        numeric: torch.Tensor,
        *,
        return_attention: bool = False,
    ):
        lengths_cpu = lengths.detach().cpu().clamp(min=1)
        packed = nn.utils.rnn.pack_padded_sequence(
            sequence, lengths_cpu, batch_first=True, enforce_sorted=False,
        )
        packed_outputs, _ = self.lstm(packed)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(
            packed_outputs, batch_first=True, total_length=sequence.shape[1],
        )

        time_index = torch.arange(sequence.shape[1], device=sequence.device).unsqueeze(0)
        mask = time_index < lengths.unsqueeze(1)
        attention_logits = self.attention(outputs).squeeze(-1)
        attention_logits = attention_logits.masked_fill(~mask, -1e9)
        attention_weights = torch.softmax(attention_logits, dim=1)
        pooled = torch.sum(outputs * attention_weights.unsqueeze(-1), dim=1)

        numeric_n = self.numeric_bn(numeric) if numeric.shape[1] > 0 else numeric
        merged = torch.cat([pooled, numeric_n], dim=1)
        risk = self.head(merged).squeeze(-1)
        if return_attention:
            return risk, attention_weights
        return risk


def cox_partial_likelihood_loss(
    log_risk: torch.Tensor,
    durations: torch.Tensor,
    events: torch.Tensor,
) -> torch.Tensor:
    """Negative Cox partial log-likelihood, sorted by descending duration."""
    order = torch.argsort(durations, descending=True)
    ordered_log_risk = log_risk[order]
    ordered_events = events[order]
    log_cum_hazard = torch.logcumsumexp(ordered_log_risk, dim=0)
    event_count = torch.clamp(ordered_events.sum(), min=1.0)
    return -torch.sum((ordered_log_risk - log_cum_hazard) * ordered_events) / event_count
