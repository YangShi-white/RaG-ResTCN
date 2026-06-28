"""PyTorch forecasting models for Phase 06."""

from __future__ import annotations

import torch
from torch import nn


class GRUForecaster(nn.Module):
    def __init__(
        self,
        input_dim: int,
        exogenous_dim: int,
        output_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        forecast_mode: str,
    ) -> None:
        super().__init__()
        self.forecast_mode = forecast_mode
        self.encoder = nn.GRU(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        head_in = hidden_dim + (exogenous_dim if forecast_mode == "controlled" else 0)
        self.head = nn.Sequential(
            nn.LayerNorm(head_in),
            nn.Linear(head_in, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, history: torch.Tensor, future_exogenous: torch.Tensor) -> torch.Tensor:
        _, h_n = self.encoder(history)
        encoded = h_n[-1]
        if self.forecast_mode == "controlled":
            encoded = torch.cat([encoded, future_exogenous], dim=-1)
        return self.head(encoded)


class CausalConvBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=self.padding,
            dilation=dilation,
        )
        self.norm = nn.BatchNorm1d(channels)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv(x)
        if self.padding:
            y = y[..., : -self.padding]
        y = self.dropout(self.act(self.norm(y)))
        return x + y


class TCNForecaster(nn.Module):
    def __init__(
        self,
        input_dim: int,
        exogenous_dim: int,
        output_dim: int,
        hidden_dim: int,
        num_layers: int,
        kernel_size: int,
        dropout: float,
        forecast_mode: str,
    ) -> None:
        super().__init__()
        self.forecast_mode = forecast_mode
        self.input_proj = nn.Conv1d(input_dim, hidden_dim, kernel_size=1)
        blocks = []
        for layer in range(num_layers):
            blocks.append(
                CausalConvBlock(
                    channels=hidden_dim,
                    kernel_size=kernel_size,
                    dilation=2**layer,
                    dropout=dropout,
                )
            )
        self.blocks = nn.Sequential(*blocks)
        head_in = hidden_dim + (exogenous_dim if forecast_mode == "controlled" else 0)
        self.head = nn.Sequential(
            nn.LayerNorm(head_in),
            nn.Linear(head_in, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, history: torch.Tensor, future_exogenous: torch.Tensor) -> torch.Tensor:
        x = history.transpose(1, 2)
        x = self.blocks(self.input_proj(x))
        encoded = x[..., -1]
        if self.forecast_mode == "controlled":
            encoded = torch.cat([encoded, future_exogenous], dim=-1)
        return self.head(encoded)


class TransformerForecaster(nn.Module):
    def __init__(
        self,
        input_dim: int,
        exogenous_dim: int,
        output_dim: int,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        dropout: float,
        forecast_mode: str,
        max_len: int = 512,
    ) -> None:
        super().__init__()
        self.forecast_mode = forecast_mode
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.positional = nn.Parameter(torch.zeros(1, max_len, hidden_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        head_in = hidden_dim + (exogenous_dim if forecast_mode == "controlled" else 0)
        self.head = nn.Sequential(
            nn.LayerNorm(head_in),
            nn.Linear(head_in, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, history: torch.Tensor, future_exogenous: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(history)
        x = x + self.positional[:, : x.shape[1], :]
        encoded = self.encoder(x)[:, -1, :]
        if self.forecast_mode == "controlled":
            encoded = torch.cat([encoded, future_exogenous], dim=-1)
        return self.head(encoded)


def build_model(
    model_family: str,
    *,
    input_dim: int,
    exogenous_dim: int,
    output_dim: int,
    forecast_mode: str,
    hyperparameters: dict,
) -> nn.Module:
    if model_family == "gru":
        return GRUForecaster(
            input_dim=input_dim,
            exogenous_dim=exogenous_dim,
            output_dim=output_dim,
            hidden_dim=int(hyperparameters["hidden_dim"]),
            num_layers=int(hyperparameters["num_layers"]),
            dropout=float(hyperparameters["dropout"]),
            forecast_mode=forecast_mode,
        )
    if model_family == "tcn":
        return TCNForecaster(
            input_dim=input_dim,
            exogenous_dim=exogenous_dim,
            output_dim=output_dim,
            hidden_dim=int(hyperparameters["hidden_dim"]),
            num_layers=int(hyperparameters["num_layers"]),
            kernel_size=int(hyperparameters["kernel_size"]),
            dropout=float(hyperparameters["dropout"]),
            forecast_mode=forecast_mode,
        )
    if model_family == "transformer":
        return TransformerForecaster(
            input_dim=input_dim,
            exogenous_dim=exogenous_dim,
            output_dim=output_dim,
            hidden_dim=int(hyperparameters["hidden_dim"]),
            num_layers=int(hyperparameters["num_layers"]),
            num_heads=int(hyperparameters["num_heads"]),
            dropout=float(hyperparameters["dropout"]),
            forecast_mode=forecast_mode,
            max_len=int(hyperparameters.get("max_len", 512)),
        )
    raise ValueError(f"Unknown model_family: {model_family}")
