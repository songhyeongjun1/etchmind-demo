"""
EtchMind 모델 아키텍처

두 가지 모드:
1. EtchMindSingle: 단일 웨이퍼 (250,) → 분류 + severity (빠른 베이스라인)
2. EtchMindSeq: 시퀀스 (W, 250) → 1D-CNN + Transformer → 분류 + severity + attention
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from model.dataset import N_CLASSES


class EtchMindSingle(nn.Module):
    """
    단일 웨이퍼 모델 (MLP 베이스라인)

    입력: (B, 250)
    출력: class_logits (B, 7), severity (B, 1)
    """

    def __init__(self, n_features: int = 250, n_classes: int = N_CLASSES,
                 hidden_dims: list[int] = [512, 256, 128], dropout: float = 0.3):
        super().__init__()

        layers = []
        in_dim = n_features
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            in_dim = h_dim

        self.backbone = nn.Sequential(*layers)
        self.class_head = nn.Linear(hidden_dims[-1], n_classes)
        self.severity_head = nn.Sequential(
            nn.Linear(hidden_dims[-1], 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        """
        Args:
            x: (B, 250) features

        Returns:
            class_logits: (B, 7)
            severity: (B,)
        """
        h = self.backbone(x)
        class_logits = self.class_head(h)
        severity = self.severity_head(h).squeeze(-1)
        return class_logits, severity


class EtchMindSeq(nn.Module):
    """
    시퀀스 모델 (1D-CNN + Transformer Encoder)

    입력: (B, W, 250)  — W개 웨이퍼의 feature 시퀀스
    출력: class_logits (B, 7), severity (B, 1), attention_weights

    1D-CNN: 로컬 패턴 (인접 웨이퍼 간 변화) 추출
    Transformer: 장기 의존성 (수십 웨이퍼에 걸친 drift) 포착
    """

    def __init__(
        self,
        n_features: int = 250,
        n_classes: int = N_CLASSES,
        # CNN
        cnn_channels: list[int] = [128, 128],
        cnn_kernel: int = 3,
        # Transformer
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.2,
    ):
        super().__init__()

        # === 1D-CNN: 로컬 패턴 추출 ===
        cnn_layers = []
        in_ch = n_features
        for out_ch in cnn_channels:
            cnn_layers.extend([
                nn.Conv1d(in_ch, out_ch, kernel_size=cnn_kernel, padding=cnn_kernel // 2),
                nn.BatchNorm1d(out_ch),
                nn.GELU(),
            ])
            in_ch = out_ch
        self.cnn = nn.Sequential(*cnn_layers)

        # Projection to d_model if needed
        self.proj = nn.Linear(cnn_channels[-1], d_model) if cnn_channels[-1] != d_model else nn.Identity()

        # === Positional Encoding ===
        self.pos_enc = PositionalEncoding(d_model, dropout=dropout, max_len=512)

        # === Transformer Encoder ===
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # === CLS token (시퀀스 요약) ===
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # === Output Heads ===
        self.class_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, n_classes),
        )
        self.severity_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

        # === Sensor Attention (해석가능성) ===
        # 어떤 feature(센서)에 주목하는지
        self.sensor_attention = nn.Sequential(
            nn.Linear(d_model, n_features),
            nn.Softmax(dim=-1),
        )

    def forward(self, x, return_attention: bool = False):
        """
        Args:
            x: (B, W, 250) — wafer sequence
            return_attention: attention weights 반환 여부

        Returns:
            class_logits: (B, 7)
            severity: (B,)
            sensor_attn: (B, 250) — optional, return_attention=True일 때
        """
        B, W, F = x.shape

        # CNN expects (B, C, L) = (B, 250, W)
        h = x.permute(0, 2, 1)          # (B, 250, W)
        h = self.cnn(h)                   # (B, 128, W)
        h = h.permute(0, 2, 1)           # (B, W, 128)
        h = self.proj(h)                  # (B, W, d_model)

        # CLS token prepend
        cls = self.cls_token.expand(B, -1, -1)  # (B, 1, d_model)
        h = torch.cat([cls, h], dim=1)           # (B, W+1, d_model)

        # Positional encoding + Transformer
        h = self.pos_enc(h)
        h = self.transformer(h)           # (B, W+1, d_model)

        # CLS token output → 시퀀스 요약
        cls_out = h[:, 0, :]              # (B, d_model)

        # Heads
        class_logits = self.class_head(cls_out)          # (B, 7)
        severity = self.severity_head(cls_out).squeeze(-1)  # (B,)

        if return_attention:
            sensor_attn = self.sensor_attention(cls_out)  # (B, 250)
            return class_logits, severity, sensor_attn

        return class_logits, severity


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding"""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 512):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class MultiTaskLoss(nn.Module):
    """
    Multi-task Loss: 분류 + severity regression

    L = α * CE(class) + β * MSE(severity, only for fault samples)
    """

    def __init__(self, alpha: float = 1.0, beta: float = 0.5,
                 class_weights: torch.Tensor | None = None):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.ce = nn.CrossEntropyLoss(weight=class_weights)
        self.mse = nn.MSELoss()

    def forward(self, class_logits, severity_pred, labels, severity_true):
        """
        Args:
            class_logits: (B, 7)
            severity_pred: (B,)
            labels: (B,) — 0~6
            severity_true: (B,) — 0~1

        Returns:
            total_loss, {ce_loss, sev_loss}
        """
        ce_loss = self.ce(class_logits, labels)

        # Severity loss: 고장 샘플(label > 0)에만 적용
        fault_mask = labels > 0
        if fault_mask.any():
            sev_loss = self.mse(severity_pred[fault_mask],
                                severity_true[fault_mask])
        else:
            sev_loss = torch.tensor(0.0, device=class_logits.device)

        total = self.alpha * ce_loss + self.beta * sev_loss

        return total, {"ce_loss": ce_loss.item(), "sev_loss": sev_loss.item()}
