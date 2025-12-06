import torch
import torch.nn as nn
import numpy as np
from torchvision.models import resnet18

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(1), :].unsqueeze(0)

class SpectroTransNet(nn.Module):
    def __init__(self, time_steps=1200, d_model=256, nhead=8, num_layers=4):
        super().__init__()
        
        # === 1. Encoder (CNN + Transformer) ===
        resnet = resnet18(weights=None)
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.cnn_encoder = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3
        ) # Output: [B, 256, 4, 75]
        
        self.feature_proj = nn.Linear(256 * 4, d_model) 
        self.pos_encoder = PositionalEncoding(d_model, max_len=time_steps)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=1024, dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # === 2. Shared Decoder ===
        # 将 Transformer 特征恢复到 120Hz 的时间分辨率
        self.fc_decode = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        self.upsample_120 = nn.Upsample(size=time_steps, mode='linear', align_corners=True)
        
        # === 3. Branch A: Base Head (120Hz) ===
        # 负责预测基础心率 (BPM, SDNN)
        # 输入: [B, 128, 1200] -> 输出: [B, 1, 1200]
        self.head_base = nn.Sequential(
            nn.Conv1d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 1, kernel_size=1)
        )
        
        # === 4. Branch B: Residual Head (480Hz Super-Res) ===
        # 负责预测高频残差 (RMSSD) 和 不确定性
        # 输入: [B, 128, 1200] -> 输出: [B, 2, 4800]
        self.super_res_body = nn.Sequential(
            # 120Hz -> 240Hz
            nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            # 240Hz -> 480Hz
            nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            # Output: Channel 0 = Residual Mean, Channel 1 = LogVar
            nn.Conv1d(32, 2, kernel_size=3, padding=1)
        )

    def forward(self, x):
        # --- Encoding ---
        feat = self.cnn_encoder(x)
        B, C, F, T = feat.shape
        feat = feat.permute(0, 3, 1, 2).reshape(B, T, -1)
        feat = self.feature_proj(feat)
        feat = self.pos_encoder(feat)
        trans_out = self.transformer(feat) # [B, 75, d_model]
        
        # --- Shared Decoding ---
        decoded = self.fc_decode(trans_out) # [B, 75, 128]
        # 上采样到 120Hz，作为共享特征基底
        base_features = self.upsample_120(decoded.permute(0, 2, 1)) # [B, 128, 1200]
        
        # --- Branch 1: Base Prediction (120Hz) ---
        base_120 = self.head_base(base_features) # [B, 1, 1200]
        
        # --- Branch 2: Residual Prediction (480Hz) ---
        # 预测相对于 base 的微小修正量
        raw_residual = self.super_res_body(base_features) # [B, 2, 4800]
        residual_mean = raw_residual[:, 0, :].unsqueeze(1) # [B, 1, 4800]
        final_logvar  = raw_residual[:, 1, :].unsqueeze(1) # [B, 1, 4800]
        
        # --- Fusion (Base + Residual) ---
        # 1. 把 120Hz Base 线性插值到 480Hz
        base_480 = torch.nn.functional.interpolate(base_120, size=4800, mode='linear', align_corners=True)
        
        # 2. 叠加
        final_mean = base_480 + residual_mean
        
        # 返回: 120Hz基准, 480Hz最终均值, 480Hz不确定性
        return base_120.squeeze(1), final_mean.squeeze(1), final_logvar.squeeze(1)