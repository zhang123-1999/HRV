import torch
import torch.nn as nn
import numpy as np
from torchvision.models import resnet18

# PositionalEncoding 类保持不变...
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
        
        # CNN Encoder 保持不变
        resnet = resnet18(weights=None)
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.cnn_encoder = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3
        )
        
        self.feature_proj = nn.Linear(256 * 4, d_model) 
        
        self.pos_encoder = PositionalEncoding(d_model, max_len=time_steps)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=1024, dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Regressor 保持不变
        self.regressor = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1)
        )
        
        self.upsample = nn.Upsample(size=time_steps, mode='linear', align_corners=True)
        
        # === 关键修改：移除 Sigmoid ===
        # 我们希望直接输出 BPM 值
        # self.output_activation = nn.Sigmoid()  <-- 删除这一行

    def forward(self, x):
        feat = self.cnn_encoder(x)
        B, C, F, T = feat.shape
        feat = feat.permute(0, 3, 1, 2).reshape(B, T, -1)
        feat = self.feature_proj(feat)
        
        feat = self.pos_encoder(feat)
        trans_out = self.transformer(feat)
        
        bpm_low = self.regressor(trans_out)
        bpm_high = self.upsample(bpm_low.permute(0, 2, 1))
        
        # === 直接输出原始数值 ===
        return bpm_high.squeeze(1) # Shape: [B, 1200]