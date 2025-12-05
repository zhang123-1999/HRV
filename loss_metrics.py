import torch
import torch.nn as nn
import torch.fft

class PhysioLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=0.5, gamma=0.1):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.alpha = alpha # MAE 权重 (准确性)
        self.beta = beta   # 相关性权重 (趋势)
        self.gamma = gamma # 频域权重 (HRV频域指标)

    def pearson_loss(self, x, y):
        # 1 - Pearson Correlation
        x_mean = x - x.mean(dim=1, keepdim=True)
        y_mean = y - y.mean(dim=1, keepdim=True)
        cov = (x_mean * y_mean).sum(dim=1)
        x_var = torch.sqrt((x_mean ** 2).sum(dim=1) + 1e-8)
        y_var = torch.sqrt((y_mean ** 2).sum(dim=1) + 1e-8)
        corr = cov / (x_var * y_var)
        return 1 - corr.mean()

    def freq_loss(self, pred, target):
        # FFT 幅度差异
        pred_fft = torch.fft.rfft(pred, dim=1).abs()
        target_fft = torch.fft.rfft(target, dim=1).abs()
        # 归一化后比较，关注能量分布形状
        pred_fft = pred_fft / (pred_fft.sum(dim=1, keepdim=True) + 1e-8)
        target_fft = target_fft / (target_fft.sum(dim=1, keepdim=True) + 1e-8)
        return self.l1(pred_fft, target_fft)

    def forward(self, pred, target):
        loss_mae = self.l1(pred, target)
        loss_corr = self.pearson_loss(pred, target)
        loss_freq = self.freq_loss(pred, target)
        
        total = self.alpha * loss_mae + self.beta * loss_corr + self.gamma * loss_freq
        return total, {"mae": loss_mae.item(), "corr": loss_corr.item(), "freq": loss_freq.item()}