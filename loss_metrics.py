import torch
import torch.nn as nn
import torch.fft

class PhysioLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=1.0, gamma=0.1, lambda_diff=10.0, delta_weight=2.0):
        super().__init__()
        # reduction='none' 关键：保留每个样本的 loss，以便后续乘上动态权重
        self.l1 = nn.L1Loss(reduction='none') 
        
        self.alpha = alpha          # MAE 权重 (BPM 准确度)
        self.beta = beta            # Pearson 权重 (整体趋势)
        self.gamma = gamma          # Freq 权重 (频谱分布)
        self.lambda_diff = lambda_diff # Diff 权重 (RMSSD 结构约束)
        
        self.delta_weight = delta_weight # 动态加权系数 (困难样本挖掘力度)

    def pearson_loss(self, x, y):
        # 计算 Pearson Correlation Loss
        x_mean = x - x.mean(dim=1, keepdim=True)
        y_mean = y - y.mean(dim=1, keepdim=True)
        cov = (x_mean * y_mean).sum(dim=1)
        
        eps = 1e-6
        x_var = torch.sqrt((x_mean ** 2).sum(dim=1) + eps)
        y_var = torch.sqrt((y_mean ** 2).sum(dim=1) + eps)
        
        corr = cov / (x_var * y_var + eps)
        return 1 - corr.mean()

    def freq_loss(self, pred, target):
        # 计算频域 Loss
        pred_fft = torch.fft.rfft(pred, dim=1).abs()
        target_fft = torch.fft.rfft(target, dim=1).abs()
        
        eps = 1e-6
        # 归一化频谱能量
        pred_fft = pred_fft / (pred_fft.sum(dim=1, keepdim=True) + eps)
        target_fft = target_fft / (target_fft.sum(dim=1, keepdim=True) + eps)
        
        # 频域通常看整体分布，使用普通的 L1 均值即可
        return nn.L1Loss()(pred_fft, target_fft)

    def diff_loss(self, pred, target):
        """
        计算一阶差分损失 (RMSSD 核心优化项)
        返回: [Batch] 大小的向量，表示每个样本的差分误差
        """
        # diff[t] = y[t] - y[t-1]
        diff_pred = pred[:, 1:] - pred[:, :-1]
        diff_target = target[:, 1:] - target[:, :-1]
        
        # 计算每个样本的平均差分误差
        return torch.abs(diff_pred - diff_target).mean(dim=1)

    def forward(self, pred, target):
        # === 1. 计算动态权重 (Hard Example Mining) ===
        # 逻辑：Target 的 SDNN 越大 (波动越剧烈)，该样本的权重就越大
        # Weight = 1.0 + delta * True_SDNN
        # detach() 很重要，我们只把 std 当作常数系数，不求它的梯度
        target_std = target.std(dim=1).detach() 
        sample_weights = 1.0 + self.delta_weight * target_std
        
        # === 2. 计算各项基础 Loss ===
        
        # A. MAE (BPM) - 形状 [Batch]
        loss_mae_raw = torch.abs(pred - target).mean(dim=1)
        
        # B. Diff (RMSSD) - 形状 [Batch]
        loss_diff_raw = self.diff_loss(pred, target)
        
        # === 3. 加权融合 (强强联合) ===
        # 我们同时对 MAE 和 Diff 应用动态权重
        # 这样既保证了难样本的 BPM 不掉队，又死磕了难样本的 RMSSD 细节
        weighted_mae = (loss_mae_raw * sample_weights).mean()
        weighted_diff = (loss_diff_raw * sample_weights).mean()
        
        # C. 其他 Loss (趋势与频域)
        loss_corr = self.pearson_loss(pred, target)
        loss_freq = self.freq_loss(pred, target)
        
        # === 4. 最终加权求和 ===
        # 注意 lambda_diff 的作用是平衡数值量级 (Diff通常很小)
        total = (self.alpha * weighted_mae) + \
                (self.beta * loss_corr) + \
                (self.gamma * loss_freq) + \
                (self.lambda_diff * weighted_diff)
        
        return total, {
            "mae": weighted_mae.item(), 
            "corr": loss_corr.item(), 
            "freq": loss_freq.item(),
            "diff": weighted_diff.item(),
            "avg_w": sample_weights.mean().item() # 监控平均权重，防止过大
        }