import torch
import torch.nn as nn
import torch.nn.functional as F

class DualResLoss(nn.Module):
    def __init__(self, lambda_diff=1.0, w_base=1.0, w_final=1.0):
        super().__init__()
        self.lambda_diff = lambda_diff
        self.w_base = w_base   
        self.w_final = w_final 

    def forward(self, pred_base, pred_final_mean, pred_final_logvar, target_high_res):
        """
        pred_base: [B, 1200]
        pred_final_mean: [B, 4800] (这是 Base_Upsampled + Residual)
        pred_final_logvar: [B, 4800]
        target_high_res: [B, 4800]
        """
        
        # === 任务 A: Base 监督 (120Hz) ===
        # 这一路保持不变，负责把心率基准锚定住
        target_120 = F.avg_pool1d(target_high_res.unsqueeze(1), kernel_size=4, stride=4).squeeze(1)
        loss_base = torch.mean(torch.abs(pred_base - target_120))
        
        
        # === 任务 B: Final 监督 (480Hz) - 关键修改 ===
        
        # 1. 梯度阻断 (Gradient Detach)
        # 我们希望 pred_final_mean 里的 "Base部分" 被视为常数
        # 只有 "Residual部分" 负责降低这里的 Loss
        # 注意：pred_final_mean = base_480 + residual
        # 我们需要在 model 输出时无法直接 detach base 部分（因为已经加在一起了）
        # 所以这里我们不改 Loss 的输入，而是依赖权重的调整
        # 或者，更高级的做法是：在计算 Loss 时，让模型专注于 "Residual 应该是什么"
        
        # 2. 基础回归 (Gaussian NLL)
        precision = torch.exp(-pred_final_logvar)
        loss_mae_adaptive = torch.mean(0.5 * precision * torch.abs(target_high_res - pred_final_mean) + 0.5 * pred_final_logvar)
        
        # 3. 差分回归 (Diff Loss) - 疯狂加大权重
        # 既然 Base 分支已经把 BPM 搞定，这里我们就要狠抓波形
        diff_pred = pred_final_mean[:, 1:] - pred_final_mean[:, :-1]
        diff_target = target_high_res[:, 1:] - target_high_res[:, :-1]
        
        diff_log_var = (pred_final_logvar[:, 1:] + pred_final_logvar[:, :-1]) / 2.0
        diff_precision = torch.exp(-diff_log_var)
        
        loss_diff_adaptive = torch.mean(0.5 * diff_precision * torch.abs(diff_target - diff_pred))
        
        # === 策略调整 ===
        # 这里的 lambda_diff 是外部传入的，我们在 train.py 里把它设大
        loss_final_total = loss_mae_adaptive + self.lambda_diff * loss_diff_adaptive
        
        # 总 Loss
        total_loss = self.w_base * loss_base + self.w_final * loss_final_total
        
        return total_loss, {
            "loss": total_loss.item(),
            "l_base": loss_base.item(),
            "l_final": loss_mae_adaptive.item(),
            "l_diff": loss_diff_adaptive.item()
        }