import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import logging
from pathlib import Path
import numpy as np

# 引入项目模块
from dataset_cwt import RadarSpectrogramDataset
from model_transformer import SpectroTransNet
from loss_metrics import DualResLoss
from test_inference import calculate_rmssd_from_curve

# Config
DATA_DIR = "results/nature_dataset_cleaned"
BATCH_SIZE = 64
EPOCHS = 100
LR = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MILESTONE_UNCERTAINTY = 10 
MILESTONE_DIFF = 30

def train():
    Path("checkpoints_trans").mkdir(exist_ok=True)
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    torch.backends.cudnn.benchmark = True
    
    # 1. Dataset (Oversample=3)
    train_set = RadarSpectrogramDataset(DATA_DIR, mode='train', oversample_factor=3)
    val_set = RadarSpectrogramDataset(DATA_DIR, mode='val')
    
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    # 2. Model
    model = SpectroTransNet().to(DEVICE)
    
    # 3. Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2)
    
    # 4. Loss 配置
    # w_base=5.0:  给基准极高的权重，确保 BPM 绝对不飘 (这是压舱石)
    # w_final=1.0: 细节修补分支
    criterion = DualResLoss(lambda_diff=0.0, w_base=5.0, w_final=1.0).to(DEVICE)
    
    scaler = GradScaler()
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=5e-4, steps_per_epoch=len(train_loader), epochs=EPOCHS)

    best_score = float('inf')

    for epoch in range(EPOCHS):
        model.train()
        
        # === 课程学习 ===
        if epoch < MILESTONE_DIFF:
            current_lambda_diff = 0.0
            phase = "Phase 1: Base Stability"
        else:
            # === 关键修改：大幅提升 Diff 权重 ===
            # 之前是 2.0，现在我们敢给到 10.0 甚至 20.0
            # 因为 Base 分支已经锁死了均值，不用担心高权重导致 BPM 跑偏
            progress = min(1.0, (epoch - MILESTONE_DIFF) / 20.0)
            current_lambda_diff = 10.0 * progress  # <--- 暴力提升细节权重
            phase = f"Phase 2: Texture Synthesis ({current_lambda_diff:.1f})"
            
        criterion.lambda_diff = current_lambda_diff
        logger.info(f"Epoch {epoch+1} | {phase}")
        
        train_loss_acc = 0
        
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            # 确保 y 是高分辨率 (4800)，如果 Dataset 没改，这里需要插值
            # 假设 Dataset 已修改为输出 4800，或在此处插值:
            if y.shape[1] == 1200:
                y = torch.nn.functional.interpolate(y.unsqueeze(1), size=4800, mode='linear', align_corners=True).squeeze(1)
            
            optimizer.zero_grad()
            
            with autocast():
                # Model returns 3 values
                p_base, p_final, p_logvar = model(x)
                
                # Warmup: Disable uncertainty initially
                if epoch < MILESTONE_UNCERTAINTY:
                    dummy_logvar = torch.full_like(p_logvar, -5.0).detach()
                    loss, _ = criterion(p_base, p_final, dummy_logvar, y)
                else:
                    loss, _ = criterion(p_base, p_final, p_logvar, y)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            if not torch.isnan(loss): train_loss_acc += loss.item()
            
        # === Validation ===
        model.eval()
        val_bpm_mae = 0
        val_rmssd_mae = 0
        val_batches = 0
        
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                if y.shape[1] == 1200:
                    y = torch.nn.functional.interpolate(y.unsqueeze(1), size=4800, mode='linear', align_corners=True).squeeze(1)
                
                with autocast():
                    p_base, p_final, _ = model(x)
                    
                    # 1. 验证 BPM: 使用 Final 输出计算 (因为它包含了 base + residual)
                    # 也可以用 p_base 验证，但最终应用肯定是用 p_final
                    val_bpm_mae += torch.mean(torch.abs(p_final.mean(dim=1) - y.mean(dim=1))).item()
                    
                    # 2. 验证 RMSSD (抽样计算)
                    pred_np = p_final.float().cpu().numpy()
                    y_np = y.float().cpu().numpy()
                    
                    batch_r_err = 0
                    cnt = 0
                    for i in range(len(pred_np)):
                        # fs=480.0
                        r_p = calculate_rmssd_from_curve(pred_np[i], fs=480.0)
                        r_t = calculate_rmssd_from_curve(y_np[i], fs=480.0)
                        if not np.isnan(r_p) and not np.isnan(r_t) and r_t < 200:
                            batch_r_err += abs(r_p - r_t)
                            cnt += 1
                    if cnt > 0: val_rmssd_mae += batch_r_err / cnt
                    
                val_batches += 1
                
        avg_bpm = val_bpm_mae / val_batches
        avg_rmssd = val_rmssd_mae / val_batches
        score = avg_bpm + 0.5 * avg_rmssd
        
        logger.info(f"Epoch {epoch+1}: Loss {train_loss_acc/len(train_loader):.4f} | "
                    f"Val BPM: {avg_bpm:.2f} | RMSSD: {avg_rmssd:.2f} | Score: {score:.2f}")
        
        if score < best_score:
            best_score = score
            torch.save(model.state_dict(), "checkpoints_trans/best_model.pth")
            logger.info("  >>> New Best Model Saved!")

if __name__ == "__main__":
    train()