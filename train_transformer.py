import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import logging
from pathlib import Path
import time

# 引入新的模块
from dataset_cwt import RadarSpectrogramDataset
from model_transformer import SpectroTransNet
from loss_metrics import PhysioLoss # 确保 loss_metrics.py 文件在目录下

# Config
DATA_DIR = "results/nature_dataset_cleaned"
BATCH_SIZE = 64
EPOCHS = 100
LR = 1e-4
DEVICE = "cuda"

def train():
    Path("checkpoints_trans").mkdir(exist_ok=True)
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    torch.backends.cudnn.benchmark = True
    
    # Dataset & Loader
    train_set = RadarSpectrogramDataset(DATA_DIR, mode='train')
    val_set = RadarSpectrogramDataset(DATA_DIR, mode='val')
    
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    # Model
    model = SpectroTransNet().to(DEVICE)
    
    # Optimizer & Loss
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2)
    
    # === 关键修改：使用 PhysioLoss ===
    # alpha=MAE, beta=Pearson, gamma=Freq
    criterion = PhysioLoss(alpha=1.0, beta=1.0, gamma=0.1).to(DEVICE)
    
    scaler = GradScaler()
    # === 建议修改 1: 降低一点学习率 ===
    # Transformer 对 LR 很敏感，1e-3 有点激进，改成 5e-4 或 1e-4 更稳
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=5e-4, steps_per_epoch=len(train_loader), epochs=EPOCHS)

    best_mae = float('inf')

    for epoch in range(EPOCHS):
        model.train()
        train_loss_acc = 0
        
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(DEVICE), y.to(DEVICE)
            
            optimizer.zero_grad()
            
            # 1. Forward
            with autocast():
                pred = model(x)
                loss, _ = criterion(pred, y)
            
            # 2. Backward
            scaler.scale(loss).backward()
            
            # === 关键修改：梯度裁剪 (Gradient Clipping) ===
            # 先将梯度从缩放状态还原 (Unscale)
            scaler.unscale_(optimizer)
            
            # 对梯度进行裁剪，最大范数设为 1.0 (常用值)
            # 这能有效防止梯度爆炸导致的 NaN
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # 3. Step
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            # === 调试：检测是否出现 NaN ===
            if torch.isnan(loss):
                logger.error(f"Loss is NaN at batch {batch_idx}!")
                logger.error(f"Pred range: {pred.min().item()} - {pred.max().item()}")
                logger.error(f"Target range: {y.min().item()} - {y.max().item()}")
                # 遇到 NaN 跳过这一步，不要让模型权重被污染 (可选 break)
                # break 
            
            train_loss_acc += loss.item()
            
        # Validation
        model.eval()
        val_loss_acc = 0
        mae_acc = 0
        corr_acc = 0
        
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                with autocast():
                    pred = model(x)
                    loss, metrics = criterion(pred, y)

                val_loss_acc += loss.item()
                # 记录具体的物理指标
                mae_acc += metrics['mae']
                corr_acc += (1 - metrics['corr']) # 还原回 correlation (越高越好)
        
        avg_loss = train_loss_acc / len(train_loader)
        avg_mae = mae_acc / len(val_loader)
        avg_corr = corr_acc / len(val_loader)
        
        logger.info(f"Epoch {epoch+1}: Loss {avg_loss:.4f} | "
                    f"Val MAE: {avg_mae:.2f} BPM | Corr: {avg_corr:.3f}")
        
        if avg_mae < best_mae:
            best_mae = avg_mae
            torch.save(model.state_dict(), "checkpoints_trans/best_model.pth")

if __name__ == "__main__":
    train()