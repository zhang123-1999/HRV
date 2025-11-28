import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import logging

# 引入必要的模块
from dataset_loader import RadarECGDataset
from model import IncResUnet

# === 配置区域 ===
DATA_DIR = "results/nature_dataset_120hz" # 确保路径正确
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_DIR = "checkpoints"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ... (Imports) ...
  
def train():
    Path(SAVE_DIR).mkdir(exist_ok=True)

    # 1. Dataset (Loader 现在返回 3 个值)
    train_set = RadarECGDataset(DATA_DIR, mode='train', split_ratio=0.8, test_subjects=5)
    val_set = RadarECGDataset(DATA_DIR, mode='val', split_ratio=0.8, test_subjects=5)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # 2. Model
    model = IncResUnet(in_channels=1, out_channels=1, base_filters=16).to(DEVICE)
    
    # 3. 损失函数
    criterion_dist = nn.SmoothL1Loss() # 距离图损失
    criterion_count = nn.SmoothL1Loss() # 计数损失
    
    # 权重系数：距离图Loss一般在0.01级别，计数Loss可能在1.0级别(差1个峰)
    # 我们希望两者梯度贡献均衡，给计数Loss乘一个小系数
    LAMBDA_COUNT = 0.05 

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    best_val_loss = float('inf')

    for epoch in range(EPOCHS):
        model.train()
        total_loss_train = 0.0
        
        # 注意：这里解包出 3 个变量
        for batch_idx, (data, target_dist, target_count) in enumerate(train_loader):
            data = data.to(DEVICE)
            target_dist = target_dist.to(DEVICE)
            target_count = target_count.to(DEVICE) # [B, 1]

            optimizer.zero_grad()
            
            # Forward 返回两个结果
            pred_dist, pred_count = model(data)
            
            # 计算两个 Loss
            loss_d = criterion_dist(pred_dist, target_dist)
            loss_c = criterion_count(pred_count, target_count)
            
            # 组合 Loss
            loss = loss_d + LAMBDA_COUNT * loss_c
            
            loss.backward()
            optimizer.step()
            
            total_loss_train += loss.item()
        
        avg_train_loss = total_loss_train / len(train_loader)

        # Validation
        model.eval()
        total_loss_val = 0.0
        with torch.no_grad():
            for data, target_dist, target_count in val_loader:
                data = data.to(DEVICE)
                target_dist = target_dist.to(DEVICE)
                target_count = target_count.to(DEVICE)
                
                pred_dist, pred_count = model(data)
                
                loss_d = criterion_dist(pred_dist, target_dist)
                loss_c = criterion_count(pred_count, target_count)
                loss = loss_d + LAMBDA_COUNT * loss_c
                
                total_loss_val += loss.item()
        
        avg_val_loss = total_loss_val / len(val_loader)
        scheduler.step(avg_val_loss)

        logger.info(f"Epoch [{epoch+1}/{EPOCHS}] Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), f"{SAVE_DIR}/best_rpnet.pth")
            logger.info("  >>> Best model saved.")

if __name__ == "__main__":
    train()