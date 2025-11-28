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

def train():
    Path(SAVE_DIR).mkdir(exist_ok=True)

    # 1. 加载数据集
    logger.info("Loading datasets...")
    
    # === 修正点：参数名改为 test_subjects ===
    # 训练集：加载除了最后 5 个受试者之外的所有数据 (Resting + Tilt + ...)
    train_set = RadarECGDataset(
        DATA_DIR, 
        mode='train', 
        split_ratio=0.8, 
        test_subjects=5 
    )
    
    # 验证集：同上，从训练受试者中分出 20%
    val_set = RadarECGDataset(
        DATA_DIR, 
        mode='val', 
        split_ratio=0.8, 
        test_subjects=5
    )

    # DataLoader
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    logger.info(f"Train samples: {len(train_set)}, Val samples: {len(val_set)}")

    # 2. 模型、损失、优化器
    # 使用包含 Attention 的 IncResUnet (确保 model.py 也是最新的)
    model = IncResUnet(in_channels=1, out_channels=1, base_filters=16).to(DEVICE)
    
    criterion = nn.SmoothL1Loss()
    # 加入 L2 正则化 (weight_decay) 防止过拟合
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    
    # 可选：学习率调度器 (如果 Loss 不降，自动减小 LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    best_val_loss = float('inf')

    # 3. 训练循环
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(DEVICE), target.to(DEVICE)

            optimizer.zero_grad()
            output = model(data)
            
            loss = criterion(output, target)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)

        # 4. 验证循环
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(DEVICE), target.to(DEVICE)
                output = model(data)
                loss = criterion(output, target)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        
        # 更新学习率
        scheduler.step(avg_val_loss)

        current_lr = optimizer.param_groups[0]['lr']
        logger.info(f"Epoch [{epoch+1}/{EPOCHS}] Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f} | LR: {current_lr:.6e}")

        # 保存最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), f"{SAVE_DIR}/best_rpnet.pth")
            logger.info("  >>> Best model saved.")

if __name__ == "__main__":
    train()