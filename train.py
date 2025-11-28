import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import logging

from dataset_loader import RadarECGDataset
from model import IncResUnet

# Configuration
DATA_DIR = "results/nature_dataset_120hz" # 确保这里是你的 .npz 文件夹路径
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_DIR = "checkpoints"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train():
    Path(SAVE_DIR).mkdir(exist_ok=True)

    # 1. Dataset & DataLoader
    # ... (前面的代码不变)
    
    # 修改 Dataset 加载部分
    logger.info("Loading datasets...")
    
    # 训练集：使用 mode='train' (剩下的80%)
    train_set = RadarECGDataset(DATA_DIR, mode='train', split_ratio=0.8, test_size=20)
    
    # 验证集：使用 mode='val' (剩下的20%)  <-- 注意这里改成了 'val'
    val_set = RadarECGDataset(DATA_DIR, mode='val', split_ratio=0.8, test_size=20)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    # 验证集不需要 shuffle
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # ... (后面的代码不变，注意要把下面的 test_loader 变量名改成 val_loader 对应的逻辑)

    logger.info(f"Train samples: {len(train_set)}, Val samples: {len(val_set)}")

    # 2. Model, Loss, Optimizer
    model = IncResUnet(in_channels=1, out_channels=1).to(DEVICE)
    
    # Paper uses SmoothL1Loss [cite: 89]
    criterion = nn.SmoothL1Loss()
    # 修改后：增加 weight_decay
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    best_val_loss = float('inf')

    # 3. Training Loop
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(DEVICE), target.to(DEVICE)

            optimizer.zero_grad()
            output = model(data)
            
            # target is [B, 1, L], output is [B, 1, L]
            loss = criterion(output, target)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)

        # 4. Validation Loop
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(DEVICE), target.to(DEVICE)
                output = model(data)
                loss = criterion(output, target)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)

        logger.info(f"Epoch [{epoch+1}/{EPOCHS}] Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")

        # Save Best Model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), f"{SAVE_DIR}/best_rpnet.pth")
            logger.info("  >>> Best model saved.")

if __name__ == "__main__":
    train()