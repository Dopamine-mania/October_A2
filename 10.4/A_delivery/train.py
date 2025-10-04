# train.py
import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from diffusers import DDIMScheduler
from tqdm import tqdm
import numpy as np

from dataset import TimeSeriesDataset
from model import DiffusionModel

# --- 配置 (Configuration) ---
BATCH_SIZE = 32
NUM_EPOCHS = 10000
LEARNING_RATE = 1e-4
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SAVE_DIR = "checkpoints_sle_baseline"  # checkpoints_sle；checkpoints_sle_no_cross；checkpoints_sle_no_self；checkpoints_sle_baseline
SAVE_EPOCH_FREQ = 500
VAL_FREQ = 100  # 每100个epoch在验证集上评估一次

# --- 准备工作 ---
os.makedirs(SAVE_DIR, exist_ok=True)

# 1. 初始化数据并分割
dataset_path = "./data/平均_SLE拉曼_500-2000_118患者加对照_标签_airPLS_smooth.xlsx"
full_dataset = TimeSeriesDataset(file_path=dataset_path, target_shape=(16, 32))

# 分割数据集为训练集和验证集 (90% 训练, 10% 验证)
train_size = int(0.9 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
print(f"Dataset loaded. Train samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")
print(f"Device set to: {DEVICE}")

# 2. 初始化模型、优化器和调度器
model = DiffusionModel(enable_self_attention=False, enable_cross_attention=False).to(DEVICE)  # 消融实验
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
noise_scheduler = DDIMScheduler(num_train_timesteps=1000, beta_schedule="linear")

# --- 训练与验证 ---
best_val_loss = float('inf')

for epoch in range(NUM_EPOCHS):
    # --- 训练部分 ---
    model.train()
    total_train_loss = 0
    progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS} [T]")
    
    for cond, image in progress_bar:
        cond, image = cond.to(DEVICE), image.to(DEVICE)
        B = cond.size(0)
        t = torch.randint(0, noise_scheduler.config.num_train_timesteps, (B,), device=DEVICE).long()
        noise = torch.randn_like(image)
        noisy_image = noise_scheduler.add_noise(image, noise, t)
        eps_pred = model(noisy_image, t, cond)
        loss = F.mse_loss(eps_pred, noise)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()
        progress_bar.set_postfix(train_loss=loss.item())
    
    avg_train_loss = total_train_loss / len(train_dataloader)

    # --- 验证部分 ---
    if (epoch + 1) % VAL_FREQ == 0:
        model.eval()
        total_val_loss = 0
        val_progress_bar = tqdm(val_dataloader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS} [V]")
        with torch.no_grad():
            for cond, image in val_progress_bar:
                cond, image = cond.to(DEVICE), image.to(DEVICE)
                B = cond.size(0)
                t = torch.randint(0, noise_scheduler.config.num_train_timesteps, (B,), device=DEVICE).long()
                noise = torch.randn_like(image)
                noisy_image = noise_scheduler.add_noise(image, noise, t)
                eps_pred = model(noisy_image, t, cond)
                val_loss = F.mse_loss(eps_pred, noise)
                total_val_loss += val_loss.item()
                val_progress_bar.set_postfix(val_loss=val_loss.item())
        
        avg_val_loss = total_val_loss / len(val_dataloader)
        print(f"Epoch {epoch + 1} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")

        # --- 保存最佳模型 ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_path = os.path.join(SAVE_DIR, "best_model.pt")
            torch.save(model.state_dict(), best_model_path)
            print(f"✨ New best model saved with Val Loss: {best_val_loss:.6f} at {best_model_path}")

    # 定期保存检查点（用于断点续训）
    if (epoch + 1) % SAVE_EPOCH_FREQ == 0:
        ckpt_path = os.path.join(SAVE_DIR, f"model_epoch_{epoch + 1}.pt")
        torch.save(model.state_dict(), ckpt_path)
        print(f"✅ Checkpoint saved to {ckpt_path}")

print("🎉 Training finished!")