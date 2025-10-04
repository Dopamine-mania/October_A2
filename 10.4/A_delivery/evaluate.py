# evaluation.py
import torch
from torch.utils.data import DataLoader, random_split
from diffusers import DDIMScheduler
import pandas as pd
import numpy as np
from tqdm import tqdm
import os

from model import DiffusionModel
from dataset import TimeSeriesDataset

# 确保你已经安装了 scipy: pip install scipy
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr, wasserstein_distance

# --- 配置 (Configuration) ---
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 16  # 可以适当增大批量，加快评估速度

# !! 关键: 确保这个路径指向你训练过程中保存的“最佳”模型 !!
MODEL_PATH = "checkpoints_sle_baseline/best_model.pt" 

# !! 关键: 确保数据文件路径正确 !!
DATASET_PATH = "./data/平均_SLE拉曼_500-2000_118患者加对照_标签_airPLS_smooth.xlsx"

# --- 函数定义 ---
def calculate_all_metrics(generated_flat, ground_truth_flat):
    """计算所有四个评估指标"""
    # 确保数据是一维的
    generated_flat = generated_flat.flatten()
    ground_truth_flat = ground_truth_flat.flatten()
    
    # 1. 均方误差 (MSE)
    mse = mean_squared_error(ground_truth_flat, generated_flat)
    
    # 2. 余弦相似度 (CS)
    cos_sim = cosine_similarity(ground_truth_flat.reshape(1, -1), generated_flat.reshape(1, -1))[0, 0]
    
    # 3. 皮尔逊相关系数 (PCC)
    pcc, _ = pearsonr(ground_truth_flat, generated_flat)
    
    # 4. 瓦瑟斯坦距离 (WD)
    wd = wasserstein_distance(ground_truth_flat, generated_flat)
    
    return mse, cos_sim, pcc, wd

# --- 主评估逻辑 ---
print(f"Device set to: {DEVICE}")

# 1. 加载数据集，并使用与训练时相同的分割方式获取验证集
full_dataset = TimeSeriesDataset(file_path=DATASET_PATH, target_shape=(16, 32))
train_size = int(0.9 * len(full_dataset))
val_size = len(full_dataset) - train_size
# 设置固定的随机种子，确保每次分割结果都一样
torch.manual_seed(0)
_, val_dataset = random_split(full_dataset, [train_size, val_size])
torch.manual_seed(torch.initial_seed()) # 恢复随机性

dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
print(f"Evaluation will be performed on {len(val_dataset)} validation samples.")

# ----------
# # evaluation.py (修改后，用于完整数据集评估)
# full_dataset = TimeSeriesDataset(file_path=DATASET_PATH, target_shape=(16, 32))


# # 直接使用 full_dataset
# dataloader = DataLoader(full_dataset, batch_size=BATCH_SIZE, shuffle=False)
# print(f"Evaluation will be performed on {len(full_dataset)} FULL samples.")


# 2. 加载模型

model = DiffusionModel(enable_self_attention=False, enable_cross_attention=False).to(DEVICE)    # model = DiffusionModel(enable_self_attention=True, enable_cross_attention=False)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()
print(f"Model loaded from {MODEL_PATH}")

# 3. 初始化调度器
scheduler = DDIMScheduler(num_train_timesteps=1000, beta_schedule="linear")
scheduler.set_timesteps(50)

# 4. 循环生成并评估
all_metrics = []
all_generated_spectra = []
all_ground_truth_spectra = []

for cond, image_gt in tqdm(dataloader, desc="Evaluating"):
    cond, image_gt = cond.to(DEVICE), image_gt.to(DEVICE)
    
    noisy_image = torch.randn(image_gt.shape, device=DEVICE)


    for t in scheduler.timesteps:
        # 获取当前批次大小
        batch_size = noisy_image.shape[0]
        
        # 创建一个与图像批次大小相同的 t 张量
        t_batch = torch.full((batch_size,), t, device=DEVICE, dtype=torch.long)
        
        with torch.no_grad():
            # 将 t_batch 送入模型，而不是单个的 t
            eps_pred = model(noisy_image, t_batch, cond)
            
        noisy_image = scheduler.step(eps_pred, t, noisy_image).prev_sample
    
    # 将一个批次的数据逐个处理
    for i in range(image_gt.shape[0]):
        gen_img = noisy_image[i]
        gt_img = image_gt[i]
        
        original_len = val_dataset.dataset.original_feature_len # 从原始数据集中获取   根据测试验证集或全集更改
        
        gen_padded = gen_img.squeeze().cpu().numpy().flatten()
        gen_original = gen_padded[:original_len]

        gt_padded = gt_img.squeeze().cpu().numpy().flatten()
        gt_original = gt_padded[:original_len]
        
        metrics = calculate_all_metrics(gen_original, gt_original)
        all_metrics.append(metrics)
        
        # 保存光谱用于后续分析（可选）
        max_val = val_dataset.dataset.raw_features_max
        all_generated_spectra.append(gen_original * max_val)
        all_ground_truth_spectra.append(gt_original * max_val)

# 5. 计算并打印平均指标
metrics_array = np.array(all_metrics)
avg_metrics = np.mean(metrics_array, axis=0)

print("\n--- 📊 Final Evaluation Results ---")
print(f"Mean Squared Error (MSE):      {avg_metrics[0]:.6f}")
print(f"Cosine Similarity (CS):        {avg_metrics[1]:.6f}")
print(f"Pearson Correlation (PCC):     {avg_metrics[2]:.6f}")
print(f"Wasserstein Distance (WD):     {avg_metrics[3]:.6f}")
print("------------------------------------")

# 保存生成的数据到Excel
df_generated = pd.DataFrame(all_generated_spectra)
df_generated.to_excel("generated_sle_spectra_baseline.xlsx", index=False, header=False)
print("✅ Generated spectra saved to generated_sle_spectra_baseline.xlsx")