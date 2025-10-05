
import torch
from torch.utils.data import DataLoader, TensorDataset
from diffusers import DDIMScheduler
import pandas as pd
import numpy as np
from tqdm import tqdm
import os

from model import DiffusionModel
from dataset import TimeSeriesDataset

# --- 1. 在这里配置你想要的生成方案 ---
generation_config = {
    # 格式: {类别标签: 生成数量}
    0: 30,  # 生成 30 个 0 类样本
    1: 59,  # 生成 59 个 1 类样本
    2: 29    # 生成 29 个 2 类样本
}
# -----------------------------------------

# --- 其他配置 ---
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 16
MODEL_PATH = "checkpoints_sle/best_model.pt"
DATASET_PATH = "./data/平均_SLE拉曼_500-2000_118患者加对照_标签_airPLS_smooth.xlsx"
OUTPUT_FILE = "generated_custom_dataset.xlsx"

# --- 主逻辑 ---
print(f"Device set to: {DEVICE}")

# 2. 手动创建生成条件的列表
conditions_list = []
for label, count in generation_config.items():
    conditions_list.extend([label] * count)

total_to_generate = len(conditions_list)
if total_to_generate == 0:
    print("生成配置为空，程序退出。")
    exit()

print(f"准备生成 {total_to_generate} 个样本，分布如下: {generation_config}")

# 将标签列表转换为Tensor
conditions_tensor = torch.tensor(conditions_list, dtype=torch.float32).unsqueeze(1)
# 使用TensorDataset和DataLoader来处理我们手动的标签，以便分批处理
condition_dataset = TensorDataset(conditions_tensor)
dataloader = DataLoader(condition_dataset, batch_size=BATCH_SIZE, shuffle=False)

# 3. 加载模型
model = DiffusionModel(enable_self_attention=True, enable_cross_attention=True).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()
print(f"Model loaded from {MODEL_PATH}")

# 4. 初始化调度器
scheduler = DDIMScheduler(num_train_timesteps=1000, beta_schedule="linear")
scheduler.set_timesteps(50)

# 5. 加载一次原始数据集，只为了获取元数据（最大值和原始长度）
#    这是为了正确地反归一化，而不会“污染”生成过程
temp_dataset = TimeSeriesDataset(file_path=DATASET_PATH, target_shape=(16, 32))
original_len = temp_dataset.original_feature_len
max_val = temp_dataset.raw_features_max

# 6. 循环生成新数据
all_generated_spectra = []
for (cond,) in tqdm(dataloader, desc="Generating Custom Data"): # DataLoader现在只提供cond
    cond = cond.to(DEVICE)
    current_batch_size = cond.shape[0]
    
    noisy_image = torch.randn((current_batch_size, 1, 16, 32), device=DEVICE)

    for t in scheduler.timesteps:
        t_batch = torch.full((current_batch_size,), t, device=DEVICE, dtype=torch.long)
        with torch.no_grad():
            eps_pred = model(noisy_image, t_batch, cond)
        noisy_image = scheduler.step(eps_pred, t, noisy_image).prev_sample
    
    for i in range(current_batch_size):
        gen_padded = noisy_image[i].squeeze().cpu().numpy().flatten()
        gen_original = gen_padded[:original_len]
        all_generated_spectra.append(gen_original * max_val)

# 7. 保存结果
df_generated = pd.DataFrame(all_generated_spectra)
# 将标签也保存为第一列，方便后续使用
df_generated.insert(0, 'label', conditions_list)
df_generated.to_excel(OUTPUT_FILE, index=False, header=False)
print(f"\n✅ 成功生成 {len(all_generated_spectra)} 条新样本，并已保存至 '{OUTPUT_FILE}'")