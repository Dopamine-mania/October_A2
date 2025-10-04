import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class TimeSeriesDataset(Dataset):
    def __init__(self, file_path, target_shape=(16, 32)):
        # *修正1: 使用 read_excel 并指定引擎
        # *注意: 这需要安装 openpyxl 包 (pip install openpyxl)
        self.df = pd.read_excel(file_path, header=None, engine='openpyxl')

        self.labels = self.df.iloc[:, 0].values.astype('float32')
        raw_features = self.df.iloc[:, 1:].values.astype('float32')
        
        # --- 数据预处理 ---
        self.raw_features_max = raw_features.max()
        if self.raw_features_max > 0:
            raw_features = raw_features / self.raw_features_max
        
        self.original_feature_len = raw_features.shape[1]
        self.target_len = target_shape[0] * target_shape[1]
        
        padded_features = np.zeros((raw_features.shape[0], self.target_len), dtype=np.float32)
        padded_features[:, :self.original_feature_len] = raw_features
        
        self.features = padded_features
        self.shape = target_shape

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = torch.tensor(self.features[idx], dtype=torch.float32).reshape(self.shape)
        x = x.unsqueeze(0)
        y = torch.tensor(self.labels[idx], dtype=torch.float32).unsqueeze(0)
        return y, x