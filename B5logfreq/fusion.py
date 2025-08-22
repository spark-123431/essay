import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.io as sio

from Model import get_global_model
import DataProgress

# ========== 参数设置 ==========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_dir = r"D:\essay\B5logfreq"
material = "3C90"
material_path = os.path.join(data_dir, 'Processed Training Data', material)
valid_file = os.path.join(material_path, "valid.mat")
model_path = os.path.join(data_dir, 'Trained Weights', f"{material}.ckpt")

# 通道名称映射
channel_map = {
    0: "B",
    1: "temp",
    2: "dB",
    3: "d2B",
    4: "h"
}

# ========== 加载数据和模型 ==========
dataset = DataProgress.MagDataset(valid_file)
inputs_all = dataset.x_data.to(device)       # [N, T, 6]
targets_all = dataset.y_data.to(device)      # [N, 1]

model = get_global_model().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# ========== 使用 DataLoader 做 batched baseline 推理 ==========
from torch.utils.data import DataLoader, TensorDataset

batch_size = 64
val_dataset = TensorDataset(inputs_all, targets_all)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

baseline_preds_all = []
targets_all_list = []

with torch.no_grad():
    for xb, yb in val_loader:
        xb = xb.to(device)
        yb = yb.to(device)
        preds = model(xb)
        baseline_preds_all.append(preds.cpu())
        targets_all_list.append(yb.cpu())

baseline_preds_cat = torch.cat(baseline_preds_all, dim=0)
targets_cat = torch.cat(targets_all_list, dim=0)
baseline_mse = torch.mean((baseline_preds_cat - targets_cat) ** 2).item()


# ========== 通道消融分析 ==========
ablation_results = {}

for c in range(5):
    # 通道置零
    inputs_ablate = inputs_all.clone()
    inputs_ablate[:, :, c] = 0.0

    # 用新的 DataLoader 做 batched 推理
    ablate_dataset = TensorDataset(inputs_ablate, targets_all)
    ablate_loader = DataLoader(ablate_dataset, batch_size=batch_size, shuffle=False)

    preds_list = []
    targets_list = []
    with torch.no_grad():
        for xb, yb in ablate_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            preds = model(xb)
            preds_list.append(preds.cpu())
            targets_list.append(yb.cpu())

    preds_all = torch.cat(preds_list, dim=0)
    targets_all_cpu = torch.cat(targets_list, dim=0)
    mse = torch.mean((preds_all - targets_all_cpu) ** 2).item()

    ablation_results[channel_map[c]] = mse


# 添加 baseline 结果
ablation_results["Baseline"] = baseline_mse

# 按 MSE 倒序排序
ablation_results_sorted = dict(sorted(ablation_results.items(), key=lambda x: x[1], reverse=True))

# ========== 输出结果为 DataFrame ==========
df_ablation = pd.DataFrame(list(ablation_results_sorted.items()), columns=["Channel", "MSE"])
print("Ablation Study Results:")
print(df_ablation)

# ========== 可视化 ==========
plt.figure(figsize=(8, 5))
sns.barplot(data=df_ablation[df_ablation["Channel"] != "Baseline"], x="Channel", y="MSE")
plt.axhline(y=baseline_mse, color='red', linestyle='--', label="Baseline")
plt.title(f"Ablation Study (Validation MSE) - {material}")
plt.ylabel("Validation MSE")
plt.xlabel("Removed Channel")
plt.legend()
plt.tight_layout()
plt.show()
