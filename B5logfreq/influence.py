import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from Model import get_global_model
from DataProgress import MagDataset
from captum.attr import IntegratedGradients

# === 设置路径与加载数据 ===
material = "3C90"
data_dir = r"D:\essay\B6log"
valid_file = os.path.join(data_dir, "Processed Training Data", material, "valid.mat")

if not os.path.exists(valid_file):
    raise FileNotFoundError(f"File not found: {valid_file}")

dataset = MagDataset(valid_file)
inputs_all = dataset.x_data  # [N, T, 6]
print(f"Input shape: {inputs_all.shape}")

# === 加载模型 ===
model = get_global_model()
model.eval()

# === 初始化 IG 解释器 ===
ig = IntegratedGradients(model)

# === 设置解释样本数 ===
num_samples = 50
attribution_list = []

for i in range(num_samples):
    input_tensor = inputs_all[i:i+1].clone().detach().requires_grad_(True)

    # 计算当前样本的 attribution
    attributions, delta = ig.attribute(input_tensor, target=0, return_convergence_delta=True)

    # 累加每个通道的 |IG| 值（对时间步求和），注意 detach()
    attr = attributions.abs().sum(dim=1).squeeze().detach().cpu().numpy()  # shape: [6]
    attribution_list.append(attr)


# === 聚合：对所有样本的 attribution 取平均 ===
attr_all = np.stack(attribution_list, axis=0)  # shape: [num_samples, 6]
attr_mean = np.mean(attr_all, axis=0)

# === 通道命名 ===
channel_names = ["B", "freq", "temp", "dB", "d2B", "H"]

# === 输出平均贡献 ===
print("\nAverage Channel Attribution over", num_samples, "samples:")
for i, name in enumerate(channel_names):
    print(f"{name}: {attr_mean[i]:.6f}")

# === 可视化 ===
plt.figure(figsize=(8, 4))
plt.bar(channel_names, attr_mean)
plt.ylabel("Average |IG| Attribution")
plt.title(f"{material}: Average Channel Importance (Integrated Gradients)")
plt.grid(axis='y', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt

attr_single = attributions.squeeze().detach().cpu().numpy()  # shape: [T, 6]

plt.figure(figsize=(10, 4))
plt.plot(attr_single[:, 0], label='B')
plt.plot(attr_single[:, 3], label='dB')
plt.plot(attr_single[:, 4], label='d2B')
plt.plot(attr_single[:, 5], label='H')
plt.legend()
plt.title("Per-timestep Attribution for One Sample")
plt.xlabel("Timestep")
plt.ylabel("IG Value")
plt.grid(True)
plt.tight_layout()
plt.show()
