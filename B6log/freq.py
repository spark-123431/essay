import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

# ========== 文件路径 ==========
base_dir = r"D:\essay\materials\3C90"
freq_file = os.path.join(base_dir, "Frequency[Hz].csv")
loss_file = os.path.join(base_dir, "Volumetric_losses[Wm-3].csv")
temp_file = os.path.join(base_dir, "Temperature[C].csv")

# ========== 数据加载 ==========
freq = pd.read_csv(freq_file, header=None).squeeze().values
loss = pd.read_csv(loss_file, header=None).squeeze().values
temp = pd.read_csv(temp_file, header=None).squeeze().values + 273.15  # 转为 K 防止 log(负数)

# ========== 对数变换 ==========
log_freq = np.log(freq + 1e-8)
log_loss = np.log(loss + 1e-8)
log_temp = np.log(temp + 1e-8)

# ========== 绘图 ==========
fig, axs = plt.subplots(3, 2, figsize=(12, 10), dpi=150)
plt.rcParams["font.family"] = "Times New Roman"

# ==== 频率 ====
axs[0, 0].hist(freq, bins=50, color='skyblue', edgecolor='black')
axs[0, 0].set_title("Original Frequency Distribution")
axs[0, 0].set_xlabel("Frequency [Hz]")
axs[0, 0].set_ylabel("Count")
axs[0, 0].grid(True)

axs[0, 1].hist(log_freq, bins=50, color='salmon', edgecolor='black')
axs[0, 1].set_title("Log-Transformed Frequency")
axs[0, 1].set_xlabel("log(Frequency)")
axs[0, 1].set_ylabel("Count")
axs[0, 1].grid(True)

# ==== 磁损 ====
axs[1, 0].hist(loss, bins=50, color='lightgreen', edgecolor='black')
axs[1, 0].set_title("Original Core Loss Distribution")
axs[1, 0].set_xlabel("Loss [W/m³]")
axs[1, 0].set_ylabel("Count")
axs[1, 0].grid(True)

axs[1, 1].hist(log_loss, bins=50, color='orange', edgecolor='black')
axs[1, 1].set_title("Log-Transformed Core Loss")
axs[1, 1].set_xlabel("log(Loss)")
axs[1, 1].set_ylabel("Count")
axs[1, 1].grid(True)

# ==== 温度 ====
axs[2, 0].hist(temp, bins=50, color='plum', edgecolor='black')
axs[2, 0].set_title("Original Temperature Distribution")
axs[2, 0].set_xlabel("Temperature [K]")
axs[2, 0].set_ylabel("Count")
axs[2, 0].grid(True)

axs[2, 1].hist(log_temp, bins=50, color='lightcoral', edgecolor='black')
axs[2, 1].set_title("Log-Transformed Temperature")
axs[2, 1].set_xlabel("log(Temperature)")
axs[2, 1].set_ylabel("Count")
axs[2, 1].grid(True)

# 布局 & 展示
fig.tight_layout()
plt.show()
