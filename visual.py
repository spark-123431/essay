import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 数据路径
data_root = r"E:\project\materials"
IMG_PATH = r"E:\project\results"
os.makedirs(IMG_PATH, exist_ok=True)

# 每隔多少个样本画一条曲线
sample_step = 10

# 每个样本中每隔4个时间点取一个（1024个点 → 256个点）
SUB_B_COLS = list(range(0, 1024, 4))
SUB_H_COLS = list(range(0, 1024, 4))

# 材料名称映射（可选）
true_mats_d = {'A': "3C92", "B": "T37", "C": "3C95", "D": "79", "E": "ML95S"}

# 收集数据
material_data = {}

for material_name in os.listdir(data_root):
    subdir = os.path.join(data_root, material_name)
    if not os.path.isdir(subdir):
        continue

    b_file = os.path.join(subdir, "B_waveform[T].csv")
    h_file = os.path.join(subdir, "H_waveform[Am-1].csv")

    if not (os.path.exists(b_file) and os.path.exists(h_file)):
        print(f"跳过 {material_name}: 缺少 B 或 H 文件")
        continue

    try:
        b_df = pd.read_csv(b_file, header=None)
        h_df = pd.read_csv(h_file, header=None)

        if b_df.shape != h_df.shape:
            print(f"跳过 {material_name}: B 和 H 样本形状不匹配")
            continue

        # 下采样选取样本
        b = b_df.iloc[::sample_step, SUB_B_COLS].to_numpy()
        h = h_df.iloc[::sample_step, SUB_H_COLS].to_numpy()

        material_data[material_name] = (b, h)

    except Exception as e:
        print(f"读取失败: {material_name}, 错误: {e}")

# 单个大图
fig, ax = plt.subplots(dpi=150, figsize=(6, 5))

for mat_name, (b, h) in material_data.items():
    try:
        for i in range(b.shape[0]):
            ax.plot(h[i], b[i] * 1000, alpha=0.6)
        # 添加图例用的标记线（空数据但有label）
        label = true_mats_d.get(mat_name, mat_name)
        ax.plot([], [], label=label)
    except Exception as e:
        print(f"{mat_name} 绘图失败: {e}")

# 设置坐标轴
ax.set_xlim(-100, 100)
ax.set_xticks(np.linspace(-80, 80, 3))
ax.set_xlabel("$H$ in A/m")
ax.set_ylabel("$B$ in mT")
ax.grid(True)
ax.legend()
fig.tight_layout()

# 显示或保存
# fig.savefig(os.path.join(IMG_PATH, "merged_bh_curves.png"), dpi=200, bbox_inches="tight")
plt.show()