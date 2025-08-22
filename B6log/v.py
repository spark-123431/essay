import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ===== 根目录设置（里面每个子文件夹是一个材料） =====
materials_root = r"D:\essay\B6log\materials"
result_dir = r"D:\essay\results"
os.makedirs(result_dir, exist_ok=True)

# ===== 波形判别函数 =====
def is_trapezoid(waveform, threshold_ratio=0.1):
    grad = np.gradient(waveform)
    mid = len(grad) // 2
    mid_avg = np.abs(np.mean(grad[mid - 10:mid + 10]))
    max_grad = np.max(np.abs(grad))
    return mid_avg < threshold_ratio * max_grad

def is_triangle(waveform, flatness_threshold=0.02):
    grad = np.gradient(waveform)
    max_grad = np.max(np.abs(grad))
    return np.all(np.abs(grad) > flatness_threshold * max_grad)

# ===== 每个材料的统计结果 =====
material_waveform_counts = {}

for material_dir in os.listdir(materials_root):
    mat_path = os.path.join(materials_root, material_dir)
    if not os.path.isdir(mat_path):
        continue  # 跳过非文件夹

    b_file = os.path.join(mat_path, "B_waveform[T].csv")
    if not os.path.exists(b_file):
        continue

    # 初始化统计字典
    waveform_counts = {"Sine": 0, "Triangle": 0, "Trapezoid": 0}

    # 读取并压缩数据
    b_df = pd.read_csv(b_file, header=None)
    sub_cols = list(range(0, 1024, 4))
    b_data = b_df[sub_cols].to_numpy()

    # 波形分类
    for b in b_data:
        if is_trapezoid(b):
            waveform_counts["Trapezoid"] += 1
        elif is_triangle(b):
            waveform_counts["Triangle"] += 1
        else:
            waveform_counts["Sine"] += 1

    material_waveform_counts[material_dir] = waveform_counts

# ===== 绘制两个材料的饼图（左右并列） =====
fig, axes = plt.subplots(1, 2, figsize=(10, 5), dpi=150)

colors = ["#ff9999", "#66b3ff", "#99ff99"]

for ax, (material, counts) in zip(axes, material_waveform_counts.items()):
    labels = list(counts.keys())
    sizes = list(counts.values())
    ax.pie(
        sizes,
        labels=labels,
        autopct="%1.1f%%",
        startangle=90,
        colors=colors
    )
    ax.set_title(f"Waveform Distribution - {material}")
    ax.axis("equal")  # 确保是正圆

plt.tight_layout()
plt.savefig(os.path.join(result_dir, "waveform_distribution_two_materials.png"), dpi=300)
plt.show()
