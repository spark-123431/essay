import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ========== 路径设置 ==========
material_dir = r"D:\essay\materials\3C90"
result_dir = r"D:\essay\results"
os.makedirs(result_dir, exist_ok=True)

b_file = os.path.join(material_dir, "B_waveform[T].csv")

# ========== 加载数据 ==========
b_df = pd.read_csv(b_file, header=None)
sub_cols = list(range(0, 1024, 4))  # 下采样为 256 点
b_data = b_df[sub_cols].to_numpy()
N, T = b_data.shape

# ========== 判别函数 ==========
def is_trapezoid(waveform, threshold_ratio=0.1):
    grad = np.gradient(waveform)
    mid = len(grad) // 2
    mid_avg = np.abs(np.mean(grad[mid - 10:mid + 10]))
    max_grad = np.max(np.abs(grad))
    return mid_avg < threshold_ratio * max_grad

def is_triangle(waveform, linearity_thresh=0.98):
    grad = np.gradient(waveform)
    sign_changes = np.diff(np.sign(grad))
    # 理想三角波只有一次拐点（上升→下降）
    return np.sum(sign_changes != 0) == 1

# ========== 分类 ==========
classified = {
    "trapezoid": [],
    "triangle": [],
    "other": []
}

for idx, b in enumerate(b_data):
    if is_trapezoid(b):
        classified["trapezoid"].append(idx)
    elif is_triangle(b):
        classified["triangle"].append(idx)
    else:
        classified["other"].append(idx)

# ========== 输出统计 ==========
print("波形类型统计：")
for k, v in classified.items():
    print(f"  {k}: {len(v)} 条")

# ========== 画图 ==========（修改为每类画 5 条）
fig, axes = plt.subplots(1, 3, figsize=(14, 4), dpi=150)
time = np.linspace(0, 1, T)

for ax, (shape_name, idx_list) in zip(axes, classified.items()):
    ax.set_title(f"{shape_name.capitalize()} ({len(idx_list)} samples)")

    for i in range(min(5, len(idx_list))):
        idx = idx_list[i]
        b_wave = b_data[idx] * 1000  # 单位 T → mT
        ax.plot(time, b_wave, label=f"#{idx}")

    ax.set_xlabel("Normalized Time")
    ax.set_ylabel("B [mT]")
    ax.grid(True)
    ax.legend(fontsize=8)

fig.suptitle("Representative B Waveforms by Shape Type", fontsize=14)
fig.tight_layout(rect=[0, 0, 1, 0.92])

# 保存拼接图
save_path = os.path.join(result_dir, "b_waveform_triple_panel.png")
plt.savefig(save_path, dpi=300)
plt.show()

