import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ===== 路径设置 =====
material_dir = r"D:\essay\materials\3C90"
img_output_dir = r"D:\essay\results\3C90_bh_grid"
os.makedirs(img_output_dir, exist_ok=True)

# ===== 读取数据文件 =====
b_path = os.path.join(material_dir, "B_waveform[T].csv")
h_path = os.path.join(material_dir, "H_waveform[Am-1].csv")
f_path = os.path.join(material_dir, "Frequency[Hz].csv")
t_path = os.path.join(material_dir, "Temperature[C].csv")

for path in [b_path, h_path, f_path, t_path]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"缺少文件: {path}")

# 加载数据
b_df = pd.read_csv(b_path, header=None)
h_df = pd.read_csv(h_path, header=None)
f_array = pd.read_csv(f_path, header=None).squeeze().values
t_array = pd.read_csv(t_path, header=None).squeeze().values

# 校验维度
assert b_df.shape == h_df.shape, "B 和 H 波形维度不一致"
assert len(f_array) == b_df.shape[0] == len(t_array), "频率、温度数量与样本数不匹配"

# 压缩波形点数
sub_cols = list(range(0, 1024, 4))  # 每条波形变为 256 点
b = b_df[sub_cols].to_numpy()
h = h_df[sub_cols].to_numpy()

# 按温度分组
unique_temps = np.unique(t_array)
temp_groups = [unique_temps[i:i+4] for i in range(0, len(unique_temps), 4)]

for group_idx, temp_group in enumerate(temp_groups):
    fig, axs = plt.subplots(2, 2, figsize=(10, 8), dpi=150)
    axs = axs.flatten()

    for i, temp in enumerate(temp_group):
        ax = axs[i]

        # 找出该温度对应的样本索引
        temp_idxs = np.where(t_array == temp)[0]
        temp_idxs = [i for i in temp_idxs if f_array[i] <= 300000]
        if len(temp_idxs) == 0:
            ax.set_title(f"{temp:.1f} °C - 无有效频率数据")
            ax.axis('off')
            continue

        # 频率排序并选择样本
        temp_freqs = f_array[temp_idxs]
        sorted_idxs = np.array(temp_idxs)[np.argsort(temp_freqs)]
        last_freq = -1e9

        for idx in sorted_idxs:
            f = f_array[idx]
            if f - last_freq >= 100_000:
                ax.plot(h[idx], b[idx] * 1000, label=f"{f/1000:.0f} kHz", alpha=0.7)
                last_freq = f

        ax.set_xlim(-350, 350)
        ax.set_ylim(-300, 300)  # 固定纵坐标范围
        ax.set_aspect('equal', adjustable='box')  # 保证横纵坐标比例一致
        ax.set_title(f"{temp:.1f} °C")
        ax.set_xlabel("$H$ [A/m]")
        ax.set_ylabel("$B$ [mT]")
        ax.grid(True)
        ax.legend(fontsize=7, title="Freq", loc="best")

    # 补足不足4个子图时的空白子图
    for j in range(len(temp_group), 4):
        axs[j].axis('off')

    fig.suptitle("B-H Loops at Various Temperatures", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    save_name = f"bh_group_{group_idx}.png"
    fig.savefig(os.path.join(img_output_dir, save_name), dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"已保存图像: {save_name}")
