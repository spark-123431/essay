import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ===== 路径设置 =====
material_dir = r"D:\essay\materials\3C90"
result_dir = r"D:\essay\results"
os.makedirs(result_dir, exist_ok=True)

b_file = os.path.join(material_dir, "B_waveform[T].csv")
h_file = os.path.join(material_dir, "H_waveform[Am-1].csv")
t_file = os.path.join(material_dir, "Temperature[C].csv")
f_file = os.path.join(material_dir, "Frequency[Hz].csv")

# ===== 加载数据 =====
b_df = pd.read_csv(b_file, header=None)
h_df = pd.read_csv(h_file, header=None)
t_array = pd.read_csv(t_file, header=None).squeeze().values
f_array = pd.read_csv(f_file, header=None).squeeze().values
assert b_df.shape == h_df.shape
assert len(t_array) == len(f_array) == b_df.shape[0]

# ===== 压缩点数 =====
sub_cols = list(range(0, 1024, 4))
b_data = b_df[sub_cols].to_numpy()
h_data = h_df[sub_cols].to_numpy()


# ===== 判别波形函数 =====
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

def is_sine(waveform):
    # 简单判断正弦波：梯形和三角都不是
    return not is_trapezoid(waveform) and not is_triangle(waveform)

# ===== 相同温度下找出三种样本 =====
trapezoid_idx = None
triangle_idx = None
sine_idx = None
common_temp = None

for temp in np.unique(t_array):
    temp_idxs = np.where(t_array == temp)[0]
    for idx in temp_idxs:
        b = b_data[idx]
        if is_trapezoid(b) and trapezoid_idx is None:
            trapezoid_idx = idx
            common_temp = temp
        elif is_triangle(b) and triangle_idx is None:
            triangle_idx = idx
            common_temp = temp
        elif is_sine(b) and sine_idx is None:
            sine_idx = idx
            common_temp = temp
        # 如果三种都找齐了并且温度一致
        if all(x is not None for x in [trapezoid_idx, triangle_idx, sine_idx]):
            if len({t_array[trapezoid_idx], t_array[triangle_idx], t_array[sine_idx]}) == 1:
                break
    if all(x is not None for x in [trapezoid_idx, triangle_idx, sine_idx]):
        break

assert trapezoid_idx and triangle_idx and sine_idx, "未找到三种波形的样本"

# ===== 绘图 =====
plt.figure(figsize=(6, 5), dpi=150)

# 梯形波
plt.plot(h_data[trapezoid_idx], b_data[trapezoid_idx]*1000,
         label=f"Trapezoid (#{trapezoid_idx}, {f_array[trapezoid_idx]/1e3:.0f} kHz)", linewidth=1.5)

# 三角波
plt.plot(h_data[triangle_idx], b_data[triangle_idx]*1000,
         label=f"Triangle (#{triangle_idx}, {f_array[triangle_idx]/1e3:.0f} kHz)", linewidth=1.5)

# 正弦波
plt.plot(h_data[sine_idx], b_data[sine_idx]*1000,
         label=f"Sine (#{sine_idx}, {f_array[sine_idx]/1e3:.0f} kHz)", linewidth=1.5)

plt.title(f"B-H Loops at {common_temp:.1f} °C")
plt.xlabel("H [A/m]")
plt.ylabel("B [mT]")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(result_dir, "bh_compare_same_temp.png"), dpi=300)
plt.show()

