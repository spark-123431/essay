import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import matplotlib.colors as mcolors

# ===== 路径设置 =====
raw_data_path = r"D:\essay\B6log\materials"
output_dir = r"D:\essay\B6log"
os.makedirs(output_dir, exist_ok=True)

# ===== 加载单个材料数据 =====
def load_material_data(material_path):
    b = np.loadtxt(os.path.join(material_path, 'B_waveform[T].csv'), delimiter=',').astype(np.float32)
    try:
        h = np.loadtxt(os.path.join(material_path, 'H_waveform[Am-1].csv'), delimiter=',').astype(np.float32)
    except FileNotFoundError:
        h = np.array([]).astype(np.float32)
    # 原始代码（删除 +273.15 的部分）
    temp = np.loadtxt(os.path.join(material_path, 'Temperature[C].csv'), delimiter=',').astype(np.float32)  # 直接读取为摄氏度
    freq = np.loadtxt(os.path.join(material_path, 'Frequency[Hz].csv'), delimiter=',').astype(np.float32)
    loss = np.loadtxt(os.path.join(material_path, 'Volumetric_losses[Wm-3].csv'), delimiter=',').astype(np.float32)

    if freq.ndim == 1:
        freq = freq[:, np.newaxis]
    if loss.ndim == 1:
        loss = loss[:, np.newaxis]

    return {
        'freq': freq,
        'loss': loss,
        'temp': temp,
        'material': os.path.basename(material_path)
    }

# ===== 加载全部材料数据 =====
all_data = []
for material in os.listdir(raw_data_path):
    material_path = os.path.join(raw_data_path, material)
    if not os.path.isdir(material_path):
        continue
    try:
        data = load_material_data(material_path)
        all_data.append(data)
        print(f"成功加载材料: {material}")
    except Exception as e:
        print(f"加载失败: {material}，错误信息: {str(e)}")

# ===== 提取温度范围 & 设置 colormap =====
all_temps = np.concatenate([d['temp'].flatten() for d in all_data])
vmin, vmax = np.min(all_temps), np.max(all_temps)
norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
# ===== 改进后的颜色设置 =====
# 使用更科学的颜色映射（从冷色到暖色，清晰区分温度变化）
cmap = plt.get_cmap('viridis')  # 替代原来的magma_r，更清晰的颜色渐变
# 或者使用自定义颜色（蓝→青→黄→红）
custom_colors = ["#2a4c7d", "#1d8c9f", "#6bb392", "#fed71b", "#f39019"]
cmap = mcolors.LinearSegmentedColormap.from_list("custom", custom_colors)

# ===== 绘图主过程（修改点） =====
plt.figure(figsize=(12, 8), dpi=150)

for data in all_data:
    freq = data['freq'].flatten()
    loss = data['loss'].flatten()
    temp = data['temp'].flatten()
    material = data['material']

    # 增加点的大小和边缘颜色
    sc = plt.scatter(
        freq, loss,
        c=temp,
        cmap=cmap,
        norm=norm,
        alpha=0.85,          # 稍微降低透明度
        s=45,               # 增大点的大小
        edgecolor='white',   # 白色边缘增加对比度
        linewidth=0.3,      # 边缘线宽
        label=material if len(all_data) <= 5 else None
    )

# ===== 改进后的颜色条 =====
cbar = plt.colorbar(sc, pad=0.02)
# 修改颜色条标签
cbar.set_label('Temperature (°C)', fontsize=12)  # 将 (K) 改为 (°C)
cbar.formatter = ScalarFormatter()
cbar.update_ticks()

# ===== 图表格式优化 =====
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Frequency (Hz)', fontsize=13, labelpad=10)
plt.ylabel('Volumetric Losses (W/m³)', fontsize=13, labelpad=10)
plt.title('Iron Loss vs Frequency (Temperature Color Mapping)',
          fontsize=15, pad=20)

# 优化网格和背景
plt.grid(True, which="both", ls="--", alpha=0.4)
plt.gca().set_facecolor('#f8f8f8')  # 浅灰色背景

# 优化图例显示
if len(all_data) <= 15:
    legend = plt.legend(
        bbox_to_anchor=(1.05, 1),
        loc='upper left',
        frameon=True,
        framealpha=1,
        edgecolor='#444444'
    )
    # 设置图例背景色
    legend.get_frame().set_facecolor('white')

plot_path = os.path.join(output_dir, 'iron_loss_vs_frequency_custom_colormap.png')
plt.tight_layout()
plt.savefig(plot_path, bbox_inches='tight')
print(f"图表已保存至: {plot_path}")
plt.show()

# ===== 输出统计信息（前 3 个材料） =====
print("\n频率-铁损统计信息:")
for data in all_data[:3]:
    freq = data['freq']
    loss = data['loss']
    print(f"\nMaterial: {data['material']}")
    print(f"Frequency range: {np.min(freq):.1f} - {np.max(freq):.1f} Hz")
    print(f"Loss range: {np.min(loss):.2f} - {np.max(loss):.2f} W/m³")
    print(f"Pearson correlation (freq vs loss): {np.corrcoef(freq.flatten(), loss.flatten())[0, 1]:.3f}")
