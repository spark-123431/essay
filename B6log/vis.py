import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

# 数据目录设置
raw_data_path = r"D:\essay\B6log\materials"
output_dir = r"D:\essay\B6log"
os.makedirs(output_dir, exist_ok=True)

def load_material_data(material_path):
    b = np.loadtxt(os.path.join(material_path, 'B_waveform[T].csv'), delimiter=',').astype(np.float32)
    try:
        h = np.loadtxt(os.path.join(material_path, 'H_waveform[Am-1].csv'), delimiter=',').astype(np.float32)
    except FileNotFoundError:
        h = np.array([]).astype(np.float32)
    temp = (np.loadtxt(os.path.join(material_path, 'Temperature[C].csv'), delimiter=',') + 273.15).astype(np.float32)
    freq = np.loadtxt(os.path.join(material_path, 'Frequency[Hz].csv'), delimiter=',').astype(np.float32)
    loss = np.loadtxt(os.path.join(material_path, 'Volumetric_losses[Wm-3].csv'), delimiter=',').astype(np.float32)

    if freq.ndim == 1:
        freq = freq[:, np.newaxis]
    if loss.ndim == 1:
        loss = loss[:, np.newaxis]
    return {'freq': freq, 'loss': loss, 'temp': temp, 'material': os.path.basename(material_path)}

# 加载所有数据
all_data = []
for material in os.listdir(raw_data_path):
    material_path = os.path.join(raw_data_path, material)
    if not os.path.isdir(material_path):
        continue
    try:
        material_data = load_material_data(material_path)
        all_data.append(material_data)
        print(f"成功加载材料: {material}")
    except Exception as e:
        print(f"加载材料 {material} 失败: {str(e)}")

# 计算全部温度值
temp_values = np.unique(np.concatenate([d['temp'] for d in all_data]))
cmap = plt.get_cmap('viridis', len(temp_values)) if len(temp_values) > 1 else None

# 绘图
plt.figure(figsize=(12, 8), dpi=150)
for data in all_data:
    freq = data['freq'].flatten()
    loss = data['loss'].flatten()
    temp = data['temp'].flatten()
    material = data['material']

    if len(temp_values) > 1:
        sc = plt.scatter(freq, loss, c=temp, cmap=cmap, alpha=0.7,
                         label=material if len(all_data) <= 5 else None)
    else:
        plt.scatter(freq, loss, alpha=0.7, label=material)

# 添加颜色条
if len(temp_values) > 1:
    cbar = plt.colorbar(sc)
    cbar.set_label('Temperature (K)')
    cbar.formatter = ScalarFormatter()
    cbar.update_ticks()

plt.xscale('log')
plt.yscale('log')
plt.xlabel('Frequency (Hz)', fontsize=12)
plt.ylabel('Volumetric Losses (W/m³)', fontsize=12)
plt.title('Iron Loss vs Frequency', fontsize=14)
plt.grid(True, which="both", ls="--")

if len(all_data) <= 15:
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
else:
    plt.text(0.95, 0.95, f"{len(all_data)} materials",
             transform=plt.gca().transAxes, ha='right')

plot_path = os.path.join(output_dir, 'iron_loss_vs_frequency.png')
plt.tight_layout()
plt.savefig(plot_path, bbox_inches='tight')
print(f"图表已保存至: {plot_path}")
plt.show()

# 统计信息输出（前 3 个材料）
print("\n频率-铁损统计信息:")
for data in all_data[:3]:
    freq = data['freq']
    loss = data['loss']
    print(f"\nMaterial: {data['material']}")
    print(f"Frequency range: {np.min(freq):.1f} - {np.max(freq):.1f} Hz")
    print(f"Loss range: {np.min(loss):.2f} - {np.max(loss):.2f} W/m³")
    print(f"Pearson correlation (freq vs loss): {np.corrcoef(freq.flatten(), loss.flatten())[0, 1]:.3f}")
