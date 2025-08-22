import numpy as np
import os
import scipy.io as sio
import matplotlib.pyplot as plt
import torch
from torch.utils.data import random_split
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

# ===== Linear standardization class =====
class linear_std:
    #x_std = k * x + b
    def __init__(self, min_raw=None, max_raw=None, min_std=0.0, max_std=1.0):
        if min_raw is not None and max_raw is not None:
            if max_raw == min_raw:
                raise ValueError("max_raw must be different from min_raw to avoid division by zero.")
            self.k = (max_std - min_std) / (max_raw - min_raw)
            self.b = min_std - self.k * min_raw
        else:
            # 默认值（若从文件加载则会覆盖）
            self.k = 1.0
            self.b = 0.0

    def std(self, x):
        return self.k * x + self.b

    def inverse(self, x_std):
        return (x_std - self.b) / self.k

    def save(self, path):
        np.save(path, {"k": self.k, "b": self.b})

    def load(self, path):
        d = np.load(path, allow_pickle=True).item()
        self.k = d['k']
        self.b = d['b']

    def unstd(self, y):
        return (y - self.b) / self.k

    @staticmethod
    def get_std_range(min_raw, max_raw, min_std=0.0, max_std=1.0):
        return linear_std(min_raw, max_raw, min_std, max_std)

# ===== MagLoader and MagPlot =====
class MagLoader:
    def __init__(self, material_path='', data_type='numpy', data_source='mat'):
        if material_path:
            if data_source == 'mat':
                data = sio.loadmat(material_path)
                self.b = np.array(data['b'])
                self.temp = np.array(data['temp'])
                self.loss = np.array(data['loss'])
                self.freq = np.array(data['freq'])
            elif data_source == 'csv':
                self.b = np.loadtxt(os.path.join(material_path, 'B_waveform[T].csv'), delimiter=',').astype(np.float32)
                self.temp = (np.loadtxt(os.path.join(material_path, 'Temperature[C].csv'), delimiter=',') + 273.15).astype(np.float32)
                self.freq = np.loadtxt(os.path.join(material_path, 'Frequency[Hz].csv'), delimiter=',').astype(np.float32)
                self.loss = np.loadtxt(os.path.join(material_path, 'Volumetric_losses[Wm-3].csv'), delimiter=',').astype(np.float32)

            if data_type == 'torch':
                self.b = torch.from_numpy(self.b)
                self.temp = torch.from_numpy(self.temp)
                self.loss = torch.from_numpy(self.loss)
                self.freq = torch.from_numpy(self.freq)
        else:
            self.b = self.temp = self.loss = self.freq = np.array([])
        return

    def save2mat(self, save_path):
        sio.savemat(save_path, {'b': self.b, 'temp': self.temp, 'loss': self.loss, 'freq': self.freq})


def magplot(material_name, relative_error, save_path="", xlim=30):
    # 转换为百分比
    relv_err = np.abs(relative_error) * 100

    # 计算关键统计指标
    avg = np.mean(relv_err)
    p95 = np.percentile(relv_err, 95)
    p99 = np.percentile(relv_err, 99)
    max_err = np.max(relv_err)

    # 创建图形
    plt.figure(figsize=(6, 3), dpi=300)
    plt.rcParams["font.family"] = "Times New Roman"

    # 设置主标题和副标题
    plt.suptitle(f"Error Distribution for {material_name}", fontsize=18, y=1.05)
    plt.title(f"Avg={avg:.2f}%, 95-Prct={p95:.2f}%, 99-Prct={p99:.2f}%, Max={max_err:.2f}%", fontsize=10)

    # 绘制误差直方图
    plt.hist(relv_err, bins=20, edgecolor='black', density=True, linewidth=0.5)

    # 辅助函数：获取直方图密度高度
    def get_density(x, data):
        hist, edges = np.histogram(data, bins=20, density=True)
        for i in range(len(edges) - 1):
            if edges[i] <= x < edges[i + 1]:
                return hist[i]
        return 0

    # 标注统计线和文本
    for stat_func, label in [(np.mean, "Avg"), (lambda x: np.percentile(x, 95), "95-Prct"), (np.max, "Max")]:
        val = stat_func(relv_err)
        if val < xlim:
            y = get_density(val, relv_err) + 0.001
            plt.plot([val, val], [0, y], '--', color="red", linewidth=1)
            plt.text(val + 0.25, y, f'{label}={val:.2f}%', color="red", fontsize=10)

    # 坐标轴设置
    plt.xlim(0, xlim)
    plt.xlabel("Relative Error of Core Loss [%]", fontsize=14)
    plt.ylabel("Ratio of Data Points", fontsize=14)
    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.tick_params(labelsize=12)

    # 保存或显示图形
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"[{material_name}] Error histogram saved to: {save_path}")
    else:
        plt.show()

    plt.close()


# ===== Data Transform and Split =====
def dataTransform(raw_data, newStep, savePath, plot=False):
    # 不做插值，直接复制原始数据
    b_buff = raw_data.b.copy()

    raw_data.b = b_buff

    if plot:
        plt.plot(np.linspace(0, raw_data.b.shape[1], raw_data.b.shape[1], endpoint=True), raw_data.b[0], 'x')
        plt.show()

    # 对数处理 temp / freq / loss
    eps = 1e-8
    raw_data.temp = np.log(raw_data.temp + eps)
    raw_data.loss = np.log(raw_data.loss + eps)
    raw_data.freq = np.log(raw_data.freq + eps)

    # 类型转换
    raw_data.freq = raw_data.freq.astype(np.float32)
    raw_data.b = raw_data.b.astype(np.float32)
    raw_data.temp = raw_data.temp.astype(np.float32)
    raw_data.loss = raw_data.loss.astype(np.float32)

    # 保存数据（无标准化参数）
    raw_data.save2mat(savePath + r"\data_processed.mat")

    return raw_data



def dataSplit(raw_data, savePath, indice=[0.7, 0.2, 0.1]):
    generator = torch.Generator().manual_seed(0)
    total_size = raw_data.b.shape[0]
    lengths = [int(total_size * r) for r in indice]
    lengths[-1] = total_size - sum(lengths[:-1])  # 保证和为总样本数

    # 随机打乱并获取索引划分
    all_indices = torch.randperm(total_size, generator=generator)
    train_idx, valid_idx, test_idx = torch.utils.data.random_split(
        all_indices, lengths, generator=generator
    )
    stepLen = raw_data.b.shape[1]

    # 辅助函数：根据索引获取子集
    def get_subset(indices):
        idx = indices.indices if isinstance(indices, torch.utils.data.Subset) else indices
        dataset = MagLoader()
        dataset.b = raw_data.b[idx]
        dataset.temp = raw_data.temp[idx]
        dataset.loss = raw_data.loss[idx]
        dataset.freq = raw_data.freq[idx]
        return dataset

    for name, subset_idx in zip(['train', 'valid', 'test'], [train_idx, valid_idx, test_idx]):
        subset = get_subset(subset_idx)
        subset.save2mat(os.path.join(savePath, f"{name}.mat"))


#将.mat 格式的磁损数据集封装成 PyTorch 可以训练使用的 Dataset 和 DataLoader 对象
class MagDataset(Dataset):
    def __init__(self, file_path):
        mag_data = MagLoader(file_path)

        num_samples = mag_data.b.shape[0]
        seq_len = mag_data.b.shape[1]

        # 计算 B 的一阶导数和二阶导数（中心差分）
        dB = np.gradient(mag_data.b, axis=1)
        d2B = np.gradient(dB, axis=1)

        # 构造输入张量：B, freq, temp, dB, d2B → 共 5 通道
        self.x_data = np.zeros((num_samples, seq_len, 6), dtype=np.float32)
        self.x_data[:, :, 0] = mag_data.b
        self.x_data[:, :, 1] = mag_data.freq  # broadcast
        self.x_data[:, :, 2] = mag_data.temp  # broadcast
        self.x_data[:, :, 3] = dB
        self.x_data[:, :, 4] = d2B

        self.y_data = torch.tensor(mag_data.loss, dtype=torch.float32)
        self.x_data = torch.tensor(self.x_data, dtype=torch.float32)

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        return self.x_data[idx], self.y_data[idx]



def get_dataloader(file_path, batch_size=64, shuffle=False):
    dataset = MagDataset(file_path)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False)


if __name__ == '__main__':
    # 设定路径
    raw_data_path = r"D:\essay\B5log\materials"
    processed_data_dir = r"D:\essay\B5log\Processed Training Data"
    newStep = 128

    for material in os.listdir(raw_data_path):
        raw_path = os.path.join(raw_data_path, material)
        save_path = os.path.join(processed_data_dir, material)

        if not os.path.isdir(raw_path):# or os.path.exists(os.path.join(save_path, 'train.mat')):
            continue

        os.makedirs(save_path, exist_ok=True)

        data = MagLoader(raw_path, data_source='csv')
        data.temp = data.temp[:, np.newaxis] if data.temp.ndim == 1 else data.temp
        data.freq = data.freq[:, np.newaxis] if data.freq.ndim == 1 else data.freq
        data.loss = data.loss[:, np.newaxis] if data.loss.ndim == 1 else data.loss

        data = dataTransform(data, newStep, save_path)
        dataSplit(data, save_path)