import numpy as np
from scipy.interpolate import interp1d
import os
import torch
import scipy.io as sio
from torch.utils.data import TensorDataset, random_split

# 归一化类
class linear_std:
    def __init__(self, min_raw=None, max_raw=None, min_std=0.0, max_std=1.0):
        if min_raw is not None and max_raw is not None:
            if max_raw == min_raw:
                raise ValueError("max_raw must be different from min_raw to avoid division by zero.")
            self.k = (max_std - min_std) / (max_raw - min_raw)
            self.b = min_std - self.k * min_raw
        else:
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

# 修改后的 MagLoader 类（核心部分）
class MagLoader:
    def __init__(self, material_path='', data_type='numpy', data_source='mat', waveform_length=1024):
        self.waveform_length = waveform_length
        self.data_type = data_type

        if material_path:
            if data_source == 'mat':
                data = sio.loadmat(material_path)
                self.b = np.array(data['b'])
                self.temp = np.array(data['temp']).flatten()
                self.loss = np.array(data['loss']).flatten()
                self.freq = np.array(data['freq']).flatten()
            elif data_source == 'csv':
                self.b = np.loadtxt(os.path.join(material_path, 'B_waveform[T].csv'), delimiter=',').astype(np.float32)
                self.temp = np.loadtxt(os.path.join(material_path, 'Temperature[C].csv'), delimiter=',').astype(np.float32) + 273.15
                self.freq = np.loadtxt(os.path.join(material_path, 'Frequency[Hz].csv'), delimiter=',').astype(np.float32)
                self.loss = np.loadtxt(os.path.join(material_path, 'Volumetric_losses[Wm-3].csv'), delimiter=',').astype(np.float32)

            self.b = self._preprocess_waveform(self.b)
            self.bm = np.max(np.abs(self.b), axis=1)
            self.loss = np.log(self.loss + 1e-12)

            self.freq_std = linear_std.get_std_range(np.min(self.freq), np.max(self.freq))
            self.temp_std = linear_std.get_std_range(np.min(self.temp), np.max(self.temp))

            self.freq = self.freq_std.std(self.freq)
            self.temp = self.temp_std.std(self.temp)

            if data_type == 'torch':
                self._to_tensor()

            self._split_data()

        else:
            self.b = self.bm = self.temp = self.loss = self.freq = np.array([])

    def _preprocess_waveform(self, b_array):
        if isinstance(b_array, torch.Tensor):
            b_array = b_array.numpy()
        processed = []
        for b in b_array:
            b = (b - np.mean(b)) / (np.max(np.abs(b)) + 1e-6)
            t_orig = np.linspace(0, 1, num=len(b))
            t_new = np.linspace(0, 1, num=self.waveform_length)
            b_interp = interp1d(t_orig, b, kind='linear')(t_new)
            processed.append(b_interp)
        b_std = np.stack(processed, axis=0)

        self.b_std = linear_std.get_std_range(np.min(b_std), np.max(b_std))
        return self.b_std.std(b_std)

    def _split_data(self):
        assert isinstance(self.b, torch.Tensor), "self.b must be torch.Tensor before calling _split_data()"
        full_dataset = TensorDataset(
            self.b.unsqueeze(1),
            self.bm,
            self.freq,
            self.temp,
            self.loss
        )
        total = len(full_dataset)
        n_train = int(total * 0.7)
        n_val = int(total * 0.2)
        n_test = total - n_train - n_val
        self.train, self.val, self.test = random_split(full_dataset, [n_train, n_val, n_test],
                                                       generator=torch.Generator().manual_seed(42))

    def _to_tensor(self):
        def ensure_tensor(x):
            return torch.from_numpy(x).float() if isinstance(x, np.ndarray) else x
        self.b = ensure_tensor(self.b)
        self.bm = ensure_tensor(self.bm)
        self.freq = ensure_tensor(self.freq)
        self.temp = ensure_tensor(self.temp)
        self.loss = ensure_tensor(self.loss)


if __name__ == '__main__':
    # 设置路径
    raw_data_path = r"E:\project\cnn_BiLSTM\materials"
    processed_data_dir = r"E:\project\cnn_BiLSTM\Processed Training Data"
    os.makedirs(processed_data_dir, exist_ok=True)

    for material in os.listdir(raw_data_path):
        raw_path = os.path.join(raw_data_path, material)
        save_path = os.path.join(processed_data_dir, material)

        if not os.path.isdir(raw_path):
            continue

        os.makedirs(save_path, exist_ok=True)

        # 加载并预处理
        data = MagLoader(raw_path, data_type='torch', data_source='csv')

        # 保存数据
        def save_split(split_data, filename):
            b, bm, freq, temp, loss = [], [], [], [], []
            for b_, bm_, f_, t_, l_ in split_data:
                b.append(b_.numpy().squeeze())  # [1,1024] -> [1024]
                bm.append(bm_.item())
                freq.append(f_.item())
                temp.append(t_.item())
                loss.append(l_.item())
            sio.savemat(os.path.join(save_path, filename), {
                'b': np.stack(b),
                'bm': np.array(bm),
                'freq': np.array(freq),
                'temp': np.array(temp),
                'loss': np.array(loss)
            })

        save_split(data.train, 'train.mat')
        save_split(data.val, 'val.mat')
        save_split(data.test, 'test.mat')

        print(f"{material}: 数据处理完毕，保存至 {save_path}")
