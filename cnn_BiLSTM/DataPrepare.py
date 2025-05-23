import os
import numpy as np
import scipy.io as sio
import torch
from scipy.interpolate import interp1d
from torch.utils.data import TensorDataset, random_split

class MagLoader:
    def __init__(self, material_path='', data_type='numpy', data_source='mat', waveform_length=1024):
        self.waveform_length = waveform_length

        if material_path:
            if data_source == 'mat':
                data = sio.loadmat(material_path)
                self.b = np.array(data['b'])               # [N, T]
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

            self._split_data()

            if data_type == 'torch':
                self._to_tensor()
        else:
            self.b = self.bm = self.temp = self.loss = self.freq = np.array([])

    def _preprocess_waveform(self, b_array):
        processed = []
        for b in b_array:
            b = (b - np.mean(b)) / (np.max(np.abs(b)) + 1e-6)
            t_orig = np.linspace(0, 1, num=len(b))
            t_new = np.linspace(0, 1, num=self.waveform_length)
            b_interp = interp1d(t_orig, b, kind='linear')(t_new)
            processed.append(b_interp)
        return np.stack(processed, axis=0)

    def _split_data(self):
        # 转换为 tensor（如未完成转换）
        self._to_tensor()

        # 创建全集合
        full_dataset = TensorDataset(
            self.b.unsqueeze(1),     # [N, 1, 1024]
            self.bm,                 # [N]
            self.freq,               # [N]
            self.temp,               # [N]
            self.loss                # [N]
        )

        total = len(full_dataset)
        n_train = int(total * 0.7)
        n_val = int(total * 0.2)
        n_test = total - n_train - n_val  # 防止舍入误差

        self.train, self.val, self.test = random_split(full_dataset, [n_train, n_val, n_test], generator=torch.Generator().manual_seed(42))

    def _get_subset(self, idx):
        return {
            'b': self.b[idx],
            'bm': self.bm[idx],
            'temp': self.temp[idx],
            'freq': self.freq[idx],
            'loss': self.loss[idx]
        }

    def _to_tensor(self):
        for d in [self.train, self.val, self.test]:
            for k in d:
                d[k] = torch.from_numpy(d[k]).float()
