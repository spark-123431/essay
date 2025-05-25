import numpy as np
import os
import torch
import scipy.io as sio
from torch.utils.data import TensorDataset, random_split


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

            # 不做插值，不归一化，只取 Bm
            self.bm = np.max(np.abs(self.b), axis=1)

            # log 处理（不做归一化）
            self.freq = np.log(self.freq + 1e-12)
            self.temp = np.log(self.temp + 1e-12)
            self.loss = np.log(self.loss + 1e-12)

            if data_type == 'torch':
                self._to_tensor()

            self._split_data()

        else:
            self.b = self.bm = self.temp = self.loss = self.freq = np.array([])

    def _split_data(self):
        assert isinstance(self.b, torch.Tensor), "self.b must be tensor"
        full_dataset = TensorDataset(
            self.b.unsqueeze(1),  # [N, 1, waveform_length]
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
    raw_data_path = r"E:\project\cnn_BiLSTM\materials"
    processed_data_dir = r"E:\project\cnn_BiLSTM\Processed Training Data"
    os.makedirs(processed_data_dir, exist_ok=True)

    for material in os.listdir(raw_data_path):
        raw_path = os.path.join(raw_data_path, material)
        save_path = os.path.join(processed_data_dir, material)

        if not os.path.isdir(raw_path):
            continue

        os.makedirs(save_path, exist_ok=True)

        data = MagLoader(raw_path, data_type='torch', data_source='csv')

        def save_split(split_data, filename):
            b, bm, freq, temp, loss = [], [], [], [], []
            for b_, bm_, f_, t_, l_ in split_data:
                b.append(b_.numpy().squeeze())
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
