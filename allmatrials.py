import DataProgress
import os
import scipy.io as sio
import numpy as np

if __name__ == '__main__':
    # 路径设定
    raw_data_path = r"E:\project\allmaterials"
    processed_data_dir = os.path.join(r"E:\project\Processed Training Data", "allmaterials")
    newStep = 128

    os.makedirs(processed_data_dir, exist_ok=True)

    # 全材料数据缓存
    all_b, all_h, all_freq, all_temp, all_loss = [], [], [], [], []
    std_saved = False
    for material in os.listdir(raw_data_path):
        raw_path = os.path.join(raw_data_path, material)

        if not os.path.isdir(raw_path):
            continue

        print(f"处理材料: {material}")
        data = DataProgress.MagLoader(raw_path, data_source='csv')
        data.temp = data.temp[:, np.newaxis] if data.temp.ndim == 1 else data.temp
        data.freq = data.freq[:, np.newaxis] if data.freq.ndim == 1 else data.freq
        data.loss = data.loss[:, np.newaxis] if data.loss.ndim == 1 else data.loss

        # 数据归一化、插值
        data, std_b, std_h, std_freq, std_temp, std_loss = DataProgress.dataTransform(data, newStep, savePath=None)  # 不保存中间文件

        # 累积数据
        all_b.append(data.b)
        all_h.append(data.h)
        all_freq.append(data.freq)
        all_temp.append(data.temp)
        all_loss.append(data.loss)

        # 数据处理，返回归一化器
        data, std_b, std_h, std_freq, std_temp, std_loss = DataProgress.dataTransform(data, newStep, savePath=None)

        # 保存全局标准化参数（只保存一次）
        if not std_saved:
            std_b.save(os.path.join(processed_data_dir, 'std_b.npy'))
            std_h.save(os.path.join(processed_data_dir, 'std_h.npy'))
            std_freq.save(os.path.join(processed_data_dir, 'std_freq.npy'))
            std_temp.save(os.path.join(processed_data_dir, 'std_temp.npy'))
            std_loss.save(os.path.join(processed_data_dir, 'std_loss.npy'))
            std_saved = True

    # 拼接所有样本
    all_b = np.concatenate(all_b, axis=0)
    all_h = np.concatenate(all_h, axis=0)
    all_freq = np.concatenate(all_freq, axis=0)
    all_temp = np.concatenate(all_temp, axis=0)
    all_loss = np.concatenate(all_loss, axis=0)

    # 构造 MagLoader 对象用于切分
    merged = DataProgress.MagLoader()
    merged.b = all_b
    merged.h = all_h
    merged.freq = all_freq
    merged.temp = all_temp
    merged.loss = all_loss

    # 划分并保存为 allmaterials/train.mat、valid.mat、test.mat
    DataProgress.dataSplit(merged, processed_data_dir)

    print(f"已合并并保存为统一数据集，路径：{processed_data_dir}")

