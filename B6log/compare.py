import os
import numpy as np
import scipy.io as sio

# 设置路径
base_dir = r"D:\essay\B6log\Processed Training Data\3C90"
train_file = os.path.join(base_dir, "train.mat")
valid_file = os.path.join(base_dir, "valid.mat")

# 加载 freq 数据
def load_freq(path):
    mat_data = sio.loadmat(path)
    freq = mat_data['freq']
    freq = freq.flatten()  # 保证是一维
    return np.unique(freq)

train_freq = load_freq(train_file)
valid_freq = load_freq(valid_file)

# 比较重合（浮点比较加一定容差）
def find_overlap(a, b, tol=1e-6):
    overlap = []
    for f in a:
        if np.any(np.isclose(f, b, atol=tol)):
            overlap.append(f)
    return np.array(overlap)


train_freq_set = set(train_freq)
valid_freq_set = set(valid_freq)

overlap = train_freq_set & valid_freq_set

print("=== Frequency Overlap Check ===")
print(f"Train freq count: {len(train_freq_set)}")
print(f"Valid freq count: {len(valid_freq_set)}")
print(f"Overlap count: {len(overlap)}")
print(f"Overlap rate (valid): {len(overlap) / len(valid_freq_set) * 100:.2f}%")


# 如需打印重合频率，取消注释：
# print("Overlapping frequencies (Hz):", overlap_freqs)
