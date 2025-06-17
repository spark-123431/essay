import scipy.io
import matplotlib.pyplot as plt
import numpy as np

# 载入 .mat 文件
mat = scipy.io.loadmat('E:/project/B6copy/Processed Training Data/3C90/test.mat')

# 打印所有变量名，排除内置变量
data_keys = [key for key in mat.keys() if not key.startswith('__')]

# 提取每个变量并画出散点图
for key in data_keys:
    data = mat[key]
    # 展平为一维向量（如需）
    if data.ndim > 1:
        data = data.flatten()
    x = np.arange(len(data))
    plt.figure()
    plt.scatter(x, data, s=2)
    plt.title(f'Scatter Plot of {key}')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
