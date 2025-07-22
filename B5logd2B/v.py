# import numpy as np
# import matplotlib.pyplot as plt
# import torch
# import torch.nn as nn
# import DataProgress  # 确保这个模块里有 MagLoader
#
# # ===== 1. 加载数据 =====
# material_path = r"E:\project\B6copy\materials\3C90"
# data = DataProgress.MagLoader(material_path, data_source='csv')
#
# # ===== 2. 选择一条样本，计算导数 =====
# sample_index = 0
# b = data.b[sample_index]
# h = data.h[sample_index]
# dB = np.gradient(b)
# d2B = np.gradient(dB)
# t = np.arange(len(b))
#
# # ===== 3. 在一张图中分4个子图可视化原始信号 =====
# plt.figure(figsize=(12, 8))
#
# plt.subplot(4, 1, 1)
# plt.plot(t, b)
# plt.title("B")
#
# plt.subplot(4, 1, 2)
# plt.plot(t, dB)
# plt.title("dB/dt")
#
# plt.subplot(4, 1, 3)
# plt.plot(t, d2B)
# plt.title("d²B/dt²")
#
# plt.subplot(4, 1, 4)
# plt.plot(t, h)
# plt.title("H")
#
# plt.tight_layout()
# plt.show()
#
# # ===== 4. 构造 CNN 输入并定义模型 =====
# x = np.stack([b, dB, d2B, h], axis=0)  # shape: (4, T)
# x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(0)  # shape: (1, 4, T)
#
# class MyCNN(nn.Module):
#     def __init__(self):
#         super(MyCNN, self).__init__()
#         self.cnn = nn.Sequential(
#             nn.Conv1d(in_channels=4, out_channels=128, kernel_size=25, padding=12),
#             nn.ReLU(),
#             nn.MaxPool1d(kernel_size=2, stride=2)
#         )
#     def forward(self, x):
#         return self.cnn(x)
#
# model = MyCNN()
#
# # ===== 5. 获取 CNN 输出并可视化前6个特征图 =====
# with torch.no_grad():
#     out = model(x_tensor)  # shape: (1, 128, T_out)a
#     out_np = out.squeeze(0).numpy()  # shape: (128, T_out)
#
# plt.figure(figsize=(12, 10))
# for i in range(6):
#     plt.subplot(6, 1, i + 1)
#     plt.plot(out_np[i])
#     plt.title(f"Feature Map {i+1}")
#     plt.grid(True)
# plt.tight_layout()
# plt.show()

import torch
print(torch.cuda.is_available())  # 输出应为 True
print(torch.cuda.get_device_name(0))  # 应显示你的 GPU 型号
