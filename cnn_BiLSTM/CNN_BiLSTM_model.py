import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# CNN-BiLSTM 模型
class CNNBiLSTM(nn.Module):
    def __init__(self, input_channels=1, cnn_out_channels=16, kernel_size=25,
                 pool_size=10, lstm_hidden_size=32, lstm_layers=1, fc1_dims=[64, 32, 32, 13], fc2_dims=[16, 16, 16, 1]):
        super(CNNBiLSTM, self).__init__()

        # 一维卷积层和池化层
        self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=cnn_out_channels, kernel_size=kernel_size)
        self.pool = nn.MaxPool1d(kernel_size=pool_size, stride=pool_size)

        # BiLSTM 层
        self.lstm = nn.LSTM(input_size=cnn_out_channels,
                            hidden_size=lstm_hidden_size,
                            num_layers=lstm_layers,
                            batch_first=True,
                            bidirectional=True)

        # 第一个全连接层（FC1）用于整合 LSTM 特征
        fc1_input_dim = 2 * lstm_hidden_size  # 双向 LSTM 输出维度
        fc1_layers = []
        for dim in fc1_dims:
            fc1_layers.append(nn.Linear(fc1_input_dim, dim))
            fc1_layers.append(nn.ReLU())
            fc1_input_dim = dim
        self.fc1 = nn.Sequential(*fc1_layers)

        # 第二个全连接层（FC2）加入 Bm、f、T
        fc2_input_dim = fc1_dims[-1] + 3  # 加入 Bm、频率、温度
        fc2_layers = []
        for dim in fc2_dims:
            fc2_layers.append(nn.Linear(fc2_input_dim, dim))
            fc2_layers.append(nn.ReLU())
            fc2_input_dim = dim
        fc2_layers.pop()  # 去掉最后一个 ReLU
        self.fc2 = nn.Sequential(*fc2_layers)

    def forward(self, b_waveform, bm, freq, temp):
        # b_waveform: [batch_size, 1, 1024]
        x = self.pool(F.relu(self.conv1(b_waveform)))  # -> [batch, out_channels, new_seq_len]
        x = x.permute(0, 2, 1)  # 转为 LSTM 输入 [batch, seq_len, feature]
        x, _ = self.lstm(x)
        x = x[:, -1, :]  # 获取 LSTM 最后一个时间步的输出
        x = self.fc1(x)  # [batch, feature]

        # 拼接 bm, freq, temp
        meta = torch.stack([bm, freq, temp], dim=1)  # [batch, 3]
        x = torch.cat((x, meta), dim=1)
        x = self.fc2(x)
        return torch.exp(x)  # 输出 Pv，对数反变换

def train_model(model, train_loader, val_loader, epochs=200, lr=0.01):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        for b_wave, bm, f, t, y in train_loader:
            optimizer.zero_grad()
            pred = model(b_wave, bm, f, t).squeeze()
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
        scheduler.step()

        if (epoch+1) % 20 == 0:
            model.eval()
            with torch.no_grad():
                val_losses = []
                for b_wave, bm, f, t, y in val_loader:
                    pred = model(b_wave, bm, f, t).squeeze()
                    val_losses.append(criterion(pred, y).item())
                print(f"Epoch {epoch+1}, Val MSE: {np.mean(val_losses):.4f}")


# 示例初始化模型
model = CNNBiLSTM()
