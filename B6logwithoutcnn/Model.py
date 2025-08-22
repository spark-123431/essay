import torch
import torch.nn as nn

class LSTMNet(nn.Module):
    def __init__(self, hidden_size, lstm_num_layers=1, input_size=4, output_size=1):
        super().__init__()
        self.hidden_size = hidden_size

        # LSTM：直接接收动态通道 [B, dB, d2B, h]，每步输入维度=4
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=lstm_num_layers,
                            batch_first=True)

        # 全连接头部（与原始保持一致）：拼接静态特征 freq、temp
        self.regression_head = nn.Sequential(
            nn.Linear(hidden_size + 2, 196),
            nn.ELU(),
            nn.Linear(196, 128),
            nn.ELU(),
            nn.Linear(128, 96),
            nn.ELU(),
            nn.Linear(96, 32),
            nn.ELU(),
            nn.Linear(32, 32),
            nn.ELU(),
            nn.Linear(32, 16),
            nn.Linear(16, output_size)
        )

    def forward(self, x):
        """
        x.shape == [B, T, 6]
        - x[:,:,0]: B
        - x[:,:,1]: freq (scalar per sequence)
        - x[:,:,2]: temp (scalar per sequence)
        - x[:,:,3]: dB
        - x[:,:,4]: d2B
        - x[:,:,5]: h
        """
        # 静态特征
        in_freq = x[:, 0, 1]
        in_temp = x[:, 0, 2]

        # 动态通道：直接送入 LSTM，形状 [B, T, 4]
        dynamic = x[:, :, [0, 3, 4, 5]]

        # LSTM
        lstm_out, _ = self.lstm(dynamic)         # [B, T, hidden_size]
        last_hidden = lstm_out[:, -1, :]         # [B, hidden_size]

        # 拼接静态特征
        out_combined = torch.cat(
            [last_hidden, in_freq.unsqueeze(1), in_temp.unsqueeze(1)], dim=1
        )

        return self.regression_head(out_combined)

    def valid(self, x):
        return self.forward(x)


def get_global_model():
    return LSTMNet(hidden_size=126, lstm_num_layers=3, input_size=4, output_size=1)


# === Loss Functions ===

class RelativeLoss(nn.Module):
    def forward(self, output, target):
        return torch.mean(torch.pow((target - output) / target, 2))


class RelativeLossAbs(nn.Module):
    def forward(self, output, target):
        return torch.mean(torch.abs((target - output) / target))


class RelativeLoss95(nn.Module):
    def forward(self, output, target):
        error = torch.pow((target - output) / target, 2)
        error, _ = torch.sort(error)
        cutoff = int(error.shape[0] * 0.97)
        return torch.mean(error[:cutoff])


# === Demo Run ===
if __name__ == '__main__':
    model = get_global_model()
    wave_step = 128
    batch_size = 64

    # 构造 ramp 波形
    wave = torch.linspace(0, 1, wave_step).unsqueeze(0).repeat(batch_size, 1)  # shape [64, 128]
    wave_h = torch.linspace(0, 1, wave_step).unsqueeze(0).repeat(batch_size, 1)

    # 一阶导数（差分）
    dB = torch.gradient(wave, dim=1)[0]     # 沿着 dim=1 求导
    dB[:, 0] = dB[:, 1]  # 边界处理
    dB = dB

    # 二阶导数（再次差分）
    d2B = torch.gradient(dB, dim=1)[0]
    d2B[:, 0] = d2B[:, 1]

    # 拼接 freq 和 temp，都是常量，广播生成
    freq = torch.full((batch_size, wave_step), 10.0)      # 频率
    temp = torch.full((batch_size, wave_step), 100.0)     # 温度

    # 构造最终 6 通道张量
    inputs = torch.stack([wave, freq, temp, dB, d2B, wave_h], dim=2)  # shape [B, T, 6]

    # 推理
    outputs = model.valid(inputs)  # valid() 无数据增强版本
    print("Output shape:", outputs.shape)
    print("First output sample:", outputs[0].item())

    # 打印参数数量
    total_params = sum(p.numel() for p in model.parameters())
    print("Total model parameters:", total_params)