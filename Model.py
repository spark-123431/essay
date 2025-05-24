import torch
import torch.nn as nn
import matplotlib.pyplot as plt


class LSTMSeq2One(nn.Module):
    """
    LSTM-based sequence-to-one regression model.
    Inputs:
        - B(t) waveform sequence: shape (batch, seq_len, 1)
        - Frequency and temperature: injected as scalar features at final stage
    """

    def __init__(self, hidden_size, lstm_num_layers=1, input_size=1, output_size=1):
        super().__init__()

        self.hidden_size = hidden_size

        # LSTM: processes B waveform
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=lstm_num_layers,
                            batch_first=True)

        # Fully connected regression head
        self.regression_head = nn.Sequential(
            nn.Linear(hidden_size + 2, 128),
            nn.ELU(),
            nn.Linear(128, 196),
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

        # Default normalization coefficients (mean, std)
        self.std_b = (1.0, 0.0)
        self.std_h = (1.0, 0.0)
        self.std_freq = (1.0, 0.0)
        self.std_loss = (1.0, 0.0)
        self.std_temp = (1.0, 0.0)

    def forward(self, x):
        """
        x.shape == [B, T, 5]
        - x[:,:,0]: B
        - x[:,:,1]: freq (scalar, constant per sequence)
        - x[:,:,2]: temp (scalar)
        - x[:,:,3]: dB
        - x[:,:,4]: d2B
        """
        batch_size, seq_len, _ = x.shape

        # 取出静态特征
        in_freq = x[:, 0, 1]  # shape [B]
        in_temp = x[:, 0, 2]  # shape [B]

        # 取出动态 4 通道：[B, T, 4]
        dynamic = x[:, :, [0, 3, 4, 5]]  # B, dB, d2B, h

        # ---- 数据增强 ----

        # 时间平移（对所有通道同步）
        rand_shifts = torch.randint(seq_len, (batch_size,), device=x.device)
        dynamic = torch.stack([
            dynamic[i].roll(shifts=int(rand_shifts[i].item()), dims=0)
            for i in range(batch_size)
        ], dim=0)  # shape [B, T, 4]

        # 上下翻转（对所有动态通道取负）
        flip_mask = torch.rand(batch_size, 1, 1, device=x.device) > 0.5
        dynamic = torch.where(flip_mask, -dynamic, dynamic)

        # ---- LSTM ----
        out, _ = self.lstm(dynamic)  # input: [B, T, 3]
        last_hidden = out[:, -1, :]  # [B, hidden_size]

        # ---- 拼接静态信息 ----
        out_combined = torch.cat([
            last_hidden,
            in_freq.unsqueeze(1),
            in_temp.unsqueeze(1)
        ], dim=1)

        # ---- FC 层回归 ----
        output = self.regression_head(out_combined)

        return output

    def valid(self, x):
        """
        Inference version: no data augmentation.
        x.shape == [batch_size, seq_len, 5]
        Channels: [B, freq, temp, dB, d2B]
        """
        batch_size, seq_len, _ = x.shape

        # 提取静态通道（常量，取第一个时间步即可）
        in_freq = x[:, 0, 1]  # [batch_size]
        in_temp = x[:, 0, 2]

        # 提取动态通道：B, dB, d2B, h
        dynamic = x[:, :, [0, 3, 4, 5]]  # shape [batch_size, seq_len, 3]

        # LSTM 编码（不进行任何数据增强）
        out, _ = self.lstm(dynamic)
        last_hidden = out[:, -1, :]  # shape [batch_size, hidden_size]

        # 拼接静态特征
        out_combined = torch.cat([
            last_hidden,
            in_freq.unsqueeze(1),  # [batch_size, 1]
            in_temp.unsqueeze(1)
        ], dim=1)  # [batch_size, hidden_size + 2]

        # 全连接层预测
        return self.regression_head(out_combined)

    def state_dict(self, *args, **kwargs):
        state = super().state_dict(*args, **kwargs)
        state.update({
            'std_b': self.std_b,
            'std_h': self.std_h,
            'std_freq': self.std_freq,
            'std_loss': self.std_loss,
            'std_temp': self.std_temp
        })
        return state

    def load_state_dict(self, state_dict, strict=True):
        self.std_b = state_dict.pop('std_b', (1.0, 0.0))
        self.std_h = state_dict.pop('std_h', (1.0, 0.0))
        self.std_freq = state_dict.pop('std_freq', (1.0, 0.0))
        self.std_loss = state_dict.pop('std_loss', (1.0, 0.0))
        self.std_temp = state_dict.pop('std_temp', (1.0, 0.0))
        super().load_state_dict(state_dict, strict)


def get_global_model():
    """
    Factory function for default model configuration.
    """
    return LSTMSeq2One(hidden_size=30, lstm_num_layers=3, input_size=4, output_size=1)


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
