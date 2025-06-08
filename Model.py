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

        # CNN 层：输入通道=1（B），输出通道=128
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=128, kernel_size=25, padding=12),
            nn.ReLU()
        )

        # LSTM 层：输入维度=128（CNN 输出），输出维度=hidden_size
        self.lstm = nn.LSTM(input_size=128,
                            hidden_size=hidden_size,
                            num_layers=lstm_num_layers,
                            batch_first=True)


        # Fully connected regression head
        self.regression_head = nn.Sequential(
            nn.Linear(hidden_size + 3, 128),
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
        self.std_bm = (1.0, 0.0)
        self.std_freq = (1.0, 0.0)
        self.std_loss = (1.0, 0.0)
        self.std_temp = (1.0, 0.0)

    def forward(self, x):

        # 取出静态特征
        in_b = x[:, :, 0:1]  # B waveform
        in_freq = x[:, 0, 1]  # scalar: frequency
        in_temp = x[:, 0, 2]  # scalar: temperature
        in_bm = x[:, 0, 3]  # scalar: Bm

        # --- CNN 编码 ---
        # 转为 (batch, 1, seq_len)
        x_b = in_b.permute(0, 2, 1)
        x_b = self.cnn(x_b)  # (batch, 128, seq_len)
        x_b = x_b.permute(0, 2, 1)  # (batch, seq_len, 128)

        # --- LSTM 编码 ---
        out, _ = self.lstm(x_b)
        last_hidden = out[:, -1, :]  # 取最后时刻输出

        # # --- Data Augmentation ---
        # batch_size, seq_len, _ = in_b.shape
        #
        # # Random waveform shift
        # rand_shifts = torch.randint(seq_len, (batch_size, 1, 1), device=x.device)
        # in_b = torch.cat([
        #     in_b[i].roll(shifts=int(rand_shifts[i]), dims=0).unsqueeze(0)
        #     for i in range(batch_size)
        # ], dim=0)

        # # 上下翻转（对所有动态通道取负）
        # flip_mask = torch.rand(batch_size, 1, 1, device=x.device) > 0.5
        # dynamic = torch.where(flip_mask, -dynamic, dynamic)

        # --- Concatenate scalar features ---
        out_combined = torch.cat([
            last_hidden,
            in_bm.unsqueeze(1),
            in_freq.unsqueeze(1),
            in_temp.unsqueeze(1)], dim=1)

        # --- Fully connected regression ---
        output = self.regression_head(out_combined)

        return output

    def valid(self, x):
        """
        Inference version: no data augmentation.
        x.shape == [batch_size, seq_len, 5]
        Channels: [B, freq, temp, dB, d2B]
        """
        in_b = x[:, :, 0:1]
        in_bm = x[:, 0, 1]
        in_freq = x[:, 0, 2]
        in_temp = x[:, 0, 3]

        # --- CNN 编码 ---
        x_b = in_b.permute(0, 2, 1)
        x_b = self.cnn(x_b)
        x_b = x_b.permute(0, 2, 1)

        # --- LSTM 编码 ---
        out, _ = self.lstm(x_b)
        last_hidden = out[:, -1, :]

        # --- 拼接静态特征 ---
        out_combined = torch.cat([
            last_hidden,
            in_bm.unsqueeze(1),
            in_freq.unsqueeze(1),
            in_temp.unsqueeze(1)
        ], dim=1)

        return self.regression_head(out_combined)


    def state_dict(self, *args, **kwargs):
        state = super().state_dict(*args, **kwargs)
        state.update({
            'std_b': self.std_b,
            'std_bm': self.std_bm,
            'std_freq': self.std_freq,
            'std_loss': self.std_loss,
            'std_temp': self.std_temp
        })
        return state

    def load_state_dict(self, state_dict, strict=True):
        self.std_b = state_dict.pop('std_b', (1.0, 0.0))
        self.std_bm = state_dict.pop('std_bm', (1.0, 0.0))
        self.std_freq = state_dict.pop('std_freq', (1.0, 0.0))
        self.std_loss = state_dict.pop('std_loss', (1.0, 0.0))
        self.std_temp = state_dict.pop('std_temp', (1.0, 0.0))
        super().load_state_dict(state_dict, strict)


def get_global_model():
    """
    Factory function for default model configuration.
    """
    return LSTMSeq2One(hidden_size=30, lstm_num_layers=3, input_size=1, output_size=1)


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

    # 构造简单 ramp 波形 B
    wave = torch.linspace(0, 1, wave_step).unsqueeze(0).repeat(batch_size, 1)  # [B, T]

    # 构造静态特征：Bm, freq, temp（每条样本一个值）
    bm = wave.abs().max(dim=1, keepdim=True)[0]  # [B, 1]
    freq = torch.full((batch_size, 1), 10.0)  # [B, 1]
    temp = torch.full((batch_size, 1), 100.0)  # [B, 1]

    # 广播这些静态特征到 [B, T] 再堆叠为 4 通道
    bm_seq = bm.repeat(1, wave_step)  # [B, T]
    freq_seq = freq.repeat(1, wave_step)
    temp_seq = temp.repeat(1, wave_step)

    # 构造最终输入：B, Bm, freq, temp
    inputs = torch.stack([wave, bm_seq, freq_seq, temp_seq], dim=2)  # [B, T, 4]

    # 模型推理（不含数据增强）
    outputs = model.valid(inputs)
    print("Output shape:", outputs.shape)
    print("First output sample:", outputs[0].item())

    # 打印参数数量
    total_params = sum(p.numel() for p in model.parameters())
    print("Total model parameters:", total_params)