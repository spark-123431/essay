import torch
import torch.nn as nn
import matplotlib.pyplot as plt

class CNNLSTMNet(nn.Module):
    def __init__(self, hidden_size, lstm_num_layers=1, input_size=1, output_size=1):
        super().__init__()

        self.hidden_size = hidden_size

        # CNN 层：输入通道=3（B, dB, d2B），输出通道=128
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=3, out_channels=128, kernel_size=25, padding=12),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        # LSTM 层：输入维度=128（CNN 输出），输出维度=hidden_size
        self.lstm = nn.LSTM(input_size=128,
                            hidden_size=hidden_size,
                            num_layers=lstm_num_layers,
                            batch_first=True)

        # 全连接层
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
        - x[:,:,1]: freq (scalar, constant per sequence)
        - x[:,:,2]: temp (scalar)
        - x[:,:,3]: dB
        - x[:,:,4]: d2B
        - x[:,:,5]: h  (已弃用)
        """
        batch_size, seq_len, _ = x.shape

        # 静态通道 freq, temp
        in_freq = x[:, 0, 1]
        in_temp = x[:, 0, 2]

        # 动态通道：B, dB, d2B → [B, T, 3] → [B, 3, T]
        dynamic = x[:, :, [0, 3, 4]].permute(0, 2, 1)

        # ---- CNN ----
        cnn_out = self.cnn(dynamic)         # [B, 128, T']
        cnn_out = cnn_out.permute(0, 2, 1)  # → [B, T', 128]

        # ---- LSTM ----
        lstm_out, _ = self.lstm(cnn_out)    # [B, T', hidden_size]
        last_hidden = lstm_out[:, -1, :]    # [B, hidden_size]

        # ---- 拼接静态特征 ----
        out_combined = torch.cat([
            last_hidden,
            in_freq.unsqueeze(1),
            in_temp.unsqueeze(1)
        ], dim=1)

        return self.regression_head(out_combined)

    def valid(self, x):
        return self.forward(x)

def get_global_model():
    return CNNLSTMNet(hidden_size=126, lstm_num_layers=3, input_size=3, output_size=1)

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
    wave = torch.linspace(0, 1, wave_step).unsqueeze(0).repeat(batch_size, 1)  # [64, 128]

    # 一阶导数（差分）
    dB = torch.gradient(wave, dim=1)[0]
    dB[:, 0] = dB[:, 1]

    # 二阶导数
    d2B = torch.gradient(dB, dim=1)[0]
    d2B[:, 0] = d2B[:, 1]

    # 静态特征
    freq = torch.full((batch_size, wave_step), 10.0)
    temp = torch.full((batch_size, wave_step), 100.0)

    # 构造最终 6 通道张量（虽然 h 被弃用，但维度还是 6）
    dummy_h = torch.zeros_like(wave)
    inputs = torch.stack([wave, freq, temp, dB, d2B, dummy_h], dim=2)  # [B, T, 6]

    outputs = model.valid(inputs)
    print("Output shape:", outputs.shape)
    print("First output sample:", outputs[0].item())

    total_params = sum(p.numel() for p in model.parameters())
    print("Total model parameters:", total_params)
