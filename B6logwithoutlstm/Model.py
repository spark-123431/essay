import torch
import torch.nn as nn
import matplotlib.pyplot as plt


class CNNOnlyNet(nn.Module):
    def __init__(self, input_size=1, output_size=1):
        super().__init__()

        # CNN 层：输入通道=4（B, dB, d2B, h），输出通道=128
        # 使用 1D 卷积 + 最大池化降低时序长度
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=4, out_channels=128, kernel_size=25, padding=12),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        # 时域全局平均池化，将 [B, 128, T'] → [B, 128]
        self.global_pool = nn.AdaptiveAvgPool1d(output_size=1)

        # 回归头：输入 128(动态) + 2(静态 freq, temp)
        self.regression_head = nn.Sequential(
            nn.Linear(128 + 2, 196),
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
        - x[:,:,1]: freq (标量，序列内常量)
        - x[:,:,2]: temp (标量)
        - x[:,:,3]: dB
        - x[:,:,4]: d2B
        - x[:,:,5]: h
        """
        batch_size, seq_len, _ = x.shape

        # 静态通道 freq, temp（取序列首帧）
        in_freq = x[:, 0, 1]
        in_temp = x[:, 0, 2]

        # 动态通道：B, dB, d2B, h → [B, T, 4] → [B, 4, T]
        dynamic = x[:, :, [0, 3, 4, 5]].permute(0, 2, 1)

        # ---- CNN ----
        cnn_feat = self.cnn(dynamic)                  # [B, 128, T']
        pooled = self.global_pool(cnn_feat).squeeze(-1)  # [B, 128]

        # ---- 拼接静态特征 ----
        out_combined = torch.cat([
            pooled,
            in_freq.unsqueeze(1),
            in_temp.unsqueeze(1)
        ], dim=1)  # [B, 130]

        return self.regression_head(out_combined)

    def valid(self, x):
        # 验证逻辑与 forward 一致
        return self.forward(x)


def get_global_model():
    """
    Factory function for default model configuration (无 LSTM 版本).
    """
    return CNNOnlyNet(input_size=4, output_size=1)


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
    wave_h = torch.linspace(0, 1, wave_step).unsqueeze(0).repeat(batch_size, 1)

    # 一阶导数（差分）
    dB = torch.gradient(wave, dim=1)[0]
    dB[:, 0] = dB[:, 1]

    # 二阶导数（再次差分）
    d2B = torch.gradient(dB, dim=1)[0]
    d2B[:, 0] = d2B[:, 1]

    # freq 和 temp 常量
    freq = torch.full((batch_size, wave_step), 10.0)
    temp = torch.full((batch_size, wave_step), 100.0)

    # 构造最终 6 通道张量
    inputs = torch.stack([wave, freq, temp, dB, d2B, wave_h], dim=2)  # [B, T, 6]

    # 推理
    outputs = model.valid(inputs)
    print("Output shape:", outputs.shape)
    print("First output sample:", outputs[0].item())

    # 打印参数数量
    total_params = sum(p.numel() for p in model.parameters())
    print("Total model parameters:", total_params)
