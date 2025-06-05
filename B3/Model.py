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
        self.std_freq = (1.0, 0.0)
        self.std_loss = (1.0, 0.0)
        self.std_temp = (1.0, 0.0)

    def forward(self, x):
        """
        x: Tensor of shape [batch_size, seq_len, 3]
           - x[:,:,0]: B(t)
           - x[:,:,1]: frequency (scalar per sample)
           - x[:,:,2]: temperature (scalar per sample)
        """
        in_b = x[:, :, 0:1]              # B waveform
        in_freq = x[:, 0, 1]             # scalar: frequency
        in_temp = x[:, 0, 2]             # scalar: temperature

        # --- Data Augmentation ---
        batch_size, seq_len, _ = in_b.shape

        # Random waveform shift
        rand_shifts = torch.randint(seq_len, (batch_size, 1, 1), device=x.device)
        in_b = torch.cat([
            in_b[i].roll(shifts=int(rand_shifts[i]), dims=0).unsqueeze(0)
            for i in range(batch_size)
        ], dim=0)

        # Random vertical flip
        vertical_flip_mask = torch.rand(batch_size, 1, 1, device=x.device) > 0.5
        in_b = torch.where(vertical_flip_mask, -in_b, in_b)

        # Random horizontal flip (i.e., reverse waveform)
        horizontal_flip_mask = torch.rand(batch_size, device=x.device) > 0.5
        if horizontal_flip_mask.any():
            # 找出需要翻转的样本索引
            flip_indices = horizontal_flip_mask.nonzero(as_tuple=True)[0]
            in_b[flip_indices] = torch.flip(in_b[flip_indices], dims=[1])

        # --- LSTM Encoding ---
        out, _ = self.lstm(in_b)
        last_hidden = out[:, -1, :]  # last timestep output

        # --- Concatenate scalar features ---
        out_combined = torch.cat([last_hidden, in_freq.unsqueeze(1), in_temp.unsqueeze(1)], dim=1)

        # --- Fully connected regression ---
        output = self.regression_head(out_combined)

        return output

    def valid(self, x):
        """
        Inference version: no data augmentation.
        """
        in_b = x[:, :, 0:1]
        in_freq = x[:, 0, 1]
        in_temp = x[:, 0, 2]

        out, _ = self.lstm(in_b)
        last_hidden = out[:, -1, :]
        out_combined = torch.cat([last_hidden, in_freq.unsqueeze(1), in_temp.unsqueeze(1)], dim=1)

        return self.regression_head(out_combined)

    def state_dict(self, *args, **kwargs):
        state = super().state_dict(*args, **kwargs)
        state.update({
            'std_b': self.std_b,
            'std_freq': self.std_freq,
            'std_loss': self.std_loss,
            'std_temp': self.std_temp
        })
        return state

    def load_state_dict(self, state_dict, strict=True):
        self.std_b = state_dict.pop('std_b', (1.0, 0.0))
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

    # Synthetic input: ramp wave + fixed freq/temp
    inputs = torch.zeros(batch_size, wave_step, 3)
    wave = torch.linspace(0, 1, wave_step)
    inputs[:, :, 0] = wave  # B(t)
    inputs[:, :, 1] = 10    # frequency
    inputs[:, :, 2] = 100   # temperature

    outputs = model(inputs)
    print("Output shape:", outputs.shape)
    print("First output sample:", outputs[0])

    total_params = sum(p.numel() for p in model.parameters())
    print("Total model parameters:", total_params)
