import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader, TensorDataset

from Model import get_global_model
import DataProgress

# ==== 参数 ====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_dir = r"D:\essay\B6log"
material = "3C90"
channel_map = {0: "B", 1: "freq", 2: "temp", 3: "dB", 4: "d2B", 5: "h"}
batch_size = 64
ablation_strategy = 'random'  # 可选：'zero', 'mean', 'random', 'constant'
constant_value = 0.0

# ==== 加载模型 ====
model_path = os.path.join(data_dir, "Trained Weights", f"{material}.ckpt")
model = get_global_model().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# ==== 加载数据 ====
data_path = os.path.join(data_dir, "Processed Training Data", material)
train_set = DataProgress.MagDataset(os.path.join(data_path, "train.mat"))
test_set = DataProgress.MagDataset(os.path.join(data_path, "test.mat"))

inputs_train = train_set.x_data.to(device)
inputs_test = test_set.x_data.to(device)
targets_test = test_set.y_data.to(device)

# ==== 封装分析函数 ====
def run_ablation_analysis(model, inputs, targets, channel_map, strategy='zero', batch_size=64,
                          reference_inputs=None, constant_value=0.0, title=None):
    model.eval()
    device = next(model.parameters()).device

    inputs_all = inputs.to(device)
    targets_all = targets.to(device)

    val_dataset = TensorDataset(inputs_all, targets_all)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    preds_all, targets_list = [], []
    with torch.no_grad():
        for xb, yb in val_loader:
            preds = model(xb)
            preds_all.append(preds.cpu())
            targets_list.append(yb.cpu())
    baseline_preds = torch.cat(preds_all, dim=0)
    targets_cpu = torch.cat(targets_list, dim=0)
    baseline_mse = torch.mean((baseline_preds - targets_cpu) ** 2).item()

    ablation_results = {}
    for c in range(inputs.shape[2]):
        inputs_ablate = inputs_all.clone()
        if strategy == 'zero':
            inputs_ablate[:, :, c] = 0.0
        elif strategy == 'mean':
            mean_val = inputs_all[:, :, c].mean()
            inputs_ablate[:, :, c] = mean_val
        elif strategy == 'constant':
            inputs_ablate[:, :, c] = constant_value
        elif strategy == 'random':
            if reference_inputs is None:
                raise ValueError("reference_inputs must be provided for 'random' strategy.")
            rand_vals = reference_inputs[:, :, c].flatten()
            sampled = rand_vals[torch.randint(0, len(rand_vals), (inputs.shape[0]*inputs.shape[1],))]
            inputs_ablate[:, :, c] = sampled.reshape(inputs.shape[0], inputs.shape[1])
        else:
            raise ValueError(f"Unsupported strategy: {strategy}")

        ablate_loader = DataLoader(TensorDataset(inputs_ablate, targets_all), batch_size=batch_size)
        preds_list = []
        with torch.no_grad():
            for xb, yb in ablate_loader:
                preds = model(xb)
                preds_list.append(preds.cpu())
        preds_all = torch.cat(preds_list, dim=0)
        mse = torch.mean((preds_all - targets_cpu) ** 2).item()
        ablation_results[channel_map[c]] = mse

    ablation_results["Baseline"] = baseline_mse
    df_ablation = pd.DataFrame(list(ablation_results.items()), columns=["Channel", "MSE"])
    df_ablation = df_ablation.sort_values(by="MSE", ascending=False)

    # 输出表格
    print(df_ablation)

    # 绘图
    plt.figure(figsize=(8, 5))
    sns.barplot(data=df_ablation[df_ablation["Channel"] != "Baseline"], x="Channel", y="MSE")
    plt.axhline(y=baseline_mse, color='red', linestyle='--', label="Baseline")
    plt.title(title or f"Ablation Study (Strategy={strategy})")
    plt.ylabel("Validation MSE")
    plt.xlabel("Removed Channel")
    plt.legend()
    plt.tight_layout()
    plt.show()

    return df_ablation

# ==== 执行消融分析 ====
df_result = run_ablation_analysis(
    model=model,
    inputs=inputs_test,
    targets=targets_test,
    channel_map=channel_map,
    strategy=ablation_strategy,
    reference_inputs=inputs_train if ablation_strategy == 'random' else None,
    constant_value=constant_value,
    batch_size=batch_size,
    title=f"Ablation Study on {material} ({ablation_strategy})"
)
