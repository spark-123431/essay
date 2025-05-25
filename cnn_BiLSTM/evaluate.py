import os
import torch
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from CNN_BiLSTM_model import CNNBiLSTM

def load_test_dataset(mat_path):
    data = sio.loadmat(mat_path)
    b = torch.tensor(data['b'], dtype=torch.float32).unsqueeze(1)
    bm = torch.tensor(data['bm'].squeeze(), dtype=torch.float32)
    freq = torch.tensor(data['freq'].squeeze(), dtype=torch.float32)
    temp = torch.tensor(data['temp'].squeeze(), dtype=torch.float32)
    loss = torch.tensor(data['loss'].squeeze(), dtype=torch.float32)
    return TensorDataset(b, bm, freq, temp, loss)

def evaluate_model(model, dataloader, device):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for b, bm, f, t, y in dataloader:
            b, bm, f, t, y = b.to(device), bm.to(device), f.to(device), t.to(device), y.to(device)
            pred = model(b, bm, f, t).squeeze()
            preds.append(pred.cpu().numpy())
            targets.append(y.cpu().numpy())
    pred_all = np.concatenate(preds)
    true_all = np.concatenate(targets)
    return abs(pred_all - true_all) / (true_all + 1e-12)

def magplot(material_name, relative_error, save_path="", xlim=50):
    relv_err = np.abs(relative_error) * 100
    avg = np.mean(relv_err)
    p95 = np.percentile(relv_err, 95)
    p99 = np.percentile(relv_err, 99)
    max_err = np.max(relv_err)
    subtitle = f"Avg={avg:.2f}%, 95-Prct={p95:.2f}%, 99-Prct={p99:.2f}%, Max={max_err:.2f}%"
    plt.figure(figsize=(6, 3), dpi=300)
    plt.rcParams["font.family"] = "Times New Roman"
    plt.suptitle(f"Error Distribution for {material_name}", fontsize=18, y=1.05)
    plt.title(f"Avg={avg:.2f}%, 95-Prct={p95:.2f}%, 99-Prct={p99:.2f}%, Max={max_err:.2f}%", fontsize=10)
    plt.hist(relv_err, bins=20, edgecolor='black', density=True, linewidth=0.5)
    def get_density(x, data):
        hist, edges = np.histogram(data, bins=20, density=True)
        for i in range(len(edges) - 1):
            if edges[i] <= x < edges[i + 1]:
                return hist[i]
        return 0
    for stat_func, label in [(np.mean, "Avg"), (lambda x: np.percentile(x, 95), "95-Prct"), (np.max, "Max")]:
        val = stat_func(relv_err)
        if val < xlim:
            y = get_density(val, relv_err) + 0.001
            plt.plot([val, val], [0, y], '--', color="red", linewidth=1)
            plt.text(val + 0.25, y, f'{label}={val:.2f}%', color="red", fontsize=10)
    plt.xlim(0, xlim)
    plt.xlabel("Relative Error of Core Loss [%]", fontsize=14)
    plt.ylabel("Ratio of Data Points", fontsize=14)
    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.tick_params(labelsize=12)
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    else:
        plt.show()
    plt.close()

if __name__ == "__main__":
    # 主程序
    data_dir = r'E:\project\cnn_BiLSTM'
    weights_dir = os.path.join(data_dir, 'Trained Weights')
    processed_data_dir = os.path.join(data_dir, 'Processed Training Data')
    results_dir = os.path.join(data_dir, 'Validation')
    os.makedirs(results_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for fname in os.listdir(weights_dir):
        if not fname.endswith('.ckpt'):
            continue
        material = fname.replace(".ckpt", "")
        model_path = os.path.join(weights_dir, fname)
        test_path = os.path.join(processed_data_dir, material, "test.mat")
        if not os.path.exists(test_path):
            continue
        model = CNNBiLSTM().to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        test_set = load_test_dataset(test_path)
        test_loader = DataLoader(test_set, batch_size=4096)
        rel_err = evaluate_model(model, test_loader, device)
        save_path = os.path.join(results_dir, f"{material}_err.png")
        magplot(material, rel_err, save_path)
