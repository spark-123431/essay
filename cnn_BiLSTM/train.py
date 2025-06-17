import os
import torch
import scipy.io as sio
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from CNN_BiLSTM_model import CNNBiLSTM, RelativeLoss
import numpy as np

def load_dataset(mat_path):
    data = sio.loadmat(mat_path)
    b = torch.tensor(data['b'], dtype=torch.float32).unsqueeze(1)
    bm = torch.tensor(data['bm'].squeeze(), dtype=torch.float32)
    freq = torch.tensor(data['freq'].squeeze(), dtype=torch.float32)
    temp = torch.tensor(data['temp'].squeeze(), dtype=torch.float32)
    loss = torch.tensor(data['loss'].squeeze(), dtype=torch.float32)
    return TensorDataset(b, bm, freq, temp, loss)

def evaluate(model, loader, loss_fn, device):
    model.eval()
    losses = []
    with torch.no_grad():
        for b, bm, f, t, y in loader:
            b, bm, f, t, y = b.to(device), bm.to(device), f.to(device), t.to(device), y.to(device)
            pred = model(b, bm, f, t).squeeze()
            losses.append(loss_fn(pred, y).item())
    return sum(losses) / len(losses)

def train_material(data_root, material, base_model_path, device, epochs=200, batch_size=32, verbose=False):
    material_dir = os.path.join(data_root, "Processed Training Data", material)
    train_set = load_dataset(os.path.join(material_dir, "train.mat"))
    val_set = load_dataset(os.path.join(material_dir, "val.mat"))
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size)

    model = CNNBiLSTM().to(device)
    if base_model_path and os.path.exists(base_model_path):
        model.load_state_dict(torch.load(base_model_path, map_location=device))
        if verbose:
            print(f"[{material}] Loaded base model from {base_model_path}")

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    loss_fn = torch.nn.MSELoss()
    train_losses, valid_losses, epochs_list = [], [], []
    min_valid_loss = float('inf')

    weight_dir = os.path.join(data_root, "Trained Weights")
    progress_dir = os.path.join(data_root, "Training Progress")
    os.makedirs(weight_dir, exist_ok=True)
    os.makedirs(progress_dir, exist_ok=True)

    for epoch in range(epochs):
        model.train()
        for b, bm, f, t, y in train_loader:
            b, bm, f, t, y = b.to(device), bm.to(device), f.to(device), t.to(device), y.to(device)
            pred = model(b, bm, f, t).squeeze()
            loss = loss_fn(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        val_loss = evaluate(model, val_loader, loss_fn, device)

        if val_loss < min_valid_loss:
            min_valid_loss = val_loss
            torch.save(model.state_dict(), os.path.join(weight_dir, f"{material}.ckpt"))
            print(f"[{material}] Model saved. Val Loss: {val_loss:.3e}")

        train_losses.append(loss.item())
        valid_losses.append(val_loss)
        epochs_list.append(epoch + 1)
        scheduler.step()

        if verbose:
            print(f"[{material}] Epoch {epoch + 1}/{epochs}, Train Loss: {loss.item():.3e}, Val Loss: {val_loss:.3e}")

    # 保存训练曲线图
    plt.figure()
    plt.plot(epochs_list, train_losses, label='Train Loss')
    plt.plot(epochs_list, valid_losses, label='Val Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Training Progress: {material}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(progress_dir, f"{material}.pdf"))
    plt.close()

    # 保存 loss 数据
    loss_dict = {
        "train_losses": np.array(train_losses, dtype=np.float32),
        "validation_losses": np.array(valid_losses, dtype=np.float32)
    }
    sio.savemat(os.path.join(progress_dir, f"{material}.mat"), loss_dict)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data_dir = r"E:\project\cnn_BiLSTM"
    weight_dir = os.path.join(data_dir, "Trained Weights")
    base_material = ""  # 设置为 "" 表示不使用 base model
    base_model_path = os.path.join(weight_dir, f"{base_material}.ckpt") if base_material else None

    epochs = 700
    batch_size = 32

    materials_root = os.path.join(data_dir, "Processed Training Data")
    all_materials = sorted(os.listdir(materials_root))
    print(f"Found {len(all_materials)} materials: {all_materials}")

    if base_material:
        if not os.path.exists(base_model_path):
            print(f"[{base_material}] Base model not found. Training from scratch...")
            train_material(data_dir, base_material, None, device, epochs, batch_size, verbose=True)
        else:
            print(f"[{base_material}] Base model exists.")
        for mat in all_materials:
            if mat == base_material:
                continue
            print(f"[{mat}] Training based on base model: {base_material}")
            train_material(data_dir, mat, base_model_path, device, epochs, batch_size, verbose=False)
    else:
        print("No base model specified. Training all materials independently.")
        for mat in all_materials:
            print(f"[{mat}] Training from scratch.")
            train_material(data_dir, mat, None, device, epochs, batch_size, verbose=True)
