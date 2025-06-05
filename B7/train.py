import os
import torch
import time
import matplotlib.pyplot as plt
import scipy.io as sio
import torch.optim as optim

from Model import get_global_model, RelativeLoss
import DataProgress


def evaluate(model, data_loader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    total_count = 0
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            total_loss += loss.item() * inputs.size(0)
            total_count += inputs.size(0)
    return total_loss / total_count


def train_model(data_dir, material, base_model_path, device, epochs, valid_batch_size, verbose=False):
    training_data_dir = os.path.join(data_dir, 'Processed Training Data')
    weight_dir = os.path.join(data_dir, 'Trained Weights')
    progress_dir = os.path.join(data_dir, 'Training Progress')
    os.makedirs(weight_dir, exist_ok=True)
    os.makedirs(progress_dir, exist_ok=True)

    material_path = os.path.join(training_data_dir, material)
    train_file = os.path.join(material_path, "train.mat")
    valid_file = os.path.join(material_path, "valid.mat")

    if not os.path.exists(train_file) or not os.path.exists(valid_file):
        print(f"[{material}] Skipped: Missing data files.")
        return

    model = get_global_model().to(device)
    if verbose:
        print(f"\n========== Training {material} ==========")
        print(model)
        print("Total parameters:", sum(p.numel() for p in model.parameters()))

    if base_model_path:
        model.load_state_dict(torch.load(base_model_path, map_location=device))
        print("Pre-trained model loaded.")

    try:
        std_b = DataProgress.linear_std(); std_b.load(os.path.join(material_path, "std_b.npy"))
        std_h = DataProgress.linear_std(); std_h.load(os.path.join(material_path, "std_h.npy"))
        std_freq = DataProgress.linear_std(); std_freq.load(os.path.join(material_path, "std_freq.npy"))
        std_temp = DataProgress.linear_std(); std_temp.load(os.path.join(material_path, "std_temp.npy"))
        std_loss = DataProgress.linear_std(); std_loss.load(os.path.join(material_path, "std_loss.npy"))
    except Exception:
        print(f"[{material}] Skipped: Missing std files.")
        return

    model.std_b = (std_b.k, std_b.b)
    model.std_h = (std_h.k, std_h.b)
    model.std_freq = (std_freq.k, std_freq.b)
    model.std_temp = (std_temp.k, std_temp.b)
    model.std_loss = (std_loss.k, std_loss.b)

    loss_fn = RelativeLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200, eta_min=0)

    train_loader = DataProgress.get_dataloader(train_file, batch_size=128, shuffle=True)
    valid_loader = DataProgress.get_dataloader(valid_file)

    min_valid_loss = evaluate(model, valid_loader, loss_fn, device)  # 初始验证 loss

    train_losses, valid_losses, epochs_list = [], [], []

    for epoch in range(epochs):
        model.train()
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # 完整验证集评估
        val_loss = evaluate(model, valid_loader, loss_fn, device)

        if val_loss < min_valid_loss:
            min_valid_loss = val_loss
            ckpt_path = os.path.join(weight_dir, f"{material}.ckpt")
            torch.save(model.state_dict(), ckpt_path)
            print(f"[{material}] Model saved. Val Loss: {val_loss:.3e}")

        train_losses.append(loss.item())
        valid_losses.append(val_loss)
        epochs_list.append(epoch + 1)

        scheduler.step()

        if (epoch + 1) % 10 == 0 and verbose:
            print(f"[{material}] Epoch {epoch+1}/{epochs}, Train Loss: {loss.item():.3e}, Val Loss: {val_loss:.3e}")

    # 保存 loss 曲线
    fig, ax = plt.subplots()
    ax.plot(epochs_list, train_losses, label='Train Loss')
    ax.plot(epochs_list, valid_losses, label='Val Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title(f'Training Progress: {material}')
    ax.legend()
    fig.tight_layout()
    fig_path = os.path.join(progress_dir, f"{material}.pdf")
    fig.savefig(fig_path)
    plt.close()

    # 保存 loss 数据
    loss_dict = {'train_losses': train_losses, 'validation_losses': valid_losses}
    mat_path = os.path.join(progress_dir, f"{material}.mat")
    sio.savemat(mat_path, loss_dict)


# === 主程序入口 ===
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 通用配置
    data_dir = "E:/project/B7/"
    weight_dir = os.path.join(data_dir, 'Trained Weights')
    base_material = "3C90"
    base_model_path = os.path.join(weight_dir, f"{base_material}.ckpt")
    epochs = 700
    valid_batch_size = 5000
    verbose = False

    # 所有材料路径
    materials_root = os.path.join(data_dir, 'Processed Training Data')
    all_materials = sorted(os.listdir(materials_root))
    print(f"\nFound {len(all_materials)} materials: {all_materials}")

    # === 第一步：训练 base_material（如果模型文件不存在）===
    if not os.path.exists(base_model_path):
        print(f"[{base_material}] Base model not found. Training from scratch...")
        train_model(data_dir, base_material, None, device, epochs, valid_batch_size, verbose=True)
    else:
        print(f"[{base_material}] Base model already exists.")

    # === 第二步：遍历其余材料，基于 base model 微调训练 ===
    for material in all_materials:
        if material == base_material:
            continue  # 已处理，无需再训练

        print(f"\n[{material}] Training based on base model: {base_material}")
        train_model(data_dir, material, base_model_path, device, epochs, valid_batch_size, verbose=False)
