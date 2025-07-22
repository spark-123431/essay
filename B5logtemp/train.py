import os
import torch
import time
import matplotlib.pyplot as plt
import scipy.io as sio
import torch.optim as optim

from Model import get_global_model, RelativeLoss
import DataProgress

def compute_input_channel_importance(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    was_training = model.training
    model.train()  # 切换到训练模式，确保 RNN 反向传播支持

    x = x.clone().detach().requires_grad_(True)
    output = model(x)
    loss = output.sum()
    loss.backward()

    grad = x.grad[:, :, [0, 2, 3, 4]]
    importance = torch.mean(torch.abs(grad), dim=[0, 1])

    if not was_training:
        model.eval()  # 恢复原模式

    return importance.detach()


import csv

def log_importance_to_csv(csv_path: str, epoch: int, importance: torch.Tensor):
    header = ["epoch", "B", "dB", "d2B", "h"]
    write_header = not os.path.exists(csv_path)
    row = [epoch] + importance.tolist()

    with open(csv_path, mode="a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(header)
        writer.writerow(row)


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


def train_model(data_dir, material, base_model_path, device, epochs, verbose=False):
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


    loss_fn = torch.nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200, eta_min=0)

    train_loader = DataProgress.get_dataloader(train_file, batch_size=128, shuffle=True)
    valid_loader = DataProgress.get_dataloader(valid_file)

    min_valid_loss = evaluate(model, valid_loader, loss_fn, device)  # 初始验证 loss

    train_losses, valid_losses, epochs_list = [], [], []

    # === Early Stopping 相关参数 ===
    patience = 200  # 容忍多少轮无提升
    best_epoch = 0  # 记录最佳模型所在的 epoch
    early_stop_counter = 0  # 累计无提升的轮数
    early_stop_triggered = False

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0.0
        total_train_count = 0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item() * inputs.size(0)
            total_train_count += inputs.size(0)

        avg_train_loss = total_train_loss / total_train_count  # 全局平均训练 loss

        # 完整验证集评估
        val_loss = evaluate(model, valid_loader, loss_fn, device)

        train_losses.append(avg_train_loss)
        valid_losses.append(val_loss)
        epochs_list.append(epoch + 1)

        # === 检查是否保存模型 + 更新最佳记录 ===
        if val_loss < min_valid_loss:
            min_valid_loss = val_loss
            best_epoch = epoch + 1
            early_stop_counter = 0
            ckpt_path = os.path.join(weight_dir, f"{material}.ckpt")
            torch.save(model.state_dict(), ckpt_path)
            print(f"[{material}] Model saved. Val Loss: {val_loss:.3e}")
        else:
            early_stop_counter += 1

        # === 判断是否早停 ===
        if early_stop_counter >= patience:
            print(f"[{material}] Early stopping at epoch {epoch + 1}. Best at epoch {best_epoch}.")
            early_stop_triggered = True
            break  # 提前跳出训练循环

        scheduler.step()
        # === 每 N 轮分析通道重要性并记录 ===
        if (epoch + 1) % 10 == 0:
            try:
                # 用前一批数据估计（或者从 train_loader 再取一批）
                inputs_sample, _ = next(iter(train_loader))
                inputs_sample = inputs_sample.to(device)
                importance = compute_input_channel_importance(model, inputs_sample[:16])
                csv_path = os.path.join(progress_dir, f"{material}_importance.csv")
                log_importance_to_csv(csv_path, epoch + 1, importance)
            except Exception as e:
                print(f"[{material}] Importance logging failed at epoch {epoch + 1}: {e}")

        if (epoch + 1) % 10 == 0 and verbose:
            print(
                f"[{material}] Epoch {epoch + 1}/{epochs}, Train Loss: {avg_train_loss:.3e}, Val Loss: {val_loss:.3e}")

    # === 补充提示 ===
    if not early_stop_triggered:
        print(f"[{material}] Training completed full {epochs} epochs. Best epoch: {best_epoch}")

    # 绘图前
    best_epoch_idx = valid_losses.index(min(valid_losses))
    best_val_loss = valid_losses[best_epoch_idx]
    # 保存 loss 曲线
    fig, ax = plt.subplots()
    ax.plot(epochs_list, train_losses, label='Train Loss')
    ax.plot(epochs_list, valid_losses, label='Val Loss')
    ax.scatter(epochs_list[best_epoch_idx], best_val_loss, color='red', label='Best Val Loss')

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

    # === 附加图：从第 50 轮开始绘制 ===
    if len(epochs_list) >= 50:
        # 筛选 epoch >= 50 的数据点
        filtered_epochs = [e for e in epochs_list if e >= 50]
        filtered_train = train_losses[49:]  # 索引从 epoch 50 起，即 index=49
        filtered_valid = valid_losses[49:]

        fig, ax = plt.subplots()
        ax.plot(filtered_epochs, filtered_train, label='Train Loss (Epoch ≥ 50)')
        ax.plot(filtered_epochs, filtered_valid, label='Val Loss (Epoch ≥ 50)')

        # 如果 best_epoch 也在这个区间，就标记出来
        if best_epoch >= 50:
            ax.scatter(best_epoch, best_val_loss, color='red', label='Best Val Loss')

        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title(f'Training Progress (From Epoch 50): {material}')
        ax.legend()
        fig.tight_layout()
        zoom_fig_path = os.path.join(progress_dir, f"{material}_from_epoch50.pdf")
        fig.savefig(zoom_fig_path)
        plt.close()

    # === 通道重要性曲线图（可选） ===
    importance_csv = os.path.join(progress_dir, f"{material}_importance.csv")
    if os.path.exists(importance_csv):
        import pandas as pd
        df = pd.read_csv(importance_csv)
        fig, ax = plt.subplots()
        for col in ["B", "dB", "d2B", "h"]:
            ax.plot(df["epoch"], df[col], label=col)

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Gradient Importance")
        ax.set_title(f"Input Channel Importance: {material}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(progress_dir, f"{material}_importance.pdf"))
        plt.close()


# === 主程序入口 ===
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 通用配置
    data_dir = r"D:\essay\B5logtemp"
    weight_dir = os.path.join(data_dir, 'Trained Weights')
    base_material = "3C90"
    base_model_path = os.path.join(weight_dir, f"{base_material}.ckpt")
    epochs = 600
    verbose = False

    # 所有材料路径
    materials_root = os.path.join(data_dir, 'Processed Training Data')
    all_materials = sorted(os.listdir(materials_root))
    print(f"\nFound {len(all_materials)} materials: {all_materials}")

    # === 第一步：训练 base_material（如果模型文件不存在）===
    if not os.path.exists(base_model_path):
        print(f"[{base_material}] Base model not found. Training from scratch...")
        train_model(data_dir, base_material, None, device, epochs, verbose=True)
    else:
        print(f"[{base_material}] Base model already exists.")

    # === 第二步：遍历其余材料，基于 base model 微调训练 ===
    for material in all_materials:
        if material == base_material:
            continue  # 已处理，无需再训练

        print(f"\n[{material}] Training based on base model: {base_material}")
        train_model(data_dir, material, base_model_path, device, epochs, verbose=False)
