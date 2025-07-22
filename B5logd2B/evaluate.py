import numpy as np
import torch
import os
import pandas as pd
import matplotlib.pyplot as plt
import DataProgress
import Model

data_dir = r'D:\essay\B5logd2B'

weights_dir = os.path.join(data_dir, 'Trained Weights') # Directory containing weighted material models
training_data_dir = os.path.join(data_dir, 'Processed Training Data') # Directory of pre-processed training data
validation_dir = os.path.join(data_dir, 'Validation') # Directory where plots and files for model validation will be stored
os.makedirs(validation_dir, exist_ok=True) # Creates folder if it does not already exist

# Check if 'Trained Weights' folder exists and contains material models
if os.path.isdir(weights_dir):
    # Get the list of models from the weights_dir folder, extracting the name of each .ckpt model which conventionally corresponds to the material name
    weights = [os.path.splitext(item)[0] for item in os.listdir(weights_dir)
                if item.endswith('.ckpt') and os.path.isfile(os.path.join(weights_dir, item))]
    if weights == []:
        raise RuntimeError(f'No .ckpt models found in "{weights_dir}", ensure "Trained Weights" subfolder in data_dir contains individual .ckpt files corresponding to each trained material.')
else:
    print(f'No subfolder labeled "Trained Weights" found at "{data_dir}", please ensure this exists and contains individual .ckpt files corresponding to each trained material.')

# Print the list of trained models found
print("Identified weightings for the following materials:", weights)

device = torch.device("cpu")
print("Device set to", device)

max_samples = 2000

error_summary = []
data = []

for material in weights:
    magData = DataProgress.MagLoader(os.path.join(training_data_dir, material, 'test.mat'))

    # Instantiate the model with appropriate dimensions
    model = Model.get_global_model().to(device)
    model.load_state_dict(torch.load(os.path.join(weights_dir, f'{material}.ckpt'), map_location=device)) # Load trained material model from .ckpt file

    num_samples = min(magData.b.shape[0], max_samples)  # Limits number of samples from validation dataset being used

    step_len = magData.b.shape[1]

    # 一阶导
    dB = np.gradient(magData.b[:num_samples], axis=1)
    dB[:, 0] = dB[:, 1]  # 边界处理

    # 构造 6 通道输入
    x_data = np.zeros([num_samples, step_len, 6], dtype=np.float32)
    x_data[:, :, 0] = magData.b[:num_samples]
    x_data[:, :, 1] = magData.freq[:num_samples]
    x_data[:, :, 2] = magData.temp[:num_samples]
    x_data[:, :, 3] = dB
    x_data[:, :, 4] = magData.h[:num_samples]

    y_data = magData.loss[:num_samples]

    # Now we can pass a batch of sequences through the model
    inputs = torch.tensor(x_data, dtype=torch.float32)
    with torch.no_grad():
        outputs = model.valid(inputs)
    total_params = sum(p.numel() for p in model.parameters())

    print('Data size ', magData.b.shape[0])
    print('model parameters: ', total_params)

    # get model performance
    pred = outputs.detach().numpy() # Get loss prediction
    real = y_data # Actual losses

    real = np.exp(real)
    pred = np.exp(pred)
    relv_error = np.abs(pred - real) / (np.abs(real) + 1e-8)

    # 直接在 log(loss) 空间比较
    #relv_error = np.abs(pred - real) / (np.abs(real) + 1e-8)  # 避免分母为 0

    mean_relv_error = np.mean(relv_error)
    errors = {
        'mean_error': mean_relv_error,
        '95_percentile_error': np.percentile(relv_error, 95),
        '99_percentile_error': np.percentile(relv_error, 99),
        'max_error': relv_error.max()
    }
    error_summary.append(errors)

    record = {'material': material}
    record.update(errors)
    data.append(record)

    # Plot and save error histogram
    DataProgress.magplot(material, relv_error, os.path.join(validation_dir, f'{material}.pdf'))

    # 只取前 50 个样本
    n_plot = min(50, len(real))  # 防止样本不足
    real_50 = real[:n_plot].flatten()
    pred_50 = pred[:n_plot].flatten()

    plt.figure(dpi=150, figsize=(6, 3))
    plt.plot(real_50, label='Actual Loss', linewidth=1.5)
    plt.plot(pred_50, label='Predicted Loss', linewidth=1.5, linestyle='--')
    plt.title(f'Loss Prediction (First {n_plot} Samples) - {material}')
    plt.xlabel('Sample Index')
    plt.ylabel('Total Core Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(validation_dir, f'{material}_loss_first_{n_plot}_samples.pdf'))
    plt.close()

dataframe = pd.DataFrame(data) # Create error dataframe of dictionary lists
dataframe.to_csv(os.path.join(validation_dir, 'model_errors.csv'), index=False)