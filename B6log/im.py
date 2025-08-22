import pandas as pd
import matplotlib.pyplot as plt

# 读取 CSV 文件
file_path = r"D:\essay\B6log\Training Progress\3C90_importance.csv"  # 修改为你的本地路径
df = pd.read_csv(file_path)

# 提取第二到第五列
columns_to_plot = df.columns[4:5]  # 注意：列索引从0开始，1:5表示第二到第五列

# 绘制折线图
plt.figure(figsize=(10, 6), dpi=150)
for col in columns_to_plot:
    plt.plot(df.index, df[col], label=col, linewidth=1.5)

plt.xlabel("Epoch", fontsize=12)
plt.ylabel("Value", fontsize=12)
plt.title("Channels (2nd to 5th Columns) vs Epoch", fontsize=14)
plt.legend()
plt.grid(True, linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.show()