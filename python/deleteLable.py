import pandas as pd

# 读取 CSV 文件
df = pd.read_csv('your_file.csv')  # 替换为你的 CSV 文件路径

# 删除最后一列（即标签列 "Label"）
df_without_label = df.iloc[:, :-1]

# 打印结果（可选）
print(df_without_label)

# 如果需要保存新文件（不含标签）
df_without_label.to_csv('your_file_without_label.csv', index=False)
