import matplotlib.pyplot as plt
import pandas as pd
# 画图
plt.figure(figsize=(20, 10))

data_12k_1797_10c = pd.read_csv("DataSet/origin_sample_data.csv")
# 循环遍历每列数据并生成子图
for index, column in enumerate(data_12k_1797_10c.columns):
    plt.subplot(5, 2, index + 1)  # 5行2列的子图排列方式
    plt.plot(data_12k_1797_10c[column][:1024])
    plt.title(column)  # 每个子图的标题是对应的列名
    plt.xlabel('Index')
    plt.ylabel('Amplitude')

# 调整子图布局，使其不重叠
plt.tight_layout()

# 显示图形
plt.show()
