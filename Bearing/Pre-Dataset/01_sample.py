# 采样并数据归一化处理

import scipy.io
import pandas as pd
from sklearn.preprocessing import StandardScaler

# 数据列名和文件夹路径
data_columns = ['X097_DE_time', 'X105_DE_time', 'X118_DE_time', 'X130_DE_time', 'X169_DE_time',
                'X185_DE_time', 'X197_DE_time', 'X209_DE_time', 'X222_DE_time', 'X234_DE_time']
columns_name = ['normal', 'IR007', 'B007', 'OR007', 'IR014', 'B014', 'OR014', 'IR021', 'B021', 'OR021']
root_path = r'../DataSet'

# 初始化一个空的 DataFrame 来存储所有数据
data_12k_1797_10c = pd.DataFrame()
length_limit = []

# 初始化 scaler
scaler = StandardScaler()

# 循环处理每一个文件
for index in range(10):
    # 拼接文件路径
    file_path = f"{root_path}//{columns_name[index]}.mat"

    # 加载 .mat 文件
    data = scipy.io.loadmat(file_path)

    # 记录原始长度信息
    length_limit.append(data[data_columns[index]].shape)

    # 提取对应的通道数据
    data_list = data[data_columns[index]].reshape(-1)

    # 截取统一长度（如 102656）
    data_segment = data_list[:102656].reshape(-1, 1)

    # 这里不进行标准化（Z-score）
    # scaled_segment = scaler.fit_transform(data_segment).flatten()

    # 添加到 DataFrame
    data_12k_1797_10c[columns_name[index]] = data_segment.flatten()

# 打印采样点数信息
print(length_limit)
'''
[(243938, 1), (121265, 1), (122571, 1), (121991, 1), (121846, 1), (121846, 1), (121846, 1), (122136, 1), (121991, 1), (122426, 1)]
'''
# 打印归一化后的 DataFrame 预览
print(data_12k_1797_10c.head(),data_12k_1797_10c.shape)
'''
     normal     IR007      B007  ...     IR021      B021     OR021
0  0.053197 -0.083004 -0.002761  ...  1.189431 -0.007959  0.104365
1  0.088662 -0.195734 -0.096324  ... -0.177866  0.025340  0.017462
2  0.099718  0.233419  0.113705  ... -0.774816  0.000162  0.116547
3  0.058621  0.103958  0.257297  ...  0.501518  0.092913  0.371164
4 -0.004590 -0.181115 -0.058314  ...  0.993697 -0.007797  0.356951

[5 rows x 10 columns] (102656, 10)
'''

# 保存到 CSV
data_12k_1797_10c.to_csv(r'../DataSet/origin_sample_data.csv', index=False)


