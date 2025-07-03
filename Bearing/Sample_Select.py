# 选择每个Mat文件中的采样点

import scipy.io
import pandas as pd

# 数据列名和文件夹路径
data_columns = ['X097_DE_time', 'X105_DE_time', 'X118_DE_time', 'X130_DE_time', 'X169_DE_time',
                'X185_DE_time', 'X197_DE_time', 'X209_DE_time', 'X222_DE_time', 'X234_DE_time']
columns_name = ['normal', 'IR007', 'B007', 'OR007', 'IR014', 'B014', 'OR014', 'IR021', 'B021', 'OR021']
root_path = r'DataSet'

# 初始化一个空的 DataFrame 来存储所有数据
data_12k_1797_10c = pd.DataFrame()
length_limit = []
# 循环遍历每一个文件
for index in range(10):
    # 正确的文件路径拼接
    file_path = f"{root_path}//{columns_name[index]}.mat"

    # 读取MAT文件
    data = scipy.io.loadmat(file_path)

    # 获取数据的形状信息，便于统一长度。
    length_limit.append(data[data_columns[index]].shape)

    # 提取对应的数据列
    data_list = data[data_columns[index]].reshape(-1)

    # 将提取的数据添加到 DataFrame 中,这里统一只提取前120000个数据。
    data_12k_1797_10c[columns_name[index]] = data_list[:102656]

# 输出所有数据的形状
# [(243938, 1), (121265, 1), (122571, 1), (121991, 1), (121846, 1), (121846, 1), (121846, 1), (122136, 1), (121991, 1), (122426, 1)]
print(length_limit)

# 输出最终的 DataFrame（包含所有提取的数据）
print(data_12k_1797_10c)

# 将数据保存到 csv
data_12k_1797_10c.to_csv(r'DataSet/origin_sample_data.csv',index=False)

