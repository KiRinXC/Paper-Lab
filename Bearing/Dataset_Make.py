import pandas as pd

def make_segment(path):
    win_size = 512  # 窗口大小
    overlap_radio = 0.5  # 重叠率
    stride = int(win_size*(1-overlap_radio)) # 步距
    count = 400  # 样本数量
    sample = pd.read_csv(path)

    segment_data = pd.DataFrame(columns=[x for x in range(win_size+1)])   # 总共513列  窗口大小加标签
    label = 0

    for name, data in sample.items():
        for i in range(count):
            seg = data[i*stride:i*stride+win_size]
            one_row = pd.DataFrame([list(seg) + [label]])      # 划分一行的数据
            segment_data = pd.concat([segment_data, one_row],ignore_index=True)  # 按行拼接
        label += 1
    segment_data.to_csv('DataSet/12k_1797_10c.csv')
    print(segment_data)

if __name__ == '__main__':
    path = 'DataSet/origin_sample_data.csv'
    make_segment(path)

    '''segment_data
             0         1         2         3    ...       509       510       511  512
0   0.053197  0.088662  0.099718  0.058621  ... -0.048607 -0.002295  0.049233    0
0   0.106602  0.127881  0.128090  0.101387  ...  0.040263  0.075102  0.068217    0
0   0.077396  0.096589  0.098258  0.083863  ...  0.104516  0.151872  0.160217    0
0   0.052988  0.019610 -0.000209  0.003964  ...  0.056952  0.089496  0.100344    0
0   0.134766  0.085115  0.015020 -0.029623  ...  0.108897  0.047564 -0.002295    0
..       ...       ...       ...       ...  ...       ...       ...       ...  ...
0   0.161217 -0.212790 -0.116141  0.177054  ...  0.019492  0.080405 -0.121420    9
0   0.005279  0.096649 -0.225379 -0.312688  ... -0.455224 -0.306190  0.206293    9
0  -0.150659  0.145379  0.044670 -0.282637  ...  0.084466  0.011777  0.152689    9
0  -0.134009 -0.467407 -0.095431  0.194922  ...  0.050355  0.051573 -0.122232    9
0   0.221724 -0.035736 -0.125481  0.105989  ...  0.067817  0.109644 -0.015431    9

[4000 rows x 513 columns]
    '''