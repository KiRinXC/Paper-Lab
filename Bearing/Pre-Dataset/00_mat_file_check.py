import scipy.io

mat_path = r'../DataSet/normal.mat'
# 读取MAT文件
mat_data = scipy.io.loadmat(mat_path)

# 查看MAT文件的内容
print(mat_data)
'''
{'__header__': b'MATLAB 5.0 MAT-file, Platform: PCWIN, Created on: Mon Jan 31 15:28:20 2000', '__version__': '1.0', '__globals__': [], 'X097_DE_time': array([[ 0.05319692],
       [ 0.08866154],
       [ 0.09971815],
       ...,
       [-0.03463015],
       [ 0.01668923],
       [ 0.04693846]], shape=(243938, 1)), 'X097_FE_time': array([[0.14566727],
       [0.09779636],
       [0.05485636],
       ...,
       [0.14053091],
       [0.09553636],
       [0.09019455]], shape=(243938, 1)), 'X097RPM': array([[1796]], dtype=uint16)}
'''