import scipy.io

mat_path = r'DataSet/normal.mat'
# 读取MAT文件
mat_data = scipy.io.loadmat(mat_path)

# 查看MAT文件的内容
print(mat_data)
