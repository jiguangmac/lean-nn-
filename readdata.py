import struct
import numpy as np
# python 3.11.7


# data预处理,
# 寻找固定的关系，接下来每次都保存前后5次读出数据以后，以前596行作为基准，
# 计算每一个原子与其他原子的距离，形成一个对角线为0的对称邻接矩阵，然后从矩阵中排列，找出例如top20，的xyz，每次读入以后都要计算xyz的变化，产生一个变化矩阵.
struct_fmt = '=iiiiddd'
struct_len = struct.calcsize(struct_fmt)
struck_unpack = struct.Struct(struct_fmt).unpack_from

result = []
arr = []
num_vertex = 596
time_stamp = 1000  # 1000个时刻的数据
inf = 1e5
k = 10  # 可变的参数，每一个原子只要topk小的欧几里得距离
# 读入数据
with open("time_point.byte", mode='rb') as file:
    while (True):
        data = file.read(struct_len)
        if not data:
            break
        s = struck_unpack(data)
        result.append(s)

# 将t=0时刻所有的xyz加入到数组中，只读x，y，z
for i in range(num_vertex):
    arr = np.append(arr, [[result[i][4], result[i][5], result[i][6]]])  # 596*3

arr = np.reshape(arr, (596, 3), order='C')  # 596*3

m = arr.shape[0]
dists = np.zeros((m, m))
index = np.zeros((m, m))
# 计算出二维数组，图的关系矩阵
for i in range(m):
    # broadcasting mechanism
    dists[i, :] = np.sqrt(np.sum((arr[i]-arr) ** 2, axis=1))

np.fill_diagonal(dists, inf)  # 排除自身
index = np.argsort(dists, axis=-1, kind="quicksort")  # 对数组进行排序

weightarray = np.zeros((1, m, k))
indexarray = np.zeros((1, m, k))

for i in range(m):
    weightarray[0, i, :] = dists[i, index[i][:k]]
    indexarray[0, i, :] = index[i][:k]
print(weightarray.shape)
print(indexarray.shape)


# 设计一个算法来更新数据，利用上一时刻的数据来更新
# 例如现在有第0时刻的top10[596,10]的数据，同时还有索引，还有第一时刻的xyz的变化量，
# 但是要注意索引不能重复
# for t_val in range(1,time_stamp,1):


# 目标问题1：596个原子之间的两两稳定关系（共价键检测）
# 目标问题2：稳定基团的认知（多原子稳定基团）
# 目标问题3：基团的频谱系
