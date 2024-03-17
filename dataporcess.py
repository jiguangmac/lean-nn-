import struct
import numpy as np
# python 3.11.7


# data预处理,
# 寻找固定的关系以前596行作为基准，
# 计算每一个原子与其他原子的距离，形成一个对角线为0的对称邻接矩阵，然后从矩阵中排列，找出例如top20，的xyz，每次读入以后都要计算xyz的变化，产生一个变化矩阵.
struct_fmt = '=iiiiddd'
struct_len = struct.calcsize(struct_fmt)
struck_unpack = struct.Struct(struct_fmt).unpack_from
time_stamp = 1001  # 1001个时刻的数据
num_vertex = 596
inf = 1e5
k = 10  # 可变的参数，每一个原子只要topk小的欧几里得距离
x_idx = 4
y_idx = 5
z_idx = 6


result = [[] for _ in range(time_stamp)]
arr = []
# 读入数据
with open("time_point.byte", mode='rb') as file:
    for i in range(time_stamp):
        for j in range(num_vertex):
            data = file.read(struct_len)
            s = struck_unpack(data)
            result[i].append(s)
        if not data:
            break

# 将t=0时刻所有的xyz加入到数组中，只读x，y，z
for i in range(num_vertex):
    arr = np.append(
        arr, [[result[0][i][x_idx], result[0][i][y_idx], result[0][i][z_idx]]])  # 596*3

# order
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


weightarray = np.zeros((time_stamp, m, k))
indexarray = np.zeros((time_stamp, m, k))

for i in range(m):
    weightarray[0, i, :] = dists[i, index[i][:k]]
    indexarray[0, i, :] = index[i][:k]


# 设计一个算法来更新数据，利用上一时刻的数据来更新
# 例如现在有第0时刻的top10[596,10]的数据，同时还有索引
# 这时可以利用索引将数据取出，然后再根据索引的索引每个数据引入5个新的值
# 去除重复值然后排序

for t_val in range(1, time_stamp, 1):
    for i in range(m):
        idx = indexarray[t_val-1, i, :]
        sortarr = []
        sortarr = np.append(sortarr, idx)
        for j in range(k):
            sortarr = np.append(
                sortarr, [indexarray[t_val-1, int(idx[j]), :5]])
        sortarr = np.setdiff1d(sortarr, [i])  # 去除重复元素和本身
        arr = []
        origin = []
        origin = np.append(
            origin, [[result[t_val][i][x_idx], result[t_val][i][y_idx], result[t_val][i][z_idx]]])
        for element in sortarr:
            arr = np.append(arr, [[result[t_val][int(element)][x_idx], result[t_val][int(
                element)][y_idx], result[t_val][int(element)][z_idx]]])
        arr = np.reshape(arr, (-1, 3), order='C')
        length = arr.shape[0]
        distancearr = np.sqrt(np.sum((origin - arr)**2, axis=1))
        secondidx = np.argsort(distancearr, axis=-1, kind="quicksort")
        # t_stamp,num_vertex,k
        weightarray[t_val, i, :] = distancearr[secondidx[:k]]
        # t_stamp,num_vertex,k
        indexarray[t_val, i, :] = sortarr[secondidx[:k]]
        print(f'{t_val} and {i}')

# reshape 成二维数组用savetxt存储
indexarray_reshaped = np.reshape(indexarray, (time_stamp, num_vertex*k))
np.savetxt("csv_time_point.csv", indexarray_reshaped, fmt="%d", delimiter=",")

# 目标问题1：596个原子之间的两两稳定关系（共价键检测）
# 目标问题2：稳定基团的认知（多原子稳定基团）
# 目标问题3：基团的频谱系
