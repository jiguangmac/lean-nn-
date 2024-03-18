# 数据预处理
[预处理文件](dataporcess.py)
1. 根据数据的格式读入数据，一共有1001个时间戳，每个时间戳都有596个数据
2. 随后根据第一组数据的文件进行初始化，对每一个顶点计算与其他顶点之间的距离，并且进行排序，拿出距离它最近的10个顶点的编号。（这里的编号采用了0-595）
3. 设计算法来更新数据，利用上一时刻的数据来更新
    例如现在有第0时刻的top10[596,10]的数据，同时还有索引
    这时可以利用索引将数据取出，然后再根据索引的索引每个数据引入5个新的值
    随后去除重复值然后排序
4. 最后将所有生成的数据（1001，596，10）其中1001表示时间戳，596表示原子数，10表示距离该原子最近的十个原子的编号，输出到[csv](./csv_time_point.csv)文件中，随后进行训练时直接从csv文件中读入原子之间的关系数据
# 数据导入，模型设计，训练
[训练文件](train.py)
在train.py中导入了数据，利用了torch框架中的dataset和dataloader输入了数据。
