import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np


class TimePointDataset(Dataset):
    '''dataset 每一个返回一个timestamp的数据，包括596个原子邻近的10个点'''

    def __init__(self, filepath="csv_time_point.csv", length=0):

        print(f'reading {filepath}')

        with open(filepath, encoding='utf-8') as f:
            lines = f.readlines()
        feat = [[] for _ in range(length)]

        for i, line in enumerate(lines):
            values = line.strip().split(',')  # 5960
            feat[i].append(values)
        feat = np.array(feat, dtype=np.int32)
        feat_reshaped = np.reshape(feat, (1001, 596, 10))

        self.x = torch.from_numpy(feat_reshaped)  # 1001,596,10

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], index


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        pass

    def forward(self, x):
        pass


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("using {} device".format(device))

    time_stamp = 1001
    time_point_dataset = TimePointDataset(length=time_stamp)

    time_point_dataloader = DataLoader(time_point_dataset)

    feature, time_val = next(iter(time_point_dataloader))

    print("feature")
    print("time_val")
    # for idx, batch_x in enumerate(time_point_dataloader):
    #     print(f"batch_id:{idx},shape {batch_x.shape}")
    #     print(batch_x)
