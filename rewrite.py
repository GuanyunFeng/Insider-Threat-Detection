from re import X
import torch.utils.data as data
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms
import numpy as np
from models.FCN_LSTM import LSTM_FCN
from models.Transformer import BERT
from models.SwinTrans import swin_cert
from models.vit import SimpleViT


data = np.load("../processed_data/data_{}.npy".format("day"),allow_pickle=True)
print(data.shape)

u_num, t_num = data.shape[0], data.shape[1]

#按小时进行合并
data_new = [[] for u in range(u_num)]
for u in range(u_num):
    for t in range(t_num):
        v = data[u][t]
        if not v.ndim:
            v = np.zeros(50)
        data_new[u].append(v)

data_new = np.array(data_new)
print(data_new.shape)

np.save("../processed_data/data_{}_new.npy".format("day"),data_new)