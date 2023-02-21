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
from models.vit import SimpleViT
from models.RCNN import RCNN, ConfigR
from models.TextCNN import TextCNN, ConfigT
from models.AnomalyTransformer import AnomalyTransformer
import torch.nn.functional as F
import random

time_step = 28

torch.cuda.set_device(0)
feater_weight = 10 * np.array([0.12173799, 0.02781476, 0.        , 0.        , 0.        , 0.
, 0.02672204, 0.03743288, 0.02379778, 0.        , 0.01728155, 0.
, 0.11617799, 0.        , 0.        , 0.09250731, 0.04452063, 0.02682699
, 0.01925607, 0.00255155, 0.        , 0.        , 0.        , 0.
, 0.        , 0.        , 0.        , 0.        , 0.        , 0.
, 0.00652585, 0.21970679, 0.02090572, 0.06559526, 0.07930623, 0.05133259
, 0.        , 0.        , 0.        , 0.        , 0.        , 0.
, 0.        , 0.        , 0.        , 0.        , 0.        , 0.        ])

class MyDataset(data.Dataset):
    def __init__(self, x, y):
        self.x = torch.from_numpy(x).float()
        self.y = torch.from_numpy(y).long()

    def __getitem__(self, index):#返回的是tensor
        input, output = self.x[index], self.y[index]
        return input, output

    def __len__(self):
        return len(self.x)

def load42(option = "day", time_step = 32):
    data = np.load("../processed_data/data_{}.npy".format(option),allow_pickle=True)
    print(data.shape)

    u_num, t_num = data.shape[0], data.shape[1]
    #u_num = 1000

    train_ratio = 0.8
    test_ratio = 0.5

    #按小时进行合并
    x_data, y_data, w_data = [], [], []
    for u in range(u_num):
        #用户特征归一化
        mean = np.mean(data[u],axis = 0)
        std = np.std(data[u], axis = 0)

        #min-max 归一化
        #mean = np.min(data[u],axis = 0)
        #std = np.max(data[u], axis = 0)

        u_data = []
        labels = []

        for t in range(t_num):
            v = data[u][t]
            if not v.ndim:
                #v = np.zeros(50)
                continue
            
            label = 1 if v[-1] > 0 else 0
            #特征归一化
            v = (v-mean)/(std+1e-10)
            #乘以
            u_data.append(v[:-2]*feater_weight)
            labels.append(label)
        
        for i in range(0, len(u_data)-time_step):
            x = np.array(u_data[i:i+time_step])
            y = max(labels[i:i+time_step])
            x_data.append(x)
            y_data.append(y)
            if y == 0:
                w_data.append(0)
            else:
                for j in range(4):
                    flag = max(labels[i+j*7:i+(j+1)*7])
                    if flag != 0:
                        w_data.append(j)
                        break
    
    x_data, y_data, w_data = np.array(x_data), np.array(y_data), np.array(w_data)
    return x_data, y_data, w_data

def augment(x_data, y_data, w_data):
    data_len = x_data.shape[0]
    new_x, new_y = [], []
    for i in range(data_len):
        old_x = x_data[i]
        old_y = y_data[i]
        if old_y == 0:
            continue
        #按大小进行随机缩放
        #scale = np.random.random()+0.5#区间0.5-1.5
        #new_x.append(old_x*scale)
        #new_y.append(old_y)
        #随机交换
        week1 = random.randint(0, 4)
        week2 = (random.randint(0, 3)+week1)%4
        old_x_copy = old_x.copy()
        old_x_copy[:,week2*7:week2*7+7] = old_x[:,week1*7:week1*7+7]
        old_x_copy[:,week1*7:week1*7+7] = old_x[:,week2*7:week2*7+7]
        new_x.append(old_x_copy)
        new_y.append(old_y)
        #随机复制
        week1 = w_data[i]
        week2 = (random.randint(0, 3)+week1)%4
        old_x_copy = old_x.copy()
        old_x_copy[:,week2*7:week2*7+7] = old_x[:,week1*7:week1*7+7]
        new_x.append(old_x_copy)
        new_y.append(old_y)
        '''
        for j in range(4):
            for k in range(j+1, 4):
                old_x_copy = old_x.copy()
                old_x_copy[:,k*7:k*7+7] = old_x[:,j*7:j*7+7]
                old_x_copy[:,j*7:j*7+7] = old_x[:,k*7:k*7+7]
                new_x.append(old_x_copy)
                new_y.append(old_y)
        #复制
        week1 = w_data[i]
        for j in range(3):
            week2 = (j+week1)%4
            old_x_copy = old_x.copy()
            old_x_copy[:,week2*7:week2*7+7] = old_x[:,week1*7:week1*7+7]
            new_x.append(old_x_copy)
            new_y.append(old_y)
        '''
    new_x, new_y = np.array(new_x), np.array(new_y)
    x_data, y_data = np.concatenate([x_data, new_x],axis=0), np.concatenate([y_data, new_y],axis=0)
    return x_data, y_data

if __name__=="__main__":
    recalls, precs = [], []
    i=7
    batch_size = 64
    x_data, y_data, w_data = load42(time_step=28)
    data_len = x_data.shape[0]

    if i == 0:
        x_train = x_data[int(0.1*data_len):]
        y_train = y_data[int(0.1*data_len):]
        w_train = w_data[int(0.1*data_len):]
    elif i == 9:
        x_train = x_data[:int(0.9*data_len)]
        y_train = y_data[:int(0.9*data_len)]
        w_train = w_data[:int(0.9*data_len)]
    else:
        x_train = np.concatenate([x_data[:int(0.1*i*data_len)],x_data[int(0.1*(i+1)*data_len):]],axis=0)
        y_train = np.concatenate([y_data[:int(0.1*i*data_len)],y_data[int(0.1*(i+1)*data_len):]],axis=0)
        w_train = np.concatenate([w_data[:int(0.1*i*data_len)],w_data[int(0.1*(i+1)*data_len):]],axis=0)
    x_test = x_data[int(0.1*i*data_len):int(0.1*(i+1)*data_len)]
    y_test = y_data[int(0.1*i*data_len):int(0.1*(i+1)*data_len)]

    #数据增强
    aug_x, aug_y = augment(x_train, y_train, w_train)

    train_loader = torch.utils.data.DataLoader(
        MyDataset(x_train, y_train), batch_size=batch_size, shuffle=True, drop_last = True)
    test_loader = torch.utils.data.DataLoader(
        MyDataset(x_test, y_test), batch_size=batch_size, shuffle=True, drop_last = True)

    att_net = AnomalyTransformer(win_size=32, enc_in=48, c_out=48, e_layers=3)
    att_net.load_state_dict(torch.load('checkpoints/4_checkpoint.pth'))
    att_net.eval()
    att_net.to('cuda')
    att_criterion = nn.MSELoss(reduce=False)

    net = LSTM_FCN(time_step, 48, 128, 2)
    #net = SimpleViT(time_step, 28, 2)
    #net = BERT()
    #Conf = ConfigR()
    #net = RCNN(Conf)
    #Conf = ConfigT()
    #net = TextCNN(Conf)

    net = net.to('cuda')
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.0001, weight_decay=0.001)

    att = torch.from_numpy(feater_weight).float().to('cuda')
    #训练
    for _ in range(10):
        for i, (x, y) in enumerate(train_loader):
            # forward
            x, y = x.to('cuda'), y.to('cuda')
            
            #x_out, _, _, _ = att_net(x)
            #diff = att_criterion(x, x_out)
            #att = torch.mean(diff.view(batch_size*time_step, -1),dim=0)
            #scale = torch.max(att)
            #att = F.softmax(att/scale*10, dim=0)
            #确定最小异常的值
            #min_ano = torch.min(att)

            #ones = torch.ones_like(att)
            #zeros = torch.zeros_like(att)
            #att = torch.where(att>min_ano*10, ones, zeros)
            #print(att)

            
            #x = torch.mul(x, att)
            #print(x.shape)
            out = net(x)
            loss = criterion(out, y)
            
            #对filter进行l1正则化
            #all_linear1_params = torch.cat([x.view(-1) for x in net.filter1.parameters()])
            #l1_regularization1 = 0.1 * torch.norm(all_linear1_params, 1)

            #loss += l1_regularization1
            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    #测试
    fprs, tprs = [], []
    for k in range(10):
        print("start eval")
        TP, FP, TN, FN = 0, 0, 0, 0
        net.eval()
        for i, (x, y) in enumerate(test_loader):
            # forward
            x, y = x.to('cuda'), y.to('cuda')
            zero = torch.zeros_like(y)
            one = torch.ones_like(y)

            out = net(x)
            #pre = torch.where(out[:,0]>k*0.1, zero, one)
            pre = torch.argmax(out, 1)
            
            FN += ((pre==zero)&(y==one)).sum().cpu().numpy()
            TN += ((pre==zero)&(y==zero)).sum().cpu().numpy()
            TP += ((pre==one)&(y==one)).sum().cpu().numpy()
            FP += ((pre==one)&(y==zero)).sum().cpu().numpy()

        print(TP, FP, TN, FN)
        net.train()
        recall = TP/(TP+FN)
        precision = TP/(TP+FP)
        f1 = 2*recall*precision/(recall+precision)
        fpr = FP/(TN+FP)
        tpr = TP/(TP+FN)
        recalls.append(recall)
        precs.append(precision)
        fprs.append(fpr)
        tprs.append(tpr)
        print("recall:{}, precision:{}, f1:{}".format(recall, precision, f1))
        print("end eval")
    
    print(fprs, tprs)


    '''
    print("recall")
    print(recalls)
    sum = 0
    for r in recalls:
        sum += r
    print(sum/10.)

    print("prec")
    print(precs)
    sum = 0
    for r in precs:
        sum += r
    print(sum/10.)
    
    print("fpr")
    print(fprs)
    sum = 0
    for r in fprs:
        sum += r
    print(sum/10.)
    '''