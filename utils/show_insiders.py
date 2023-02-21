# -*- coding:utf-8 -*-
import os
import numpy as np
import pandas as pd

#返回所有内鬼操作编码
def get_insiders():
    insiders = [[], [], []]
    ins_actions = []
    for root,_,files in os.walk("../insiders"): 
        for file in files: 
            file_path = os.path.join(root,file)
            #print(file[7:-4])
            insiders[int(file[5])-1].append(file[7:-4])
            with open(file_path,"r") as f:
                for line in f.readlines():
                    #print(line)
                    ins_actions.append(line.split(",")[1])
    return insiders, ins_actions




#返回所有员工编码
def get_users():
    df = pd.read_csv('../data/logon.csv')
    users = df["user"].drop_duplicates().values.tolist()
    return users

users = get_users()
user_id = {}
id = 0
for u in users:
    user_id[u] = id
    id += 1
insiders,ins_actions  = get_insiders()
for i in range(3):
    insider_id = []
    for u in insiders[i]:
        print(u, user_id[u])
        insider_id.append(user_id[u])
    insider_id.sort()
    print(insider_id)