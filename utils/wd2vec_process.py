from gensim.models import Word2Vec
from readline import insert_text
import pandas as pd
import os
import numpy as np
import re

#返回所有内鬼操作编码
def get_insiders():
    insiders = []
    ins_actions = []
    for root,_,files in os.walk("../insiders"): 
        for file in files: 
            file_path = os.path.join(root,file)
            #print(file[7:-4])
            insiders.append(file[7:-4])
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




#返回 员工编码-电脑编码 的键值对
def get_user_pc(users):
    #user_pc表示 员工-个人电脑 编码的键值对
    #shared_pc表示 共享电脑 列表
    user_pc = {}
    df = pd.read_csv('../data/logon.csv')
    shared_pc = df["pc"].drop_duplicates().values.tolist()

    tmp_dict = {}
    for u in users:
        tmp_dict[u] = {}

    #记录员工-电脑的登录次数
    for idex, row in df.iterrows():
        cur_u = row["user"]
        cur_pc = row["pc"]
        if cur_pc not in tmp_dict[cur_u]:
            tmp_dict[cur_u][cur_pc] = 1
        else:
            tmp_dict[cur_u][cur_pc] += 1

    #取使用次数最多的为员工的个人电脑
    for u in users:
        for k,v in tmp_dict[u].items():
            if v == max(tmp_dict[u].values()):
                user_pc[u] = k
                #从共享电脑中删除
                shared_pc.remove(k)

    return user_pc,shared_pc




def get_time_embedding():
    m_dict = [31,28,31,30,31,30,31,31,30,31,30,31]
    count = 0
    time_embedding = {}
    for y in [0,1]:
        for m in range(1,13):
            ms = str(m)
            if m < 10:
                ms = "0"+ms        
            for d in range(1,m_dict[m-1]+1):
                ds = str(d)
                if d < 10:
                    ds = "0"+ds
                for h in range(24):
                    hs = str(h)
                    if h < 10:
                        hs = "0"+ hs
                    time_str = "201{}/{}/{} {}".format(str(y), ms, ds, hs)
                    time_embedding[time_str] = count
                    count += 1
    return time_embedding


if __name__ == "__main__":
    users = get_users()
    user_id = {}
    id = 0
    for u in users:
        user_id[u] = id
        id += 1
    insiders,ins_actions  = get_insiders()
    insider_id = []
    for u in users:
        if u in insiders:
            insider_id.append(user_id[u])
    print(insider_id)
    user_pc, shared_pc = get_user_pc(users)
    time_embedding = get_time_embedding()

    actions = [[] for x in range(len(users))]

    for table_name in ["email","device","email", "file", "http", "logon"]:
        # 读取本地CSV文件
        df = pd.read_csv('../data/{}.csv'.format(table_name))
        df["date"] = df["date"].apply(lambda x:x[6:10]+"/"+x[:5]+x[10:])

        #0-5
        if table_name == "device":
            for index, row in df.iterrows():
                word = table_name

                u = row["user"]
                uid = user_id[u]

                word += "_"
                word += row["activity"]
                word += "_"
                #记录使用pc的归属
                if row["pc"] == user_pc[u]:
                    word += "1"
                elif row["pc"] in shared_pc:
                    word += "2"
                else:
                    word += "3"

                actions[uid].append((word, row["date"]))
        
        #登录特征 6-11
        elif table_name == "logon":
            #特征6-11
            for index, row in df.iterrows():
                word = table_name

                u = row["user"]
                uid = user_id[u]

                word += "_"
                word += row["activity"]
                word += "_"
                #记录使用pc的归属
                if row["pc"] == user_pc[u]:
                    word += "1"
                elif row["pc"] in shared_pc:
                    word += "2"
                else:
                    word += "3"

                actions[uid].append((word, row["date"]))

        #邮件特征 12-14
        elif table_name == "email":
            for index, row in df.iterrows():
                word = table_name

                u = row["user"]
                uid = user_id[u]

                word += "_"
                word += row["to"][-7:]
                word += "_"
                #记录使用pc的归属
                if row["pc"] == user_pc[u]:
                    word += "1"
                elif row["pc"] in shared_pc:
                    word += "2"
                else:
                    word += "3"

                actions[uid].append((word, row["date"]))
        
        
        #文件访问特征15-29
        elif table_name == "file":
            for index, row in df.iterrows():
                word = table_name

                u = row["user"]
                uid = user_id[u]

                word += "_"
                word += row["filename"][-3:]
                word += "_"
                #记录使用pc的归属
                if row["pc"] == user_pc[u]:
                    word += "1"
                elif row["pc"] in shared_pc:
                    word += "2"
                else:
                    word += "3"

                actions[uid].append((word, row["date"]))
        

        #http特征 30-
        elif table_name == "http":
            for index, row in df.iterrows():
                word = table_name

                u = row["user"]
                uid = user_id[u]


                actions[uid].append((word, row["date"]))
                
                domainname = re.findall("//(.*?)/", row["url"])[0]
                domainname.replace("www.","")
                dn = domainname.split(".")
                if len(dn) > 2 and not any([x in domainname for x in ["google.com", '.co.uk', '.co.nz', 'live.com']]):
                    domainname = ".".join(dn[-2:])
                # other 1, socnet 2, cloud 3, job 4, leak 5, hack 6
                word += "_"
                word += domainname
                word += "_"
                #记录使用pc的归属
                if row["pc"] == user_pc[u]:
                    word += "1"
                elif row["pc"] in shared_pc:
                    word += "2"
                else:
                    word += "3"

                actions[uid].append((word, row["date"]))


    
    f = open("log.txt", "w")
    for u in range(len(users)):
        tmp = actions[u]
        tmp = sorted(tmp, key=lambda x:x[1])
        tmp = [x[0] for x in tmp]
        line = []
        for t in tmp:
            line.append(t)
            if "device_logoff" in t:
                f.writeline(" ".join(line))
                line = []
        actions[u] = tmp
    f.close()
    

model = Word2Vec(LineSentence(inp), size=100, window=10, min_count=3, workers=multiprocessing.cpu_count(), sg=1, iter=10,negative=20)

model.save("wd2vec.model")