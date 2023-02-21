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

    actions = [[[] for _ in range((365+180)*24)] for x in range(len(users))]

    for table_name in ["email","device","email", "file", "http", "logon"]:
        # 读取本地CSV文件
        df = pd.read_csv('../data/{}.csv'.format(table_name))
        df["date"] = df["date"].apply(lambda x:x[6:10]+"/"+x[:5]+x[10:])

        #0-5
        if table_name == "device":
            for index, row in df.iterrows():
                v = np.zeros(50)
                tid = time_embedding[row["date"][:13]]
                min = int(row["date"][14:16])
                sec = int(row["date"][17:19])
                v[-2] = min*60+sec
                u = row["user"]
                uid = user_id[u]

                w = 0
                #记录使用pc的归属
                if row["pc"] == user_pc[u]:
                    w1 = 0
                elif row["pc"] in shared_pc:
                    w1 = 1
                else:
                    w1 = 2

                #记录
                if row["activity"] == "Connect":
                    v[w1*2] = 1
                else:
                    v[w1*2+1] = 1
                if row["id"] in ins_actions:
                    v[-1] = 1
                
                actions[uid][tid].append(v)
        
        #登录特征 6-11
        elif table_name == "logon":
            #特征6-11
            for index, row in df.iterrows():
                v = np.zeros(50)
                tid = time_embedding[row["date"][:13]]
                min = int(row["date"][14:16])
                sec = int(row["date"][17:19])
                v[-2] = min*60+sec
                u = row["user"]
                uid = user_id[u]

                w1 = 0
                #记录使用pc的归属
                if row["pc"] == user_pc[u]:
                    w1 = 0
                elif row["pc"] in shared_pc:
                    w1 = 1
                else:
                    w1 = 2

                if row["activity"] == "Logon":
                    v[6+w1*2] = 1
                else:
                    v[6+w1*2+1] = 1
                if row["id"] in ins_actions:
                    v[-1] = 1
                
                actions[uid][tid].append(v)

        #邮件特征 12-14
        elif table_name == "email":
            for index, row in df.iterrows():
                v = np.zeros(50)
                tid = time_embedding[row["date"][:13]]
                min = int(row["date"][14:16])
                sec = int(row["date"][17:19])
                v[-2] = min*60+sec
                u = row["user"]
                uid = user_id[u]

                w1 = 0
                #记录使用pc的归属
                if row["pc"] == user_pc[u]:
                    w1 = 0
                elif row["pc"] in shared_pc:
                    w1 = 1
                else:
                    w1 = 2

                v[12+w1] = 1
                
                if row["id"] in ins_actions:
                    v[-1] = 1
                
                actions[uid][tid].append(v)
        
        
        #文件访问特征15-29
        elif table_name == "file":
            for index, row in df.iterrows():
                v = np.zeros(50)
                tid = time_embedding[row["date"][:13]]
                min = int(row["date"][14:16])
                sec = int(row["date"][17:19])
                v[-2] = min*60+sec
                u = row["user"]
                uid = user_id[u]

                w1 = 0
                #记录使用pc的归属
                if row["pc"] == user_pc[u]:
                    w1 = 0
                elif row["pc"] in shared_pc:
                    w1 = 1
                else:
                    w1 = 2


                if ".doc" in row["filename"]:
                    v[15+w1*5] = 1
                elif ".pdf" in row["filename"]:
                    v[15+w1*5+1] = 1
                elif ".txt" in row["filename"]:
                    v[15+w1*5+2] = 1
                elif ".jpg" in row["filename"]:
                    v[15+w1*5+3] = 1
                elif ".zip" in row["filename"]:
                    v[15+w1*5+4] = 1
                if row["id"] in ins_actions:
                    v[-1] += 1
                
                actions[uid][tid].append(v)
        

        #http特征 30-
        elif table_name == "http":
            for index, row in df.iterrows():
                v = np.zeros(50)
                tid = time_embedding[row["date"][:13]]
                min = int(row["date"][14:16])
                sec = int(row["date"][17:19])
                v[-2] = min*60+sec
                u = row["user"]
                uid = user_id[u]

                w1 = 0
                #记录使用pc的归属
                if row["pc"] == user_pc[u]:
                    w1 = 0
                elif row["pc"] in shared_pc:
                    w1 = 1
                else:
                    w1 = 2
                
                domainname = re.findall("//(.*?)/", row["url"])[0]
                domainname.replace("www.","")
                dn = domainname.split(".")
                if len(dn) > 2 and not any([x in domainname for x in ["google.com", '.co.uk', '.co.nz', 'live.com']]):
                    domainname = ".".join(dn[-2:])
                # other 1, socnet 2, cloud 3, job 4, leak 5, hack 6
                if domainname in ['dropbox.com', 'drive.google.com', 'mega.co.nz', 'account.live.com']:
                    v[30+w1*6] = 1
                elif domainname in ['wikileaks.org','freedom.press','theintercept.com']:
                    v[30+w1*6+1] = 1
                elif domainname in ['facebook.com','twitter.com','plus.google.com','instagr.am','instagram.com',
                                    'flickr.com','linkedin.com','reddit.com','about.com','youtube.com','pinterest.com',
                                    'tumblr.com','quora.com','vine.co','match.com','t.co']:
                    v[30+w1*6+2] = 1
                elif (domainname in ['indeed.com','monster.com', 'careerbuilder.com','simplyhired.com']) \
                    or ('job' in domainname and ('hunt' in domainname or 'search' in domainname)) \
                    or ('aol.com' in domainname and ("recruit" in row["url"] or "job" in row["url"])):
                    v[30+w1*6+3] = 1
                elif (domainname in ['webwatchernow.com','actionalert.com', 'relytec.com','refog.com','wellresearchedreviews.com',
                                    'softactivity.com', 'spectorsoft.com','best-spy-soft.com']) or ('keylog' in domainname):
                    v[30+w1*6+4] = 1
                else:
                    v[30+w1*6+5] = 1
                if row["id"] in ins_actions:
                    v[-1] += 1
                
                actions[uid][tid].append(v)


    for u in range(len(users)):
        for t in range((365+180)*24):
            tmp = actions[u][t]
            tmp = sorted(tmp, key=lambda x:x[-2])
            actions[u][t] = tmp
    
    np.save("../processed_data2/data.npy",np.array(actions))