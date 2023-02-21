import numpy as np

data = np.load("../processed_data2/data_day.npy",allow_pickle=True)
print(data.shape)
u_num, t_num = data.shape[0], data.shape[1]

day_month = [[] for _ in range(u_num)]

for u in range(u_num):
    for t in range(0, t_num, 30):
        v = np.sum(data[u][t:t+30], axis=0)
        day_month[u].append(v)

np.save("../processed_data2/data_month.npy", day_month)