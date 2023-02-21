import numpy as np

data = np.load("../processed_data2/data_hour.npy",allow_pickle=True)
print(data.shape)
u_num, t_num = data.shape[0], data.shape[1]

day_data = [[] for _ in range(u_num)]

for u in range(u_num):
    for t in range(0, t_num, 24):
        v = np.sum(data[u][t:t+24], axis=0)
        day_data[u].append(v)

np.save("../processed_data2/data_day.npy", day_data)