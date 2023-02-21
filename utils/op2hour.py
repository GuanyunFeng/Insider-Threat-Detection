import numpy as np

data = np.load("../processed_data2/data.npy",allow_pickle=True)
print(data.shape)
u_num, t_num = data.shape[0], data.shape[1]

hour_data = [[] for _ in range(u_num)]

for u in range(u_num):
    for t in range(t_num):
        v = np.sum(data[u][t], axis=0)
        hour_data[u].append(v)

np.save("../processed_data2/data_hour.npy", hour_data)