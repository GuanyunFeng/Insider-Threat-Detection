import numpy as np
from PIL import Image

data = np.load("../../processed_data/data_day.npy",allow_pickle=True)

print(data.shape)
u_num, t_num = data.shape[0], data.shape[1]

u=989

mean = np.mean(data[u],axis = 0)
std = np.std(data[u], axis = 0)

data_fig = np.zeros((3, t_num))
label_fig = np.zeros(t_num)

for t in range(t_num):
    v = data[u][t]
    if not v.ndim:
        data_fig[0][t] = 0
        data_fig[1][t] = 0
        data_fig[2][t] = 0
        label_fig[t] = 0
    else:
        label = 1 if v[-1] > 0 else 0
        label_fig[t] = label
        #特征归一化
        v = (v-mean)/(std+1e-10)

        data_fig[0][t] = v[0]
        data_fig[1][t] = v[6]
        data_fig[2][t] = v[31]


data_fig = np.transpose(data_fig)
max_pix = np.max(data_fig, axis=0)
min_pix = np.min(data_fig, axis=0)
data_fig = (data_fig-min_pix)/(max_pix-min_pix+1e-10)*255
data_fig = 255 - data_fig

data_fig = np.resize(data_fig,(1, t_num, 3))

data_img = Image.fromarray(data_fig.astype("uint8"))
#data_img = data_img.resize((100, t_num))
data_img.save("data.jpg")

label_fig = np.resize(label_fig, (1,t_num))
label_fig = 255*(1-label_fig)
label_img = Image.fromarray(label_fig.astype("uint8"))
#label_img = label_img.resize((100, t_num))
label_img.save("label.jpg")
