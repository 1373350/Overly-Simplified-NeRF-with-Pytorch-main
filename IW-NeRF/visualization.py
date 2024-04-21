import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import torch
from lib.nerf.model import BigModel
import matplotlib.pyplot as plt
import numpy as np

# 实例化模型
model = BigModel(hidden_dim=256)
model_weight_path = "./models/big.pth"
model.load_state_dict(torch.load(model_weight_path))

weights_keys = model.state_dict().keys()  # 获取训练参数字典里面keys
for key in weights_keys:
    # remove num_batches_tracked para(in bn)
    if "num_batches_tracked" in key:  # bn层也有参数
        continue

    # [卷积核个数,卷积核的深度, 卷积核 h,卷积核 w]
    weight_value = model.state_dict()[key].numpy()  # 返回 key 里面具体的值

    # mean, std, min, max
    weight_mean = weight_value.mean()
    weight_std = weight_value.std()
    weight_min = weight_value.min()
    weight_max = weight_value.max()
    print("{} layer:mean:{}, std:{}, min:{}, max:{}".format(key, weight_mean, weight_std, weight_min, weight_max))

    # 绘制参数的直方图
    plt.close()
    weight_vec = np.reshape(weight_value, [-1])
    plt.hist(weight_vec, bins=50)  # 将 min-max分成50份
    plt.title(key)
    plt.show()