import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pickle
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image
from PIL import Image
from lib.nerf.model import SmallModel, BigModel

# 保存数组列表到文件中
with open('key1.pkl', 'rb') as f:
    key = pickle.load(f)

# 创建模型实例
model = SmallModel(hidden_dim=128).to('cuda')

big_model = BigModel(hidden_dim=256).to('cuda')
big_model.load_state_dict(torch.load('./models/big.pth'))

# 提取参数
count_key = 0
for new_name, new_param in big_model.named_parameters():
    for name, param in model.named_parameters():
        if new_name == name:
            if 'weight' in name:
                count_param = 0
                for i in range(new_param.shape[0]):
                    if key[count_key][i] == 1:
                        if new_param[0].shape[0] == param[0].shape[0]:

                            param[count_param].data.copy_(new_param[i])
                        else:

                            param[count_param].data.copy_(new_param[i, :param.shape[1]])

                        count_param = count_param + 1

            elif 'bias' in name:
                count_param = 0
                for i in range(new_param.shape[0]):
                    if key[count_key][i] == 1:
                        param[count_param].data.copy_(new_param[i])
                        count_param = count_param + 1
                count_key = count_key + 1
# 保存模型
torch.save(model.state_dict(), './models/recover.pth')
