import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pickle
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image
from PIL import Image
from lib.nerf.model import SmallModel, BigModel
with open('key.pkl', 'rb') as f:
    key = pickle.load(f)

# 生成一个长度为128的随机数组，其中包含64个0和64个1
#np.concatenate是numpy库中用于数组拼接的函数。它可以按照指定的轴将多个数组拼接在一起，生成一个新的数组
#numpy.zeros(shape, dtype=float)
"""key_1 = np.concatenate((np.zeros(128), np.ones(128)))
key_2 = np.concatenate((np.zeros(128), np.ones(128)))
key_3 = np.concatenate((np.zeros(128), np.ones(128)))
key_4 = np.concatenate((np.zeros(128), np.ones(128)))

key_5 = np.concatenate((np.zeros(128), np.ones(128)))
key_6 = np.concatenate((np.zeros(128), np.ones(128)))
key_7 = np.concatenate((np.zeros(128), np.ones(128)))
key_8 = np.concatenate((np.zeros(128), np.ones(129)))

key_9 = np.concatenate((np.zeros(64), np.ones(64)))
key_10 = np.ones(3)"""
#print(key_1,key_2,key_3,key_4,key_5,key_6,key_7,key_8,key_9,key_10)
# 将数组打乱顺序
#np.random.shuffle(key[8])
"""np.random.shuffle(key_2)
np.random.shuffle(key_3)
np.random.shuffle(key_4)
np.random.shuffle(key_5)
np.random.shuffle(key_6)
np.random.shuffle(key_7)
np.random.shuffle(key_8)
np.random.shuffle(key_9)
np.random.shuffle(key_10)"""
m=0
n=0
for i in range (len(key[1])):
    #print(key[8][i])
    if key[1][i]==0:
        if m<=99:
            key[1][i]=1
            m=m+1
    else:
       if n<=99:
        key[1][i]=0
        n=n+1

print(m)
print(n)
print(key[1])
print(len(key[1]))

print(len(key))
# 保存数组列表到文件中
# #使用 with open() as f 语句可以实现文件的打开和关闭操作，这样可以避免忘记关闭文件的情况发生。
with open('key1.pkl', 'wb') as f:
    pickle.dump(key, f)