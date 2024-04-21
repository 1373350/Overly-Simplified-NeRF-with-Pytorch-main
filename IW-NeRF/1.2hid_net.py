import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pickle
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image
from PIL import Image
from lib.nerf.model import SmallModel, BigModel


# 生成一个长度为128的随机数组，其中包含64个0和64个1
#np.concatenate是numpy库中用于数组拼接的函数。它可以按照指定的轴将多个数组拼接在一起，生成一个新的数组
#numpy.zeros(shape, dtype=float)
key_1 = np.concatenate((np.zeros(128), np.ones(128)))
key_2 = np.concatenate((np.zeros(128), np.ones(128)))
key_3 = np.concatenate((np.zeros(128), np.ones(128)))
key_4 = np.concatenate((np.zeros(128), np.ones(128)))

key_5 = np.concatenate((np.zeros(128), np.ones(128)))
key_6 = np.concatenate((np.zeros(128), np.ones(128)))
key_7 = np.concatenate((np.zeros(128), np.ones(128)))
key_8 = np.concatenate((np.zeros(128), np.ones(129)))

key_9 = np.concatenate((np.zeros(64), np.ones(64)))
key_10 = np.ones(3)
#print(key_1,key_2,key_3,key_4,key_5,key_6,key_7,key_8,key_9,key_10)
# 将数组打乱顺序
np.random.shuffle(key_1)
np.random.shuffle(key_2)
np.random.shuffle(key_3)
np.random.shuffle(key_4)
np.random.shuffle(key_5)
np.random.shuffle(key_6)
np.random.shuffle(key_7)
np.random.shuffle(key_8)
np.random.shuffle(key_9)
np.random.shuffle(key_10)


key = [key_1, key_2, key_3, key_4, key_5, key_6, key_7, key_8, key_9, key_10]
#print(key)
# 保存数组列表到文件中
# #使用 with open() as f 语句可以实现文件的打开和关闭操作，这样可以避免忘记关闭文件的情况发生。
with open('key.pkl', 'wb') as f:
    pickle.dump(key, f)
#序列化对象，将对象obj保存到文件file中去。参数protocol是序列化模式，默认是0（ASCII协议，表示以文本的形式进行序列化），
# protocol的值还可以是1和2（1和2表示以二进制的形式进行序列化。其中，1是老式的二进制协议；2是新二进制协议）。
# file表示保存到的类文件对象，file必须有write()接口，file可以是一个以'w'打开的文件或者是一个StringIO对象，
# 也可以是任何可以实现write()接口的对象。

# 创建模型实例
model = SmallModel(hidden_dim=128).to('cuda')
model.load_state_dict(torch.load('./models/secret.pth'))

big_model = BigModel(hidden_dim=256).to('cuda')

# 传导参数，把小模型的参数传导到大模型
#named_parameters()返回的list中，每个元组（与list相似，只是数据不可修改）打包了2个内容，分别是layer-name和layer-param（网络层的名字和参数的迭代器）；
count_key = 0
for new_name, new_param in big_model.named_parameters():#大模型参数
    for name, param in model.named_parameters():#小模型参数
        if new_name == name:#如果大模型网络名称和小模型网络层名称相同
            print(name)
            if 'weight' in name:
                for i in range(new_param.shape[0]):#遍历大模型的参数
                    if key[count_key][i] == 1:  #1为秘密参数
                        if new_param[i].shape[0] == param[0].shape[0]:#如果大模型和小模型参数维度相同，大模型复制小模型参数值
                            new_param[i].data.copy_(param[0])
                        else:
                            new_param[i].data.copy_(torch.cat((param[0], param[0]))[:new_param.shape[1]])
                        param = param[1:, :]#去掉第一个参数

            elif 'bias' in name:
                for i in range(new_param.shape[0]):
                    if key[count_key][i] == 1:
                        new_param[i].data.copy_(param[0])
                        param = param[1:]

                count_key = count_key + 1

# 保存模型
torch.save(big_model.state_dict(), './models/big.pth')
print("Finished")
