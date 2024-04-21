import torch.nn as nn
import numpy as np
import math
from copy import deepcopy
import torch
from matplotlib import pyplot as plt
import torch.nn.utils.prune as prune
from lib.nerf.model import SmallModel, BigModel
from lib.nerf.utils import get_rays, render_rays, pose_spherical, compute_accumulated_transmittance
import pickle
from torch import nn
import torch.nn.utils.prune as prune
import torch.nn.functional as F

model = BigModel(hidden_dim=256)
model_weight_path = "./models/big.pth"
model.load_state_dict(torch.load(model_weight_path))





parameters_to_prune = (
    (model.block1_1[0], 'weight'),
    (model.block1_2[0], 'weight'),
    (model.block1_3[0], 'weight'),
    (model.block1_5[0], 'weight'),
    (model.block1_6[0], 'weight'),
    (model.block1_7[0], 'weight'),
(model.block1_8[0], 'weight'),
(model.block2_1[0], 'weight'),
(model.block2_2[0], 'weight'),
(model.block2_3[0], 'weight'),
(model.block2_4[0], 'weight'),
(model.block2_5[0], 'weight'),
(model.block2_6[0], 'weight'),
(model.block2_7[0], 'weight'),
(model.block2_8[0], 'weight'),
(model.block3_1[0], 'weight'),
(model.block3_2[0], 'weight'),
(model.block3_3[0], 'weight'),
(model.block4_1[0], 'weight'),
(model.block4_2[0], 'weight'),
(model.block4_3[0], 'weight'),
)

prune.global_unstructured(
    parameters_to_prune,
    pruning_method=prune.L1Unstructured,
    amount=0.5,
)
prune.remove(model.block1_1[0], 'weight')
prune.remove(model.block1_2[0], 'weight')
prune.remove(model.block1_3[0], 'weight')
prune.remove(model.block1_5[0], 'weight')
prune.remove(model.block1_6[0], 'weight')
prune.remove(model.block1_7[0], 'weight')
prune.remove(model.block1_8[0], 'weight')
prune.remove(model.block2_1[0], 'weight')
prune.remove(model.block2_2[0], 'weight')
prune.remove(model.block2_3[0], 'weight')
prune.remove(model.block2_4[0], 'weight')
prune.remove(model.block2_5[0], 'weight')
prune.remove(model.block2_6[0], 'weight')
prune.remove(model.block2_7[0], 'weight')
prune.remove(model.block2_8[0], 'weight')
prune.remove(model.block3_1[0], 'weight')
prune.remove(model.block3_2[0], 'weight')
prune.remove(model.block3_3[0], 'weight')
prune.remove(model.block4_1[0], 'weight')
prune.remove(model.block4_2[0], 'weight')
prune.remove(model.block4_3[0], 'weight')

torch.save(model.state_dict(), './models/shiyan1.pth')
print(model.state_dict().keys())