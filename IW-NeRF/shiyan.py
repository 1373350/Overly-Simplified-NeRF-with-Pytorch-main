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



big_model = BigModel(hidden_dim=256).to('cuda')
big_model.load_state_dict(torch.load('./models/big.pth'))

module=big_model.block1_1
print(list(module.named_parameters()))
print(list(module.named_buffers()))

