import os
import cv2
import json
import time
import imageio
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from tqdm import tqdm, trange

from lib.utils import isfloat, seed_everything, get_lr
from lib.nerf.model import MyNerfModel
from lib.nerf.utils import get_rays, render_rays, pose_spherical, compute_accumulated_transmittance

seed_everything(seed=42)