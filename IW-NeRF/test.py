import os
import cv2
import json
import time
import mcubes
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

size = 100
H, W = size, size
hn = 2
hf = 6
nb_bins = 192
batch_size = 512
model_path = "./models/big.pth"
device = torch.device("cuda")

my_model = MyNerfModel(hidden_dim=256).to(device)
my_model.load_state_dict(torch.load(model_path, map_location=str(device)))
for name, param in my_model.named_parameters():
    print(name)
    print(param.shape)
my_model = my_model.eval()


render_poses = []
n_frames = 30
for i in range(n_frames):
    focal = size * 1.5 + size * 0.5 * np.sin(i * np.pi * 2 / n_frames)
    angle_x = - 180 + i * 360 / n_frames
    angle_y = - 45 + 20 * np.sin(i * np.pi * 4 / n_frames)
    render_poses.append((focal, pose_spherical(angle_x, angle_y, 4.0)))

reconstructed = []
with torch.no_grad():
    for focal, t_mat in tqdm(render_poses):
        rays_o, rays_d = get_rays(H, W, focal, t_mat[:3, :4])
        rays_o, rays_d = rays_o.reshape([-1, 3]), rays_d.reshape([-1, 3])
        rec = []
        dataset = np.concatenate([rays_o, rays_d], axis=1).astype(np.float32)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        for xs in dataloader:
            ray_origins = xs[:,:3].to(device)
            ray_directions = xs[:,3:6].to(device)
            regenerated_pixels = render_rays(
                my_model, ray_origins, ray_directions, hn=hn, hf=hf, nb_bins=nb_bins
            )
            rec.append(regenerated_pixels.detach().cpu().numpy())
        rec = np.concatenate(rec, axis=0)[..., ::-1] # BGR to RGB
        rec = np.clip(rec.reshape([H, W, 3]) * 255, 0, 255).astype(np.uint8)
        reconstructed.append(rec)

imageio.mimsave("./outputs/video.gif", reconstructed, fps = 8)
print("Finished.")


sigma_threshold = 30.0
N = 128
x, y, z = [np.linspace(-1., 1., N) for _ in range(3)]
rgbd = []
rays_o = np.stack(np.meshgrid(x, y, z), -1).reshape([-1, 3])
rays_d = np.zeros(rays_o.shape)
dataset = np.concatenate([rays_o, rays_d], axis=1).astype(np.float32)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
with torch.no_grad():
    for xs in tqdm(dataloader):
        ray_origins = xs[:,:3].to(device)
        ray_directions = xs[:,3:6].to(device)
        colors, sigma = my_model(ray_origins, ray_directions)
        rgbd.append(torch.cat([colors, sigma.reshape([-1,1])], dim=-1))
rgbd = torch.cat(rgbd, 0)
sigma = np.maximum(rgbd[:,-1].detach().cpu().numpy(), 0).reshape([N,N,N])

vertices, triangles = mcubes.marching_cubes(sigma, sigma_threshold)
mcubes.export_obj(vertices, triangles, "./outputs/mesh.obj")
print("Finished.")





