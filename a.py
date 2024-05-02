# %%
# %%
import torch
import pytorch3d.transforms.so3 as so3
from pytorch3d.transforms.so3 import (
    so3_exp_map,
    so3_relative_angle,
)
from pytorch3d.renderer.cameras import (
    SfMPerspectiveCameras,
)

# add path for demo utils
import sys
import os
sys.path.append(os.path.abspath(''))

# set for reproducibility
torch.manual_seed(42)

from utils import plot_camera_scene

# %%
log_R = torch.randn(1, 3, dtype=torch.float32)
R = so3_exp_map(log_R)
T = torch.tensor([[[1,2,3]]], dtype=torch.float32)

# %%
print(R.shape, T.shape)
cam = SfMPerspectiveCameras(R=R, T=T)
plot_camera_scene(cam, cam, "")

# %%
M = cam.get_world_to_view_transform()
M.get_matrix()

# %%
p_world = torch.tensor([[1, 0, 0]], dtype=torch.float32)
print(M.transform_points(p_world))

# %%
p_WH = torch.hstack([p_world, torch.ones((1,1))])
print(p_WH @ M.get_matrix())

# %%

print(T @ R.mT)
