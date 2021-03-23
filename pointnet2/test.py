import sys
sys.path.append('/home/yifan/Code/sess-master/pointnet2')
import torch
from pointnet2_utils import *

radius=3
nsample=10
xyz=torch.randn(12,128,3).cuda()
new_xyz=torch.randn(12,128,3).cuda()
new_xyz.copy_(xyz)

idx = ball_query(radius, nsample, xyz, new_xyz)

print(idx.shape)
print(idx)
print(idx[0])