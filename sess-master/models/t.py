import torch
import torch.nn.functional as F
import sys
sys.path.append('/home/yifan/Code/sess-master/models')
end_points=torch.load('/home/yifan/Code/sess-master/scripts/end_points.pth')
from loss_helper_sess import get_intra_loss,get_aff_mtx,get_inter_loss
from time import *
import os
from itertools import *
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
'''

ema_end_points=end_points.copy()
ema_end_points['net2']=torch.randn(12,128,128).cuda()
begin=time()
loss=get_inter_loss(end_points,ema_end_points)
end=time()'''

def g1et_intra_loss(end_points):
    num_proposal = end_points['objectness_scores'].shape[1]
    batch_size = end_points['objectness_scores'].shape[0]
    L = torch.zeros(batch_size, num_proposal, num_proposal).cuda()
    # L.requires_grad=True
    object_soft = F.softmax(end_points['objectness_scores'], dim=2)
    class_soft = F.softmax(end_points['sem_cls_scores'], dim=2)
    place = torch.nonzero(object_soft[..., 1] > 0.5)
    #List = []
    Mask=torch.zeros(batch_size,num_proposal,num_proposal).cuda()
    for i in place:
        Mask[i[0].item(),i[1].item()]+=0.5
        Mask[i[0].item(),:,i[1].item()]+= 0.5
    Mask=(Mask==1).float()

    key1,key2=torch.max(class_soft,dim=2)

    for i in range(num_proposal):
        k2=key2[:,i].unsqueeze(1)
        mask=(k2==key2).float()
        num=key1[:,i].unsqueeze(1)*key1
        L[:,i,:]=num*mask
    L=L*Mask

    S = get_aff_mtx(end_points,1)

    #key1 = torch.sum(L!= 0, dim=2)
    #denominator = torch.sum(key1, dim=1).float() + 1e-6  #k power of 2?
    denominator = num_proposal ** 2
    numerator = torch.sum(L * (1 - S), dim=2)
    numerator = torch.sum(numerator, dim=1)
    # print(numerator)
    intra_loss = torch.sum(numerator / denominator) / batch_size
    end_points['intra_loss']=intra_loss
    return intra_loss

begin=time()
#num_proposal = end_points['objectness_scores'].shape[1]
#batch_size = end_points['objectness_scores'].shape[0]
ema_end_points=end_points.copy()
ema_end_points['net2']=torch.randn(12,128,128).cuda()
loss=get_intra_loss(end_points)
#S = get_aff_mtx(end_points, 1)
end=time()
run_time=end-begin
print('time of running:',run_time)
print(loss)

begin=time()

intra_loss=g1et_intra_loss(end_points)
'''
for k in range(batch_size):
    inter_loss+=(torch.norm(S[k] - T[k])**2)/(num_proposal**2)

inter_loss/=batch_size
'''
end=time()
run_time=end-begin
print('new time of running:',run_time)
print(intra_loss)
'''num_proposal = end_points['objectness_scores'].shape[1]
batch_size = end_points['objectness_scores'].shape[0]
time1=time()
S = get_aff_mtx(end_points,1)
time2=time()
T = get_aff_mtx(ema_end_points,1)
time3=time()
Cal=S-T

norm1=torch.norm(Cal,dim=2)
norm2=torch.norm(norm1,dim=1)**2/(num_proposal**2)

inter_loss=torch.sum(norm2)/batch_size'''