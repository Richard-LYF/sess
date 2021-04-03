# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Modified by Zhao Na, 2019

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'pointnet2'))
sys.path.append(BASE_DIR)
from pointnet2_modules import PointnetSAModuleVotes
from GCN_module import *
from GAT_module import *
import pointnet2_utils

def decode_scores(net, end_points, num_class, num_heading_bin, num_size_cluster, mean_size_arr):
    net_transposed = net.transpose(2,1) # (batch_size, 1024, ..)
    batch_size = net_transposed.shape[0]
    num_proposal = net_transposed.shape[1]

    objectness_scores = net_transposed[:,:,0:2]
    end_points['objectness_scores'] = objectness_scores
    
    base_xyz = end_points['aggregated_vote_xyz'] # (batch_size, num_proposal, 3)
    center = base_xyz + net_transposed[:,:,2:5] # (batch_size, num_proposal, 3)
    end_points['center'] = center

    heading_scores = net_transposed[:,:,5:5+num_heading_bin]
    heading_residuals_normalized = net_transposed[:,:,5+num_heading_bin:5+num_heading_bin*2]
    end_points['heading_scores'] = heading_scores # Bxnum_proposalxnum_heading_bin
    end_points['heading_residuals_normalized'] = heading_residuals_normalized # Bxnum_proposalxnum_heading_bin (should be -1 to 1)
    end_points['heading_residuals'] = heading_residuals_normalized * (np.pi/num_heading_bin) # Bxnum_proposalxnum_heading_bin

    size_scores = net_transposed[:,:,5+num_heading_bin*2:5+num_heading_bin*2+num_size_cluster]
    size_residuals_normalized = net_transposed[:,:,5+num_heading_bin*2+num_size_cluster:5+num_heading_bin*2+num_size_cluster*4].view([batch_size, num_proposal, num_size_cluster, 3]) # Bxnum_proposalxnum_size_clusterx3
    end_points['size_scores'] = size_scores
    end_points['size_residuals_normalized'] = size_residuals_normalized
    end_points['size_residuals'] = size_residuals_normalized * torch.from_numpy(mean_size_arr.astype(np.float32)).cuda().unsqueeze(0).unsqueeze(0)

    sem_cls_scores = net_transposed[:,:,5+num_heading_bin*2+num_size_cluster*4:] # Bxnum_proposalx10
    end_points['sem_cls_scores'] = sem_cls_scores
    return end_points

def get_adj_matrix_knn(end_points):
    n = 16 #number of the nearest points to find
    points=end_points['aggregated_vote_xyz']# (batch_size, num_proposal, 3)
    num_points=points.shape[1]
    batch_size=points.shape[0]
    D=torch.zeros(batch_size, num_points, num_points).cuda()
    A=torch.zeros(batch_size, num_points, num_points).cuda()

    for i in range(num_points):
        batch_point=points[:,i,:].unsqueeze(1)
        distance=torch.norm(points-batch_point,dim=2)
        D[:,i,:]=distance
    '''
    for k in range(batch_size):
        for i in range(num_points):
            for j in range(num_points):
                D[k,i,j]=torch.norm(points[k,i]-points[k,j])'''

    list,idx=D.sort(dim=2)
    idx=idx[...,:n]
    '''
    for k in range(batch_size):
        for i in range(num_points):
            ind=idx[k,i]
            A[k,i,ind]=1'''
    A = A.scatter(2, idx.long(), 1)
    A = ((A + A.transpose(1, 2)) != 0).float()
    return A

def get_adj_mtx(end_points):
    '''
    radius = 1.2
    nsample = 16
    xyz = end_points['aggregated_vote_xyz']
    batch_size=end_points['aggregated_vote_xyz'].shape[0]
    num_proposal=end_points['aggregated_vote_xyz'].shape[1]
    '''
    #new_xyz = torch.randn(12, 128, 3).cuda()
    #new_xyz.copy_(xyz)

    #idx = pointnet2_utils.ball_query(radius, nsample, xyz, xyz)

    #print(idx.shape)
    #print(idx)
    #print(idx[0])
    #A = torch.zeros(batch_size, num_proposal, num_proposal).cuda()

    #A = A.scatter(2, idx.long(), 1)
    #eye=torch.eye(num_proposal,num_proposal).cuda()
    #A=A+eye
    #A=((A+A.transpose(1,2))!=0).float()



    A = get_adj_matrix_knn(end_points)
    '''
    diag = torch.diagonal(A,dim1=-2,dim2=-1)
    a_diag = torch.diag_embed(diag)
    A=A-a_diag'''
    
    #print(A)
    #print(A.shape)
    eps = np.finfo(float).eps
    '''
    D = A.sum(2)  # (num_nodes,)
    D_sqrt_inv = torch.sqrt(1.0 / (D + eps))
    D_sqrt_inv = torch.diag_embed(D_sqrt_inv).cuda()
    A= D_sqrt_inv @ A @ D_sqrt_inv
    '''
    D = A.sum(2)  # (num_nodes,)
    D_sqrt_inv = 1.0 / (D + eps)
    D_sqrt_inv = torch.diag_embed(D_sqrt_inv).cuda()
    A = D_sqrt_inv @ A

    diag = torch.diagonal(A, dim1=-2, dim2=-1)-1
    a_diag = torch.diag_embed(diag)
    A=A-a_diag

    return A

class ProposalModule(nn.Module):
    def __init__(self, num_class, num_heading_bin, num_size_cluster, mean_size_arr, num_proposal, sampling, seed_feat_dim=256):
        super().__init__() 

        self.num_class = num_class
        self.num_heading_bin = num_heading_bin
        self.num_size_cluster = num_size_cluster
        self.mean_size_arr = mean_size_arr
        self.num_proposal = num_proposal
        self.sampling = sampling
        self.seed_feat_dim = seed_feat_dim

        # Vote clustering
        self.vote_aggregation = PointnetSAModuleVotes( 
                npoint=self.num_proposal,
                radius=0.3,
                nsample=16,
                mlp=[self.seed_feat_dim, 128, 128, 128],
                use_xyz=True,
                normalize_xyz=True
            )

        # Object proposal/detection
        # Objectness scores (2), center residual (3),
        # heading class+residual (num_heading_bin*2), size class+residual(num_size_cluster*4)
        self.conv1 = torch.nn.Conv1d(128,128,1)
        self.conv2 = torch.nn.Conv1d(128,128,1)
        #self.gcn=GCN(128,128,128,0.3)
        self.gat=GAT(128,128,128,0.3,0.2,4)
        self.conv3 = torch.nn.Conv1d(128,2+3+num_heading_bin*2+num_size_cluster*4+self.num_class,1) #2+3+1*2+18*4+18
        self.bn1 = torch.nn.BatchNorm1d(128)
        self.bn2 = torch.nn.BatchNorm1d(128)

    def forward(self, xyz, features, end_points, stu_end_points=None):  # 2020.2.28
        """
        Args:
            xyz: (B,K,3)
            features: (B,C,K)
        Returns:
            scores: (B,num_proposal,2+3+NH*2+NS*4) 
        """
        if self.sampling == 'vote_fps':
            # Farthest point sampling (FPS) on votes
            if stu_end_points is not None:
                sample_inds = stu_end_points['aggregated_vote_inds']
                xyz, features, _ = self.vote_aggregation(xyz, features, sample_inds)
            else:
                xyz, features, fps_inds = self.vote_aggregation(xyz, features)
                sample_inds = fps_inds
        elif self.sampling == 'seed_fps': 
            # FPS on seed and choose the votes corresponding to the seeds
            # This gets us a slightly better coverage of *object* votes than vote_fps (which tends to get more cluster votes)
            if stu_end_points is not None:#2021.2.28
                #print('3. inside the proposal module, into the add layer')
                sample_inds = stu_end_points['aggregated_vote_inds']
                xyz, features, _ = self.vote_aggregation(xyz, features, sample_inds)
            else:
                #print('3. old layer')
                sample_inds = pointnet2_utils.furthest_point_sample(end_points['seed_xyz'], self.num_proposal)
                xyz, features, _ = self.vote_aggregation(xyz, features, sample_inds)
        elif self.sampling == 'random':
            if stu_end_points is not None:
                sample_inds = stu_end_points['aggregated_vote_inds']
                xyz, features, _ = self.vote_aggregation(xyz, features, sample_inds)
            else:
                # Random sampling from the votes
                num_seed = end_points['seed_xyz'].shape[1]
                sample_inds = torch.randint(0, num_seed, (batch_size, self.num_proposal), dtype=torch.int).cuda()
                xyz, features, _ = self.vote_aggregation(xyz, features, sample_inds)
        else:
            log_string('Unknown sampling strategy: %s. Exiting!'%(self.sampling))
            exit()
        end_points['aggregated_vote_xyz'] = xyz # (batch_size, num_proposal, 3)
        end_points['aggregated_vote_inds'] = sample_inds # (batch_size, num_proposal,) # should be 0,1,2,...,num_proposal

        # --------- PROPOSAL GENERATION ---------
        net = F.relu(self.bn1(self.conv1(features))) 
        net = F.relu(self.bn2(self.conv2(net)))
        net=net.transpose(2,1)
        end_points['net2']=net #2021.2.28

        A=get_adj_mtx(end_points)
        #net = self.gcn(net, A) ## feed into gcn
        net = self.gat(net, A)
        #end_points['gcn_feature'] =net
        end_points['gat_feature'] = net
        net=net.transpose(2,1)

        #torch.save(net,'net2.pth') #2+3+num_heading_bin*2+num_size_cluster*4+self.num_class
        net = self.conv3(net) # (batch_size, 2+3+num_heading_bin*2+num_size_cluster*4, num_proposal) #2+3+1*2+18*4+18
        #torch.save(net,'net3.pth')
        #torch.save(end_points,'end_points_before_score.pth')
        end_points = decode_scores(net, end_points, self.num_class, self.num_heading_bin, self.num_size_cluster, self.mean_size_arr)


        return end_points

if __name__=='__main__':
    sys.path.append(os.path.join(ROOT_DIR, 'sunrgbd'))
    from sunrgbd_detection_dataset import SunrgbdDetectionVotesDataset, DC
    net = ProposalModule(DC.num_class, DC.num_heading_bin,
        DC.num_size_cluster, DC.mean_size_arr,
        128, 'seed_fps').cuda()
    end_points = {'seed_xyz': torch.rand(8,1024,3).cuda()}
    out = net(torch.rand(8,1024,3).cuda(), torch.rand(8,256,1024).cuda(), end_points)
    for key in out:
        print(key, out[key].shape)
