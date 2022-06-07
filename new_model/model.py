import time
import gc
import sys,os
from prometheus_client import Metric
import numpy as np
import torch
import torch.nn as nn
#from model_utils.metrics import findSize,findSizeModel
from model_utils.utils import findKNN
import pynanoflann
from torchsummary import summary

sys.path.append(os.path.dirname(os.getcwd()))

class SharedMLP(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        transpose=False,
        padding_mode='zeros',
        bn=False,
        activation_fn=None
    ):
        super(SharedMLP, self).__init__()

        conv_fn = nn.ConvTranspose2d if transpose else nn.Conv2d

        self.conv = conv_fn(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding_mode=padding_mode
        )
        self.batch_norm = nn.BatchNorm2d(out_channels, eps=1e-6, momentum=0.99) if bn else None
        self.activation_fn = activation_fn

    def forward(self, input):
        r"""
            Forward pass of the network

            Parameters
            ----------
            input: torch.Tensor, shape (B, d_in, N, K)

            Returns
            -------
            torch.Tensor, shape (B, d_out, N, K)
        """
        x = self.conv(input)
        if self.batch_norm:
            x = self.batch_norm(x)
        if self.activation_fn:
            x = self.activation_fn(x)
        return x


class LocalSpatialEncoding(nn.Module):
    def __init__(self, d, num_neighbors, device):
        super(LocalSpatialEncoding, self).__init__()

        self.num_neighbors = num_neighbors
        self.mlp = SharedMLP(10, d, bn=True, activation_fn=nn.ReLU())

        self.device = device

    def forward(self, coords, features, knn_output):
        r"""
            Forward pass

            Parameters
            ----------
            coords: torch.Tensor, shape (B, N, 3)
                coordinates of the point cloud
            features: torch.Tensor, shape (B, d, N, 1)
                features of the point cloud
            neighbors: tuple

            Returns
            -------
            torch.Tensor, shape (B, 2*d, N, K)
        """
        # finding neighboring points
        idx,dist= knn_output
        features=features.cpu()

        B, N, K = idx.size()
        # idx(B, N, K), coords(B, N, 3)
        # neighbors[b, i, n, k] = coords[b, idx[b, n, k], i] = expanded_coords[b, i, extended_idx[b, i, n, k], k]
        expanded_idx = idx.unsqueeze(1).expand(B, 3, N, K)
        expanded_coords = coords.transpose(-2,-1).unsqueeze(-1).expand(B, 3, N, K)
        neighbor_coords = torch.gather(expanded_coords, 2, expanded_idx) # shape (B, 3, N, K)

        expanded_idx = idx.unsqueeze(1).expand(B, features.size(1), N, K)
        expanded_features = features.expand(B, -1, N, K)
        neighbor_features = torch.gather(expanded_features, 2, expanded_idx)
        # if USE_CUDA:
        #     neighbors = neighbors.cuda()

        # relative point position encoding
        concat = torch.cat((
            expanded_coords,
            neighbor_coords,
            expanded_coords - neighbor_coords,
            dist.unsqueeze(-3)
        ), dim=-3).to(self.device)

        del idx,dist,expanded_idx,neighbor_coords,expanded_coords
        concat=self.mlp(concat)
        concat=concat.cpu()
        concat=torch.cat((concat,neighbor_features),dim=-3)
        del expanded_features,neighbor_features
        # Free Cached tensors
        gc.collect()
        torch.cuda.empty_cache()
        return concat



class AttentivePooling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AttentivePooling, self).__init__()

        self.score_fn = nn.Sequential(
            nn.Linear(in_channels, in_channels, bias=False),
            nn.Softmax(dim=-2)
        )
        self.mlp = SharedMLP(in_channels, out_channels, bn=True, activation_fn=nn.ReLU())

    def forward(self, x):
        r"""
            Forward pass

            Parameters
            ----------
            x: torch.Tensor, shape (B, d_in, N, K)

            Returns
            -------
            torch.Tensor, shape (B, d_out, N, 1)
        """
        # computing attention scores
        scores = self.score_fn(x.permute(0,2,3,1)).permute(0,3,1,2)

        # sum over the neighbors
        features = torch.sum(scores * x, dim=-1, keepdim=True) # shape (B, d_in, N, 1)
        del scores
       

        return self.mlp(features)



class LocalFeatureAggregation(nn.Module):
    def __init__(self, d_in, d_out, num_neighbors, device):
        super(LocalFeatureAggregation, self).__init__()

        self.num_neighbors = num_neighbors

        self.mlp1 = SharedMLP(d_in, d_out//2, activation_fn=nn.LeakyReLU(0.2))
        self.mlp2 = SharedMLP(d_out, 2*d_out)
        self.shortcut = SharedMLP(d_in, 2*d_out, bn=True)

        self.lse1 = LocalSpatialEncoding(d_out//2, num_neighbors, device)
        self.lse2 = LocalSpatialEncoding(d_out//2, num_neighbors, device)

        self.pool1 = AttentivePooling(d_out, d_out//2)
        self.pool2 = AttentivePooling(d_out, d_out)

        self.lrelu = nn.LeakyReLU()
        self.device=device

    def forward(self, coords, features):
        r"""
            Forward pass

            Parameters
            ----------
            coords: torch.Tensor, shape (B, N, 3)
                coordinates of the point cloud
            features: torch.Tensor, shape (B, d_in, N, 1)
                features of the point cloud

            Returns
            -------
            torch.Tensor, shape (B, 2*d_out, N, 1)
        """
        knn_output=findKNN(coords,coords,self.num_neighbors)
        x = self.mlp1(features)

        x = self.lse1(coords, x, knn_output)
        #print(f"After LSA1, Memory Allocation {torch.cuda.memory_allocated(0)/(1024**3)}")
        gc.collect()
        torch.cuda.empty_cache()
        x = self.pool1(x.to(self.device))
        #print(f"After Pool1, Memory Allocation {torch.cuda.memory_allocated(0)/(1024**3)}")
        gc.collect()
        torch.cuda.empty_cache()
        x= self.lse2(coords, x, knn_output)
        #print(f"After LSA2, Memory Allocation {torch.cuda.memory_allocated(0)/(1024**3)}")
        gc.collect()
        torch.cuda.empty_cache()
        x = self.pool2(x.to(self.device))
        #print(f"After Pool2, Memory Allocation {torch.cuda.memory_allocated(0)/(1024**3)}")
        x=self.lrelu(self.mlp2(x)+self.shortcut(features))
        #print(f"After Relu, Memory Allocation {torch.cuda.memory_allocated(0)/(1024**3)}")
        # Free Cached Tensors
        gc.collect()
        torch.cuda.empty_cache()
        
        return x


class FeatureExtractor(nn.Module):
    def __init__(self,in_channels=1):
        super(FeatureExtractor, self).__init__()
        self.conv=nn.Sequential(
            nn.Conv3d(in_channels,8,kernel_size=7),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(8,8),
            nn.ReLU()
        )
    
    def forward(self,x):
        x=self.conv(x)
        return x

class SAMS_NET(nn.Module):
    def __init__(self,num_classes,num_neighbors,device=torch.device('cpu')):
        super(SAMS_NET, self).__init__()
        self.num_classes = num_classes
        self.num_neighbors = num_neighbors
        self.device = device

        # Postion Encoding
        self.pe=nn.Linear(3,8)
        self.pe_bn = nn.Sequential(
            nn.BatchNorm2d(16, eps=1e-6, momentum=0.99),
            nn.LeakyReLU(0.2)
        )
        # Feature Extractor
        self.fe=FeatureExtractor()

        """# Latent Vector Initialization
        z=torch.empty(B,N,self.latent_size) 
        z=nn.init.xavier_uniform_(z, gain=nn.init.calculate_gain('relu')) # shape (B,N,D)
        z=z.unsqueeze(dim=-1)"""

        # Decoding layers
        self.decoder = nn.ModuleList([
            LocalFeatureAggregation(35, 16, num_neighbors, device),
            LocalFeatureAggregation(32, 32, num_neighbors, device),
            LocalFeatureAggregation(64, 64, num_neighbors, device),
            LocalFeatureAggregation(128, 128, num_neighbors, device)
        ])

        self.sdf=SharedMLP(256,1)
        self.seg=SharedMLP(256,num_classes)
        
        
    def forward(self,latent,input):
        """
           Forward pass

            Parameters
            ----------
            latent_vector: torch.Tensor, shape (B,N,D)
            input: torch.Tensor, shape (B, N, d_in)
                input points
            Returns
            -------
            torch.Tensor, shape (B,1,N)
            torch.Tensor, shape (B, num_classes, N)
                segmentation scores for each point)
        """
        B,N,d_in=input.shape
        coords=input.clone().cpu()

        if not latent.requires_grad:
            latent.requires_grad=True
        latent=latent.permute((0,2,1))
        latent=latent.unsqueeze(dim=-1)
        x=coords.permute((0,2,1))
        x=x.unsqueeze(dim=-1)

        """# Positional Encoding of shape
        x = self.fc_start(input).transpose(-2,-1).unsqueeze(-1)
        x = self.bn_start(x) # shape (B, d, N, 1)"""

        """# Feature Extration of Neighbouring Intensity
        features=[[] for i in range(B)]  # shape (B, N, d1)
        for i in range(B):
            features[i]=self.convModel(nbr_intensity[i].unsqueeze(1)).unsqueeze(0) # shape(1, N, d1)
        features=torch.cat(features) # shape(1, N d1)
        features=features.unsqueeze(-1) # shape(1 ,N, d1, -1)
        features=torch.permute(features,(0,2,1,3)) # shape(B, d1, N, 1)"""

        #x=torch.cat((x,features),dim=1) # shape (B, 32, N, 1)
        x=torch.cat((latent,x),dim=1) # shape (B, 35, N, 1)

        # Decoding Operation
        x_stack = []
        x=x.to("cpu")

        for lfa in self.decoder:
            x = lfa(coords, x.to(self.device))
            x=x.detach().cpu()
            x_stack.append(x)
            # Free Cached Tensors
            gc.collect()
            torch.cuda.empty_cache()
        
        sdf=self.sdf(x) # Signed Distance Function Prediction
        seg=self.seg(x) # Segmentation Output

        return latent,sdf,seg
        


if __name__== "__main__":
    model=SAMS_NET(2,8)
    print(summary(model,((6000000,32),(6000000,3))))


