import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.data import DataLoader
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from datasets import traffic_dataset
from utils import *
import argparse
import yaml
import time
from pathlib import Path
from tqdm import tqdm
import sys
sys.path.append('./model')
sys.path.append('./model/TSFormer')
sys.path.append('./model/Meta_Models')
from meta_patch import *
from TSmodel_TSFormerTST import *
import copy
# from pykeops.torch import LazyTensor
# import faiss
from kmeans_pytorch import kmeans

parser = argparse.ArgumentParser()
parser.add_argument('--data_list', default='chengdu_shenzhen_metr',type=str)
parser.add_argument('--test_dataset', default = 'pems-bay', type=str)
parser.add_argument('--gpu', default=0, type = int)
parser.add_argument('--sim', default='cosine',type = str)
parser.add_argument('--K', default=10, type = int)
args = parser.parse_args()
args.gpu=0

use_cuda = False


if __name__ == "__main__":
    if torch.cuda.is_available():
        args.device = torch.device('cuda:{}'.format(args.gpu))
        use_cuda = True
    else:
        args.device = torch.device('cpu')
    # args.device = torch.device('cpu')
    print("INFO: {}".format(args.device))
    
    patch_pool = torch.load('./pattern/{}/patch.pt'.format(args.data_list)).to(args.device)
    unnorm_patch_pool = torch.load('./pattern/{}/unorm_patch.pt'.format(args.data_list)).to(args.device)
    hour_list = [1, 3, 6, 12, 24]
    for hour in hour_list:
        emb_pool = torch.load('./pattern/{}/emb_hour_{}.pt'.format(args.data_list, hour)).to(args.device)
        
        N,D = emb_pool.shape
        sample_ratio = 0.1
        indices = torch.randperm(N)[:int(N * sample_ratio)]
        emb_pool = emb_pool[indices]
        print("Use {}. After sampling {}, N = {}".format(args.sim,sample_ratio, emb_pool.shape[0]))
        
        c, cl = kmeans(
            X=emb_pool, num_clusters=args.K, distance=args.sim, device=torch.device('cuda:0')
        )
        print("c shape is : {}, cl shape is : {}".format(c.shape, cl.shape))
        torch.save(c,'./pattern/{}/{}_{}_c_{}.pt'.format(args.data_list,args.sim,args.K,hour))
        torch.save(cl,'./pattern/{}/{}_{}_cl_{}.pt'.format(args.data_list,args.sim,args.K,hour))
    
    
    
