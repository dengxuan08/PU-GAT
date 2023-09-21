import argparse
import os, sys
sys.path.append("../")

parser = argparse.ArgumentParser(description="Arg parser")
parser.add_argument('--gpu', type=int, default=0, help='GPU to use', required=False)
parser.add_argument('--resume', type=str, default='/G_iter_99.pth', required=False)
parser.add_argument('--exp_name',type=str,default='exp',required=False)

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

import torch
from torch.utils.data import DataLoader
from upsampling.utils.xyz_util import save_xyz_file

from upsampling.networks import Generator
from data.data_loader import Test_Dataset



def index_points(points,idx):
    raw_size = idx.size()
    idx = idx.reshape(raw_size[0],-1)
    res = torch.gather(points,1,idx[...,None].expand(-1,-1,points.size(-1)))  #torch.gather(input, dim, index, out=None) → Tensor沿给定轴 dim ,将输入索引张量 index 指定位置的值进行聚合
    return res.reshape(*raw_size,-1)

def fps(xyz,points):
    device = xyz.device
    print(xyz.shape)
    B,N,C = xyz.shape
    centroids = torch.zeros(B,points,dtype=torch.long).to(device)
    distance = torch.ones(B,N).to(device)*1e10
    farthest = torch.randint(0,N,(B,),dtype=torch.long).to(device)
    batch_indices = torch.arange(B,dtype=torch.long).to(device)
    for i in range(points):
        centroids[:,i] = farthest
        centroids = xyz[batch_indices,farthest,:].view(B,1,3)
        dist = torch.sum((xyz-centroids) **2,-1)
        distance = torch.min(distance,dist)
        farthest = torch.max(distance,-1)[1]
    return centroids

if __name__ == '__main__':
    model = Generator()

    checkpoint = torch.load(args.resume)  #load model
    model.load_state_dict(checkpoint, strict = False)
    model.eval().cuda()

    eval_dst = Test_Dataset(npoints=2048)
    eval_loader = DataLoader(eval_dst, batch_size=1,
                             shuffle=False, pin_memory=True, num_workers=0)

    names = eval_dst.names
    exp_name=args.exp_name
    save_dir=os.path.join('../outputs',exp_name)
    if os.path.exists(save_dir) == False:
        os.makedirs(save_dir)
    print("initialize")

    with torch.no_grad():
        for itr, batch in enumerate(eval_loader):
            print(itr)
            name = names[itr]
            print(names[itr])
            points = batch[:, :, 0:3].permute(0, 2, 1).float().cuda()
            preds = model(points)
            print(preds.shape)
            # radius=radius.float().cuda()
            # centroid=centroid.float().cuda()
            # print(preds.shape,radius.shape,centroid.shape)
            # preds=preds*radius+centroid.unsqueeze(2).repeat(1,1,4096)

            preds = preds.permute(0, 2, 1).data.cpu().numpy()[0]
            points = points.permute(0, 2, 1).data.cpu().numpy()
            save_file = '../outputs/{}/{}.xyz'.format(exp_name, name)

            save_xyz_file(preds, save_file)

    print("success")



