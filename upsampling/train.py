import os,sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--exp_name', '-e', type=str, required=False, default='exp', help='experiment name')
parser.add_argument('--debug', action='store_true', help='specify debug mode')
parser.add_argument('--use_gan',action='store_true', required=False)
parser.add_argument('--batch_size',type=int, required=False, default=4)
parser.add_argument('--gpu',type=str,default='0')

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
sys.path.append('../')
import torch
torch.cuda.set_device(0)
device = 'cuda'

from upsampling.networks import Generator, Discriminator
from data.data_loader import PUNAN_Dataset
import time
from option.train_option import get_train_options
from torch.utils import data
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
from loss.loss import Loss
import datetime
import torch.nn as nn


def xavier_init(m):
    classname = m.__class__.__name__
    #print(classname)
    if classname.find('Conv') != -1:
        nn.init.xavier_normal_(m.weight)
    elif classname.find('Linear')!=-1:
        nn.init.xavier_normal_(m.weight)
    elif classname.find('BatchNorm') != -1:
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

def train(args):
    start_t=time.time()
    params=get_train_options()
    params["exp_name"]=args.exp_name
    params["patch_num_point"]=1024
    params["batch_size"]=args.batch_size
    params['use_gan']=args.use_gan

    if args.debug:
        params["nepoch"]=2
        params["model_save_interval"]=3
        params['model_vis_interval']=3

    log_dir=os.path.join(params["model_save_dir"],args.exp_name)
    if os.path.exists(log_dir)==False:
        os.makedirs(log_dir)

    trainloader=PUNAN_Dataset(split_dir=params['train_split'])
    #print(params["dataset_dir"])
    num_workers=4
    train_data_loader=data.DataLoader(dataset=trainloader,batch_size=params["batch_size"],shuffle=True,
                                      num_workers=num_workers,pin_memory=True,drop_last=True)

    G_model=Generator(params)
    G_model.apply(xavier_init)
    G_model=torch.nn.DataParallel(G_model).to(device)
    D_model=Discriminator(params,in_channels=3)
    D_model.apply(xavier_init)
    D_model=torch.nn.DataParallel(D_model).to(device)

    G_model.train()
    D_model.train()

    optimizer_D=Adam(D_model.parameters(),lr=params["lr_D"],betas=(0.9,0.999))
    optimizer_G=Adam(G_model.parameters(),lr=params["lr_G"],betas=(0.9,0.999))

    D_scheduler = MultiStepLR(optimizer_D,[50,80],gamma=0.2)
    G_scheduler = MultiStepLR(optimizer_D,[50,80],gamma=0.2)

    Loss_fn=Loss()

    print("preparation time is %fs" % (time.time() - start_t))
    iter=0
    for e in range(params["nepoch"]):
        D_scheduler.step()
        G_scheduler.step()
        for batch_id,(input_data, gt_data, radius_data) in enumerate(train_data_loader):
            optimizer_G.zero_grad()
            optimizer_D.zero_grad()

            input_data=input_data[:,:,0:3].permute(0,2,1).float().cuda()
            gt_data=gt_data[:,:,0:3].permute(0,2,1).float().cuda()
            #print(input_data.shape, gt_data.shape)

            start_t_batch=time.time()
            output_point_cloud=G_model(input_data)
            # print(output_point_cloud.shape)

            repulsion_loss = Loss_fn.get_repulsion_loss(output_point_cloud.permute(0, 2, 1))
            # print(repulsion_loss)
            uniform_loss = Loss_fn.get_uniform_loss(output_point_cloud.permute(0, 2, 1))
            # print(output_point_cloud.shape,gt_data.shape)
            emd_loss = Loss_fn.get_emd_loss(output_point_cloud.permute(0, 2, 1), gt_data.permute(0, 2, 1))

            if params['use_gan'] == True:
                fake_pred = D_model(output_point_cloud.detach())
                d_loss_fake = Loss_fn.get_discriminator_loss_single(fake_pred,label=False)
                d_loss_fake.backward()
                optimizer_D.step()

                real_pred = D_model(gt_data.detach())
                d_loss_real = Loss_fn.get_discriminator_loss_single(real_pred, label=True)
                d_loss_real.backward()
                optimizer_D.step()

                d_loss=d_loss_real+d_loss_fake

                fake_pred=D_model(output_point_cloud)
                g_loss=Loss_fn.get_generator_loss(fake_pred)

                # print(repulsion_loss,uniform_loss,emd_loss)
                total_G_loss = params['uniform_w']*uniform_loss + params['emd_w']*emd_loss + repulsion_loss*params['repulsion_w'] + g_loss*params['gan_w']
            else:
                #total_G_loss = params['uniform_w'] * uniform_loss + params['emd_w'] * emd_loss + \
                #               repulsion_loss * params['repulsion_w']
                total_G_loss = params['emd_w'] * emd_loss + repulsion_loss * params['repulsion_w']
            #
            # total_G_loss = params['emd_w'] * emd_loss + repulsion_loss * params['repulsion_w']

            total_G_loss.backward()
            optimizer_G.step()

            current_lr_D=optimizer_D.state_dict()['param_groups'][0]['lr']
            current_lr_G=optimizer_G.state_dict()['param_groups'][0]['lr']

            msg="{:0>8},{}:{}, [{}/{}], {}: {},{}:{}".format(
                str(datetime.timedelta(seconds=round(time.time() - start_t))),
                "epoch",
                e,
                batch_id + 1,
                len(train_data_loader),
                "total_G_loss",
                total_G_loss.item(),
                "iter time",
                (time.time() - start_t_batch)
            )
            print(msg)

            iter+=1
        if (e+1) % params['model_save_interval'] == 0 and e > 0:
            model_save_dir = os.path.join(params['model_save_dir'], params['exp_name'])
            if os.path.exists(model_save_dir) == False:
                os.makedirs(model_save_dir)
            D_ckpt_model_filename = "D_iter_%d.pth" % (e)
            G_ckpt_model_filename = "G_iter_%d.pth" % (e)
            D_model_save_path = os.path.join(model_save_dir, D_ckpt_model_filename)
            G_model_save_path = os.path.join(model_save_dir, G_ckpt_model_filename)
            torch.save(D_model.module.state_dict(), D_model_save_path)
            torch.save(G_model.module.state_dict(), G_model_save_path)


if __name__=="__main__":
    train(args)
