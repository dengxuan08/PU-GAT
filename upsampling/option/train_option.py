import os

def get_train_options():
    opt = {}

    opt['project_dir'] = "/home/dx/PycharmProjects/PU-GAT"
    opt['model_save_dir'] = opt['project_dir'] + '/checkpoints'
    opt["test_save_dir"] = opt['project_dir'] + '/test_results'
    opt['test_log_dir'] = opt['project_dir'] + '/log_results'
    opt['dataset_dir'] = os.path.join(opt["project_dir"],"/data/PUGAN_poisson_256_poisson_1024.h5")
    opt['test_split'] = os.path.join(opt['project_dir'],'data','test_list.txt')
    opt['train_split'] = os.path.join(opt['project_dir'],'data','train_list.txt')
    opt['isTrain']=True
    opt['batch_size'] = 16
    opt['nepoch'] = 100
    opt['model_save_interval'] = 10
    opt['model_vis_interval'] = 200
    opt["up_ratio"] = 4
    opt["patch_num_point"]=1024
    opt['lr_D']=1e-4
    opt['lr_G']=1e-3
    opt['emd_w']=100
    opt['uniform_w']=10
    opt['gan_w']=0.5
    opt['repulsion_w']=10.0
    opt['use_gan']=False
    return opt
