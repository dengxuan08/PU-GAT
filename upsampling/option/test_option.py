import os

def get_test_options():
    opt = {}

    opt['project_dir'] = "/PU-GAT"
    opt['model_save_dir'] = opt['project_dir'] + '/checkpoints'
    opt["test_save_dir"]=opt['project_dir'] + '/test_results'
    opt['test_log_dir']=opt['project_dir'] + '/log_results'
    opt['dataset_dir'] = os.path.join(opt["project_dir"],"/data/PUGAN_poisson_256_poisson_1024.h5")
    opt['isTrain']=False
    opt['batch_size'] = 12
    opt["patch_num_point"]=1024
    opt['lr_D']=1e-4
    opt['lr_G']=1e-3
    opt['emd_w']=100.0
    opt['uniform_w']=10.0
    opt['gan_w']=0.5
    opt['repulsion_w']=5.0
    opt['use_gan']=False
    return opt
