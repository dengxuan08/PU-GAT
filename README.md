# PU-GAT
Pytorch official code of "PU-GAT: Point cloud upsampling with graph attention network". The code is modified from [PU-GAN](https://github.com/UncleMEDM/PUGAN-pytorch).

#### Install some packages
This repository is based on PyTorch (1.10.1) and python==3.8, simply by 
```
pip install -r requirement.txt
```
#### Install Pointnet2 module
```
cd pointnet2
python setup.py install
```
#### Install KNN_cuda
```
pip install --upgrade https://github.com/unlimblue/KNN_CUDA/releases/download/0.2/KNN_CUDA-0.2-py3-none-any.whl
```
#### dataset
We use the PU-GAN dataset for training, you can download the training patches in HDF5 format from [GoogleDrive](https://drive.google.com/open?id=13ZFDffOod_neuF3sOM0YiqNbIJEeSKdZ) and put it in folder `data/train`.

#### modify some setting in the upsampling/option/train_option.py
change opt['project_dir'] to where this project is located, and change opt['dataset_dir'] to where you store the dataset.
<br/>
also change params['train_split'] and params['test_split'] to where you save the train/test split txt files.

#### training
```
cd upsampling
python train.py --exp_name=the_project_name --gpu=gpu_number --use_gan --batch_size=12

```
### Evaluation code
We provide the code to calculate the uniform metric in the evaluation code folder. In order to use it, you need to install the CGAL library. Please refer [this link](https://www.cgal.org/download/linux.html) and  [PU-Net](https://github.com/yulequan/PU-Net) to install this library.
Then:
   ```shell
   cd evaluation_code
   cmake .
   make
   source eval.sh
   ```
The second argument is the mesh, and the third one is the predicted points.

### Questions
Please contact 'dengxuan08@hdu.edu.cn'
