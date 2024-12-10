import torch
import os

# Check GPU availability
use_cuda = torch.cuda.is_available()
gpu_ids = [0] if use_cuda else []
device = torch.device('cuda' if use_cuda else 'cpu')


#dataset_name = 'DRIVE'  # DRIVE
dataset_name = 'ALL'  # LES
#dataset_name = 'hrf'  # HRF
dataset = dataset_name

max_step = 40000  # 30000 for ukbb
if dataset_name=='DRIVE':
    patch_size_list = [96, 128, 256]
elif dataset_name=='LES':
    patch_size_list = [128,256, 512]
elif dataset_name=='hrf':
    patch_size_list = [128, 256, 512]
elif dataset_name=='STU':
    #patch_size_list = [96, 384, 512]    
    patch_size_list = [128, 256, 512]    
elif dataset_name=='ukbb':
    #patch_size_list = [96, 384, 512]    
    patch_size_list = [128, 256, 512]
else:
    patch_size_list = [128, 384, 256]  
patch_size = patch_size_list[2]
batch_size = 2 # default: 4
print_iter = 100 # default: 100
display_iter = 100 # default: 100
save_iter = 5000 # default: 5000
first_display_metric_iter = max_step-save_iter # default: 25000
lr = 0.001 #if dataset_name!='LES' else 0.00005 # default: 0.0002
step_size = 7000  # 7000 for DRIVE
lr_decay_gamma = 0.5  # default: 0.5
use_SGD = False # default:False

input_nc = 3
ndf = 32
netD_type = 'basic'
n_layers_D = 5
norm = 'instance'
no_lsgan = False
init_type = 'normal'
init_gain = 0.02
use_sigmoid = no_lsgan
use_noise_input_D = False
use_dropout_D = False

# torch.cuda.set_device(gpu_ids[0])
use_GAN = True # default: True

# adam
beta1 = 0.5

# settings for GAN loss
num_classes_D = 1
lambda_GAN_D = 0.01
lambda_GAN_G = 0.01
lambda_GAN_gp = 100
lambda_BCE = 5
lambda_DICE = 1

input_nc_D = input_nc + 3

# settings for centerness
use_centerness =True # default: True
lambda_centerness = 1
center_loss_type = 'centerness'
centerness_map_size =  [patch_size//2,patch_size//2]

# pretrained model
use_pretrained_G =  False
use_pretrained_D = False

#model_path_pretrained_G = r'./log/RIP'

#model_path_pretrained_G = r'./log/UKBB-2023_11_02_23_22_07(5000)'
#model_path_pretrained_G = r'./log/2024_05_26_23_00_12' #40000
#model_path_pretrained_G = r'./log/2024_05_27_23_21_28'
#model_path_pretrained_G = r'./log/2024_05_29_11_26_46'
#model_path_pretrained_G = r'./log/2024_06_08_16_37_29'  #no bn
#model_path_pretrained_G = r'./log/2024_06_09_11_02_11'  # bn
#model_path_pretrained_G = r'./log/2024_06_11_20_31_40'  # bn
#model_path_pretrained_G = r'./log/2024_06_27_23_17_43'  # dive
#model_path_pretrained_G = r'./log/2024_07_01_20_02_24'  # dive
#model_path_pretrained_G = r'./log/resum'  # bn
#model_path_pretrained_G = r'./log/2024_09_04_16_00_44'
model_path_pretrained_G = r'./log/RIP'  # bn
#model_path_pretrained_G = r'./log/2024_11_01_16_15_16'


#model_step_pretrained_G = 'all'
model_step_pretrained_G = '512'


# path for dataset0
stride_height = 150
stride_width = 150


n_classes = 3

model_step = 0

# use CAM
use_CAM = False

#use resize
use_resize = False
resize_w_h = (256,256)

#use av_cross
use_av_cross = False

use_high_semantic = False
lambda_high = 0 # A,V,Vessel

# use global semantic
use_global_semantic = False
global_warmup_step = 0 if use_pretrained_D else max_step//2

# use network
use_network = 'convnext_tiny' # swin_t,convnext_tiny

dataset_path = {'DRIVE': './data/AV_DRIVE/training/',

                'hrf': './data/hrf/training/',

                'ukbb':'./data/ukbb/training/',

                'LES': './data/LES_AV/training/',
                'STU': './data/STU/training/',
                #'STU_patch': './data/STU_patch/training/',

                }
trainset_path = dataset_path


print("Dataset:")
print(trainset_path)
print(use_network)




