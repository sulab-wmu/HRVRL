import os

import torch.autograd as autograd
import torch
import cv2
import numpy as np
from tqdm import tqdm
import os
from Tools.ImageResize import creatMask,shift_rgb
from models.network import PGNet
from lib.Utils import *
from Tools.AVclassifiationMetrics import AVclassifiation, AVclassifiation_vein
from PIL import Image


def CAM(x,rate=1,ind=0):
    """
    :param dataset_path: 计算整个训练数据集的平均RGB通道值
    :param image:  array， 单张图片的array 形式
    :return: array形式的cam后的结果
    """
    # 每次使用新数据集时都需要重新计算前面的RBG平均值
    # RGB-->Rshift-->CLAHE
    print(x.shape)
    x = np.uint8(x)
    _,Mask0 = creatMask(x,threshold=10)
    Mask = np.zeros((x.shape[0],x.shape[1]),np.float32)
    Mask[Mask0>0] = 1

    resize=False
    R_mea_num, G_mea_num, B_mea_num = [], [], []

    dataset_paths = [
                     r'D:\su-lab\code\RIP-AV-main\AV\data\out\test\test_18.png'
                     ]
    dataset_path=dataset_paths[ind]
    image = np.array(Image.open(dataset_path))
    R_mea_num.append(np.mean(image[:, :, 0]))
    G_mea_num.append(np.mean(image[:, :, 1]))
    B_mea_num.append(np.mean(image[:, :, 2]))
    print(f'target RGB = [{np.mean(image[:, :, 0])},{np.mean(image[:, :, 1])},{np.mean(image[:, :, 2])}]')
    print(f'source RGB = [{np.mean(x[:, :, 0])},{np.mean(x[:, :, 1])},{np.mean(x[:, :, 2])}]')
    mea2stand = int((np.mean(R_mea_num)-np.mean(x[:,:,0]))*rate)
    mea2standg = int((np.mean(G_mea_num)-np.mean(x[:,:,1]))*rate)
    mea2standb = int((np.mean(B_mea_num)-np.mean(x[:,:,2]))*rate)

    y = shift_rgb(x,mea2stand,mea2standg,mea2standb)
    print(y.shape)

   
    
    y[Mask==0,:]=0
    print(f'source to target RGB = [{np.mean(y[:, :, 0])},{np.mean(y[:, :, 1])},{np.mean(y[:, :, 2])}]')

    
    return y


def modelEvalution_out(i,net,savePath, use_cuda=False, dataset='DRIVE', is_kill_border=False, input_ch=3, strict_mode=True,config=None, output_dir=''):

    # path for images to save
    dataset_dict = {'out':'out', 'LES': 'LES_AV', 'DRIVE': 'AV_DRIVE', 'hrf': 'hrf'}
    dataset_name = dataset_dict[dataset]
    print(f'evaluating {dataset_name} dataset...')
    image_basename = sorted(os.listdir(f'./data/{dataset_name}/test/{output_dir}'))

    image0 = cv2.imread(f'./data/{dataset_name}/test/{output_dir}/{image_basename[0]}')

    data_path = os.path.join(savePath, dataset,output_dir)

    if not os.path.exists(data_path):
        os.mkdir(data_path)
    print(f'num of test images: {len(image_basename)}')
    test_image_num = len(image_basename)
    #test_image_num = test_image_num//5
    test_image_height = image0.shape[0]
    test_image_width = image0.shape[1]
    if config.use_resize:

        test_image_width = config.resize_w_h[0]
        test_image_height = int(test_image_width*(image0.shape[0]/image0.shape[1]))

        # test_image_height = config.resize_w_h[1]
        # test_image_width = config.resize_w_h[0]
    ArteryPredAll = np.zeros((test_image_num, 1, test_image_height, test_image_width), np.float32)
    VeinPredAll = np.zeros((test_image_num, 1, test_image_height, test_image_width), np.float32)
    VesselPredAll = np.zeros((test_image_num, 1, test_image_height, test_image_width), np.float32)
    ProMap = np.zeros((test_image_num, 3, test_image_height, test_image_width), np.float32)
    MaskAll = np.zeros((test_image_num, 1, test_image_height, test_image_width), np.float32)

    #Vessel = VesselProMap('./data/AV_DRIVE/test/images')

    n_classes = 3
    Net = PGNet(resnet=config.use_network,use_global_semantic=True, input_ch=input_ch, num_classes= n_classes, use_cuda=use_cuda, pretrained=False, centerness=config.use_centerness, centerness_map_size=config.centerness_map_size)
    Net.load_state_dict(net)

    if use_cuda:
        Net.cuda()
    Net.eval()

    for k in tqdm(range(test_image_num)):
        ArteryPred,VeinPred,VesselPred,Mask = GetResult_out_big(Net,k,
                                                                                          use_cuda=use_cuda,
                                                                                          dataset_name=dataset_name,
                                                                                          is_kill_border=is_kill_border,
                                                                                          input_ch=input_ch,
                                                                                          config=config,resize_w_h=(test_image_width,test_image_height),output_dir=output_dir
                                                                                          )
        ArteryPredAll[k,:,:,:] = ArteryPred
        VeinPredAll[k,:,:,:] = VeinPred
        VesselPredAll[k,:,:,:] = VesselPred

        MaskAll[k,:,:,:] = Mask


    ProMap[:,0,:,:] = ArteryPredAll[:,0,:,:]
    ProMap[:,1,:,:] = VeinPredAll[:,0,:,:]
    ProMap[:,2,:,:] = VesselPredAll[:,0,:,:]

    # filewriter = centerline_eval(ProMap, config)
    #np.save(os.path.join(savePath, "ProMap_testset.npy"), ProMap)
    AVclassifiation(data_path,ArteryPredAll,VeinPredAll,VesselPredAll,test_image_num,image_basename)


def modelEvalution_out_global(i,net,savePath, use_cuda=False, dataset='DRIVE', is_kill_border=False, input_ch=3, strict_mode=True,config=None, output_dir=''):

    # path for images to save
    dataset_dict = {'out':'out', 'LES': 'LES_AV', 'DRIVE': 'AV_DRIVE', 'hrf': 'hrf'}
    dataset_name = dataset_dict[dataset]
    print(f'evaluating {dataset_name} dataset...')
    image_basename = sorted(os.listdir(f'./data/{dataset_name}/test/{output_dir}'))

    image0 = cv2.imread(f'./data/{dataset_name}/test/{output_dir}/{image_basename[0]}')

    data_path = os.path.join(savePath, dataset,output_dir)

    if not os.path.exists(data_path):
        os.mkdir(data_path)
    print(f'num of test images: {len(image_basename)}')
    test_image_num = len(image_basename)
    #test_image_num = test_image_num//5
    test_image_height = image0.shape[0]
    test_image_width = image0.shape[1]
    if config.use_resize:

        # test_image_width = config.resize_w_h[0]
        # test_image_height = int(test_image_width*(image0.shape[0]/image0.shape[1]))

        test_image_height = config.resize_w_h[1]
        test_image_width = config.resize_w_h[0]
    ArteryPredAll = np.zeros((test_image_num, 1, test_image_height, test_image_width), np.float32)
    VeinPredAll = np.zeros((test_image_num, 1, test_image_height, test_image_width), np.float32)
    VesselPredAll = np.zeros((test_image_num, 1, test_image_height, test_image_width), np.float32)
    ProMap = np.zeros((test_image_num, 3, test_image_height, test_image_width), np.float32)
    MaskAll = np.zeros((test_image_num, 1, test_image_height, test_image_width), np.float32)

    #Vessel = VesselProMap('./data/AV_DRIVE/test/images')

    n_classes = 3
    Net = PGNet(resnet=config.use_network,use_global_semantic=True, input_ch=input_ch, num_classes= n_classes, use_cuda=use_cuda, pretrained=False, centerness=config.use_centerness, centerness_map_size=config.centerness_map_size)
    Net.load_state_dict(net)

    if use_cuda:
        Net.cuda()
    Net.eval()

    for k in tqdm(range(test_image_num)):
        ArteryPred,VeinPred,VesselPred,Mask = GetResult_out_global(Net,k,
                                                                                          use_cuda=use_cuda,
                                                                                          dataset_name=dataset_name,
                                                                                          is_kill_border=is_kill_border,
                                                                                          input_ch=input_ch,
                                                                                          config=config,resize_w_h=(test_image_width,test_image_height),output_dir=output_dir
                                                                                          )
        ArteryPredAll[k,:,:,:] = ArteryPred
        VeinPredAll[k,:,:,:] = VeinPred
        VesselPredAll[k,:,:,:] = VesselPred

        # MaskAll[k,:,:,:] = Mask


    ProMap[:,0,:,:] = ArteryPredAll[:,0,:,:]
    ProMap[:,1,:,:] = VeinPredAll[:,0,:,:]
    ProMap[:,2,:,:] = VesselPredAll[:,0,:,:]

    # filewriter = centerline_eval(ProMap, config)
    #np.save(os.path.join(savePath, "ProMap_testset.npy"), ProMap)
    AVclassifiation(data_path,ArteryPredAll,VeinPredAll,VesselPredAll,test_image_num,image_basename)


def GetResult_out(Net, k, use_cuda=False, dataset_name='DRIVE', is_kill_border=True, input_ch=3, config=None,resize_w_h=None):
    image_basename = sorted(os.listdir(f'./data/{dataset_name}/test/images'))[k]
    # label_basename = sorted(os.listdir(f'./data/{dataset_name}/test/av'))[k]
    # assert image_basename.split('.')[0] == label_basename.split('.')[0]  # check if the image and label are matched

    ImgName = os.path.join(f'./data/{dataset_name}/test/images/', image_basename)
    # LabelName = os.path.join(f'./data/{dataset_name}/test/av/', label_basename)

    Img0 = cv2.imread(ImgName)
    # Label0 = cv2.imread(LabelName)
    _, Mask0 = creatMask(Img0, threshold=10)
    Mask = np.zeros((Img0.shape[0], Img0.shape[1]), np.float32)
    Mask[Mask0 > 0] = 1

    if config.use_resize:
        
        Img0 = cv2.resize(Img0, resize_w_h)
        # Label0 = cv2.resize(Label0, config.resize_w_h, interpolation=cv2.INTER_NEAREST)
        Mask = cv2.resize(Mask, resize_w_h, interpolation=cv2.INTER_NEAREST)

    Img = Img0
    height, width = Img.shape[:2]
    n_classes = 3
    patch_height = config.patch_size
    patch_width = config.patch_size
    stride_height = config.stride_height
    stride_width = config.stride_width

    Img = cv2.cvtColor(Img, cv2.COLOR_BGR2RGB)
    # rgb2rgg
    if cfg.use_CAM:
        Img  = CAM(Img)

    Img = np.float32(Img / 255.)
    Img_enlarged = paint_border_overlap(Img, patch_height, patch_width, stride_height, stride_width)
    patch_size = config.patch_size
    batch_size = 8
    patches_imgs,_ = extract_ordered_overlap(Img_enlarged, patch_height, patch_width, stride_height, stride_width)
    patches_imgs = np.transpose(patches_imgs, (0, 3, 1, 2))
    patches_imgs = Normalize(patches_imgs)
    patchNum = patches_imgs.shape[0]
    max_iter = int(np.ceil(patchNum / float(batch_size)))
    if config.use_global_semantic:
        golbal_img = Img_enlarged.copy()
        golbal_img = cv2.resize(golbal_img, (config.patch_size, config.patch_size))
        golbal_img_batch = [golbal_img] * batch_size
        golbal_img_batch = np.stack(golbal_img_batch, axis=0)
        golbal_img_batch = np.transpose(golbal_img_batch, (0, 3, 1, 2))
        golbal_img_batch = Normalize(golbal_img_batch)
    pred_patches = np.zeros((patchNum, n_classes, patch_size, patch_size), np.float32)
    print(f'pred_patches shape {pred_patches.shape}')
    for i in range(max_iter):
        begin_index = i * batch_size
        end_index = (i + 1) * batch_size

        patches_temp1 = patches_imgs[begin_index:end_index, :, :, :]

        patches_input_temp1 = torch.FloatTensor(patches_temp1)
        if config.use_global_semantic:
            global_temp1 = golbal_img_batch[0:patches_temp1.shape[0], :, :, :]
            global_input_temp1 = torch.FloatTensor(global_temp1)
        if use_cuda:
            patches_input_temp1 = autograd.Variable(patches_input_temp1.cuda())
            if config.use_global_semantic:
                global_input_temp1 = autograd.Variable(global_input_temp1.cuda())
        else:
            patches_input_temp1 = autograd.Variable(patches_input_temp1)
            if config.use_global_semantic:
                global_input_temp1 = autograd.Variable(global_input_temp1)

        output_temp, _1, = Net(patches_input_temp1, global_input_temp1)

        pred_patches_temp = np.float32(output_temp.data.cpu().numpy())

        pred_patches_temp_sigmoid = sigmoid(pred_patches_temp)

        pred_patches[begin_index:end_index, :, :, :] = pred_patches_temp_sigmoid

        del patches_input_temp1
        del pred_patches_temp
        del patches_temp1
        del output_temp
        del pred_patches_temp_sigmoid

    new_height, new_width = Img_enlarged.shape[0], Img_enlarged.shape[1]
    pred_img = recompone_overlap(pred_patches, new_height, new_width, stride_height, stride_width)  # predictions
    pred_img = pred_img[:, 0:height, 0:width]
    if is_kill_border:
        pred_img = kill_border(pred_img, Mask)

    ArteryPred = np.float32(pred_img[0, :, :])
    VeinPred = np.float32(pred_img[2, :, :])
    VesselPred = np.float32(pred_img[1, :, :])

    ArteryPred = ArteryPred[np.newaxis, :, :]
    VeinPred = VeinPred[np.newaxis, :, :]
    VesselPred = VesselPred[np.newaxis, :, :]

    Mask = Mask[np.newaxis, :, :]

    return ArteryPred, VeinPred, VesselPred, Mask

def GetResult_out_big(Net, k, use_cuda=False, dataset_name='DRIVE', is_kill_border=False, input_ch=3, config=None,resize_w_h=None,output_dir='images'):
    image_basename = sorted(os.listdir(f'./data/{dataset_name}/test/{output_dir}'))[k]
    # label_basename = sorted(os.listdir(f'./data/{dataset_name}/test/av'))[k]
    # assert image_basename.split('.')[0] == label_basename.split('.')[0]  # check if the image and label are matched

    ImgName = os.path.join(f'./data/{dataset_name}/test/{output_dir}/', image_basename)
    # LabelName = os.path.join(f'./data/{dataset_name}/test/av/', label_basename)

    Img0 = cv2.imread(ImgName)
    # Label0 = cv2.imread(LabelName)
    _, Mask0 = creatMask(Img0, threshold=10)
    Mask = np.zeros((Img0.shape[0], Img0.shape[1]), np.float32)
    Mask[Mask0 > 0] = 1

    if config.use_resize:
        
        Img0 = cv2.resize(Img0, resize_w_h)
        # Label0 = cv2.resize(Label0, config.resize_w_h, interpolation=cv2.INTER_NEAREST)
        Mask = cv2.resize(Mask, resize_w_h, interpolation=cv2.INTER_NEAREST)

    Img = Img0
    height, width = Img.shape[:2]
    n_classes = 3
    patch_height = config.patch_size
    patch_width = config.patch_size
    stride_height = config.stride_height
    stride_width = config.stride_width

    Img = cv2.cvtColor(Img, cv2.COLOR_BGR2RGB)
    # rgb2rgg
    if cfg.use_CAM:
        Img  = CAM(Img)

    Img = np.float32(Img / 255.)
    Img_enlarged = paint_border_overlap(Img, patch_height, patch_width, stride_height, stride_width)
    patch_size = config.patch_size
    batch_size = 8
    patches_imgs,global_images = extract_ordered_overlap_big(Img_enlarged, patch_height, patch_width, stride_height, stride_width)
    patches_imgs = np.transpose(patches_imgs, (0, 3, 1, 2))
    patches_imgs = Normalize(patches_imgs)
    global_images = np.transpose(global_images, (0, 3, 1, 2))
    global_images = Normalize(global_images)
    patchNum = patches_imgs.shape[0]
    max_iter = int(np.ceil(patchNum / float(batch_size)))
    
    
    pred_patches = np.zeros((patchNum, n_classes, patch_size, patch_size), np.float32)
    print(f'pred_patches shape {pred_patches.shape}')
    for i in range(max_iter):
        begin_index = i * batch_size
        end_index = (i + 1) * batch_size

        patches_temp1 = patches_imgs[begin_index:end_index, :, :, :]

        patches_input_temp1 = torch.FloatTensor(patches_temp1)
        if config.use_global_semantic:
            global_temp1 = global_images[begin_index:end_index, :, :, :]
            global_input_temp1 = torch.FloatTensor(global_temp1)
        if use_cuda:
            patches_input_temp1 = autograd.Variable(patches_input_temp1.cuda())
            if config.use_global_semantic:
                global_input_temp1 = autograd.Variable(global_input_temp1.cuda())
        else:
            patches_input_temp1 = autograd.Variable(patches_input_temp1)
            if config.use_global_semantic:
                global_input_temp1 = autograd.Variable(global_input_temp1)

        output_temp, _1, = Net(patches_input_temp1, global_input_temp1)

        pred_patches_temp = np.float32(output_temp.data.cpu().numpy())

        pred_patches_temp_sigmoid = sigmoid(pred_patches_temp)

        pred_patches[begin_index:end_index, :, :, :] = pred_patches_temp_sigmoid

        del patches_input_temp1
        del pred_patches_temp
        del patches_temp1
        del output_temp
        del pred_patches_temp_sigmoid

    new_height, new_width = Img_enlarged.shape[0], Img_enlarged.shape[1]
    pred_img = recompone_overlap(pred_patches, new_height, new_width, stride_height, stride_width)  # predictions
    pred_img = pred_img[:, 0:height, 0:width]
    if is_kill_border:
        pred_img = kill_border(pred_img, Mask)

    ArteryPred = np.float32(pred_img[0, :, :])
    VeinPred = np.float32(pred_img[2, :, :])
    VesselPred = np.float32(pred_img[1, :, :])

    ArteryPred = ArteryPred[np.newaxis, :, :]
    VeinPred = VeinPred[np.newaxis, :, :]
    VesselPred = VesselPred[np.newaxis, :, :]

    Mask = Mask[np.newaxis, :, :]

    return ArteryPred, VeinPred, VesselPred, Mask
def GetResult_out_global(Net, k, use_cuda=False, dataset_name='DRIVE', is_kill_border=True, input_ch=3, config=None,resize_w_h=None,output_dir='images'):
    image_basename = sorted(os.listdir(f'./data/{dataset_name}/test/{output_dir}'))[k]
    # label_basename = sorted(os.listdir(f'./data/{dataset_name}/test/av'))[k]
    # assert image_basename.split('.')[0] == label_basename.split('.')[0]  # check if the image and label are matched

    ImgName = os.path.join(f'./data/{dataset_name}/test/{output_dir}/', image_basename)
    # LabelName = os.path.join(f'./data/{dataset_name}/test/av/', label_basename)

    Img0 = cv2.imread(ImgName)
    # Label0 = cv2.imread(LabelName)
    _, Mask0 = creatMask(Img0, threshold=10)
    Mask = np.zeros((Img0.shape[0], Img0.shape[1]), np.float32)
    Mask[Mask0 > 0] = 1

    # if config.use_resize:
        
    #     Img0 = cv2.resize(Img0, resize_w_h)
    #     # Label0 = cv2.resize(Label0, config.resize_w_h, interpolation=cv2.INTER_NEAREST)
    #     Mask = cv2.resize(Mask, resize_w_h, interpolation=cv2.INTER_NEAREST)

    Img = Img0
    height, width = Img.shape[:2]
    n_classes = 3
    patch_height = config.patch_size
    patch_width = config.patch_size
    stride_height = config.stride_height
    stride_width = config.stride_width

    Img = cv2.cvtColor(Img, cv2.COLOR_BGR2RGB)
    # rgb2rgg
    if cfg.use_CAM:
        Img  = CAM(Img)

    Img = np.float32(Img / 255.)
    Img_enlarged = paint_border_overlap(Img, patch_height, patch_width, stride_height, stride_width)
    

    if config.use_resize:
        
        Img_enlarged = cv2.resize(Img_enlarged, resize_w_h)
        # Label0 = cv2.resize(Label0, config.resize_w_h, interpolation=cv2.INTER_NEAREST)

    patch_size = config.patch_size
    batch_size = 8
    patches_imgs = Img_enlarged[np.newaxis,:]
    print(f'shape: {patches_imgs.shape}')
    patches_imgs = np.transpose(patches_imgs, (0, 3, 1, 2))
    patches_imgs = Normalize(patches_imgs)
    patchNum = patches_imgs.shape[0]
    max_iter = int(np.ceil(patchNum / float(batch_size)))
    if config.use_global_semantic:
        golbal_img = Img_enlarged.copy()
        golbal_img = cv2.resize(golbal_img, (config.patch_size, config.patch_size))
        golbal_img_batch = [golbal_img] * batch_size
        golbal_img_batch = np.stack(golbal_img_batch, axis=0)
        golbal_img_batch = np.transpose(golbal_img_batch, (0, 3, 1, 2))
        golbal_img_batch = Normalize(golbal_img_batch)
    pred_patches = np.zeros((patchNum, n_classes, patch_size, patch_size), np.float32)
    print(f'pred_patches shape {pred_patches.shape}')
    for i in range(max_iter):
        begin_index = i * batch_size
        end_index = (i + 1) * batch_size

        patches_temp1 = patches_imgs[begin_index:end_index, :, :, :]

        patches_input_temp1 = torch.FloatTensor(patches_temp1)
        if config.use_global_semantic:
            global_temp1 = golbal_img_batch[0:patches_temp1.shape[0], :, :, :]
            global_input_temp1 = torch.FloatTensor(global_temp1)
        if use_cuda:
            patches_input_temp1 = autograd.Variable(patches_input_temp1.cuda())
            if config.use_global_semantic:
                global_input_temp1 = autograd.Variable(global_input_temp1.cuda())
        else:
            patches_input_temp1 = autograd.Variable(patches_input_temp1)
            if config.use_global_semantic:
                global_input_temp1 = autograd.Variable(global_input_temp1)

        output_temp, _1, = Net(patches_input_temp1, global_input_temp1)

        pred_patches_temp = np.float32(output_temp.data.cpu().numpy())

        pred_patches_temp_sigmoid = sigmoid(pred_patches_temp)

        pred_patches[begin_index:end_index, :, :, :] = pred_patches_temp_sigmoid

        del patches_input_temp1
        del pred_patches_temp
        del patches_temp1
        del output_temp
        del pred_patches_temp_sigmoid

    new_height, new_width = Img_enlarged.shape[0], Img_enlarged.shape[1]
    pred_img = recompone_overlap(pred_patches, new_height, new_width, stride_height, stride_width)  # predictions
    pred_img = pred_img[:, 0:height, 0:width]
    # if is_kill_border:
    #     pred_img = kill_border(pred_img, Mask)

    ArteryPred = np.float32(pred_img[0, :, :])
    VeinPred = np.float32(pred_img[2, :, :])
    VesselPred = np.float32(pred_img[1, :, :])

    ArteryPred = ArteryPred[np.newaxis, :, :]
    VeinPred = VeinPred[np.newaxis, :, :]
    VesselPred = VesselPred[np.newaxis, :, :]

    Mask = Mask[np.newaxis, :, :]

    return ArteryPred, VeinPred, VesselPred, Mask



def out_test(cfg,output_dir=''):
    device = torch.device("cuda" if cfg.use_cuda else "cpu")
    model_root = cfg.model_path_pretrained_G
    model_path = os.path.join(model_root, 'G_' + str(cfg.model_step_pretrained_G) + '.pkl')
    net = torch.load(model_path, map_location=device)
    result_folder = os.path.join(model_root, 'running_result')
    modelEvalution_out(cfg.model_step_pretrained_G, net,
                   result_folder,
                   use_cuda=cfg.use_cuda,
                   dataset='out',
                   input_ch=cfg.input_nc,
                   config=cfg,
                   strict_mode=True, output_dir=output_dir)


if __name__ == '__main__':
    from config import config_test_general as cfg
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--output','-o', type=str, default='', help='output dir')
    args = parser.parse_args()
    
    output_dir = args.output
    print(output_dir)

    out_test(cfg, output_dir=output_dir)