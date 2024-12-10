# -*- coding: utf-8 -*-
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.autograd as autograd
import os
import cv2
from Tools.ImageResize import creatMask,shift_rgb
from Tools.data_augmentation import data_aug5
from lib.Utils import *
from models.network import PGNet
from sklearn.metrics import roc_auc_score
from Preprocessing.generate_patch_selection import patch_select
from Tools.evalution_vessel import evalue
from skimage import morphology
from Tools.AVclassifiationMetrics import AVclassifiationMetrics_skeletonPixles
from scipy import ndimage
from torchvision import transforms

def CAM(x,rate=0.5,ind=3):
    """
    :param dataset_path: 计算整个训练数据集的平均RGB通道值
    :param image:  array， 单张图片的array 形式
    :return: array形式的cam后的结果
    """
    # 每次使用新数据集时都需要重新计算前面的RBG平均值
    # RGB-->Rshift-->CLAHE
    x = np.transpose(x, (1, 2, 0))
    x = np.uint8(x)
    _,Mask0 = creatMask(x,threshold=10)
    Mask = np.zeros((x.shape[0],x.shape[1]),np.float32)
    Mask[Mask0>0] = 1

    resize=False
    R_mea_num, G_mea_num, B_mea_num = [], [], []

    dataset_paths = [r'data/AV_DRIVE/test/images/01_test.tif',
                     r'data/LES_AV/test/images/10.png',
                     r'data/hrf/test/images/11_dr.png',
                     r'data/ukbb/training/images/training_15.png',
                     r'data/STU/training/images/1919150042.jpg'
                     ]
    dataset_path=dataset_paths[ind]
    image = np.array(Image.open(dataset_path))
    R_mea_num.append(np.mean(image[:, :, 0]))
    G_mea_num.append(np.mean(image[:, :, 1]))
    B_mea_num.append(np.mean(image[:, :, 2]))

    mea2stand = int((np.mean(R_mea_num)-np.mean(x[:,:,0]))*rate)
    mea2standg = int((np.mean(G_mea_num)-np.mean(x[:,:,1]))*rate)
    mea2standb = int((np.mean(B_mea_num)-np.mean(x[:,:,2]))*rate)

    y = shift_rgb(x,mea2stand,mea2standg,mea2standb)
    y[Mask==0,:]=0
    y = np.float32(y / 255.)
    
    y = np.transpose(y, (2, 0, 1))
    return y




def generate_a_v(image,av):
    ##hwc
    #image_t = np.transpose(image, (1, 2, 0))
    kernel = np.ones((3,3))
    patch_select_a = np.uint8(av[0])
    patch_select_v = np.uint8(av[1])

    # patch_select_image_oa = image.copy()
    # patch_select_image_oa[patch_select_a==0]=0

    # patch_select_image_ov = image.copy()
    # patch_select_image_ov[patch_select_v==0]=0

    # patch_select_image_oav = image.copy()
    # patch_select_image_oav[(patch_select_v==0) & (patch_select_a==0)]=0
    

    patch_select_a_d = cv2.dilate(patch_select_a, kernel=kernel)
    patch_select_v_d = cv2.dilate(patch_select_v, kernel=kernel)
    patch_select_image_na = image.copy()
    patch_select_image_na[patch_select_a_d==1]=0
    #patch_select_image_na = cv2.inpaint(patch_select_image_na, patch_select_a_d, 5, cv2.INPAINT_TELEA)
    patch_select_av_na = av.copy()
    patch_select_av_na[0,:,:]=0
    patch_select_av_na[2]=patch_select_av_na[1]


    patch_select_image_nv = image.copy()
    patch_select_image_nv[patch_select_v_d==1]=0
    
    #patch_select_image_nv = cv2.inpaint(patch_select_image_nv, patch_select_v_d, 5, cv2.INPAINT_TELEA)
    patch_select_av_nv = av.copy()
    patch_select_av_nv[1,:,:]=0
    patch_select_av_nv[2]=patch_select_av_nv[0]

    
    
    #patch_select_image_na = np.transpose(patch_select_image_na, (2, 0, 1))
    #patch_select_image_nv = np.transpose(patch_select_image_nv, (2, 0, 1))
    return patch_select_image_nv, patch_select_image_na, patch_select_av_nv,patch_select_av_na


def get_patch_trad_5(batch_size, patch_size, train_data=None, train_label_data=None, train_label_data_centerness=None,
                     train_data_mask=None,train_data_mask2=None, train_patch_data=None, train_patch_label=None, patch_size1=96,
                     patch_size2=128, use_patch_selection=True,use_global_semantic=False):
    if use_patch_selection:
        ratio=5
    else:
        ratio=1
    data = np.zeros((2*batch_size*ratio, 3, patch_size, patch_size), np.float32)
    label = np.zeros((2*batch_size*ratio, 3, patch_size, patch_size), np.float32)
    label_data_centerness = np.zeros((2*batch_size*ratio, 3, patch_size, patch_size), np.float32)
    patch_data = np.zeros((2*batch_size*ratio, 3, patch_size, patch_size), np.float32)
    patch_label = np.zeros((2*batch_size*ratio, 3, patch_size, patch_size), np.float32)
    total = 3
    
    for i in range(len(train_data)):
        
        
        random_sizes = []
        Img_len = train_data[i].shape[0]//3

        for j in range(batch_size):

            choice = np.random.randint(0, 6)
            random_size = np.random.randint(0, total)
            random_type = np.random.randint(0, 2)
            
            if use_patch_selection:
                while random_size in random_sizes:
                    random_size = np.random.randint(0, total)
                random_sizes.append(random_size)
                if len(random_sizes) == total:
                    random_sizes = []
                find_patch = False
                if random_size == 0:
                    # only for av together in the patch
                    while not find_patch:
                        z = np.random.randint(0, Img_len)
                        x, y, patch_size, find_patch,roi_area_y_left,roi_area_x_left,roi_area_y_right,roi_area_x_right = patch_select(ind=z, patch_size=patch_size,
                                                                    Mask=train_data_mask[i][z, :, :, :],
                                                                    Mask2 = train_data_mask2[i][z, 0, :, :],
                                                                    LabelA=train_label_data[i][z, 0, :, :],
                                                                    LabelV=train_label_data[i][z, 1, :, :],
                                                                    LabelVessel=train_label_data[i][z, 2, :, :], type=2)

                    patch_data_mat = train_data[i][z, :, x:x + patch_size, y:y + patch_size]
                    patch_label_mat = train_label_data[i][z, :, x:x + patch_size, y:y + patch_size]
                    label_centerness_mat = train_label_data_centerness[i][z, :, x:x + patch_size, y:y + patch_size]

                    if random_type==0:
                        av_ind = z+Img_len
                    else:
                        av_ind = z+2*Img_len
                    patch_data_mat_av = train_data[i][av_ind, :, x:x + patch_size, y:y + patch_size]
                    patch_label_mat_av = train_label_data[i][av_ind, :, x:x + patch_size, y:y + patch_size]
                    label_centerness_mat_av = train_label_data_centerness[i][av_ind, :, x:x + patch_size, y:y + patch_size]        
                         
                    label_high_mat = np.array([2])
                elif random_size == 1:
                    # only for a,v
                    while not find_patch:
                        z = np.random.randint(0, Img_len)
                        if random_type==0:
                            z = z+Img_len
                        else:
                            z = z+2*Img_len
                        x, y, patch_size1, find_patch,roi_area_y_left,roi_area_x_left,roi_area_y_right,roi_area_x_right = patch_select(ind=z, patch_size=patch_size1,
                                                                    Mask=train_data_mask[i][z, :, :, :],
                                                                    Mask2 = train_data_mask2[i][z, 0, :, :],
                                                                    LabelA=train_label_data[i][z, 0, :, :],
                                                                    LabelV=train_label_data[i][z, 1, :, :],
                                                                    LabelVessel=train_label_data[i][z, 2, :, :],
                                                                    type=random_type)
                    patch_data_mat = np.transpose(
                        cv2.resize(np.transpose(train_data[i][z, :, x:x + patch_size1, y:y + patch_size1], (1, 2, 0)),
                                (patch_size, patch_size)), (2, 0, 1))
                    patch_label_mat = np.transpose(
                        cv2.resize(np.transpose(train_label_data[i][z, :, x:x + patch_size1, y:y + patch_size1], (1, 2, 0)),
                                #(patch_size, patch_size)), (2, 0, 1))
                                (patch_size, patch_size),interpolation=cv2.INTER_NEAREST), (2, 0, 1))
                    label_centerness_mat = np.transpose(
                        cv2.resize(
                            np.transpose(train_label_data_centerness[i][z, :, x:x + patch_size1, y:y + patch_size1],
                                        (1, 2, 0)),
                            (patch_size, patch_size),interpolation=cv2.INTER_NEAREST), (2, 0, 1))
                    

                    if random_type==0:
                        av_ind = z-Img_len
                    else:
                        av_ind = z-2*Img_len 
                    patch_data_mat_av = np.transpose(
                        cv2.resize(np.transpose(train_data[i][av_ind, :, x:x + int(patch_size1//1.5), y:y + int(patch_size1//1.5)], (1, 2, 0)),
                                (patch_size, patch_size)), (2, 0, 1))
                    patch_label_mat_av = np.transpose(
                        cv2.resize(np.transpose(train_label_data[i][av_ind, :, x:x + int(patch_size1//1.5), y:y + int(patch_size1//1.5)], (1, 2, 0)),
                                #(patch_size, patch_size)), (2, 0, 1))
                                (patch_size, patch_size),interpolation=cv2.INTER_NEAREST), (2, 0, 1))
                    label_centerness_mat_av = np.transpose(
                        cv2.resize(
                            np.transpose(train_label_data_centerness[i][av_ind, :, x:x + int(patch_size1//1.5), y:y + int(patch_size1//1.5)],
                                        (1, 2, 0)),
                            (patch_size, patch_size),interpolation=cv2.INTER_NEAREST), (2, 0, 1))

                    label_high_mat = np.array([random_type])

                else:  # random
                    while not find_patch:
                        z = np.random.randint(0, Img_len)
                        x, y, patch_size2, find_patch,roi_area_y_left,roi_area_x_left,roi_area_y_right,roi_area_x_right = patch_select(ind=z, patch_size=patch_size2,
                                                                    Mask=train_data_mask[i][z, :, :, :],
                                                                    Mask2 = train_data_mask2[i][z, 0, :, :],
                                                                    LabelA=train_label_data[i][z, 0, :, :],
                                                                    LabelV=train_label_data[i][z, 1, :, :],
                                                                    LabelVessel=train_label_data[i][z, 2, :, :], type=2)
                    patch_data_mat = np.transpose(
                        cv2.resize(np.transpose(train_data[i][z, :, x:x + patch_size2, y:y + patch_size2], (1, 2, 0)),
                                (patch_size, patch_size)), (2, 0, 1))
                    patch_label_mat = np.transpose(
                        cv2.resize(np.transpose(train_label_data[i][z, :, x:x + patch_size2, y:y + patch_size2], (1, 2, 0)),
                                #(patch_size, patch_size)), (2, 0, 1))
                                (patch_size, patch_size),interpolation=cv2.INTER_NEAREST), (2, 0, 1))


                    #patch_data_mat_a,patch_data_mat_v,patch_label_mat_a,patch_label_mat_v = generate_patch_a_v(patch_data_mat,patch_label_mat)
                    label_centerness_mat = np.transpose(
                        cv2.resize(
                            np.transpose(train_label_data_centerness[i][z, :, x:x + patch_size2, y:y + patch_size2],
                                        (1, 2, 0)),
                            (patch_size, patch_size),interpolation=cv2.INTER_NEAREST), (2, 0, 1))
                    


                    if random_type==0:
                        av_ind = z+Img_len
                    else:
                        av_ind = z+2*Img_len
                    patch_data_mat_av = np.transpose(
                        cv2.resize(np.transpose(train_data[i][av_ind, :, x:x + patch_size2, y:y + patch_size2], (1, 2, 0)),
                                (patch_size, patch_size)), (2, 0, 1))
                    patch_label_mat_av = np.transpose(
                        cv2.resize(np.transpose(train_label_data[i][av_ind, :, x:x + patch_size2, y:y + patch_size2], (1, 2, 0)),
                                #(patch_size, patch_size)), (2, 0, 1))
                                (patch_size, patch_size),interpolation=cv2.INTER_NEAREST), (2, 0, 1))


                    #patch_data_mat_a,patch_data_mat_v,patch_label_mat_a,patch_label_mat_v = generate_patch_a_v(patch_data_mat,patch_label_mat)
                    label_centerness_mat_av = np.transpose(
                        cv2.resize(
                            np.transpose(train_label_data_centerness[i][av_ind, :, x:x + patch_size2, y:y + patch_size2],
                                        (1, 2, 0)),
                            (patch_size, patch_size),interpolation=cv2.INTER_NEAREST), (2, 0, 1))
                    
                    
                    label_high_mat = np.array([2])

                if use_global_semantic:
                    #roi_area_y_left,roi_area_x_left,roi_area_y_right,roi_area_x_right
                    #print(roi_area_y_left,roi_area_x_left,roi_area_y_right,roi_area_x_right)
                    #print(train_data.shape)
                    #print(train_data[z, roi_area_y_left:roi_area_y_right, roi_area_x_left:roi_area_x_right, :])
                    
                    
                    data_mat = np.transpose(
                        cv2.resize(np.transpose(train_data[i][z,:, roi_area_y_left:roi_area_y_right, roi_area_x_left:roi_area_x_right], (1, 2, 0)),
                                (patch_size, patch_size)), (2, 0, 1))
                    label_mat = np.transpose(
                        cv2.resize(np.transpose(train_label_data[i][z, :,roi_area_y_left:roi_area_y_right, roi_area_x_left:roi_area_x_right], (1, 2, 0)),
                                #(patch_size, patch_size)), (2, 0, 1))
                                (patch_size, patch_size), interpolation=cv2.INTER_NEAREST), (2, 0, 1))


                    if choice==6:
                        patch_data_mat,data_mat = data_aug5(patch_data_mat,data_mat,choice=6)
                    else:
                        patch_data_mat, patch_label_mat, label_centerness_mat, data_mat, label_mat = data_aug5(patch_data_mat,
                                                                                                        patch_label_mat,
                                                                                                        label_centerness_mat,
                                                                                                        data_mat,
                                                                                                        label_mat,
                                                                                                        
                                                                                                        choice=choice)

                else:
                    patch_data_mat = Image.fromarray(np.uint8(np.transpose(patch_data_mat,(1,2,0))*255))
                    patch_data_mat = transforms.RandomGrayscale(p=0.1)(patch_data_mat)

                    
                    patch_data_mat = np.transpose(np.array(patch_data_mat)/255.0,(2,0,1))
                    if choice==6:
                        patch_data_mat,_ = data_aug5(patch_data_mat,patch_data_mat,choice=6)
                    else:
                        patch_data_mat, patch_label_mat, label_centerness_mat,patch_data_mat_av, patch_label_mat_av,label_centerness_mat_av = data_aug5(patch_data_mat, patch_label_mat,label_centerness_mat,patch_data_mat_av, patch_label_mat_av,label_centerness_mat_av, choice=choice)
                
            else:
                #print(f'===================use_patch_selection: {use_patch_selection}========================')
                z = np.random.randint(0, train_data.shape[0])
                # x = np.random.randint(0, train_data.shape[2] - patch_size + 1)
                # y = np.random.randint(0, train_data.shape[3] - patch_size + 1)
                # patch_data_mat = train_data[z, :, x:x + patch_size, y:y + patch_size]
                # patch_label_mat = train_label_data[z, :, x:x + patch_size, y:y + patch_size]
                # label_centerness_mat = train_label_data_centerness[z, :, x:x + patch_size, y:y + patch_size]
                # patch_data_mat, patch_label_mat, label_centerness_mat = data_aug5(patch_data_mat, patch_label_mat,
                #                                                                     label_centerness_mat, choice=choice)
                if random_size == 0:
                    x = np.random.randint(0, train_data.shape[2] - patch_size + 1)
                    y = np.random.randint(0, train_data.shape[3] - patch_size + 1)
                    patch_data_mat = train_data[z, :, x:x + patch_size, y:y + patch_size]
                    patch_label_mat = train_label_data[z, :, x:x + patch_size, y:y + patch_size]
                    label_centerness_mat = train_label_data_centerness[z, :, x:x + patch_size, y:y + patch_size]

                elif random_size == 1:
                    x = np.random.randint(0, train_data.shape[2] - patch_size1 + 1)
                    y = np.random.randint(0, train_data.shape[3] - patch_size1 + 1)
                    patch_data_mat = np.transpose(
                        cv2.resize(np.transpose(train_data[z, :, x:x + patch_size1, y:y + patch_size1], (1, 2, 0)),
                                (patch_size, patch_size)), (2, 0, 1))
                    patch_label_mat = np.transpose(
                        cv2.resize(np.transpose(train_label_data[z, :, x:x + patch_size1, y:y + patch_size1], (1, 2, 0)),
                                #(patch_size, patch_size)), (2, 0, 1))
                                (patch_size, patch_size),interpolation=cv2.INTER_NEAREST), (2, 0, 1))
                    label_centerness_mat = np.transpose(
                        cv2.resize(np.transpose(train_label_data_centerness[z, :, x:x + patch_size1, y:y + patch_size1], (1, 2, 0)),
                                (patch_size, patch_size),interpolation=cv2.INTER_NEAREST), (2, 0, 1))
                else:
                    x = np.random.randint(0, train_data.shape[2] - patch_size2 + 1)
                    y = np.random.randint(0, train_data.shape[3] - patch_size2 + 1)
                    patch_data_mat = np.transpose(
                        cv2.resize(np.transpose(train_data[z, :, x:x + patch_size2, y:y + patch_size2], (1, 2, 0)),
                                (patch_size, patch_size)), (2, 0, 1))
                    patch_label_mat = np.transpose(
                        cv2.resize(np.transpose(train_label_data[z, :, x:x + patch_size2, y:y + patch_size2], (1, 2, 0)),
                                #(patch_size, patch_size)), (2, 0, 1))
                                (patch_size, patch_size),interpolation=cv2.INTER_NEAREST), (2, 0, 1))
                    label_centerness_mat = np.transpose(
                        cv2.resize(np.transpose(train_label_data_centerness[z, :, x:x + patch_size2, y:y + patch_size2], (1, 2, 0)),
                                (patch_size, patch_size),interpolation=cv2.INTER_NEAREST), (2, 0, 1))


                patch_data_mat, patch_label_mat, label_centerness_mat = data_aug5(patch_data_mat, patch_label_mat,
                                                                                    label_centerness_mat, choice=choice)

            #print(f'=========random_size: {random_size}=========={random_type}')
            patch_data[j+i*batch_size, :, :, :] = patch_data_mat
            patch_label[j+i*batch_size, :, :, :] = patch_label_mat
            label_data_centerness[j+i*batch_size, :, :, :] = label_centerness_mat
            
            patch_data[j+i*batch_size+batch_size*ratio, :, :, :] = patch_data_mat_av
            patch_label[j+i*batch_size+batch_size*ratio, :, :, :] = patch_label_mat_av
            label_data_centerness[j+i*batch_size+batch_size*ratio, :, :, :] = label_centerness_mat_av
            
            if use_global_semantic:
                data[j+i*batch_size, :, :, :] = data_mat
                label[j+i*batch_size, :, :, :] = label_mat
                
    # np.save('test_data.npy',patch_data)
    # np.save('test_label.npy',patch_label)
    # np.save('test_label_cen.npy',label_data_centerness)            
    patch_data = Normalize(patch_data)
    if use_global_semantic:
        data = Normalize(data)
        return patch_data, patch_label, label_data_centerness, data, label
    else:
        return patch_data, patch_label, label_data_centerness,patch_data, patch_label

def get_patch_trad_5_no_av(batch_size, patch_size, train_data=None, train_label_data=None, train_label_data_centerness=None,
                     train_data_mask=None,train_data_mask2=None, train_patch_data=None, train_patch_label=None, patch_size1=96,
                     patch_size2=128, use_patch_selection=True,use_global_semantic=False):
    data = np.zeros((batch_size, 3, patch_size, patch_size), np.float32)
    label = np.zeros((batch_size, 3, patch_size, patch_size), np.float32)
    label_data_centerness = np.zeros((batch_size, 3, patch_size, patch_size), np.float32)
    patch_data = np.zeros((batch_size, 3, patch_size, patch_size), np.float32)
    patch_label = np.zeros((batch_size, 3, patch_size, patch_size), np.float32)
    total = 3
    random_sizes = []

    #train_data = train_data.copy()
    rate = np.random.randint(0,11)/10
    ind = np.random.randint(0,5)

    # for k in range(train_data.shape[0]):
    #     train_data[k,:,:,:] = CAM(train_data[k,:,:,:].copy(),rate=rate,ind=ind)
    #train_data = Normalize(train_data)
    for j in range(batch_size):

        choice = np.random.randint(0, 6)
        random_size = np.random.randint(0, total)
        random_type = np.random.randint(0, 2)

        if use_patch_selection:
            while random_size in random_sizes:
                random_size = np.random.randint(0, total)
            random_sizes.append(random_size)
            if len(random_sizes) == total:
                random_sizes = []
            find_patch = False
            if random_size == 0:
                # only for av together in the patch
                while not find_patch:
                    z = np.random.randint(0, train_data.shape[0])
                    x, y, patch_size, find_patch,roi_area_y_left,roi_area_x_left,roi_area_y_right,roi_area_x_right = patch_select(ind=z, patch_size=patch_size,
                                                                Mask=train_data_mask[z, :, :, :],
                                                                Mask2 = train_data_mask2[z, 0, :, :],
                                                                LabelA=train_label_data[z, 0, :, :],
                                                                LabelV=train_label_data[z, 1, :, :],
                                                                LabelVessel=train_label_data[z, 2, :, :], type=2)

                patch_data_mat = train_data[z, :, x:x + patch_size, y:y + patch_size]
                patch_label_mat = train_label_data[z, :, x:x + patch_size, y:y + patch_size]

                #patch_data_mat_a,patch_data_mat_v,patch_label_mat_a,patch_label_mat_v = generate_patch_a_v(patch_data_mat,patch_label_mat)

                label_centerness_mat = train_label_data_centerness[z, :, x:x + patch_size, y:y + patch_size]
                label_high_mat = np.array([2])
            elif random_size == 1:
                # only for a,v
                while not find_patch:
                    z = np.random.randint(0, train_data.shape[0])
                    x, y, patch_size1, find_patch,roi_area_y_left,roi_area_x_left,roi_area_y_right,roi_area_x_right = patch_select(ind=z, patch_size=patch_size1,
                                                                 Mask=train_data_mask[z, :, :, :],
                                                                 Mask2 = train_data_mask2[z, 0, :, :],
                                                                 LabelA=train_label_data[z, 0, :, :],
                                                                 LabelV=train_label_data[z, 1, :, :],
                                                                 LabelVessel=train_label_data[z, 2, :, :],
                                                                 type=random_type)
                patch_data_mat = np.transpose(
                    cv2.resize(np.transpose(train_data[z, :, x:x + patch_size1, y:y + patch_size1], (1, 2, 0)),
                               (patch_size, patch_size)), (2, 0, 1))
                patch_label_mat = np.transpose(
                    cv2.resize(np.transpose(train_label_data[z, :, x:x + patch_size1, y:y + patch_size1], (1, 2, 0)),
                               #(patch_size, patch_size)), (2, 0, 1))
                               (patch_size, patch_size),interpolation=cv2.INTER_NEAREST), (2, 0, 1))
                label_centerness_mat = np.transpose(
                    cv2.resize(
                        np.transpose(train_label_data_centerness[z, :, x:x + patch_size1, y:y + patch_size1],
                                     (1, 2, 0)),
                        (patch_size, patch_size),interpolation=cv2.INTER_NEAREST), (2, 0, 1))
                label_high_mat = np.array([random_type])

            else:  # random
                while not find_patch:
                    z = np.random.randint(0, train_data.shape[0])
                    x, y, patch_size2, find_patch,roi_area_y_left,roi_area_x_left,roi_area_y_right,roi_area_x_right = patch_select(ind=z, patch_size=patch_size2,
                                                                 Mask=train_data_mask[z, :, :, :],
                                                                 Mask2 = train_data_mask2[z, 0, :, :],
                                                                 LabelA=train_label_data[z, 0, :, :],
                                                                 LabelV=train_label_data[z, 1, :, :],
                                                                 LabelVessel=train_label_data[z, 2, :, :], type=2)
                patch_data_mat = np.transpose(
                    cv2.resize(np.transpose(train_data[z, :, x:x + patch_size2, y:y + patch_size2], (1, 2, 0)),
                               (patch_size, patch_size)), (2, 0, 1))
                patch_label_mat = np.transpose(
                    cv2.resize(np.transpose(train_label_data[z, :, x:x + patch_size2, y:y + patch_size2], (1, 2, 0)),
                               #(patch_size, patch_size)), (2, 0, 1))
                               (patch_size, patch_size),interpolation=cv2.INTER_NEAREST), (2, 0, 1))


                #patch_data_mat_a,patch_data_mat_v,patch_label_mat_a,patch_label_mat_v = generate_patch_a_v(patch_data_mat,patch_label_mat)
                label_centerness_mat = np.transpose(
                    cv2.resize(
                        np.transpose(train_label_data_centerness[z, :, x:x + patch_size2, y:y + patch_size2],
                                     (1, 2, 0)),
                        (patch_size, patch_size),interpolation=cv2.INTER_NEAREST), (2, 0, 1))
                label_high_mat = np.array([2])

            if use_global_semantic:
                #roi_area_y_left,roi_area_x_left,roi_area_y_right,roi_area_x_right
                #print(roi_area_y_left,roi_area_x_left,roi_area_y_right,roi_area_x_right)
                #print(train_data.shape)
                #print(train_data[z, roi_area_y_left:roi_area_y_right, roi_area_x_left:roi_area_x_right, :])
                data_mat = np.transpose(
                    cv2.resize(np.transpose(train_data[z,:, roi_area_y_left:roi_area_y_right, roi_area_x_left:roi_area_x_right], (1, 2, 0)),
                            (patch_size, patch_size)), (2, 0, 1))
                label_mat = np.transpose(
                    cv2.resize(np.transpose(train_label_data[z, :,roi_area_y_left:roi_area_y_right, roi_area_x_left:roi_area_x_right], (1, 2, 0)),
                            #(patch_size, patch_size)), (2, 0, 1))
                            (patch_size, patch_size), interpolation=cv2.INTER_NEAREST), (2, 0, 1))

                if choice==6:
                    patch_data_mat,data_mat = data_aug5(patch_data_mat,data_mat,choice=6)
                else:
                    patch_data_mat, patch_label_mat, label_centerness_mat, data_mat, label_mat = data_aug5(patch_data_mat,
                                                                                                    patch_label_mat,
                                                                                                    label_centerness_mat,
                                                                                                    data_mat,
                                                                                                    label_mat,
                                                                                                    choice=choice)

            else:
                if choice==6:
                    patch_data_mat,_ = data_aug5(patch_data_mat,patch_data_mat,choice=6)
                else:
                    patch_data_mat, patch_label_mat, label_centerness_mat = data_aug5(patch_data_mat, patch_label_mat,
                                                                                label_centerness_mat, choice=choice)
            
        else:
            #print(f'===================use_patch_selection: {use_patch_selection}========================')
            z = np.random.randint(0, train_data.shape[0])
            # x = np.random.randint(0, train_data.shape[2] - patch_size + 1)
            # y = np.random.randint(0, train_data.shape[3] - patch_size + 1)
            # patch_data_mat = train_data[z, :, x:x + patch_size, y:y + patch_size]
            # patch_label_mat = train_label_data[z, :, x:x + patch_size, y:y + patch_size]
            # label_centerness_mat = train_label_data_centerness[z, :, x:x + patch_size, y:y + patch_size]
            # patch_data_mat, patch_label_mat, label_centerness_mat = data_aug5(patch_data_mat, patch_label_mat,
            #                                                                     label_centerness_mat, choice=choice)
            if random_size == 0:
                x = np.random.randint(0, train_data.shape[2] - patch_size + 1)
                y = np.random.randint(0, train_data.shape[3] - patch_size + 1)
                patch_data_mat = train_data[z, :, x:x + patch_size, y:y + patch_size]
                patch_label_mat = train_label_data[z, :, x:x + patch_size, y:y + patch_size]
                label_centerness_mat = train_label_data_centerness[z, :, x:x + patch_size, y:y + patch_size]

            elif random_size == 1:
                x = np.random.randint(0, train_data.shape[2] - patch_size1 + 1)
                y = np.random.randint(0, train_data.shape[3] - patch_size1 + 1)
                patch_data_mat = np.transpose(
                    cv2.resize(np.transpose(train_data[z, :, x:x + patch_size1, y:y + patch_size1], (1, 2, 0)),
                               (patch_size, patch_size)), (2, 0, 1))
                patch_label_mat = np.transpose(
                    cv2.resize(np.transpose(train_label_data[z, :, x:x + patch_size1, y:y + patch_size1], (1, 2, 0)),
                               #(patch_size, patch_size)), (2, 0, 1))
                               (patch_size, patch_size),interpolation=cv2.INTER_NEAREST), (2, 0, 1))
                label_centerness_mat = np.transpose(
                    cv2.resize(np.transpose(train_label_data_centerness[z, :, x:x + patch_size1, y:y + patch_size1], (1, 2, 0)),
                               (patch_size, patch_size),interpolation=cv2.INTER_NEAREST), (2, 0, 1))
            else:
                x = np.random.randint(0, train_data.shape[2] - patch_size2 + 1)
                y = np.random.randint(0, train_data.shape[3] - patch_size2 + 1)
                patch_data_mat = np.transpose(
                    cv2.resize(np.transpose(train_data[z, :, x:x + patch_size2, y:y + patch_size2], (1, 2, 0)),
                               (patch_size, patch_size)), (2, 0, 1))
                patch_label_mat = np.transpose(
                    cv2.resize(np.transpose(train_label_data[z, :, x:x + patch_size2, y:y + patch_size2], (1, 2, 0)),
                               #(patch_size, patch_size)), (2, 0, 1))
                               (patch_size, patch_size),interpolation=cv2.INTER_NEAREST), (2, 0, 1))
                label_centerness_mat = np.transpose(
                    cv2.resize(np.transpose(train_label_data_centerness[z, :, x:x + patch_size2, y:y + patch_size2], (1, 2, 0)),
                               (patch_size, patch_size),interpolation=cv2.INTER_NEAREST), (2, 0, 1))


            patch_data_mat, patch_label_mat, label_centerness_mat = data_aug5(patch_data_mat, patch_label_mat,
                                                                                 label_centerness_mat, choice=choice)


        patch_data[j, :, :, :] = patch_data_mat
        patch_label[j, :, :, :] = patch_label_mat

        
        #patch_data = Normalize(patch_data)
        label_data_centerness[j, :, :, :] = label_centerness_mat
        if use_global_semantic:
            data[j, :, :, :] = data_mat
            label[j, :, :, :] = label_mat
            #data = Normalize(data)
    if use_global_semantic:
        return patch_data, patch_label, label_data_centerness, data, label
    else:
        return patch_data, patch_label, label_data_centerness,patch_data, patch_label
import numpy as np
import albumentations as albu
from PIL import Image


def Dataloader_general(path, use_centermap=False,use_resize=False,resize_w_h=(256,256),use_global_semantic = False):
    use_resize = False
    ImgPath = path + "images/"
    LabelPath = path + "av/"
    # PatchImagePath = path + "image_patch/"
    # PatchLabelPath = path + "av_patch/"

    if use_resize:
        resize_w_h = resize_w_h
        #resize_w_h=(256,256)

    ImgDir = os.listdir(ImgPath)
    Img_len = len(ImgDir)
    LabelDir = os.listdir(LabelPath)

    Img0 = cv2.imread(ImgPath + ImgDir[0])
    Label0 = cv2.imread(LabelPath + LabelDir[0])
    image_suffix = os.path.splitext(ImgDir[0])[1]
    label_suffix = os.path.splitext(LabelDir[0])[1]
    if use_resize:
        Img0 = cv2.resize(Img0, resize_w_h)
        Label0 = cv2.resize(Label0, resize_w_h)

    Img = np.zeros((len(ImgDir)*3, 3, Img0.shape[0], Img0.shape[1]), np.float32)
    Label = np.zeros((len(ImgDir)*3, 3, Img0.shape[0], Img0.shape[1]), np.float32)
    Label_center = np.zeros((len(ImgDir)*3, 3, Img0.shape[0], Img0.shape[1]), np.float32)
    Label_ske = np.zeros((len(ImgDir)*3, 3, Img0.shape[0], Img0.shape[1]), np.float32)
    Masks = np.zeros((len(ImgDir)*3, 1, Img0.shape[0], Img0.shape[1]), np.float32)
    # Label_patch = np.zeros((len(PatchImageDir), 3, Img0.shape[0], Img0.shape[1]), np.float32)
    # Img_patch = np.zeros((len(PatchImageDir), 3, Img0.shape[0], Img0.shape[1]), np.float32)

    for i,name in enumerate(ImgDir):


        prefix = os.path.splitext(name)[0]

        LabelArtery = np.zeros((Img0.shape[0], Img0.shape[1]), np.uint8)
        LabelVein = np.zeros((Img0.shape[0], Img0.shape[1]), np.uint8)
        LabelVessel = np.zeros((Img0.shape[0], Img0.shape[1]), np.uint8)
        Mask = np.ones((Img0.shape[0], Img0.shape[1]), np.uint8)

        Img0 = cv2.imread(os.path.join(ImgPath, f'{prefix}{image_suffix}'))
        Label0 = cv2.imread(os.path.join(LabelPath,f'{prefix}{label_suffix}'))

        if use_resize:
            Img0 = cv2.resize(Img0, resize_w_h)
            Label0 = cv2.resize(Label0, resize_w_h, interpolation=cv2.INTER_NEAREST)

        # BGR
        LabelArtery[(Label0[:, :, 2] == 255) | (Label0[:, :, 1] == 255)] = 1
        LabelArtery[(Label0[:, :, 2] == 255) & (Label0[:, :, 1] == 255) & (Label0[:, :, 0] == 255)] = 0

        LabelVein[(Label0[:, :, 1] == 255) | (Label0[:, :, 0] == 255)] = 1
        LabelVein[(Label0[:, :, 2] == 255) & (Label0[:, :, 1] == 255) & (Label0[:, :, 0] == 255)] = 0

        LabelVessel[(Label0[:, :, 2] == 255) | (Label0[:, :, 1] == 255) | (Label0[:, :, 0] == 255)] = 1
        #LabelVessel[(Label0[:, :, 2] == 255) & (Label0[:, :, 1] == 255) & (Label0[:, :, 0] == 255)] = 0
        skel_V =morphology.medial_axis(LabelVein)
        skel_A =morphology.medial_axis(LabelArtery)
        #skel =morphology.medial_axis(LabelVessel)
        #LabelVessel[(Label0[:, :, 2] == 255) & (Label0[:, :, 1] == 255) & (Label0[:, :, 0] == 255)] = 0

        _, Mask0 = creatMask(Img0, threshold=-1)
        Mask[Mask0 > 0] = 1



        Label[i, 0, :, :] = LabelArtery
        Label[i, 1, :, :] = LabelVein
        Label[i, 2, :, :] = LabelVessel
        ImgCropped = cv2.cvtColor(Img0, cv2.COLOR_BGR2RGB)
        Image_nv,Image_na, av_nv,av_na = generate_a_v(ImgCropped,Label[i])


        if use_centermap:
            Labelcenter_artery = LabelArtery.copy()
            Labelcenter_vein = LabelVein.copy()
            Labelcenter_vessel = LabelVessel.copy()

            center_img_a = ndimage.distance_transform_edt(Labelcenter_artery)
            max_dis_a = center_img_a.max()
            min_dis_a = center_img_a.min()
            if max_dis_a!=0 or min_dis_a!=0:
                center_img_a = (center_img_a - min_dis_a) / (max_dis_a - min_dis_a)

            center_img_v = ndimage.distance_transform_edt(Labelcenter_vein)
            max_dis_v = center_img_v.max()
            min_dis_v = center_img_v.min()
            if max_dis_v!=0 or min_dis_v!=0:
                center_img_v = (center_img_v - min_dis_v) / (max_dis_v - min_dis_v)

            center_img_vessel = ndimage.distance_transform_edt(Labelcenter_vessel)
            max_dis_vessel = center_img_vessel.max()
            min_dis_vessel = center_img_vessel.min()
            if max_dis_vessel!=0 or min_dis_vessel!=0:
                center_img_vessel = (center_img_vessel - min_dis_vessel) / (max_dis_vessel - min_dis_vessel)

            Label_center[i, 0, :, :] = center_img_a
            Label_center[i, 1, :, :] = center_img_v
            Label_center[i, 2, :, :] = center_img_vessel


            Labelcenter_artery = av_nv[0].copy()
            Labelcenter_vein = av_nv[1].copy()
            Labelcenter_vessel = av_nv[2].copy()
            center_img_a = ndimage.distance_transform_edt(Labelcenter_artery)
            max_dis_a = center_img_a.max()
            min_dis_a = center_img_a.min()
            if max_dis_a!=0 or min_dis_a!=0:
                center_img_a = (center_img_a - min_dis_a) / (max_dis_a - min_dis_a)
            Label_center[i+Img_len, 0, :, :] = center_img_a
            Label_center[i+Img_len, 2, :, :] = center_img_a


            Labelcenter_artery = av_na[0].copy()
            Labelcenter_vein = av_na[1].copy()
            Labelcenter_vessel = av_na[2].copy()
            center_img_v = ndimage.distance_transform_edt(Labelcenter_vein)
            max_dis_v = center_img_v.max()
            min_dis_v = center_img_v.min()
            if max_dis_v!=0 or min_dis_v!=0:
                center_img_v = (center_img_v - min_dis_v) / (max_dis_v - min_dis_v)
            Label_center[i+2*Img_len, 1, :, :] = center_img_v
            Label_center[i+2*Img_len, 2, :, :] = center_img_v
            
            
        skel = morphology.medial_axis(LabelVessel)
        Label_ske[i, 0, :, :] = skel_A
        Label_ske[i, 1, :, :] = skel_V
        Label_ske[i, 2, :, :] = skel


        #ImgCropped = np.float32(ImgCropped / 255.)

        

        ImgCropped = np.float32(ImgCropped / 255.)
        Image_nv = np.float32(Image_nv / 255.)
        Image_na = np.float32(Image_na / 255.)


        Img[i, :, :, :] = np.transpose(ImgCropped, (2, 0, 1)) # HWC to CHW
        Img[i+Img_len,:,:,:] = np.transpose(Image_nv, (2, 0, 1))
        Img[i+2*Img_len,:,:,:] = np.transpose(Image_na, (2, 0, 1))

        Label[i+Img_len] = av_nv
        Label[i+2*Img_len] = av_na

        Label_ske[i+Img_len,0] = skel_A
        Label_ske[i+Img_len,2] = skel_A

        Label_ske[i+2*Img_len,1] = skel_V
        Label_ske[i+2*Img_len,2] = skel_V
        

        Masks[i, 0, :, :] = Mask
        Masks[i+Img_len, 0, :, :] = Mask
        Masks[i+2*Img_len, 0, :, :] = Mask
        #LabelVessel[(Label0[:, :, 2] == 255) & (Label0[:, :, 1] == 255) & (Label0[:, :, 0] == 255)] = 0
        
        
    # np.save('test_data.npy',Img)
    # np.save('test_label.npy',Label)
    # np.save('test_label_cen.npy',Label_center)      


    #Img = Normalize(Img)


    return Img, Label, Label_center,Label_ske,Masks
def Dataloader_general_no_av(path, use_centermap=False,use_resize=False,resize_w_h=(256,256),use_global_semantic = False):
    use_resize = False
    ImgPath = path + "images/"
    LabelPath = path + "av/"
    centPath = path+"centerness_maps"
    # PatchImagePath = path + "image_patch/"
    # PatchLabelPath = path + "av_patch/"

    if use_resize:
        resize_w_h = resize_w_h

    ImgDir = os.listdir(ImgPath)
    LabelDir = os.listdir(LabelPath)

    Img0 = cv2.imread(ImgPath + ImgDir[0])
    Label0 = cv2.imread(LabelPath + LabelDir[0])
    image_suffix = os.path.splitext(ImgDir[0])[1]
    label_suffix = os.path.splitext(LabelDir[0])[1]
    if use_resize:
        Img0 = cv2.resize(Img0, resize_w_h)
        Label0 = cv2.resize(Label0, resize_w_h)

    Img = np.zeros((len(ImgDir), 3, Img0.shape[0], Img0.shape[1]), np.float32)
    Label = np.zeros((len(ImgDir), 3, Img0.shape[0], Img0.shape[1]), np.float32)
    Label_center = np.zeros((len(ImgDir), 3, Img0.shape[0], Img0.shape[1]), np.float32)
    Label_ske = np.zeros((len(ImgDir), 3, Img0.shape[0], Img0.shape[1]), np.float32)
    Masks = np.zeros((len(ImgDir), 1, Img0.shape[0], Img0.shape[1]), np.float32)
    # Label_patch = np.zeros((len(PatchImageDir), 3, Img0.shape[0], Img0.shape[1]), np.float32)
    # Img_patch = np.zeros((len(PatchImageDir), 3, Img0.shape[0], Img0.shape[1]), np.float32)

    for i,name in enumerate(ImgDir):


        prefix = os.path.splitext(name)[0]

        LabelArtery = np.zeros((Img0.shape[0], Img0.shape[1]), np.uint8)
        LabelVein = np.zeros((Img0.shape[0], Img0.shape[1]), np.uint8)
        LabelVessel = np.zeros((Img0.shape[0], Img0.shape[1]), np.uint8)
        Mask = np.zeros((Img0.shape[0], Img0.shape[1]), np.uint8)

        Img0 = cv2.imread(os.path.join(ImgPath, f'{prefix}{image_suffix}'))
        Label0 = cv2.imread(os.path.join(LabelPath,f'{prefix}{label_suffix}'))

        if use_resize:
            Img0 = cv2.resize(Img0, resize_w_h)
            Label0 = cv2.resize(Label0, resize_w_h, interpolation=cv2.INTER_NEAREST)

        # BGR
        LabelArtery[(Label0[:, :, 2] == 255) | (Label0[:, :, 1] == 255)] = 1
        LabelArtery[(Label0[:, :, 2] == 255) & (Label0[:, :, 1] == 255) & (Label0[:, :, 0] == 255)] = 0

        LabelVein[(Label0[:, :, 1] == 255) | (Label0[:, :, 0] == 255)] = 1
        LabelVein[(Label0[:, :, 2] == 255) & (Label0[:, :, 1] == 255) & (Label0[:, :, 0] == 255)] = 0

        LabelVessel[(Label0[:, :, 2] == 255) | (Label0[:, :, 1] == 255) | (Label0[:, :, 0] == 255)] = 1
        #LabelVessel[(Label0[:, :, 2] == 255) & (Label0[:, :, 1] == 255) & (Label0[:, :, 0] == 255)] = 0
        skel_V =morphology.medial_axis(LabelVein)
        skel_A =morphology.medial_axis(LabelArtery)
        #skel =morphology.medial_axis(LabelVessel)
        #LabelVessel[(Label0[:, :, 2] == 255) & (Label0[:, :, 1] == 255) & (Label0[:, :, 0] == 255)] = 0

        _, Mask0 = creatMask(Img0, threshold=-1)
        Mask[Mask0 > 0] = 1
        if use_centermap:
            Labelcenter_artery = cv2.imread(os.path.join(centPath,f'{prefix}_art_cen{label_suffix}') ,cv2.IMREAD_GRAYSCALE)
            Labelcenter_vein = cv2.imread(
                os.path.join(centPath, f'{prefix}_ven_cen{label_suffix}'), cv2.IMREAD_GRAYSCALE)
            Labelcenter_vessel = cv2.imread(
                os.path.join(centPath, f'{prefix}_ves_cen{label_suffix}'), cv2.IMREAD_GRAYSCALE)

            if use_resize:
                #print('=======resize==========')
                Labelcenter_artery = cv2.resize(Labelcenter_artery, resize_w_h,interpolation=cv2.INTER_NEAREST)
                Labelcenter_vein = cv2.resize(Labelcenter_vein, resize_w_h,interpolation=cv2.INTER_NEAREST)
                Labelcenter_vessel = cv2.resize(Labelcenter_vessel,resize_w_h,interpolation=cv2.INTER_NEAREST)
            #print(Labelcenter_artery.shape)
            Labelcenter_artery = Labelcenter_artery / 255.0
            Labelcenter_vein = Labelcenter_vein / 255.0
            Labelcenter_vessel = Labelcenter_vessel / 255.0

            Label_center[i, 0, :, :] = Labelcenter_artery
            Label_center[i, 1, :, :] = Labelcenter_vein
            Label_center[i, 2, :, :] = Labelcenter_vessel


        ImgCropped = cv2.cvtColor(Img0, cv2.COLOR_BGR2RGB)


          

        ImgCropped = np.float32(ImgCropped / 255.)


        Img[i, :, :, :] = np.transpose(ImgCropped, (2, 0, 1)) # HWC to CHW


        Label[i, 0, :, :] = LabelArtery
        Label[i, 1, :, :] = LabelVein
        Label[i, 2, :, :] = LabelVessel
        Masks[i, 0, :, :] = Mask
        #LabelVessel[(Label0[:, :, 2] == 255) & (Label0[:, :, 1] == 255) & (Label0[:, :, 0] == 255)] = 0
        skel = morphology.medial_axis(LabelVessel)
        Label_ske[i, 0, :, :] = skel_A
        Label_ske[i, 1, :, :] = skel_V
        Label_ske[i, 2, :, :] = skel
        


    Img = Normalize(Img)


    return Img, Label, Label_center,Label_ske,Masks

def modelEvalution(i, net, savePath, use_cuda=False, dataset='DRIVE', is_kill_border=True, input_ch=3, strict_mode=True,
                   config=None):
    # path for images to save
    dataset_dict = {'STU_patch':'STU_patch', 'ukbb':'ukbb','LES': 'LES_AV', 'DRIVE': 'AV_DRIVE', 'hrf': 'hrf','STU':'STU'}
    dataset_name = dataset_dict[dataset]

    image_basename = sorted(os.listdir(f'./data/{dataset_name}/test/images'))
    label_basename = sorted(os.listdir(f'./data/{dataset_name}/test/av'))
    assert len(image_basename) == len(label_basename)

    image0 = cv2.imread(f'./data/{dataset_name}/test/images/{image_basename[0]}')

    data_path = os.path.join(savePath, dataset)
    metrics_file_path = os.path.join(savePath, 'metrics.txt')  # _'+str(config.model_step_pretrained_G)+'.txt')
    if not os.path.exists(data_path):
        os.mkdir(data_path)

    test_image_num = len(image_basename)
    test_image_height = image0.shape[0]
    test_image_width = image0.shape[1]
    if config.use_resize:
        test_image_height = config.resize_w_h[1]
        test_image_width = config.resize_w_h[0]
    ArteryPredAll = np.zeros((test_image_num, 1, test_image_height, test_image_width), np.float32)
    VeinPredAll = np.zeros((test_image_num, 1, test_image_height, test_image_width), np.float32)
    VesselPredAll = np.zeros((test_image_num, 1, test_image_height, test_image_width), np.float32)
    LabelArteryAll = np.zeros((test_image_num, 1, test_image_height, test_image_width), np.float32)
    LabelVeinAll = np.zeros((test_image_num, 1, test_image_height, test_image_width), np.float32)
    LabelVesselAll = np.zeros((test_image_num, 1, test_image_height, test_image_width), np.float32)
    LabelVesselNoAll = np.zeros((test_image_num, 1, test_image_height, test_image_width), np.float32)

    ProMap = np.zeros((test_image_num, 3, test_image_height, test_image_width), np.float32)
    LabelMap = np.zeros((test_image_num, 3, test_image_height, test_image_width), np.float32)
    MaskAll = np.zeros((test_image_num, 1, test_image_height, test_image_width), np.float32)

    # Vessel = VesselProMap('./data/AV_DRIVE/test/images')

    n_classes = 3
    if config.use_global_semantic:
        Net = PGNet(resnet=config.use_network, input_ch=input_ch, num_classes=n_classes, use_cuda=use_cuda,
                    pretrained=False, centerness=config.use_centerness, centerness_map_size=config.centerness_map_size,
                    use_global_semantic=config.use_global_semantic)
    else:
        Net = PGNet(resnet=config.use_network, input_ch=input_ch, num_classes=n_classes, use_cuda=use_cuda,
                    pretrained=False, centerness=config.use_centerness, centerness_map_size=config.centerness_map_size)

    Net.load_state_dict(net)

    if use_cuda:
        Net.cuda()
    Net.eval()

    for k in tqdm(range(test_image_num)):
        ArteryPred, VeinPred, VesselPred, LabelArtery, LabelVein, LabelVessel, Mask, LabelVessel_no_unknow = GetResult_big(
            Net, k,
            use_cuda=use_cuda,
            dataset_name=dataset_name,
            is_kill_border=is_kill_border,
            input_ch=input_ch,
            config=config)
        ArteryPredAll[k, :, :, :] = ArteryPred
        VeinPredAll[k, :, :, :] = VeinPred
        VesselPredAll[k, :, :, :] = VesselPred
        LabelArteryAll[k, :, :, :] = LabelArtery
        LabelVeinAll[k, :, :, :] = LabelVein
        LabelVesselAll[k, :, :, :] = LabelVessel
        LabelVesselNoAll[k, :, :, :] = LabelVessel_no_unknow
        MaskAll[k, :, :, :] = Mask

    ProMap[:, 0, :, :] = ArteryPredAll[:, 0, :, :]
    ProMap[:, 1, :, :] = VeinPredAll[:, 0, :, :]
    ProMap[:, 2, :, :] = VesselPredAll[:, 0, :, :]
    LabelMap[:, 0, :, :] = LabelArteryAll[:, 0, :, :]
    LabelMap[:, 1, :, :] = LabelVeinAll[:, 0, :, :]
    LabelMap[:, 2, :, :] = LabelVesselAll[:, 0, :, :]
    
    VesselAUC, VesselAcc, VesselSp, VesselSe, VesselF1, VesselDice, VesselIou = evalue(VesselPredAll,
                                                                                                LabelVesselAll, MaskAll)
    # filewriter = centerline_eval(ProMap, config)
    #np.save(os.path.join(savePath, "ProMap_testset.npy"), ProMap)
    #np.save(os.path.join(savePath, "LabelMap_testset.npy"), LabelMap)
    #np.save(os.path.join(savePath, "MaskAll_testset.npy"), MaskAll)
    # ArteryAUC,AveAcc,VeinAcc,ArteryAcc,AveF1,AveDice,AveIou,VeinAUC = Evalution_AV_skeletonPixles(ArteryPredAll,VeinPredAll,VesselPredAll,LabelArteryAll,LabelVeinAll,LabelVesselNoAll,MaskAll,test_image_num,strict_mode=strict_mode)
    ArteryAUC, AveAcc, VeinAcc, ArteryAcc, AveF1, AveDice, AveIou, VeinAUC = Evalution_AV_skeletonPixles(ArteryPredAll,
                                                                                                         VeinPredAll,
                                                                                                         VesselPredAll,
                                                                                                         LabelArteryAll,
                                                                                                         LabelVeinAll,
                                                                                                         LabelVesselAll,
                                                                                                         MaskAll,
                                                                                                         test_image_num,
                                                                                                         strict_mode=strict_mode)

    for k in range(0, test_image_num):
        cv2.imwrite(os.path.join(data_path, f"{dataset}_Artery" + str(k).zfill(3) + ".png"),
                    ArteryPredAll[k, 0, :, :] * 255)
        cv2.imwrite(os.path.join(data_path, f"{dataset}_Vein" + str(k).zfill(3) + ".png"),
                    VeinPredAll[k, 0, :, :] * 255)
        cv2.imwrite(os.path.join(data_path, f"{dataset}_Vessel" + str(k).zfill(3) + ".png"),
                    VesselPredAll[k, 0, :, :] * 255)

    print(f"========================={dataset}=============================")
    print("Strict mode:{}".format(strict_mode))
    print(f"The {i} step Average Acc is:{AveAcc}")
    print(f"The {i} step Average F1 is:{AveF1}")
    print(f"The {i} step Average Dice is:{AveDice}")
    print(f"The {i} step Average Iou is:{AveIou}")
    print("-----------------------------------------------------------")
    print(f"The {i} step Artery AUC is:{ArteryAUC}")
    print(f"The {i} step Artery Acc is:{ArteryAcc}")
    print("-----------------------------------------------------------")
    print(f"The {i} step Vein AUC is:{VeinAUC}")
    print(f"The {i} step Vein Acc is:{VeinAcc}")
    print("-----------------------------------------------------------")
    print(f"The {i} step Vessel AUC is:{VesselAUC}")
    print(f"The {i} step Vessel Acc is:{VesselAcc}")
    print(f"The {i} step Vessel Sens is:{VesselSe}")
    print(f"The {i} step Vessel Spec is:{VesselSp}")
    print(f"The {i} step Vessel F1 is:{VesselF1}")
    print(f"The {i} step Vessel Dice is:{VesselDice}")
    print(f"The {i} step Vessel Iou is:{VesselIou}")
    print("-----------------------------------------------------------")

    if not os.path.exists(metrics_file_path):
        file_w = open(metrics_file_path, 'w')
    file_w = open(metrics_file_path, 'r+')
    file_w.read()
    file_w.write(f"========================={dataset}=============================" + '\n' +
                 "Strict mode:{}".format(strict_mode) + '\n' +
                 f"The {i} step Average Acc is:{AveAcc}" + '\n' +
                 f"The {i} step Average F1 is:{AveF1}" + '\n' +
                 f"The {i} step Average Dice is:{AveDice}" + '\n' +
                 f"The {i} step Average Iou is:{AveIou}" + '\n' +
                 "-----------------------------------------------------------" + '\n' +
                 f"The {i} step Artery AUC is:{ArteryAUC}" + '\n' +
                 f"The {i} step Artery Acc is:{ArteryAcc}" + '\n' +
                 "-----------------------------------------------------------" + '\n' +
                 f"The {i} step Vein AUC is:{VeinAUC}" + '\n' +
                 f"The {i} step Vein Acc is:{VeinAcc}" + '\n' +
                 "-----------------------------------------------------------" + '\n' +
                 f"The {i} step Vessel AUC is:{VesselAUC}" + '\n' +
                 f"The {i} step Vessel Acc is:{VesselAcc}" + '\n' +
                 f"The {i} step Vessel Sens is:{VesselSe}" + '\n' +
                 f"The {i} step Vessel Spec is:{VesselSp}" + '\n' +
                 f"The {i} step Vessel F1 is:{VesselF1}" + '\n' +
                 f"The {i} step Vessel Dice is:{VesselDice}" + '\n' +
                 f"The {i} step Vessel Iou is:{VesselIou}" + '\n' +
                 "-----------------------------------------------------------" + '\n')
    # file_w.write(filewriter)
    file_w.close()


def GetResult(Net, k, use_cuda=False, dataset_name='DRIVE', is_kill_border=True, input_ch=3, config=None):
    image_basename = sorted(os.listdir(f'./data/{dataset_name}/test/images'))[k]
    label_basename = sorted(os.listdir(f'./data/{dataset_name}/test/av'))[k]
    assert image_basename.split('.')[0] == label_basename.split('.')[0]  # check if the image and label are matched

    ImgName = os.path.join(f'./data/{dataset_name}/test/images/', image_basename)
    LabelName = os.path.join(f'./data/{dataset_name}/test/av/', label_basename)

    Img0 = cv2.imread(ImgName)
    Label0 = cv2.imread(LabelName)
    _, Mask0 = creatMask(Img0, threshold=10)
    Mask = np.zeros((Img0.shape[0], Img0.shape[1]), np.float32)
    Mask[Mask0 > 0] = 1

    if config.use_resize:
        Img0 = cv2.resize(Img0, config.resize_w_h)
        Label0 = cv2.resize(Label0, config.resize_w_h, interpolation=cv2.INTER_NEAREST)
        Mask = cv2.resize(Mask, config.resize_w_h, interpolation=cv2.INTER_NEAREST)

    LabelArtery = np.zeros((Label0.shape[0], Label0.shape[1]), np.float32)
    LabelVein = np.zeros((Label0.shape[0], Label0.shape[1]), np.float32)
    LabelVessel = np.zeros((Label0.shape[0], Label0.shape[1]), np.float32)
    LabelVessel_no_unknow = np.zeros((Label0.shape[0], Label0.shape[1]), np.float32)
    LabelArtery[(Label0[:, :, 2] >= 128) | (Label0[:, :, 1] >= 128)] = 1
    LabelArtery[(Label0[:, :, 2] >= 128) & (Label0[:, :, 1] >= 128) & (Label0[:, :, 0] >= 128)] = 0
    LabelVein[(Label0[:, :, 1] >= 128) | (Label0[:, :, 0] >= 128)] = 1
    LabelVein[(Label0[:, :, 2] >= 128) & (Label0[:, :, 1] >= 128) & (Label0[:, :, 0] >= 128)] = 0
    LabelVessel[(Label0[:, :, 2] >= 128) | (Label0[:, :, 1] >= 128) | (Label0[:, :, 0] >= 128)] = 1
    # LabelVessel[(Label0[:, :, 2] >= 128) & (Label0[:, :, 1] >= 128) & (Label0[:, :, 0] >= 128)] = 0
    LabelVessel_no_unknow[(Label0[:, :, 2] >= 128) | (Label0[:, :, 1] >= 128) | (Label0[:, :, 0] >= 128)] = 1
    LabelVessel_no_unknow[(Label0[:, :, 2] >= 128) & (Label0[:, :, 1] >= 128) & (Label0[:, :, 0] >= 128)] = 0

    Img = Img0
    height, width = Img.shape[:2]
    n_classes = 3
    patch_height = config.patch_size
    patch_width = config.patch_size
    stride_height = config.stride_height
    stride_width = config.stride_width

    Img = cv2.cvtColor(Img, cv2.COLOR_BGR2RGB)
    # rgb2rgg
    # if config.use_CAM:
    #     print('TE_CAM')
    #     Img = CAM(Img)
    Img = np.float32(Img / 255.)
    Img_enlarged = paint_border_overlap(Img, patch_height, patch_width, stride_height, stride_width)
    patch_size = config.patch_size
    batch_size = 32
    patches_imgs,_ = extract_ordered_overlap_big(Img_enlarged, patch_height, patch_width, stride_height, stride_width)
    patches_imgs = np.transpose(patches_imgs, (0, 3, 1, 2))
    patches_imgs = Normalize(patches_imgs)
    # global_images = np.transpose(global_images, (0, 3, 1, 2))
    # global_images = Normalize(global_images)
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
    for i in range(max_iter):
        begin_index = i * batch_size
        end_index = (i + 1) * batch_size

        patches_temp1 = patches_imgs[begin_index:end_index, :, :, :]

        patches_input_temp1 = torch.FloatTensor(patches_temp1)
        if config.use_global_semantic:
            #global_temp1 = global_images[begin_index:end_index, :, :, :]
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


        if config.use_global_semantic:
            output_temp, _1, = Net(patches_input_temp1, global_input_temp1)
        else:
            output_temp, _1, = Net(patches_input_temp1, None)

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
    LabelArtery = LabelArtery[np.newaxis, :, :]
    LabelVein = LabelVein[np.newaxis, :, :]
    LabelVessel = LabelVessel[np.newaxis, :, :]
    LabelVessel_no_unknow = LabelVessel_no_unknow[np.newaxis, :, :]
    Mask = Mask[np.newaxis, :, :]

    return ArteryPred, VeinPred, VesselPred, LabelArtery, LabelVein, LabelVessel, Mask, LabelVessel_no_unknow

def GetResult_big(Net, k, use_cuda=False, dataset_name='DRIVE', is_kill_border=True, input_ch=3, config=None):
    image_basename = sorted(os.listdir(f'./data/{dataset_name}/test/images'))[k]
    label_basename = sorted(os.listdir(f'./data/{dataset_name}/test/av'))[k]
    assert image_basename.split('.')[0] == label_basename.split('.')[0]  # check if the image and label are matched

    ImgName = os.path.join(f'./data/{dataset_name}/test/images/', image_basename)
    LabelName = os.path.join(f'./data/{dataset_name}/test/av/', label_basename)

    Img0 = cv2.imread(ImgName)
    Label0 = cv2.imread(LabelName)
    _, Mask0 = creatMask(Img0, threshold=10)
    Mask = np.zeros((Img0.shape[0], Img0.shape[1]), np.float32)
    Mask[Mask0 > 0] = 1

    if config.use_resize:
        Img0 = cv2.resize(Img0, config.resize_w_h)
        Label0 = cv2.resize(Label0, config.resize_w_h, interpolation=cv2.INTER_NEAREST)
        Mask = cv2.resize(Mask, config.resize_w_h, interpolation=cv2.INTER_NEAREST)

    LabelArtery = np.zeros((Label0.shape[0], Label0.shape[1]), np.float32)
    LabelVein = np.zeros((Label0.shape[0], Label0.shape[1]), np.float32)
    LabelVessel = np.zeros((Label0.shape[0], Label0.shape[1]), np.float32)
    LabelVessel_no_unknow = np.zeros((Label0.shape[0], Label0.shape[1]), np.float32)
    LabelArtery[(Label0[:, :, 2] >= 128) | (Label0[:, :, 1] >= 128)] = 1
    LabelArtery[(Label0[:, :, 2] >= 128) & (Label0[:, :, 1] >= 128) & (Label0[:, :, 0] >= 128)] = 0
    LabelVein[(Label0[:, :, 1] >= 128) | (Label0[:, :, 0] >= 128)] = 1
    LabelVein[(Label0[:, :, 2] >= 128) & (Label0[:, :, 1] >= 128) & (Label0[:, :, 0] >= 128)] = 0
    LabelVessel[(Label0[:, :, 2] >= 128) | (Label0[:, :, 1] >= 128) | (Label0[:, :, 0] >= 128)] = 1
    # LabelVessel[(Label0[:, :, 2] >= 128) & (Label0[:, :, 1] >= 128) & (Label0[:, :, 0] >= 128)] = 0
    LabelVessel_no_unknow[(Label0[:, :, 2] >= 128) | (Label0[:, :, 1] >= 128) | (Label0[:, :, 0] >= 128)] = 1
    LabelVessel_no_unknow[(Label0[:, :, 2] >= 128) & (Label0[:, :, 1] >= 128) & (Label0[:, :, 0] >= 128)] = 0

    Img = Img0
    height, width = Img.shape[:2]
    n_classes = 3
    patch_height = config.patch_size
    patch_width = config.patch_size
    stride_height = config.stride_height
    stride_width = config.stride_width

    Img = cv2.cvtColor(Img, cv2.COLOR_BGR2RGB)
    # rgb2rgg
    # if config.use_CAM:
    #     print('TE_CAM')
    #     Img = CAM(Img)
    Img = np.float32(Img / 255.)
    Img_enlarged = paint_border_overlap(Img, patch_height, patch_width, stride_height, stride_width)
    patch_size = config.patch_size
    batch_size = 4
    patches_imgs,global_images = extract_ordered_overlap_big(Img_enlarged, patch_height, patch_width, stride_height, stride_width)
    patches_imgs = np.transpose(patches_imgs, (0, 3, 1, 2))
    patches_imgs = Normalize(patches_imgs)
    global_images = np.transpose(global_images, (0, 3, 1, 2))
    global_images = Normalize(global_images)
    patchNum = patches_imgs.shape[0]
    max_iter = int(np.ceil(patchNum / float(batch_size)))


    pred_patches = np.zeros((patchNum, n_classes, patch_size, patch_size), np.float32)
    for i in range(max_iter):
        begin_index = i * batch_size
        end_index = (i + 1) * batch_size

        patches_temp1 = patches_imgs[begin_index:end_index, :, :, :]

        patches_input_temp1 = torch.FloatTensor(patches_temp1)
        if config.use_global_semantic:
            global_temp1 = global_images[begin_index:end_index, :, :, :]
            #global_temp1 = golbal_img_batch[0:patches_temp1.shape[0], :, :, :]
            global_input_temp1 = torch.FloatTensor(global_temp1)
        if use_cuda:
            patches_input_temp1 = autograd.Variable(patches_input_temp1.cuda())
            if config.use_global_semantic:
                global_input_temp1 = autograd.Variable(global_input_temp1.cuda())
        else:
            patches_input_temp1 = autograd.Variable(patches_input_temp1)
            if config.use_global_semantic:
                global_input_temp1 = autograd.Variable(global_input_temp1)

        if config.use_global_semantic:
            output_temp, _1, = Net(patches_input_temp1, global_input_temp1)
        else:
            output_temp, _1, = Net(patches_input_temp1, None)

        pred_patches_temp = np.float32(output_temp.data.cpu().numpy())

        pred_patches_temp_sigmoid = sigmoid(pred_patches_temp)
        #pred_patches_temp_sigmoid = pred_patches_temp
        #pred_patches_temp_sigmoid[:,0:1,:,:] = sigmoid(pred_patches_temp[:,0:1,:,:]*pred_patches_temp_sigmoid[:,1:2,:,:])
        #pred_patches_temp_sigmoid[:,2:3,:,:] = sigmoid(pred_patches_temp[:,2:3,:,:]*pred_patches_temp_sigmoid[:,1:2,:,:]) 
        #pred_patches_temp_sigmoid[:,1:2,:,:] = sigmoid(pred_patches_temp[:,1:2,:,:]) 

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
    LabelArtery = LabelArtery[np.newaxis, :, :]
    LabelVein = LabelVein[np.newaxis, :, :]
    LabelVessel = LabelVessel[np.newaxis, :, :]
    LabelVessel_no_unknow = LabelVessel_no_unknow[np.newaxis, :, :]
    Mask = Mask[np.newaxis, :, :]

    return ArteryPred, VeinPred, VesselPred, LabelArtery, LabelVein, LabelVessel, Mask, LabelVessel_no_unknow
def Evalution_AV_skeletonPixles(PredAll1, PredAll2, VesselPredAll, LabelAll1, LabelAll2, LabelVesselAll, MaskAll,
                                DataSet=0, onlyMeasureSkeleton=False, strict_mode=False):
    """
    OUTPUT:
    AUC1: 动脉
    AUC2: 静脉
    accuracy1: 动静脉联合准确率
    specificity1: 静脉准确率
    sensitivity1: 动脉准确率
    f_score1: 动静脉联合F1
    dice_score1: 动静脉联合Dice
    iou_score1: 动静脉联合IoU

    """

    threshold_confusion = 0.5
            #ind = np.where(np.logical_and(np.logical_or(ArteryPredAll > threshold_confusion,VeinPredAll>threshold_confusion), MaskAll > 0))
    ind = np.where(np.logical_and(np.logical_or(LabelAll1 > threshold_confusion, LabelAll2 > threshold_confusion),
                                                                       MaskAll > 0))

    y_scores1 = PredAll1[ind]
    y_true1 = LabelAll1[ind]
    y_scores2 = PredAll2[ind]
    y_true2 = LabelAll2[ind]


    #y_scores1, y_true1, y_scores2, y_true2 = pred_only_FOV_AV(PredAll1, PredAll2, LabelAll1, LabelAll2, MaskAll,
    #                                                          threshold_confusion)
    AUC1 = roc_auc_score(y_true1, y_scores1)  # 动脉
    AUC2 = roc_auc_score(y_true2, y_scores2)  # 静脉
    
    accuracy1, specificity1, sensitivity1, f_score1, dice_score1, iou_score1 = AVclassifiationMetrics_skeletonPixles(
        PredAll1, PredAll2, VesselPredAll, LabelAll1, LabelAll2, LabelVesselAll, DataSet,
        onlyMeasureSkeleton=onlyMeasureSkeleton, strict_mode=strict_mode)
    # accuracy2,specificity2,sensitivity2 = AVclassifiationMetrics(PredAll2,PredAll1,VesselPredAll,LabelAll2,LabelAll1,LabelVesselAll,DataSet)
    return AUC1, accuracy1, specificity1, sensitivity1, f_score1, dice_score1, iou_score1, AUC2


def draw_prediction(writer, pred, targs, step):
    target_artery = targs[0:1, 0, :, :]
    target_vein = targs[0:1, 1, :, :]
    target_all = targs[0:1, 2, :, :]

    pred_sigmoid = pred  # nn.Sigmoid()(pred)

    writer.add_image('artery', torch.cat([pred_sigmoid[0:1, 0, :, :], target_artery], dim=1), global_step=step)
    writer.add_image('vessel', torch.cat([pred_sigmoid[0:1, 1, :, :], target_vein], dim=1), global_step=step)
    writer.add_image('vein', torch.cat([pred_sigmoid[0:1, 2, :, :], target_all], dim=1), global_step=step)


if __name__ == '__main__':
    import pathlib

    i = sorted(os.listdir(f'./data/AV_DRIVE/test/images'), reverse=False)[0]

    print(i)
