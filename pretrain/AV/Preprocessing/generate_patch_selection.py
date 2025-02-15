# -*- coding: utf-8 -*-

import numpy as np
import os
import config.config_train_general as cfg

def patchJudge(mask, a, v, av, y, x, patch_h, patch_w, type='only_a', limit=100):
    '''
    :param mask: mask
    :param a: artery
    :param v: vein
    :param av: artery and vein
    :param y: y coordinate of the patch
    :param x: x coordinate of the patch
    :param patch_h: height of the patch
    :param patch_w: width of the patch
    :param type: 'only_a', 'only_v', 'only_av'
    :param limit: the limit of the number of pixels in the patch
    0 represents only_a
    1 represents only_v
    2 represents only_av
    '''
    # print(type)
    # print(patch_w)
    # print(limit)
    # print(f'av:{np.sum(av[y:y + patch_h, x:x + patch_w])}')
    # print(f'a:{np.sum(a[y:y + patch_h, x:x+patch_w])}')
    # print(f'v:{np.sum(v[y:y + patch_h, x:x + patch_w])}')

    # print(b_rate)

    #b_rate = np.sum(mask[y:y+patch_h,x:x+patch_w] == 0)/patch_h/patch_w
    #b_rate_threshold = 0.5
    
    if cfg.dataset == 'LES':
        limit_av = 200
    else:
        limit_av = 50
    

    if type ==0 or type=='only_a':
        return np.sum(a[y:y+patch_h,x:x+patch_w])>=limit and np.sum(v[y:y+patch_h,x:x+patch_w])==0  
    elif type ==1 or type=='only_v':
        return np.sum(v[y:y+patch_h,x:x+patch_w])>=limit and np.sum(a[y:y+patch_h,x:x+patch_w])==0 
    elif type ==2 or type=='only_av':

        return np.sum(av[y:y+patch_h,x:x+patch_w])>=limit and  np.sum(a[y:y+patch_h,x:x+patch_w])>=limit and np.sum(v[y:y+patch_h,x:x+patch_w])>=limit and np.sum(a[y:y + patch_h, x:x + patch_w])/np.sum(v[y:y + patch_h, x:x + patch_w])<4 and np.sum(v[y:y + patch_h, x:x + patch_w])/np.sum(a[y:y + patch_h, x:x + patch_w])<4

    else:
        return True


def patch_select(ind, patch_size, Mask, Mask2, LabelA, LabelV, LabelVessel, type=0):

    H, W = LabelA.shape
    patch_size = patch_size
    Ve_rate_back = np.count_nonzero(LabelVessel)/(H*W)
    V_rate_back = np.count_nonzero(LabelV)/(H*W)
    A_rate_back = np.count_nonzero(LabelA)/(H*W)
    rate = min(Ve_rate_back,V_rate_back,A_rate_back)

    if type==2 or patch_size>=256:
        limit = int(min(patch_size,256) * min(patch_size,256) * rate)
        
    else:
        if cfg.dataset=='LES':
            limit=200
        else:
            limit = 100
    
    if type == 0:
        skel = Mask[0, :, :]
    elif type == 1:
        skel = Mask[1, :, :]
    else:
        skel = Mask[2, :, :]

    skels = np.argwhere(skel)
    find_patch = False
    dot_pix = np.random.randint(0, skels.shape[0])

    y_aixs = skels[dot_pix][0]  # h
    x_aixs = skels[dot_pix][1]  # w


    roi_area_x_left = max(0,x_aixs-patch_size)
    roi_area_y_left = max(0,y_aixs-patch_size)
    roi_area_x_right = min(x_aixs+patch_size,LabelVessel.shape[1])
    roi_area_y_right = min(y_aixs+patch_size,LabelVessel.shape[0])

    
    y = np.random.randint(roi_area_y_left, roi_area_y_right - patch_size + 1)
    x = np.random.randint(roi_area_x_left, roi_area_x_right - patch_size + 1)

    cn = 0
    limit_cn = 1000
    while (not patchJudge(Mask2, LabelA, LabelV, LabelVessel, y, x, patch_size, patch_size, type=type,
                          limit=limit)) and cn < limit_cn:

        if cn % 100 == 0 and cn > 0:
            dot_pix = np.random.randint(0, skels.shape[0])
            y_aixs = skels[dot_pix][0]  # h
            x_aixs = skels[dot_pix][1]  # w
        roi_area_x_left = max(0, x_aixs - patch_size)
        roi_area_y_left = max(0, y_aixs - patch_size)
        roi_area_x_right = min(x_aixs + patch_size, LabelVessel.shape[1])
        roi_area_y_right = min(y_aixs + patch_size, LabelVessel.shape[0])

            
        y = np.random.randint(roi_area_y_left, roi_area_y_right - patch_size + 1)
        x = np.random.randint(roi_area_x_left, roi_area_x_right - patch_size + 1)

        cn = cn + 1
    if cn < limit_cn:
        find_patch = True
    # print("x,y:",x,y)
    
    return y, x, patch_size, find_patch, roi_area_y_left, roi_area_x_left, roi_area_y_right, roi_area_x_right
