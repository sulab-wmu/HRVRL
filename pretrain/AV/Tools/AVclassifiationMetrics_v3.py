import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
# import natsort
import pandas as pd
from skimage.morphology import skeletonize, erosion, square,dilation
from Tools.BinaryPostProcessing import binaryPostProcessing3
from PIL import Image
from scipy.signal import convolve2d
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, confusion_matrix,roc_curve,auc
from collections import OrderedDict
import time
#########################################
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def Skeleton(a_or_v, a_and_v):
    th = np.uint8(a_and_v)
    # Distance transform for maximum diameter
    vessels = th.copy()
    dist = cv2.distanceTransform(a_or_v, cv2.DIST_L2, 3)  
    thinned = np.uint8(skeletonize((vessels / 255))) * 255
    return thinned, dist


def cal_crosspoint(vessel):
    # Removing bifurcation points by using specially designed kernels
    # Can be optimized further! (not the best implementation)
    thinned1, dist = Skeleton(vessel, vessel)
    thh = thinned1.copy()
    thh = thh / 255
    kernel1 = np.array([[1, 1, 1], [1, 10, 1], [1, 1, 1]])

    th = convolve2d(thh, kernel1, mode="same")
    for u in range(th.shape[0]):
        for j in range(th.shape[1]):
            if th[u, j] >= 13.0:
                cv2.circle(vessel, (j, u), 2 * int(dist[u, j]), (0, 0, 0), -1)
    # thi = cv2.cvtColor(thi, cv2.COLOR_BGR2GRAY)
    return vessel

def get_cen_img_from_custom(groundtruth,sign='3-inf'):
    '''
    groundtruth: H*W
    sign: must select one of '0-2','0-3','2-3','3-inf'
    '''
    image_cen = skeletonize(groundtruth > 0)
    image_3_inf = skeletonize(dilation(erosion(groundtruth > 0)))

    # retrieve the indices for the centerline pixels [3,inf)
    if sign=='3-inf': #[3,inf)

        return skeletonize(erosion(groundtruth > 0))


    # retrieve the indices for the centerline pixels [0,3)
    image_3_inf = cv2.bitwise_and(src1=np.uint8(image_cen), src2=np.uint8(image_3_inf), mask=np.uint8(image_cen))
    image_0_3 = cv2.bitwise_xor(src1=np.uint8(image_cen),src2=np.uint8(image_3_inf),mask=np.uint8(image_cen))
    # contours, _ = cv2.findContours(image_0_3, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    # for contour in contours:
    #     peri = cv2.arcLength(contour, False)
    #     if peri < 5:
    #         cv2.drawContours(image_0_3, [contour], -1, (0, 0, 0), -1)
    if sign=='0-3':#[0,3)
        return image_0_3>0


    # retrieve the indices for the centerline pixels [0,2)
    image_2_inf = skeletonize(dilation(erosion(groundtruth > 0, square(2))))
    image_2_inf =  cv2.bitwise_and(src1=np.uint8(image_cen),src2=np.uint8(image_2_inf),mask=np.uint8(image_cen))
    image_0_2 = cv2.bitwise_xor(src1=np.uint8(image_cen),src2=np.uint8(image_2_inf),mask=np.uint8(image_cen))
    # contours, _ = cv2.findContours(image_0_2, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    # for contour in contours:
    #     peri = cv2.arcLength(contour, False)
    #     if peri < 5:
    #         cv2.drawContours(image_0_2, [contour], -1, (0, 0, 0), -1)
    if sign=='0-2':
        return image_0_2 > 0

    # retrieve the indices for the centerline pixels [2,3)

    image_2_3 = cv2.bitwise_xor(src1=np.uint8(image_0_2), src2=np.uint8(image_0_3), mask=np.uint8(image_cen))
    if sign=='2-3':
        return image_2_3 > 0

def get_full_img_from_custom(groundtruth, sign='3-inf'):
    '''
    groundtruth: H*W
    sign: must select one of '0-2','0-3','2-3','3-inf'
    '''
    image_cen = groundtruth > 0
    image_3_inf = erosion(groundtruth > 0)

    # retrieve the indices for the centerline pixels [3,inf)
    if sign == '3-inf':  # [3,inf)
        return erosion(groundtruth > 0)
    # retrieve the indices for the centerline pixels [0,3)
    image_3_inf_dilation = dilation(image_3_inf)
    image_3_inf_dilation = cv2.bitwise_and(src1=np.uint8(image_cen), src2=np.uint8(image_3_inf_dilation), mask=np.uint8(image_cen))
    image_0_3 = cv2.bitwise_xor(src1=np.uint8(image_cen), src2=np.uint8(image_3_inf_dilation), mask=np.uint8(image_cen))
    # contours, _ = cv2.findContours(image_0_3, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    # for contour in contours:
    #     peri = cv2.arcLength(contour, False)
    #     if peri < 5:
    #         cv2.drawContours(image_0_3, [contour], -1, (0, 0, 0), -1)
    if sign == '0-3':  # [0,3)
        return image_0_3 > 0
    # retrieve the indices for the centerline pixels [0,2)
    image_2_inf = erosion(groundtruth > 0, square(2))
    image_2_inf_dilation = dilation(image_2_inf)
    image_2_inf_dilation = cv2.bitwise_and(src1=np.uint8(image_cen), src2=np.uint8(image_2_inf_dilation), mask=np.uint8(image_cen))
    image_0_2 = cv2.bitwise_xor(src1=np.uint8(image_cen), src2=np.uint8(image_2_inf_dilation), mask=np.uint8(image_cen))
    # contours, _ = cv2.findContours(image_0_2, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    # for contour in contours:
    #     peri = cv2.arcLength(contour, False)
    #     if peri < 5:
    #         cv2.drawContours(image_0_2, [contour], -1, (0, 0, 0), -1)
    if sign == '0-2':
        return image_0_2 > 0

    # retrieve the indices for the centerline pixels [2,3)

    image_2_3 = cv2.bitwise_xor(src1=np.uint8(image_0_2), src2=np.uint8(image_0_3), mask=np.uint8(image_cen))
    if sign == '2-3':
        return image_2_3 > 0


def evaluation_cen_av_test(prediction, groundtruth,full_metric):
    '''
    Function to evaluate the performance of AV predictions with a given ground truth
    - prediction: should be an image array of [dim1, dim2, img_channels = 3] with arteries in red and veins in blue
    - groundtruth: same as above
    '''

    encoded_pred = np.zeros(prediction.shape[:2], dtype=int)
    encoded_gt = np.zeros(groundtruth.shape[:2], dtype=int)

    encoded_pred_a = np.zeros(prediction.shape[:2], dtype=int)
    encoded_gt_a = np.zeros(groundtruth.shape[:2], dtype=int)

    encoded_pred_v = np.zeros(prediction.shape[:2], dtype=int)
    encoded_gt_v = np.zeros(groundtruth.shape[:2], dtype=int)

    # convert white pixels to green pixels (which are ignored)
    white_ind = np.where(np.logical_and.reduce([groundtruth[:, :, 0] == 255, groundtruth[:, :, 1] == 255, groundtruth[:, :, 2] == 255]))
    if white_ind[0].size != 0:
        groundtruth[white_ind] = [0, 255, 0]

    # --- original -------
    arteriole = np.where(np.logical_and(groundtruth[:, :, 0] == 255, groundtruth[:, :, 1] == 0));encoded_gt_a[arteriole] = 1;encoded_gt[arteriole] = 1
    venule = np.where(np.logical_and(groundtruth[:, :, 2] == 255, groundtruth[:, :, 1] == 0));encoded_gt_v[venule] = 1;encoded_gt[venule] = 2
    arteriole = np.where(prediction[:, :, 0] == 255);encoded_pred_a[arteriole] = 1;encoded_pred[arteriole] = 1
    venule = np.where(prediction[:, :, 2] == 255);encoded_pred_v[venule] = 1;encoded_pred[venule] = 2

    # retrieve the indices for the centerline pixels present in the prediction
    center_a_pred = np.where(skeletonize(encoded_pred_a[:, :] > 0))
    center_v_pred = np.where(skeletonize(encoded_pred_v[:, :] > 0))
    encoded_pred_center_a = encoded_pred_a[center_a_pred]
    encoded_gt_center_a = encoded_gt_a[center_a_pred]
    encoded_pred_center_v = encoded_pred_v[center_v_pred]
    encoded_gt_center_v = encoded_gt_v[center_v_pred]

    center_av_pred = np.where(np.logical_and(
        np.logical_or((skeletonize(encoded_gt_a > 0)),(skeletonize(encoded_gt_v > 0))),
        encoded_pred[:,:] > 0))
    # center_av_pred = np.concatenate((center_a_pred, center_v_pred), axis=1)
    # center_av_pred = (center_av_pred[0], center_av_pred[1])
    encoded_pred_center_av = encoded_pred[center_av_pred]
    encoded_gt_center_av = encoded_gt[center_av_pred]

    # retrieve the indices for the centerline pixels present in the groundtruth
    center_comp_a = np.where(skeletonize(encoded_gt_a[:, :] > 0))
    center_comp_v = np.where(skeletonize(encoded_gt_v[:, :] > 0))
    encoded_pred_center_comp_a = encoded_pred_a[center_comp_a]
    encoded_gt_center_comp_a = encoded_gt_a[center_comp_a]
    encoded_pred_center_comp_v = encoded_pred_v[center_comp_v]
    encoded_gt_center_comp_v = encoded_gt_v[center_comp_v]
    center_comp_av = np.concatenate((center_comp_a, center_comp_v), axis=1)
    center_comp_av = (center_comp_av[0],center_comp_av[1])
    encoded_pred_center_comp_av = encoded_pred[center_comp_av]
    encoded_gt_center_comp_av = encoded_gt[center_comp_av]

    # retrieve the indices for centerline pixels - limited to vessels wider than two pixels [3,inf)
    center_eroded_a_3_Inf_image = get_cen_img_from_custom(encoded_gt_a[:, :], sign='3-inf')
    center_eroded_a_3_Inf = np.where(center_eroded_a_3_Inf_image)
    center_eroded_v_3_Inf_image = get_cen_img_from_custom(encoded_gt_v[:, :], sign='3-inf')
    center_eroded_v_3_Inf = np.where(center_eroded_v_3_Inf_image)
    encoded_pred_center_eroded_a_3_Inf = encoded_pred_a[center_eroded_a_3_Inf]
    encoded_gt_center_eroded_a_3_Inf = encoded_gt_a[center_eroded_a_3_Inf]
    encoded_pred_center_eroded_v_3_Inf = encoded_pred_v[center_eroded_v_3_Inf]
    encoded_gt_center_eroded_v_3_Inf = encoded_gt_v[center_eroded_v_3_Inf]
    center_eroded_av_3_Inf_pre = np.where(np.logical_and(
        np.logical_or(center_eroded_a_3_Inf_image,center_eroded_v_3_Inf_image),
        encoded_pred[:,:] > 0))
    center_eroded_av_3_Inf = np.concatenate((center_eroded_a_3_Inf, center_eroded_v_3_Inf), axis=1)
    center_eroded_av_3_Inf_gt = (center_eroded_av_3_Inf[0], center_eroded_av_3_Inf[1])
    encoded_pred_center_eroded_av_3_Inf_pre = encoded_pred[center_eroded_av_3_Inf_pre]
    encoded_gt_center_eroded_av_3_Inf_pre = encoded_gt[center_eroded_av_3_Inf_pre]
    encoded_pred_center_eroded_av_3_Inf_gt = encoded_pred[center_eroded_av_3_Inf_gt]
    encoded_gt_center_eroded_av_3_Inf_gt = encoded_gt[center_eroded_av_3_Inf_gt]

    # retrieve the indices for centerline pixels - limited to vessels [0,3),[0,2) and [2,3)
    center_eroded_a_0_3_image = get_cen_img_from_custom(encoded_gt_a[:, :], sign='0-3')
    center_eroded_a_0_3 = np.where(center_eroded_a_0_3_image)
    center_eroded_v_0_3_image = get_cen_img_from_custom(encoded_gt_v[:, :], sign='0-3')
    center_eroded_v_0_3 = np.where(center_eroded_v_0_3_image)
    center_eroded_av_0_3_pre = np.where(np.logical_and(
        np.logical_or(center_eroded_a_0_3_image, center_eroded_v_0_3_image),
        encoded_pred[:, :] > 0))
    center_eroded_av_0_3 = np.concatenate((center_eroded_a_0_3, center_eroded_v_0_3), axis=1)
    center_eroded_av_0_3_gt = (center_eroded_av_0_3[0], center_eroded_av_0_3[1])
    encoded_pred_center_eroded_av_0_3_pre = encoded_pred[center_eroded_av_0_3_pre]
    encoded_gt_center_eroded_av_0_3_pre = encoded_gt[center_eroded_av_0_3_pre]
    encoded_pred_center_eroded_av_0_3_gt = encoded_pred[center_eroded_av_0_3_gt]
    encoded_gt_center_eroded_av_0_3_pre = encoded_gt[center_eroded_av_0_3_gt]

    # retrieve the indices for centerline pixels - limited to vessels [0,2)
    center_eroded_a_0_2_image = get_cen_img_from_custom(encoded_gt_a[:, :], sign='0-2')
    center_eroded_a_0_2 = np.where(center_eroded_a_0_2_image)
    center_eroded_v_0_2_image = get_cen_img_from_custom(encoded_gt_v[:, :], sign='0-2')
    center_eroded_v_0_2 = np.where(center_eroded_v_0_2_image)
    encoded_pred_center_eroded_a_0_2 = encoded_pred_a[center_eroded_a_0_2]
    encoded_gt_center_eroded_a_0_2 = encoded_gt_a[center_eroded_a_0_2]
    encoded_pred_center_eroded_v_0_2 = encoded_pred_v[center_eroded_v_0_2]
    encoded_gt_center_eroded_v_0_2 = encoded_gt_v[center_eroded_v_0_2]

    center_eroded_av_0_2_pre = np.where(np.logical_and(
        np.logical_or(center_eroded_a_0_2_image, center_eroded_v_0_2_image),
        encoded_pred[:, :] > 0))
    center_eroded_av_0_2 = np.concatenate((center_eroded_a_0_2, center_eroded_v_0_2), axis=1)
    center_eroded_av_0_2_gt = (center_eroded_av_0_2[0], center_eroded_av_0_2[1])
    encoded_pred_center_eroded_av_0_2_pre = encoded_pred[center_eroded_av_0_2_pre]
    encoded_gt_center_eroded_av_0_2_pre = encoded_gt[center_eroded_av_0_2_pre]
    encoded_pred_center_eroded_av_0_2_gt = encoded_pred[center_eroded_av_0_2_gt]
    encoded_gt_center_eroded_av_0_2_gt = encoded_gt[center_eroded_av_0_2_gt]

    # retrieve the indices for centerline pixels - limited to vessels [2,3)
    center_eroded_a_2_3_image = get_cen_img_from_custom(encoded_gt_a[:, :], sign='2-3')
    center_eroded_a_2_3 = np.where(center_eroded_a_2_3_image)
    center_eroded_v_2_3_image = get_cen_img_from_custom(encoded_gt_v[:, :], sign='2-3')
    center_eroded_v_2_3 = np.where(center_eroded_v_2_3_image)
    encoded_pred_center_eroded_a_2_3 = encoded_pred_a[center_eroded_a_2_3]
    encoded_gt_center_eroded_a_2_3 = encoded_gt_a[center_eroded_a_2_3]
    encoded_pred_center_eroded_v_2_3 = encoded_pred_v[center_eroded_v_2_3]
    encoded_gt_center_eroded_v_2_3 = encoded_gt_v[center_eroded_v_2_3]
    center_eroded_av_2_3_pre = np.where(np.logical_and(
        np.logical_or(center_eroded_a_2_3_image, center_eroded_v_2_3_image),
        encoded_pred[:, :] > 0))
    center_eroded_av_2_3 = np.concatenate((center_eroded_a_2_3, center_eroded_v_2_3),axis=1)
    center_eroded_av_2_3_gt = (center_eroded_av_2_3[0], center_eroded_av_2_3[1])
    encoded_pred_center_eroded_av_2_3_pre = encoded_pred[center_eroded_av_2_3_pre]
    encoded_gt_center_eroded_av_2_3_pre = encoded_gt[center_eroded_av_2_3_pre]
    encoded_pred_center_eroded_av_2_3_gt = encoded_pred[center_eroded_av_2_3_gt]
    encoded_gt_center_eroded_av_2_3_gt = encoded_gt[center_eroded_av_2_3_gt]

    # compute artery detection rate
    artery_ind = np.where(encoded_gt_a > 0)
    artery_gt = encoded_gt_a[artery_ind]
    artery_pred = encoded_pred_a[artery_ind]
    vein_ind = np.where(encoded_gt_v > 0)
    vein_gt = encoded_gt_v[vein_ind]
    vein_pred = encoded_pred_v[vein_ind]
    av_ind = np.concatenate((artery_ind, vein_ind),axis=1)
    av_ind = (av_ind[0], av_ind[1])
    av_pred = encoded_pred[av_ind]
    av_gt = encoded_gt[av_ind]
    detection_rate_v = accuracy_score(vein_gt.flatten(), vein_pred.flatten())
    detection_rate_a = accuracy_score(artery_gt.flatten(), artery_pred.flatten())
    detection_rates = [detection_rate_a, detection_rate_v]

    encoded_gts = OrderedDict({'av_pre':[encoded_gt_center_av, encoded_gt_center_eroded_av_0_2_pre,encoded_gt_center_eroded_av_2_3_pre,
                   encoded_gt_center_eroded_av_3_Inf_pre],'av_gt':[encoded_gt_center_comp_av,encoded_gt_center_eroded_av_0_2_gt,encoded_gt_center_eroded_av_2_3_gt,encoded_gt_center_eroded_av_3_Inf_gt]})

    encoded_preds = OrderedDict({'av_pre':[encoded_pred_center_av, encoded_pred_center_eroded_av_0_2_pre,encoded_pred_center_eroded_av_2_3_pre,
                     encoded_pred_center_eroded_av_3_Inf_pre],'av_gt':[encoded_pred_center_comp_av,encoded_pred_center_eroded_av_0_2_gt,encoded_pred_center_eroded_av_2_3_gt,encoded_pred_center_eroded_av_3_Inf_gt]})


    av_metric_base_pre=[]
    av_metric_base_gt=[]
    for av_type in encoded_gts.keys():
        metrics_num = len(encoded_gts[av_type])
        for i in range(metrics_num):
            eps = 1e-7
            y_true = encoded_gts[av_type][i]
            y_pred = encoded_preds[av_type][i]
            cur_acc = accuracy_score(encoded_gts[av_type][i].flatten(), encoded_preds[av_type][i].flatten())
            cur_F1 = f1_score(encoded_gts[av_type][i].flatten(), encoded_preds[av_type][i].flatten(),
                              average='weighted')

            # if i==0:
            #     pred_binary = (av_pred == 2).astype(int)
            #     gt_binary = (av_gt == 2).astype(int)
            #     cur_auc = roc_auc_score(gt_binary, pred_binary)
            # else:
            #     cur_auc=0
            try:
                cur_auc = roc_auc_score(y_true, y_pred)
            except:
                cur_auc = 0
            TP = np.sum((y_true == 1) & (y_pred == 1))  #  true positive  (artery)
            FP = np.sum((y_true == 2) & (y_pred == 1))  # false positive
            TN = np.sum((y_true == 2) & (y_pred == 2))  # true negative (vein)
            FN = np.sum((y_true == 1) & (y_pred == 2))  # fasle negative

            # cur_acc = (TP + TN) / (TP + FP + TN + FN + eps)
            # cur_F1 = (2 * TP) / (2 * TP + FP + FN + eps)

            sens = TP / (TP + FN + eps)
            spec = TN / (TN + FP + eps)

            if av_type=='av_pre':
                if i == 0:
                    cur_acc = full_metric[0]
                    cur_F1 = full_metric[1]
                    sens = full_metric[2]
                    spec = full_metric[3]
                av_metric_base_pre.append([cur_acc, cur_F1,cur_auc ,sens, spec])
            else:
                av_metric_base_gt.append([cur_acc, cur_F1, cur_auc, sens, spec])

    return detection_rates, av_metric_base_pre,av_metric_base_gt

def evaluation_full_av_test(prediction,groundtruth,full_metric):
    encoded_pred = np.zeros(prediction.shape[:2], dtype=int)
    encoded_gt = np.zeros(groundtruth.shape[:2], dtype=int)

    encoded_pred_a = np.zeros(prediction.shape[:2], dtype=int)
    encoded_gt_a = np.zeros(groundtruth.shape[:2], dtype=int)

    encoded_pred_v = np.zeros(prediction.shape[:2], dtype=int)
    encoded_gt_v = np.zeros(groundtruth.shape[:2], dtype=int)

    # convert white pixels to green pixels (which are ignored)
    white_ind = np.where(
        np.logical_and.reduce([groundtruth[:, :, 0] == 255, groundtruth[:, :, 1] == 255, groundtruth[:, :, 2] == 255]))
    if white_ind[0].size != 0:
        groundtruth[white_ind] = [0, 255, 0]

    # --- original -------
    arteriole = np.where(np.logical_and(groundtruth[:, :, 0] == 255, groundtruth[:, :, 1] == 0));
    encoded_gt_a[arteriole] = 1;
    encoded_gt[arteriole] = 1
    venule = np.where(np.logical_and(groundtruth[:, :, 2] == 255, groundtruth[:, :, 1] == 0));
    encoded_gt_v[venule] = 1;
    encoded_gt[venule] = 2
    arteriole = np.where(prediction[:, :, 0] == 255);
    encoded_pred_a[arteriole] = 1;
    encoded_pred[arteriole] = 1
    venule = np.where(prediction[:, :, 2] == 255);
    encoded_pred_v[venule] = 1;
    encoded_pred[venule] = 2

    # retrieve the indices for the full pixels present in the prediction
    full_a_pred = np.where((encoded_pred_a[:, :] > 0))
    full_v_pred = np.where((encoded_pred_v[:, :] > 0))
    encoded_pred_full_a = encoded_pred_a[full_a_pred]
    encoded_gt_full_a = encoded_gt_a[full_a_pred]
    encoded_pred_full_v = encoded_pred_v[full_v_pred]
    encoded_gt_full_v = encoded_gt_v[full_v_pred]
    full_av_pred = np.where(np.logical_and(
        np.logical_or(((encoded_gt_a > 0)), ((encoded_gt_v > 0))),
        encoded_pred[:, :] > 0))
    # full_av_pred = np.concatenate((full_a_pred, full_v_pred), axis=1)
    # full_av_pred = (full_av_pred[0], full_av_pred[1])
    encoded_pred_full_av = encoded_pred[full_av_pred]
    encoded_gt_full_av = encoded_gt[full_av_pred]

    # retrieve the indices for the full pixels present in the groundtruth
    full_comp_a = np.where((encoded_gt_a[:, :] > 0))
    full_comp_v = np.where((encoded_gt_v[:, :] > 0))
    encoded_pred_full_comp_a = encoded_pred_a[full_comp_a]
    encoded_gt_full_comp_a = encoded_gt_a[full_comp_a]
    encoded_pred_full_comp_v = encoded_pred_v[full_comp_v]
    encoded_gt_full_comp_v = encoded_gt_v[full_comp_v]
    full_comp_av = np.concatenate((full_comp_a, full_comp_v), axis=1)
    full_comp_av = (full_comp_av[0], full_comp_av[1])
    encoded_pred_full_comp_av = encoded_pred[full_comp_av]
    encoded_gt_full_comp_av = encoded_gt[full_comp_av]

    # retrieve the indices for full pixels - limited to vessels wider than two pixels [3,inf)
    full_eroded_a_3_Inf_image = get_full_img_from_custom(encoded_gt_a[:, :], sign='3-inf')
    full_eroded_a_3_Inf = np.where(full_eroded_a_3_Inf_image)
    full_eroded_v_3_Inf_image = get_full_img_from_custom(encoded_gt_v[:, :], sign='3-inf')
    full_eroded_v_3_Inf = np.where(full_eroded_v_3_Inf_image)
    encoded_pred_full_eroded_a_3_Inf = encoded_pred_a[full_eroded_a_3_Inf]
    encoded_gt_full_eroded_a_3_Inf = encoded_gt_a[full_eroded_a_3_Inf]
    encoded_pred_full_eroded_v_3_Inf = encoded_pred_v[full_eroded_v_3_Inf]
    encoded_gt_full_eroded_v_3_Inf = encoded_gt_v[full_eroded_v_3_Inf]
    full_eroded_av_3_Inf_pre = np.where(np.logical_and(
        np.logical_or(full_eroded_a_3_Inf_image, full_eroded_v_3_Inf_image),
        encoded_pred[:, :] > 0))
    full_eroded_av_3_Inf = np.concatenate((full_eroded_a_3_Inf, full_eroded_v_3_Inf), axis=1)
    full_eroded_av_3_Inf_gt = (full_eroded_av_3_Inf[0], full_eroded_av_3_Inf[1])
    encoded_pred_full_eroded_av_3_Inf_pre = encoded_pred[full_eroded_av_3_Inf_pre]
    encoded_gt_full_eroded_av_3_Inf_pre = encoded_gt[full_eroded_av_3_Inf_pre]
    encoded_pred_full_eroded_av_3_Inf_gt = encoded_pred[full_eroded_av_3_Inf_gt]
    encoded_gt_full_eroded_av_3_Inf_gt = encoded_gt[full_eroded_av_3_Inf_gt]

    # retrieve the indices for full pixels - limited to vessels [0,3),[0,2) and [2,3)
    full_eroded_a_0_3_image = get_full_img_from_custom(encoded_gt_a[:, :], sign='0-3')
    full_eroded_a_0_3 = np.where(full_eroded_a_0_3_image)
    full_eroded_v_0_3_image = get_full_img_from_custom(encoded_gt_v[:, :], sign='0-3')
    full_eroded_v_0_3 = np.where(full_eroded_v_0_3_image)
    full_eroded_av_0_3_pre = np.where(np.logical_and(
        np.logical_or(full_eroded_a_0_3_image, full_eroded_v_0_3_image),
        encoded_pred[:, :] > 0))
    full_eroded_av_0_3 = np.concatenate((full_eroded_a_0_3, full_eroded_v_0_3), axis=1)
    full_eroded_av_0_3_gt = (full_eroded_av_0_3[0], full_eroded_av_0_3[1])
    encoded_pred_full_eroded_av_0_3_pre = encoded_pred[full_eroded_av_0_3_pre]
    encoded_gt_full_eroded_av_0_3_pre = encoded_gt[full_eroded_av_0_3_pre]
    encoded_pred_full_eroded_av_0_3_gt = encoded_pred[full_eroded_av_0_3_gt]
    encoded_gt_full_eroded_av_0_3_pre = encoded_gt[full_eroded_av_0_3_gt]

    # retrieve the indices for full pixels - limited to vessels [0,2)
    full_eroded_a_0_2_image = get_full_img_from_custom(encoded_gt_a[:, :], sign='0-2')
    full_eroded_a_0_2 = np.where(full_eroded_a_0_2_image)
    full_eroded_v_0_2_image = get_full_img_from_custom(encoded_gt_v[:, :], sign='0-2')
    full_eroded_v_0_2 = np.where(full_eroded_v_0_2_image)
    encoded_pred_full_eroded_a_0_2 = encoded_pred_a[full_eroded_a_0_2]
    encoded_gt_full_eroded_a_0_2 = encoded_gt_a[full_eroded_a_0_2]
    encoded_pred_full_eroded_v_0_2 = encoded_pred_v[full_eroded_v_0_2]
    encoded_gt_full_eroded_v_0_2 = encoded_gt_v[full_eroded_v_0_2]

    full_eroded_av_0_2_pre = np.where(np.logical_and(
        np.logical_or(full_eroded_a_0_2_image, full_eroded_v_0_2_image),
        encoded_pred[:, :] > 0))
    full_eroded_av_0_2 = np.concatenate((full_eroded_a_0_2, full_eroded_v_0_2), axis=1)
    full_eroded_av_0_2_gt = (full_eroded_av_0_2[0], full_eroded_av_0_2[1])
    encoded_pred_full_eroded_av_0_2_pre = encoded_pred[full_eroded_av_0_2_pre]
    encoded_gt_full_eroded_av_0_2_pre = encoded_gt[full_eroded_av_0_2_pre]
    encoded_pred_full_eroded_av_0_2_gt = encoded_pred[full_eroded_av_0_2_gt]
    encoded_gt_full_eroded_av_0_2_gt = encoded_gt[full_eroded_av_0_2_gt]

    # retrieve the indices for full pixels - limited to vessels [2,3)
    full_eroded_a_2_3_image = get_full_img_from_custom(encoded_gt_a[:, :], sign='2-3')
    full_eroded_a_2_3 = np.where(full_eroded_a_2_3_image)
    full_eroded_v_2_3_image = get_full_img_from_custom(encoded_gt_v[:, :], sign='2-3')
    full_eroded_v_2_3 = np.where(full_eroded_v_2_3_image)
    encoded_pred_full_eroded_a_2_3 = encoded_pred_a[full_eroded_a_2_3]
    encoded_gt_full_eroded_a_2_3 = encoded_gt_a[full_eroded_a_2_3]
    encoded_pred_full_eroded_v_2_3 = encoded_pred_v[full_eroded_v_2_3]
    encoded_gt_full_eroded_v_2_3 = encoded_gt_v[full_eroded_v_2_3]
    full_eroded_av_2_3_pre = np.where(np.logical_and(
        np.logical_or(full_eroded_a_2_3_image, full_eroded_v_2_3_image),
        encoded_pred[:, :] > 0))
    full_eroded_av_2_3 = np.concatenate((full_eroded_a_2_3, full_eroded_v_2_3), axis=1)
    full_eroded_av_2_3_gt = (full_eroded_av_2_3[0], full_eroded_av_2_3[1])
    encoded_pred_full_eroded_av_2_3_pre = encoded_pred[full_eroded_av_2_3_pre]
    encoded_gt_full_eroded_av_2_3_pre = encoded_gt[full_eroded_av_2_3_pre]
    encoded_pred_full_eroded_av_2_3_gt = encoded_pred[full_eroded_av_2_3_gt]
    encoded_gt_full_eroded_av_2_3_gt = encoded_gt[full_eroded_av_2_3_gt]

    # compute artery detection rate
    artery_ind = np.where(encoded_gt_a > 0)
    artery_gt = encoded_gt_a[artery_ind]
    artery_pred = encoded_pred_a[artery_ind]
    vein_ind = np.where(encoded_gt_v > 0)
    vein_gt = encoded_gt_v[vein_ind]
    vein_pred = encoded_pred_v[vein_ind]
    av_ind = np.concatenate((artery_ind, vein_ind), axis=1)
    av_ind = (av_ind[0], av_ind[1])
    av_pred = encoded_pred[av_ind]
    av_gt = encoded_gt[av_ind]
    detection_rate_v = accuracy_score(vein_gt.flatten(), vein_pred.flatten())
    detection_rate_a = accuracy_score(artery_gt.flatten(), artery_pred.flatten())
    detection_rates = [detection_rate_a, detection_rate_v]

    encoded_gts = OrderedDict(
        {'av_pre': [encoded_gt_full_av, encoded_gt_full_eroded_av_0_2_pre, encoded_gt_full_eroded_av_2_3_pre,
                    encoded_gt_full_eroded_av_3_Inf_pre],
         'av_gt': [encoded_gt_full_comp_av, encoded_gt_full_eroded_av_0_2_gt, encoded_gt_full_eroded_av_2_3_gt,
                   encoded_gt_full_eroded_av_3_Inf_gt]})

    encoded_preds = OrderedDict({'av_pre': [encoded_pred_full_av, encoded_pred_full_eroded_av_0_2_pre,
                                            encoded_pred_full_eroded_av_2_3_pre,
                                            encoded_pred_full_eroded_av_3_Inf_pre],
                                 'av_gt': [encoded_pred_full_comp_av, encoded_pred_full_eroded_av_0_2_gt,
                                           encoded_pred_full_eroded_av_2_3_gt,
                                           encoded_pred_full_eroded_av_3_Inf_gt]})

    av_metric_base_pre = []
    av_metric_base_gt = []
    for av_type in encoded_gts.keys():
        metrics_num = len(encoded_gts[av_type])
        for i in range(metrics_num):
            eps = 1e-7
            y_true = encoded_gts[av_type][i]
            y_pred = encoded_preds[av_type][i]
            cur_acc = accuracy_score(encoded_gts[av_type][i].flatten(), encoded_preds[av_type][i].flatten())
            cur_F1 = f1_score(encoded_gts[av_type][i].flatten(), encoded_preds[av_type][i].flatten(),
                              average='weighted')

            # pred_binary = (av_pred == 2).astype(int)
            # gt_binary = (av_gt == 2).astype(int)
            try:
                cur_auc = roc_auc_score(y_true, y_pred)
            except:
                cur_auc = 0

            TP = np.sum((y_true == 1) & (y_pred == 1))  # true positive  (artery)
            FP = np.sum((y_true == 2) & (y_pred == 1))  # false positive
            TN = np.sum((y_true == 2) & (y_pred == 2))  # true negative (vein)
            FN = np.sum((y_true == 1) & (y_pred == 2))  # fasle negative

            sens = TP / (TP + FN + eps)
            spec = TN / (TN + FP + eps)
            if av_type == 'av_pre':
                if i == 0:
                    cur_acc = full_metric[0]
                    cur_F1 = full_metric[1]
                    sens = full_metric[2]
                    spec = full_metric[3]
                av_metric_base_pre.append([cur_acc, cur_F1, cur_auc, sens, spec])
            else:
                av_metric_base_gt.append([cur_acc, cur_F1, cur_auc, sens, spec])

    return detection_rates, av_metric_base_pre, av_metric_base_gt


def write_results_to_file(detect_av_rate,overall_value_av_cen_pre, overall_value_av_cen_gt, overall_value_av_full_pre,overall_value_av_full_gt,image_basename=None,
                          outFileName='output.xlsx'):
    overall_value_av_cen_pre = np.transpose(overall_value_av_cen_pre, (1, 2, 0))
    overall_value_av_cen_gt = np.transpose(overall_value_av_cen_gt, (1, 2, 0))
    overall_value_av_full_pre = np.transpose(overall_value_av_full_pre, (1, 2, 0))
    overall_value_av_full_gt = np.transpose(overall_value_av_full_gt, (1, 2, 0))
    av_value_single = np.empty((5, overall_value_av_cen_pre.shape[2], 4 * overall_value_av_cen_pre.shape[0]))
    for i in range(5):
        overall_value_av_cen_pre_f1 = overall_value_av_cen_pre[:, i, :]
        overall_value_av_cen_gt_f1 = overall_value_av_cen_gt[:, i, :]
        overall_value_av_full_pre_f1 = overall_value_av_full_pre[:, i, :]
        overall_value_av_full_gt_f1 = overall_value_av_full_gt[:, i, :]
        # 沿着行合并
        empty_array = np.empty((0, overall_value_av_cen_pre_f1.shape[1]))
        for j in range(overall_value_av_cen_pre_f1.shape[0]):
            av_cen = np.concatenate((overall_value_av_cen_pre_f1[j], overall_value_av_cen_gt_f1[j]), axis=0).reshape(
                (int(overall_value_av_cen_pre_f1.shape[0] / 2), overall_value_av_cen_pre_f1.shape[1]))
            empty_array = np.append(empty_array, av_cen, axis=0)

        for k in range(overall_value_av_full_pre_f1.shape[0]):
            full = np.concatenate((overall_value_av_full_pre_f1[k], overall_value_av_full_gt_f1[k]), axis=0).reshape(
                (int(overall_value_av_full_pre_f1.shape[0] / 2), overall_value_av_full_pre_f1.shape[1]))
            empty_array = np.append(empty_array, full, axis=0)

        av_value_single[i, :, :] = empty_array.T
    mean_values_av_cen_pre = np.mean(overall_value_av_cen_pre, axis=2)
    std_dev_av_cen_pre = np.std(overall_value_av_cen_pre, axis=2)
    mean_values_av_cen_gt = np.mean(overall_value_av_cen_gt, axis=2)
    std_dev_av_cen_gt = np.std(overall_value_av_cen_gt, axis=2)

    mean_values_av_full_pre = np.mean(overall_value_av_full_pre, axis=2)
    std_dev_av_full_pre = np.std(overall_value_av_full_pre, axis=2)
    mean_values_av_full_gt = np.mean(overall_value_av_full_gt, axis=2)
    std_dev_av_full_gt = np.std(overall_value_av_full_gt, axis=2)

    # 沿着列合并
    av_cen_pre = np.concatenate((mean_values_av_cen_pre, std_dev_av_cen_pre), axis=1)
    av_cen_gt = np.concatenate((mean_values_av_cen_gt, std_dev_av_cen_gt), axis=1)
    print(av_cen_pre.shape)
    full_pre = np.concatenate((mean_values_av_full_pre, std_dev_av_full_pre), axis=1)
    full_gt = np.concatenate((mean_values_av_full_gt, std_dev_av_full_gt), axis=1)
    print(full_gt.shape)
    empty_array = np.empty((0, av_cen_pre.shape[1]))
    for i in range(av_cen_pre.shape[0]):
        av_cen = np.concatenate((av_cen_pre[i], av_cen_gt[i]), axis=0).reshape(
            (int(av_cen_pre.shape[0] / 2), av_cen_pre.shape[1]))
        empty_array = np.append(empty_array, av_cen, axis=0)
    print(empty_array.shape)
    for i in range(av_cen_pre.shape[0]):
        full = np.concatenate((full_pre[i], full_gt[i]), axis=0).reshape(
            (int(av_cen_pre.shape[0] / 2), av_cen_pre.shape[1]))
        empty_array = np.append(empty_array, full, axis=0)
    av_value_all = empty_array

    column_name = ['Artery/Vein (cen pre): Discovered centerline pixels based on prediction',
                   'Artery/Vein (cen gt): Discovered centerline pixels based on ground',
                   'Artery/Vein (cen pre): Vessels in [0,2) pixels (pre)',
                   'Artery/Vein (cen gt): Vessels in [0,2) pixels (gt)',
                   'Artery/Vein (cen pre): Vessels in [2,3) pixels (pre)',
                   'Artery/Vein (cen gt): Vessels in [2,3) pixels (gt)',
                   'Artery/Vein (cen pre): Vessels in 3-Inf pixels (pre)',
                   'Artery/Vein (cen gt): Vessels in 3-Inf pixels (gt)',
                   'Artery/Vein (full pre): Discovered full pixels based on prediction',
                   'Artery/Vein (full gt): Discovered full pixels based on ground',
                   'Artery/Vein (full pre): Vessels in [0,2) pixels (pre)',
                   'Artery/Vein (full gt): Vessels in [0,2) pixels (gt)',
                   'Artery/Vein (full pre): Vessels in [2,3) pixels (pre)',
                   'Artery/Vein (full gt): Vessels in [2,3) pixels (gt)',
                   'Artery/Vein (full pre): Vessels in 3-Inf pixels (pre)',
                   'Artery/Vein (full gt): Vessels in 3-Inf pixels (gt)']

    # 单张图片指标的行名（即每张图片的文件名，不包含文件名后缀）
    row_name = image_basename
    # 单张图片的不同指标，一个sheet一种指标
    sheet_name = ['ACC', 'F1', 'AUC', 'SN(Artery ACC)', 'SP(vein ACC)']

    # 平均值和方差指标的列名
    column_name1 = ['ACC', 'F1', 'AUC', 'SN(Artery ACC)', 'SP(vein ACC)', 'ACC_std', 'F1_std', 'AUC_std',
                    'SN_std(Artery ACC)', 'SP_std(vein ACC)']
    # 平均值和方差指标的行名
    row_name1 = column_name
    # 动静脉检测率的列名
    column_name2 = ['ratio']
    # 动静脉检测率的行名
    row_name2 = ['Artery detection', 'Vein detection']
    rates = np.array(detect_av_rate).reshape((2, 1))

    # 使用ExcelWriter保存到Excel文件，指定engine为openpyxl
    with pd.ExcelWriter(outFileName, engine='openpyxl') as writer:

        for i in range(5):
            # 将数组转换为pandas DataFrame
            df = pd.DataFrame(av_value_single[i, :, :], columns=column_name)
            df.index = row_name
            # 将df写入工作表
            df.to_excel(writer, sheet_name=sheet_name[i])

        df1 = pd.DataFrame(av_value_all, columns=column_name1)
        df1.index = row_name1
        df2 = pd.DataFrame(rates, columns=column_name2)
        df2.index = row_name2
        # 将df1写入第一个工作表
        df1.to_excel(writer, sheet_name='index')
        # 将df2写入第二个工作表
        df2.to_excel(writer, sheet_name='detection')

    # print
    metrics_names_cen_pre = ['Discovered centerline pixels based on prediction', 'Vessels in [0,2) pixels (pre)',
                             'Vessels in [2,3) pixels (pre)', 'Vessels in 3-Inf pixels (pre)']
    metrics_names_cen_gt = ['Discovered centerline pixels based on ground', 'Vessels in [0,2) pixels (gt)',
                            'Vessels in [2,3) pixels (gt)', 'Vessels in 3-Inf pixels (gt)']

    metrics_names_full_pre = ['Discovered full pixels based on prediction', 'Vessels in [0,2) pixels (pre)',
                              'Vessels in [2,3) pixels (pre)', 'Vessels in 3-Inf pixels (pre)']
    metrics_names_full_gt = ['Discovered full pixels based on ground', 'Vessels in [0,2) pixels (gt)',
                             'Vessels in [2,3) pixels (gt)', 'Vessels in 3-Inf pixels (gt)']

    print('=============metric_name: (mean,std)=====================')
    print("--------------------------Centerline---------------------------------")
    print("Artery detection - Ratio:{}".format(rates[0]))
    print("Vein detection - Ratio:{}".format(rates[1]))
    for j in range(len(metrics_names_cen_pre)):
        print(
            f'Artery/Vein (cen pre): {metrics_names_cen_pre[j]} - ACC: {(mean_values_av_cen_pre[j][0], std_dev_av_cen_pre[j][0])}, F1: {(mean_values_av_cen_pre[j][1], std_dev_av_cen_pre[j][1])},AUC: {(mean_values_av_cen_pre[j][2], std_dev_av_cen_pre[j][2])}, SN (Artery ACC): {(mean_values_av_cen_pre[j][3], std_dev_av_cen_pre[j][3])}, SP (vein ACC): {(mean_values_av_cen_pre[j][4], std_dev_av_cen_pre[j][4])}')
        print(
            f'Artery/Vein (cen gt): {metrics_names_cen_gt[j]} - ACC: {(mean_values_av_cen_gt[j][0], std_dev_av_cen_gt[j][0])}, F1: {(mean_values_av_cen_gt[j][1], std_dev_av_cen_gt[j][1])},AUC: {(mean_values_av_cen_gt[j][2], std_dev_av_cen_gt[j][2])}, SN (Artery ACC): {(mean_values_av_cen_gt[j][3], std_dev_av_cen_gt[j][3])}, SP (vein ACC): {(mean_values_av_cen_gt[j][4], std_dev_av_cen_gt[j][4])}')

    print("--------------------------groundtruth---------------------------------")
    for j in range(len(metrics_names_full_gt)):
        print(
            f'Artery/Vein (full pre): {metrics_names_full_pre[j]} - ACC: {(mean_values_av_full_pre[j][0], std_dev_av_full_pre[j][0])}, F1: {(mean_values_av_full_pre[j][1], std_dev_av_full_pre[j][1])},AUC: {(mean_values_av_full_pre[j][2], std_dev_av_full_pre[j][2])}, SN (Artery ACC): {(mean_values_av_full_pre[j][3], std_dev_av_full_pre[j][3])}, SP (vein ACC): {(mean_values_av_full_pre[j][4], std_dev_av_full_pre[j][4])}')
        print(
            f'Artery/Vein (full gt): {metrics_names_full_gt[j]} - ACC: {(mean_values_av_full_gt[j][0], std_dev_av_full_gt[j][0])}, F1: {(mean_values_av_full_gt[j][1], std_dev_av_full_gt[j][1])},AUC: {(mean_values_av_full_gt[j][2], std_dev_av_full_gt[j][2])}, SN (Artery ACC): {(mean_values_av_full_gt[j][3], std_dev_av_full_gt[j][3])}, SP (vein ACC): {(mean_values_av_full_gt[j][4], std_dev_av_full_gt[j][4])}')


def AVclassifiation_pos_ve(out_path, PredAll1, PredAll2, VesselPredAll, DataSet=0, image_basename=''):
    """
    predAll1: predition results of artery
    predAll2: predition results of vein
    VesselPredAll: predition results of vessel
    DataSet: the length of dataset
    image_basename: the name of saved mask
    """

    ImgN = DataSet

    for ImgNumber in range(ImgN):

        height, width = PredAll1.shape[2:4]

        VesselProb = VesselPredAll[ImgNumber, 0, :, :]

        ArteryProb = PredAll1[ImgNumber, 0, :, :]
        VeinProb = PredAll2[ImgNumber, 0, :, :]

        VesselSeg = (VesselProb >= 0.1) & ((ArteryProb >= 0.2) | (VeinProb >= 0.2))
        # VesselSeg = (VesselProb >= 0.5) & ((ArteryProb >= 0.5) | (VeinProb >= 0.5))
        crossSeg = (VesselProb >= 0.2) & ((ArteryProb >= 0.6) & (VeinProb >= 0.6))
        VesselSeg = binaryPostProcessing3(VesselSeg, removeArea=100, fillArea=20)

        vesselPixels = np.where(VesselSeg > 0)

        ArteryProb2 = np.zeros((height, width))
        VeinProb2 = np.zeros((height, width))
        crossProb2 = np.zeros((height, width))
        image_color = np.zeros((3, height, width), dtype=np.uint8)
        for i in range(len(vesselPixels[0])):
            row = vesselPixels[0][i]
            col = vesselPixels[1][i]
            probA = ArteryProb[row, col]
            probV = VeinProb[row, col]
            ArteryProb2[row, col] = probA
            VeinProb2[row, col] = probV

        test_use_vessel = np.zeros((height, width), np.uint8)
        ArteryPred2 = ((ArteryProb2 >= 0.2) & (ArteryProb2 > VeinProb2))
        VeinPred2 = ((VeinProb2 >= 0.2) & (VeinProb2 > ArteryProb2))

        ArteryPred2 = binaryPostProcessing3(ArteryPred2, removeArea=100, fillArea=20)
        VeinPred2 = binaryPostProcessing3(VeinPred2, removeArea=100, fillArea=20)

        image_color[0, :, :] = ArteryPred2 * 255
        image_color[2, :, :] = VeinPred2 * 255
        image_color = image_color.transpose((1, 2, 0))

        imgBin_vessel = ArteryPred2 + VeinPred2
        imgBin_vessel[imgBin_vessel[:, :] == 2] = 1
        test_use_vessel = imgBin_vessel.copy() * 255

        vessel = cal_crosspoint(test_use_vessel)

        contours_vessel, hierarchy_c = cv2.findContours(vessel, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        # inter continuity
        for vessel_seg in range(len(contours_vessel)):
            C_vessel = np.zeros(vessel.shape, np.uint8)
            C_vessel = cv2.drawContours(C_vessel, contours_vessel, vessel_seg, (255, 255, 255), cv2.FILLED)
            cli = np.mean(VeinProb2[C_vessel == 255]) / np.mean(ArteryProb2[C_vessel == 255])
            if cli < 1:
                image_color[
                    (C_vessel[:, :] == 255) & (test_use_vessel[:, :] == 255)] = [255, 0, 0]
            else:
                image_color[
                    (C_vessel[:, :] == 255) & (test_use_vessel[:, :] == 255)] = [0, 0, 255]


        Image.fromarray(image_color).save(os.path.join(out_path, f'{image_basename[ImgNumber].split(".")[0]}.png'))
        Image.fromarray(test_use_vessel).save(os.path.join(out_path, f'{image_basename[ImgNumber].split(".")[0]}_vessel.png'))

def AVclassifiation(out_path, PredAll1, PredAll2, VesselPredAll, DataSet=0, image_basename=''):
    """
    predAll1: predition results of artery
    predAll2: predition results of vein
    VesselPredAll: predition results of vessel
    DataSet: the length of dataset
    image_basename: the name of saved mask
    """

    ImgN = DataSet

    for ImgNumber in range(ImgN):

        height, width = PredAll1.shape[2:4]

        VesselProb = VesselPredAll[ImgNumber, 0, :, :]

        ArteryProb = PredAll1[ImgNumber, 0, :, :]
        VeinProb = PredAll2[ImgNumber, 0, :, :]

        VesselSeg = (VesselProb >= 0.1) & ((ArteryProb >0.2) | (VeinProb > 0.2))
        # VesselSeg = (VesselProb >= 0.5) & ((ArteryProb >= 0.5) | (VeinProb >= 0.5))
        crossSeg = (VesselProb >= 0.1) & ((ArteryProb >= 0.6) & (VeinProb >= 0.6))
        VesselSeg = binaryPostProcessing3(VesselSeg, removeArea=100, fillArea=20)

        vesselPixels = np.where(VesselSeg > 0)

        ArteryProb2 = np.zeros((height, width))
        VeinProb2 = np.zeros((height, width))
        crossProb2 = np.zeros((height, width))
        image_color = np.zeros((3, height, width), dtype=np.uint8)
        for i in range(len(vesselPixels[0])):
            row = vesselPixels[0][i]
            col = vesselPixels[1][i]
            probA = ArteryProb[row, col]
            probV = VeinProb[row, col]
            #probA,probV = softmax([probA,probV])
            ArteryProb2[row, col] = probA
            VeinProb2[row, col] = probV

        test_use_vessel = np.zeros((height, width), np.uint8)
        ArteryPred2 = ((ArteryProb2 >= 0.2) & (ArteryProb2 >= VeinProb2))
        VeinPred2 = ((VeinProb2 >= 0.2) & (VeinProb2 >= ArteryProb2))

        ArteryPred2 = binaryPostProcessing3(ArteryPred2, removeArea=100, fillArea=20)
        VeinPred2 = binaryPostProcessing3(VeinPred2, removeArea=100, fillArea=20)

        image_color[0, :, :] = ArteryPred2 * 255
        image_color[2, :, :] = VeinPred2 * 255
        image_color = image_color.transpose((1, 2, 0))

        #Image.fromarray(image_color).save(os.path.join(out_path, f'{image_basename[ImgNumber].split(".")[0]}_ori.png'))

        imgBin_vessel = ArteryPred2 + VeinPred2
        imgBin_vessel[imgBin_vessel[:, :] == 2] = 1
        test_use_vessel = imgBin_vessel.copy() * 255

        vessel = cal_crosspoint(test_use_vessel)

        contours_vessel, hierarchy_c = cv2.findContours(vessel, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        # inter continuity
        for vessel_seg in range(len(contours_vessel)):
            C_vessel = np.zeros(vessel.shape, np.uint8)
            C_vessel = cv2.drawContours(C_vessel, contours_vessel, vessel_seg, (255, 255, 255), cv2.FILLED)
            cli = np.mean(VeinProb2[C_vessel == 255]) / np.mean(ArteryProb2[C_vessel == 255])
            if cli < 1:
                image_color[
                    (C_vessel[:, :] == 255) & (test_use_vessel[:, :] == 255)] = [255, 0, 0]
            else:
                image_color[
                    (C_vessel[:, :] == 255) & (test_use_vessel[:, :] == 255)] = [0, 0, 255]
        loop=0
        while loop<2:
            # out vein continuity
            vein = image_color[:, :, 2]
            contours_vein, hierarchy_b = cv2.findContours(vein, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

            vein_size = []
            for z in range(len(contours_vein)):
                vein_size.append(contours_vein[z].size)
            vein_size = np.sort(np.array(vein_size))
            # image_color_copy = np.uint8(image_color).copy()
            for vein_seg in range(len(contours_vein)):
                judge_number = min(np.mean(vein_size),500)
                # cv2.putText(image_color_copy, str(vein_seg), (int(contours_vein[vein_seg][0][0][0]), int(contours_vein[vein_seg][0][0][1])), 3, 1,
                #             color=(255, 0, 0), thickness=2)
                if contours_vein[vein_seg].size < judge_number:
                    C_vein = np.zeros(vessel.shape, np.uint8)
                    C_vein = cv2.drawContours(C_vein, contours_vein, vein_seg, (255, 255, 255), cv2.FILLED)
                    max_diameter = np.max(Skeleton(C_vein, C_vein)[1])

                    image_color_copy_vein = image_color[:, :, 2].copy()
                    image_color_copy_arter = image_color[:, :, 0].copy()
                    # a_ori = cv2.drawContours(a_ori, contours_b, k, (0, 0, 0), cv2.FILLED)
                    image_color_copy_vein = cv2.drawContours(image_color_copy_vein, contours_vein, vein_seg,
                                                             (0, 0, 0),
                                                             cv2.FILLED)
                    # image_color[(C_cross[:, :] == 255) & (image_color[:, :, 1] == 255)] = [255, 0, 0]
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (
                        4 * int(np.ceil(max_diameter)), 4 * int(np.ceil(max_diameter))))
                    C_vein_dilate = cv2.dilate(C_vein, kernel, iterations=1)
                    # cv2.imwrite(path_out_3, C_vein_dilate)
                    C_vein_dilate_judge = np.zeros(vessel.shape, np.uint8)
                    C_vein_dilate_judge[
                        (C_vein_dilate[:, :] == 255) & (image_color_copy_vein == 255)] = 1
                    C_arter_dilate_judge = np.zeros(vessel.shape, np.uint8)
                    C_arter_dilate_judge[
                        (C_vein_dilate[:, :] == 255) & (image_color_copy_arter == 255)] = 1
                    if (len(np.unique(C_vein_dilate_judge)) == 1) & (
                            len(np.unique(C_arter_dilate_judge)) != 1) & (np.mean(VeinProb2[C_vein == 255]) < 0.6):
                        image_color[
                            (C_vein[:, :] == 255) & (image_color[:, :, 2] == 255)] = [255, 0,
                                                                                      0]

            # out artery continuity
            arter = image_color[:, :, 0]
            contours_arter, hierarchy_a = cv2.findContours(arter, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            arter_size = []
            for z in range(len(contours_arter)):
                arter_size.append(contours_arter[z].size)
            arter_size = np.sort(np.array(arter_size))
            for arter_seg in range(len(contours_arter)):
                judge_number = min(np.mean(arter_size),500)

                if contours_arter[arter_seg].size < judge_number:

                    C_arter = np.zeros(vessel.shape, np.uint8)
                    C_arter = cv2.drawContours(C_arter, contours_arter, arter_seg, (255, 255, 255), cv2.FILLED)
                    max_diameter = np.max(Skeleton(C_arter, test_use_vessel)[1])

                    image_color_copy_vein = image_color[:, :, 2].copy()
                    image_color_copy_arter = image_color[:, :, 0].copy()
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (
                        4 * int(np.ceil(max_diameter)), 4 * int(np.ceil(max_diameter))))
                    image_color_copy_arter = cv2.drawContours(image_color_copy_arter, contours_arter, arter_seg,
                                                              (0, 0, 0),
                                                              cv2.FILLED)
                    C_arter_dilate = cv2.dilate(C_arter, kernel, iterations=1)
                    # image_color[(C_cross[:, :] == 255) & (image_color[:, :, 1] == 255)] = [255, 0, 0]
                    C_arter_dilate_judge = np.zeros(arter.shape, np.uint8)
                    C_arter_dilate_judge[
                        (C_arter_dilate[:, :] == 255) & (image_color_copy_arter[:, :] == 255)] = 1
                    C_vein_dilate_judge = np.zeros(arter.shape, np.uint8)
                    C_vein_dilate_judge[
                        (C_arter_dilate[:, :] == 255) & (image_color_copy_vein[:, :] == 255)] = 1

                    if (len(np.unique(C_arter_dilate_judge)) == 1) & (
                            len(np.unique(C_vein_dilate_judge)) != 1) & (np.mean(ArteryProb2[C_arter == 255]) < 0.6):
                        image_color[
                            (C_arter[:, :] == 255) & (image_color[:, :, 0] == 255)] = [0,
                                                                                       0,
                                                                                       255]
            loop=loop+1


        Image.fromarray(image_color).save(os.path.join(out_path, f'{image_basename[ImgNumber].split(".")[0]}.png'))
        #Image.fromarray(np.uint8(VesselProb*255)).save(os.path.join(out_path, f'{image_basename[ImgNumber].split(".")[0]}_vessel.png'))



def AVclassifiation_vein(out_path, PredAll1, PredAll2, VesselPredAll, DataSet=0, image_basename=''):
    """
    predAll1: predition results of artery
    predAll2: predition results of vein
    VesselPredAll: predition results of vessel
    DataSet: the length of dataset
    image_basename: the name of saved mask
    """

    ImgN = DataSet

    for ImgNumber in range(ImgN):

        height, width = PredAll1.shape[2:4]

        VesselProb = VesselPredAll[ImgNumber, 0, :, :]

        ArteryProb = PredAll1[ImgNumber, 0, :, :]
        VeinProb = PredAll2[ImgNumber, 0, :, :]

        VesselSeg = (VesselProb >= 0.1) & ((ArteryProb >= 0.1) | (VeinProb >= 0.1))
        # VesselSeg = (VesselProb >= 0.5) & ((ArteryProb >= 0.5) | (VeinProb >= 0.5))
        crossSeg = (VesselProb >= 0.1) & ((ArteryProb >= 0.6) & (VeinProb >= 0.6))
        VesselSeg = binaryPostProcessing3(VesselSeg, removeArea=100, fillArea=20)

        vesselPixels = np.where(VesselSeg > 0)

        ArteryProb2 = np.zeros((height, width))
        VeinProb2 = np.zeros((height, width))
        crossProb2 = np.zeros((height, width))
        image_color = np.zeros((3, height, width), dtype=np.uint8)
        for i in range(len(vesselPixels[0])):
            row = vesselPixels[0][i]
            col = vesselPixels[1][i]
            probA = ArteryProb[row, col]
            probV = VeinProb[row, col]
            ArteryProb2[row, col] = probA
            VeinProb2[row, col] = probV

        test_use_vessel = np.zeros((height, width), np.uint8)
        ArteryPred2 = ((ArteryProb2 >= 0.1) & (ArteryProb2 > VeinProb2))
        VeinPred2 = ((VeinProb2 >= 0.1) & (VeinProb2 > ArteryProb2))

        ArteryPred2 = binaryPostProcessing3(ArteryPred2, removeArea=50, fillArea=20)
        VeinPred2 = binaryPostProcessing3(VeinPred2, removeArea=50, fillArea=20)

        image_color[0, :, :] = ArteryPred2 * 255
        image_color[2, :, :] = VeinPred2 * 255
        image_color = image_color.transpose((1, 2, 0))

        imgBin_vessel = ArteryPred2 + VeinPred2
        imgBin_vessel[imgBin_vessel[:, :] == 2] = 1
        test_use_vessel = imgBin_vessel.copy() * 255

        vessel = cal_crosspoint(test_use_vessel)

        contours_vessel, hierarchy_c = cv2.findContours(vessel, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        # inter continuity
        for vessel_seg in range(len(contours_vessel)):
            C_vessel = np.zeros(vessel.shape, np.uint8)
            C_vessel = cv2.drawContours(C_vessel, contours_vessel, vessel_seg, (255, 255, 255), cv2.FILLED)
            cli = np.mean(VeinProb2[C_vessel == 255]) / np.mean(ArteryProb2[C_vessel == 255])
            if cli < 1:
                image_color[
                    (C_vessel[:, :] == 255) & (test_use_vessel[:, :] == 255)] = [255, 0, 0]
            else:
                image_color[
                    (C_vessel[:, :] == 255) & (test_use_vessel[:, :] == 255)] = [0, 0, 255]

       

        # out artery continuity
        arter = image_color[:, :, 0]
        contours_arter, hierarchy_a = cv2.findContours(arter, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        arter_size = []
        for z in range(len(contours_arter)):
            arter_size.append(contours_arter[z].size)
        arter_size = np.sort(np.array(arter_size))
        for arter_seg in range(len(contours_arter)):
            judge_number = min(np.mean(arter_size),500)

            if contours_arter[arter_seg].size < judge_number:

                C_arter = np.zeros(vessel.shape, np.uint8)
                C_arter = cv2.drawContours(C_arter, contours_arter, arter_seg, (255, 255, 255), cv2.FILLED)
                max_diameter = np.max(Skeleton(C_arter, test_use_vessel)[1])

                image_color_copy_vein = image_color[:, :, 2].copy()
                image_color_copy_arter = image_color[:, :, 0].copy()
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (
                    4 * int(np.ceil(max_diameter)), 4 * int(np.ceil(max_diameter))))
                image_color_copy_arter = cv2.drawContours(image_color_copy_arter, contours_arter, arter_seg,
                                                          (0, 0, 0),
                                                          cv2.FILLED)
                C_arter_dilate = cv2.dilate(C_arter, kernel, iterations=1)
                # image_color[(C_cross[:, :] == 255) & (image_color[:, :, 1] == 255)] = [255, 0, 0]
                C_arter_dilate_judge = np.zeros(arter.shape, np.uint8)
                C_arter_dilate_judge[
                    (C_arter_dilate[:, :] == 255) & (image_color_copy_arter[:, :] == 255)] = 1
                C_vein_dilate_judge = np.zeros(arter.shape, np.uint8)
                C_vein_dilate_judge[
                    (C_arter_dilate[:, :] == 255) & (image_color_copy_vein[:, :] == 255)] = 1

                if (len(np.unique(C_arter_dilate_judge)) == 1) & (
                        len(np.unique(C_vein_dilate_judge)) != 1) & (np.mean(VeinProb2[C_arter == 255]) < 0.5):
                    image_color[
                        (C_arter[:, :] == 255) & (image_color[:, :, 0] == 255)] = [0,
                                                                                   0,
                                                                                   255]

         # out vein continuity
        vein = image_color[:, :, 2]
        contours_vein, hierarchy_b = cv2.findContours(vein, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        vein_size = []
        for z in range(len(contours_vein)):
            vein_size.append(contours_vein[z].size)
        vein_size = np.sort(np.array(vein_size))
        # image_color_copy = np.uint8(image_color).copy()
        for vein_seg in range(len(contours_vein)):
            judge_number = min(np.mean(vein_size),500)
            # cv2.putText(image_color_copy, str(vein_seg), (int(contours_vein[vein_seg][0][0][0]), int(contours_vein[vein_seg][0][0][1])), 3, 1,
            #             color=(255, 0, 0), thickness=2)
            if contours_vein[vein_seg].size < judge_number:
                C_vein = np.zeros(vessel.shape, np.uint8)
                C_vein = cv2.drawContours(C_vein, contours_vein, vein_seg, (255, 255, 255), cv2.FILLED)
                max_diameter = np.max(Skeleton(C_vein, C_vein)[1])

                image_color_copy_vein = image_color[:, :, 2].copy()
                image_color_copy_arter = image_color[:, :, 0].copy()
                # a_ori = cv2.drawContours(a_ori, contours_b, k, (0, 0, 0), cv2.FILLED)
                image_color_copy_vein = cv2.drawContours(image_color_copy_vein, contours_vein, vein_seg,
                                                         (0, 0, 0),
                                                         cv2.FILLED)
                # image_color[(C_cross[:, :] == 255) & (image_color[:, :, 1] == 255)] = [255, 0, 0]
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (
                    4 * int(np.ceil(max_diameter)), 4 * int(np.ceil(max_diameter))))
                C_vein_dilate = cv2.dilate(C_vein, kernel, iterations=1)
                # cv2.imwrite(path_out_3, C_vein_dilate)
                C_vein_dilate_judge = np.zeros(vessel.shape, np.uint8)
                C_vein_dilate_judge[
                    (C_vein_dilate[:, :] == 255) & (image_color_copy_vein == 255)] = 1
                C_arter_dilate_judge = np.zeros(vessel.shape, np.uint8)
                C_arter_dilate_judge[
                    (C_vein_dilate[:, :] == 255) & (image_color_copy_arter == 255)] = 1

                if (len(np.unique(C_vein_dilate_judge)) == 1) & (
                        len(np.unique(C_arter_dilate_judge)) != 1) :
                    image_color[
                        (C_vein[:, :] == 255) & (image_color[:, :, 2] == 255)] = [255, 0,
                                                                                  0]

        Image.fromarray(image_color).save(os.path.join(out_path, f'{image_basename[ImgNumber].split(".")[0]}.png'))
        Image.fromarray(imgBin_vessel*255).save(os.path.join(out_path, f'{image_basename[ImgNumber].split(".")[0]}_vessel.png'))


def AVclassifiationMetrics_skeletonPixles(PredAll1,PredAll2,VesselPredAll,LabelAll1,LabelAll2,LabelVesselAll,DataSet=0, onlyMeasureSkeleton=False, strict_mode=True,image_basename=None,outFileName=None):
    
    """
    predAll1: predition results of artery
    predAll2: predition results of vein
    VesselPredAll: predition results of vessel
    LabelAll1: label of artery
    LabelAll2: label of vein
    LabelVesselAll: label of vessel
    DataSet: the length of dataset
    onlyMeasureSkeleton: measure skeleton
    strict_mode: strict
    """

    ImgN = DataSet
        
    senList = []
    specList = []
    accList = []
    f1List = []
    ioulist = []
    diceList = []

    senList_sk = []
    specList_sk = []
    accList_sk = []
    f1List_sk = []
    ioulist_sk = []
    diceList_sk = []

    bad_case_count = 0
    bad_case_index = []
    ArteryVeinLabelImgs= np.zeros((ImgN,PredAll1.shape[2], PredAll1.shape[3], 3), np.uint8)
    ArteryVeinPreImgs = np.zeros((ImgN,PredAll1.shape[2], PredAll1.shape[3], 3), np.uint8)

    rates = [0, 0]
    overall_value_av_cen_pre = []
    overall_value_av_cen_gt = []
    overall_value_av_full_pre = []
    overall_value_av_full_gt = []

    for ImgNumber in range(ImgN):
        
        height, width = PredAll1.shape[2:4]
        
    
        VesselProb = VesselPredAll[ImgNumber, 0,:,:]
        VesselLabel = LabelVesselAll[ImgNumber, 0, :, :]
    
    
        ArteryLabel = LabelAll1[ImgNumber, 0, :, :]
        VeinLabel = LabelAll2[ImgNumber, 0, :, :]
    
        ArteryProb = PredAll1[ImgNumber, 0,:,:]
        VeinProb = PredAll2[ImgNumber, 0,:,:]
        
        if strict_mode:
            VesselSeg = VesselLabel
            VesselSegDraw = (VesselProb >= 0.1)
            VesselSegDraw= binaryPostProcessing3(VesselSegDraw, removeArea=50, fillArea=20)
            
        else:
            VesselSeg = (VesselProb >= 0.5) & ((ArteryProb >= 0.2) | (VeinProb >= 0.2))
            VesselSeg= binaryPostProcessing3(VesselSeg, removeArea=100, fillArea=20)
        
        vesselPixels = np.where(VesselSeg>0)
        vesselPixelsDraw = np.where(VesselSegDraw>0)
        ArteryProb2 = np.zeros((height,width))
        VeinProb2 = np.zeros((height,width))


        ArteryProb2Draw = np.zeros((height,width))
        VeinProb2Draw = np.zeros((height,width))
        
        for i in range(len(vesselPixelsDraw[0])):
            row = vesselPixelsDraw[0][i]
            col = vesselPixelsDraw[1][i]
            probA = ArteryProb[row, col]
            probV = VeinProb[row, col]
            ArteryProb2Draw[row, col] = probA
            VeinProb2Draw[row, col] = probV

        for i in range(len(vesselPixels[0])):
            row = vesselPixels[0][i]
            col = vesselPixels[1][i]
            probA = ArteryProb[row, col]
            probV = VeinProb[row, col]
            ArteryProb2[row, col] = probA
            VeinProb2[row, col] = probV

        threshold = 0.2
        ArteryLabelImg2= ArteryLabel.copy()
        VeinLabelImg2= VeinLabel.copy()


        ArteryLabelImg2 [VesselSeg == 0] = 0
        VeinLabelImg2 [VesselSeg == 0] = 0
        ArteryVeinLabelImg = np.zeros((height, width,3), np.uint8)
        ArteryVeinLabelImg[ArteryLabelImg2>0] = (255, 0, 0)
        ArteryVeinLabelImg[VeinLabelImg2>0] = (0, 0, 255)
        ArteryVeinLabelImgs[ImgNumber] = ArteryVeinLabelImg
        ArteryVeinLabelCommon = np.bitwise_and(ArteryLabelImg2>0, VeinLabelImg2>0)
        ArteryVeinLabelImg[ArteryLabelImg2 > 0] = (255, 0, 0)
        ArteryVeinLabelImg[VeinLabelImg2 > 0] = (0, 0, 255)
        ArteryVeinLabelImg[ArteryVeinLabelCommon > 0] = (0, 255, 0)
        ArteryVeinLabelImgs[ImgNumber] = ArteryVeinLabelImg

        ArteryPred2 = (ArteryProb2 > threshold) & (ArteryProb2 >= VeinProb2)
        VeinPred2 = (VeinProb2 >= threshold) &  (VeinProb2 >= ArteryProb2)
        ArteryPred2 = binaryPostProcessing3(ArteryPred2, removeArea=100, fillArea=20)
        VeinPred2 = binaryPostProcessing3(VeinPred2, removeArea=100, fillArea=20)

        TPimg =  np.bitwise_and(ArteryPred2>0, ArteryLabelImg2>0) # 真实为动脉，预测为动脉
        TNimg =  np.bitwise_and(VeinPred2>0, VeinLabelImg2>0) # 真实为静脉，预测为静脉
        FPimg = np.bitwise_and(ArteryPred2>0, VeinLabelImg2>0) # 真实为静脉，预测为动脉
        FPimg = np.bitwise_and(FPimg, np.bitwise_not(ArteryVeinLabelCommon))  # 真实为静脉，预测为动脉，且不属于动静脉共存区域
    
        FNimg = np.bitwise_and(VeinPred2>0, ArteryLabelImg2>0) # 真实为动脉，预测为静脉
        FNimg = np.bitwise_and(FNimg, np.bitwise_not(ArteryVeinLabelCommon)) # 真实为动脉，预测为静脉，且不属于动静脉共存区域

        Skeleton = np.uint8(skeletonize(VesselSeg))
        ArterySkeletonLabel = cv2.bitwise_and(ArteryLabelImg2, ArteryLabelImg2, mask=Skeleton)
        VeinSkeletonLabel = cv2.bitwise_and(VeinLabelImg2, VeinLabelImg2, mask=Skeleton)
        ArterySkeletonPred = cv2.bitwise_and(ArteryPred2, ArteryPred2, mask=Skeleton)
        VeinSkeletonPred = cv2.bitwise_and(VeinPred2, VeinPred2, mask=Skeleton)

        ArteryVeinPred_sk = np.zeros((height, width, 3), np.uint8)
        skeletonPixles = np.where(Skeleton > 0)

        TPa_sk = 0
        TNa_sk = 0
        FPa_sk = 0
        FNa_sk = 0
        for i in range(len(skeletonPixles[0])):
            row = skeletonPixles[0][i]
            col = skeletonPixles[1][i]
            if ArterySkeletonLabel[row, col] == 1 and ArterySkeletonPred[row, col] == 1:
                TPa_sk = TPa_sk + 1

            elif VeinSkeletonLabel[row, col] == 1 and VeinSkeletonPred[row, col] == 1:
                TNa_sk = TNa_sk + 1

            elif ArterySkeletonLabel[row, col] == 1 and VeinSkeletonPred[row, col] == 1 \
                    and ArteryVeinLabelCommon[row, col] == 0:
                FNa_sk = FNa_sk + 1
            elif VeinSkeletonLabel[row, col] == 1 and ArterySkeletonPred[row, col] == 1 \
                    and ArteryVeinLabelCommon[row, col] == 0:
                FPa_sk = FPa_sk + 1
            else:
                pass
        if (TPa_sk + FNa_sk) == 0 and (TNa_sk + FPa_sk) == 0 and (TPa_sk + TNa_sk + FPa_sk + FNa_sk) == 0:
            bad_case_count += 1
            bad_case_index.append(ImgNumber)
        sensitivity_sk = TPa_sk / (TPa_sk + FNa_sk)
        specificity_sk = TNa_sk / (TNa_sk + FPa_sk)
        acc_sk = (TPa_sk + TNa_sk) / (TPa_sk + TNa_sk + FPa_sk + FNa_sk)
        f1_sk = 2 * TPa_sk / (2 * TPa_sk + FPa_sk + FNa_sk)
        dice_sk = 2 * TPa_sk / (2 * TPa_sk + FPa_sk + FNa_sk)
        iou_sk = TPa_sk / (TPa_sk + FPa_sk + FNa_sk)

        senList_sk.append(sensitivity_sk)
        specList_sk.append(specificity_sk)
        accList_sk.append(acc_sk)
        f1List_sk.append(f1_sk)
        diceList_sk.append(dice_sk)
        ioulist_sk.append(iou_sk)

        TPa = np.count_nonzero(TPimg)
        TNa = np.count_nonzero(TNimg)
        FPa = np.count_nonzero(FPimg)
        FNa = np.count_nonzero(FNimg)
        sensitivity = TPa/(TPa+FNa+1e-7)
        specificity = TNa/(TNa + FPa+1e-7)
        acc = (TPa + TNa) /(TPa + TNa + FPa + FNa+1e-7)
        f1 = 2*TPa/(2*TPa + FPa + FNa+1e-7)
        dice = 2*TPa/(2*TPa + FPa + FNa+1e-7)
        iou = TPa/(TPa + FPa + FNa+1e-7)

        senList.append(sensitivity)
        specList.append(specificity)
        accList.append(acc)
        f1List.append(f1)
        diceList.append(dice)
        ioulist.append(iou)

        ArteryVeinPredImg_different_vessel_width = np.zeros((height, width, 3), np.uint8)
        ArteryPred2Image = (ArteryProb2Draw > threshold) & (ArteryProb2Draw >= VeinProb2Draw)
        VeinPred2Image = (VeinProb2Draw >= threshold) & (ArteryProb2Draw <= VeinProb2Draw)
        ArteryVeinPredImg_different_vessel_width[ArteryPred2Image > 0] = (255, 0, 0)
        ArteryVeinPredImg_different_vessel_width[VeinPred2Image > 0] = (0, 0, 255)
        ArteryVeinPredImg_different_vessel_width[np.logical_and(ArteryPred2Image > 0, VeinPred2Image > 0)] = (0, 255, 0)


        detection_rates, av_cen_metrics_pre,av_cen_metrics_gt = evaluation_cen_av_test(ArteryVeinPredImg_different_vessel_width,ArteryVeinLabelImg,full_metric=[acc_sk,f1_sk,sensitivity_sk,specificity_sk])

        _,av_full_metrics_pre,av_full_metrics_gt = evaluation_full_av_test(ArteryVeinPredImg_different_vessel_width, ArteryVeinLabelImg, full_metric=[acc, f1, sensitivity, specificity])

        rates[0] += detection_rates[0]
        rates[1] += detection_rates[1]
        overall_value_av_cen_pre.append(av_cen_metrics_pre)
        overall_value_av_cen_gt.append(av_cen_metrics_gt)
        overall_value_av_full_pre.append(av_full_metrics_pre)
        overall_value_av_full_gt.append(av_full_metrics_gt)

    rates[0] /= ImgN
    rates[1] /= ImgN

    write_results_to_file(detect_av_rate=rates,
                          overall_value_av_cen_pre=overall_value_av_cen_pre,
                          overall_value_av_cen_gt=overall_value_av_cen_gt,
                          overall_value_av_full_pre=overall_value_av_full_pre,
                          overall_value_av_full_gt=overall_value_av_full_gt,
                          image_basename=image_basename,
                          outFileName=outFileName)



    print('Avg Pixel-wise Performance:', (np.mean(accList),np.std(accList)), (np.mean(senList),np.std(senList)), (np.mean(specList),np.std(specList)))
    print('Avg centerline Performance:', (np.mean(accList_sk),np.std(accList_sk)), (np.mean(senList_sk),np.std(senList_sk)), (np.mean(specList_sk),np.std(specList_sk)))

    return (np.mean(accList),np.std(accList)), (np.mean(specList),np.std(specList)),(np.mean(senList),np.std(senList)),(np.mean(f1List),np.std(f1List)),(np.mean(diceList),np.std(diceList)),(np.mean(ioulist),np.std(ioulist))



if __name__ == '__main__':


    pro_path = r'F:\dw\RIP-AV\AV\log\DRIVE\running_result\ProMap_testset.npy'
    ps = np.load(pro_path)
    AVclassifiation(r'./', ps[:, 0:1, :, :], ps[:, 1:2, :, :], ps[:, 2:, :, :], DataSet=ps.shape[0], image_basename=[str(i)+'.png' for i in range(20)])