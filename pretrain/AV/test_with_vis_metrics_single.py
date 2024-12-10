import os

import torch.autograd as autograd
import torch
import cv2
import numpy as np
from tqdm import tqdm
import os
from Tools.ImageResize import creatMask, shift_rgb
from models.network import PGNet
from lib.Utils import *
from Tools.AVclassifiationMetrics_v3 import AVclassifiationMetrics_skeletonPixles, AVclassifiation
from PIL import Image
from sklearn.metrics import roc_auc_score
from Tools.evalution_vessel import evalue

import logging

logger = logging.getLogger('dev')
logging.basicConfig(filename=os.path.join(f'./app_hy_re'), level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


def CAM(x, rate=0.8, ind=0):
    """
    :param dataset_path: 计算整个训练数据集的平均RGB通道值
    :param image:  array， 单张图片的array 形式
    :return: array形式的cam后的结果
    """
    # 每次使用新数据集时都需要重新计算前面的RBG平均值
    # RGB-->Rshift-->CLAHE
    #print(x.shape)
    x = np.uint8(x)
    _, Mask0 = creatMask(x, threshold=10)
    Mask = np.zeros((x.shape[0], x.shape[1]), np.float32)
    Mask[Mask0 > 0] = 1

    resize = False
    R_mea_num, G_mea_num, B_mea_num = [], [], []

    dataset_paths = [
        r'./data/out/test/test_18.png',
        r'D:\su-lab\code\RIP-AV-main\AV\data\STU\test\images\2001140005.jpg',
    ]
    dataset_path = dataset_paths[ind]
    image = np.array(Image.open(dataset_path))
    R_mea_num.append(np.mean(image[:, :, 0]))
    G_mea_num.append(np.mean(image[:, :, 1]))
    B_mea_num.append(np.mean(image[:, :, 2]))
    print(f'target RGB = [{np.mean(image[:, :, 0])},{np.mean(image[:, :, 1])},{np.mean(image[:, :, 2])}]')
    print(f'source RGB = [{np.mean(x[:, :, 0])},{np.mean(x[:, :, 1])},{np.mean(x[:, :, 2])}]')
    mea2stand = int((np.mean(R_mea_num) - np.mean(x[:, :, 0])) * rate)
    mea2standg = int((np.mean(G_mea_num) - np.mean(x[:, :, 1])) * rate)
    mea2standb = int((np.mean(B_mea_num) - np.mean(x[:, :, 2])) * rate)

    y = shift_rgb(x, mea2stand, mea2standg, mea2standb)
    #print(y.shape)

    y[Mask == 0, :] = 0
    print(f'source to target RGB = [{np.mean(y[:, :, 0])},{np.mean(y[:, :, 1])},{np.mean(y[:, :, 2])}]')

    return y


def modelEvalution_out_big(i, net, savePath, use_cuda=False, dataset='DRIVE', is_kill_border=True, input_ch=3,
                           strict_mode=True, config=None, output_dir='', evaluate_metrics=False):
    # path for images to save
    n_classes = 3
    Net = PGNet(resnet=config.use_network, use_global_semantic=config.use_global_semantic, input_ch=input_ch,
                num_classes=n_classes, use_cuda=use_cuda, pretrained=False, centerness=config.use_centerness,
                centerness_map_size=config.centerness_map_size)
    msg = Net.load_state_dict(net, strict=False)
    logger.info(msg)
    if use_cuda:
        Net.cuda()
    Net.eval()

    dataset_dict = {'STU': 'STU', 'out': 'out', 'LES': 'LES_AV', 'DRIVE': 'AV_DRIVE', 'hrf': 'hrf', 'ukbb': 'ukbb'}
    dataset_name = dataset_dict[dataset]
    logger.info(f'evaluating {dataset_name} dataset...')
    image_basename = sorted(os.listdir(f'./data/{dataset_name}/test/{output_dir}'))
    logger.info(f'num of test images: {len(image_basename)}')
    image0 = cv2.imread(f'./data/{dataset_name}/test/{output_dir}/{image_basename[0]}')
    data_path = os.path.join(savePath, dataset, output_dir)

    if not os.path.exists(data_path):
        os.makedirs(data_path)
    start_end_list = []
    step = 1
    # every step of between star and end for loop until len(image_basename)
    for i in range(0, len(image_basename), step):
        if i + step < len(image_basename):
            start_end_list.append((i, i + step))
        else:
            start_end_list.append((i, len(image_basename)))
    for start_end in start_end_list:
        logger.info(f'num of test images {start_end}')
        image0 = cv2.imread(f'./data/{dataset_name}/test/{output_dir}/{image_basename[start_end[0]]}')
        image_basename_start_end = image_basename[start_end[0]:start_end[1]]
        test_image_num = start_end[1] - start_end[0]
        test_image_height = image0.shape[0]
        test_image_width = image0.shape[1]

        logger.info(f'num of test images {start_end[0]} ori shape is {test_image_height},{test_image_width}')
        if config.use_resize:
            if min(test_image_height, test_image_width) <= 256:
                test_image_width = 512
                test_image_height = int(test_image_width * (image0.shape[0] / image0.shape[1]))
                logger.info(f'num of test images {start_end[0]}  reshape is {test_image_height},{test_image_width}')
            elif max(test_image_height, test_image_width) >= 2000:
                test_image_width = 1536
                test_image_height = int(test_image_width * (image0.shape[0] / image0.shape[1]))
                logger.info(f'num of test images {start_end[0]}  reshape is {test_image_height},{test_image_width}')
        if evaluate_metrics:
            label_basename = sorted(os.listdir(f'./data/{dataset_name}/test/av'))
            assert len(image_basename) == len(label_basename)
            LabelMap = np.zeros((test_image_num, 3, test_image_height, test_image_width), np.float32)
            LabelArteryAll = np.zeros((test_image_num, 1, test_image_height, test_image_width), np.float32)
            LabelVeinAll = np.zeros((test_image_num, 1, test_image_height, test_image_width), np.float32)
            LabelVesselAll = np.zeros((test_image_num, 1, test_image_height, test_image_width), np.float32)
            LabelVesselNoAll = np.zeros((test_image_num, 1, test_image_height, test_image_width), np.float32)

        # Vessel = VesselProMap('./data/AV_DRIVE/test/images')

        for k in tqdm(range(start_end[0], start_end[1])):
            ArteryPredAll = np.zeros((test_image_num, 1, test_image_height, test_image_width), np.float32)
            VeinPredAll = np.zeros((test_image_num, 1, test_image_height, test_image_width), np.float32)
            VesselPredAll = np.zeros((test_image_num, 1, test_image_height, test_image_width), np.float32)
            ProMap = np.zeros((test_image_num, 3, test_image_height, test_image_width), np.float32)
            MaskAll = np.zeros((test_image_num, 1, test_image_height, test_image_width), np.float32)
            ArteryPred, VeinPred, VesselPred, Mask, LabelArtery, LabelVein, LabelVessel = GetResult_out_big(Net, k,
                                                                                                            use_cuda=use_cuda,
                                                                                                            dataset_name=dataset_name,
                                                                                                            is_kill_border=is_kill_border,
                                                                                                            input_ch=input_ch,
                                                                                                            config=config,
                                                                                                            resize_w_h=(
                                                                                                                test_image_width,
                                                                                                                test_image_height),
                                                                                                            output_dir=output_dir,
                                                                                                            evaluate_metrics=evaluate_metrics
                                                                                                            )
            ArteryPredAll[k % step, :, :, :] = ArteryPred
            VeinPredAll[k % step, :, :, :] = VeinPred
            VesselPredAll[k % step, :, :, :] = VesselPred

            MaskAll[k % step, :, :, :] = Mask
            if evaluate_metrics:
                LabelArteryAll[k % step, :, :, :] = LabelArtery
                # print(np.unique(LabelArtery))
                LabelVeinAll[k % step, :, :, :] = LabelVein
                LabelVesselAll[k % step, :, :, :] = LabelVessel

        # ProMap[:, 0, :, :] = ArteryPredAll[:, 0, :, :]
        # ProMap[:, 1, :, :] = VeinPredAll[:, 0, :, :]
        # ProMap[:, 2, :, :] = VesselPredAll[:, 0, :, :]

        if evaluate_metrics:
            LabelMap[:, 0, :, :] = LabelArteryAll[:, 0, :, :]
            LabelMap[:, 1, :, :] = LabelVeinAll[:, 0, :, :]
            LabelMap[:, 2, :, :] = LabelVesselAll[:, 0, :, :]

            VesselAUC, VesselAcc, VesselSp, VesselSe, VesselF1, VesselDice, VesselIou = evalue(VesselPredAll,
                                                                                               LabelVesselAll, MaskAll)

            AveAcc, VeinAcc, ArteryAcc, AveF1, AveDice, AveIou = AVclassifiationMetrics_skeletonPixles(ArteryPredAll,
                                                                                                       VeinPredAll,
                                                                                                       VesselPredAll,
                                                                                                       LabelArteryAll,
                                                                                                       LabelVeinAll,
                                                                                                       LabelVesselAll,
                                                                                                       DataSet=test_image_num,

                                                                                                       strict_mode=strict_mode,
                                                                                                       image_basename=image_basename,
                                                                                                       outFileName=os.path.join(
                                                                                                           savePath,
                                                                                                           f'{dataset_name}_{config.model_step_pretrained_G}_output.xlsx'))

            threshold_confusion = 0.0
            # ind = np.where(np.logical_and(np.logical_or(ArteryPredAll > threshold_confusion,VeinPredAll>threshold_confusion), MaskAll > 0))
            ind = np.where(
                np.logical_and(np.logical_or(LabelArteryAll > threshold_confusion, LabelVeinAll > threshold_confusion),
                               MaskAll > 0))

            y_scores1 = ArteryPredAll[ind]
            y_true1 = LabelArteryAll[ind]
            y_scores2 = VeinPredAll[ind]
            y_true2 = LabelVeinAll[ind]

            AUC1 = roc_auc_score(y_true1, y_scores1)  # 动脉
            AUC2 = roc_auc_score(y_true2, y_scores2)  # 静脉

            print(f"========================={dataset}=============================")
            print("Strict mode:{}".format(strict_mode))
            print(f"The {i} step Average artery AUC is:{AUC1}")
            print(f"The {i} step Average vein AUC is:{AUC2}")

            print(f"The {i} step Average Acc is:{AveAcc}")
            print(f"The {i} step Average F1 is:{AveF1}")
            print(f"The {i} step Average Dice is:{AveDice}")
            print(f"The {i} step Average Iou is:{AveIou}")
            print("-----------------------------------------------------------")

            print(f"The {i} step Artery Acc is:{ArteryAcc}")
            print("-----------------------------------------------------------")

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

        # filewriter = centerline_eval(ProMap, config)
        # np.save(os.path.join(savePath, f"{dataset}_{output_dir}_ProMap_testset.npy"), ProMap)

        AVclassifiation(data_path, ArteryPredAll, VeinPredAll, VesselPredAll, test_image_num, image_basename_start_end)
        del ArteryPredAll, VeinPredAll, VesselPredAll, ProMap, MaskAll


def GetResult_out_big(Net, k, use_cuda=False, dataset_name='DRIVE', is_kill_border=False, input_ch=3, config=None,
                      resize_w_h=None, output_dir='images', evaluate_metrics=False):
    image_basename = sorted(os.listdir(f'./data/{dataset_name}/test/{output_dir}'))[k]
    if evaluate_metrics:
        label_basename = sorted(os.listdir(f'./data/{dataset_name}/test/av'))[k]
        assert image_basename.split('.')[0] == label_basename.split('.')[0]  # check if the image and label are matched
        LabelName = os.path.join(f'./data/{dataset_name}/test/av/', label_basename)

    ImgName = os.path.join(f'./data/{dataset_name}/test/{output_dir}/', image_basename)
    logger.info(f'processing {k}: {ImgName}')
    Img0 = cv2.imread(ImgName)
    # Label0 = cv2.imread(LabelName)
    _, Mask0 = creatMask(Img0, threshold=-1)
    Mask = np.zeros((Img0.shape[0], Img0.shape[1]), np.float32)
    Mask[Mask0 > 0] = 1

    if config.use_resize:
        Img0 = cv2.resize(Img0, resize_w_h)
        # Label0 = cv2.resize(Label0, config.resize_w_h, interpolation=cv2.INTER_NEAREST)
        Mask = cv2.resize(Mask, resize_w_h, interpolation=cv2.INTER_NEAREST)
    if evaluate_metrics:
        Label0 = cv2.imread(LabelName)
        if config.use_resize:
            Label0 = cv2.resize(Label0, resize_w_h, interpolation=cv2.INTER_NEAREST)
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

    # # # 将图像转换为 LAB 颜色空间
    # lab = cv2.cvtColor(Img, cv2.COLOR_RGB2LAB)

    # # 拆分 LAB 通道
    # l, a, b = cv2.split(lab)

    # # 创建 CLAHE 对象并应用到 L 通道
    # clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(8, 8))
    # l_clahe = clahe.apply(l)

    # # 将 CLAHE 处理后的 L 通道与原始的 A 和 B 通道合并
    # lab_clahe = cv2.merge((l_clahe, a, b))

    # # 将图像转换回 BGR 颜色空间
    # Img = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2RGB)

    # rgb2rgg
    if cfg.use_CAM:
        Img  = CAM(Img)

    Img = np.float32(Img / 255.)
    Img_enlarged = paint_border_overlap(Img, patch_height, patch_width, stride_height, stride_width)
    patch_size = config.patch_size
    batch_size = 16
    patches_imgs, global_images = extract_ordered_overlap_big(Img_enlarged, patch_height, patch_width, stride_height,
                                                              stride_width)

    patches_imgs = np.transpose(patches_imgs, (0, 3, 1, 2))
    patches_imgs = Normalize(patches_imgs)
    global_images = np.transpose(global_images, (0, 3, 1, 2))
    global_images = Normalize(global_images)
    patchNum = patches_imgs.shape[0]
    max_iter = int(np.ceil(patchNum / float(batch_size)))

    pred_patches = np.zeros((patchNum, n_classes, patch_size, patch_size), np.float32)
    # print(f'pred_patches shape {pred_patches.shape}')
    for i in range(max_iter):
        begin_index = i * batch_size
        end_index = (i + 1) * batch_size

        patches_temp1 = patches_imgs[begin_index:end_index, :, :, :]

        patches_input_temp1 = torch.FloatTensor(patches_temp1)
        global_input_temp1 = patches_input_temp1
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

        pred_patches[begin_index:end_index, :, :, :] = pred_patches_temp_sigmoid[:, :, :patch_size, :patch_size]

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
    if evaluate_metrics:
        LabelArtery = LabelArtery[np.newaxis, :, :]
        LabelVein = LabelVein[np.newaxis, :, :]
        LabelVessel = LabelVessel[np.newaxis, :, :]
        LabelVessel_no_unknow = LabelVessel_no_unknow[np.newaxis, :, :]

        return ArteryPred, VeinPred, VesselPred, Mask, LabelArtery, LabelVein, LabelVessel

    return ArteryPred, VeinPred, VesselPred, Mask, ArteryPred, VeinPred, VesselPred,


def out_test(cfg, output_dir='', evaluate_metrics=0):
    device = torch.device("cuda" if cfg.use_cuda else "cpu")
    model_root = cfg.model_path_pretrained_G
    model_path = os.path.join(model_root, 'G_' + str(cfg.model_step_pretrained_G) + '.pkl')
    net = torch.load(model_path, map_location=device)
    result_folder = os.path.join(model_root, 'running_result')
    modelEvalution_out_big(cfg.model_step_pretrained_G, net,
                           result_folder,
                           use_cuda=cfg.use_cuda,
                           dataset='out',
                           input_ch=cfg.input_nc,
                           config=cfg,
                           strict_mode=True, output_dir=output_dir, evaluate_metrics=evaluate_metrics)


if __name__ == '__main__':
    from config import config_train_general as cfg
    import argparse
    from timm.utils import accuracy, ModelEma

    parser = argparse.ArgumentParser()
    parser.add_argument('--output', '-o', type=str, default='hy_re_av', help='output dir')
    parser.add_argument('--eval', '-e', type=int, default=0, help='eval')

    args = parser.parse_args()

    output_dir = args.output
    evaluate_metrics = args.eval
    logger.info(output_dir)
    logger.info(evaluate_metrics)
    out_test(cfg, output_dir=output_dir, evaluate_metrics=evaluate_metrics)
