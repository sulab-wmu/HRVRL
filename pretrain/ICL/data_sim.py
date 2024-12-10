import os
from tqdm import tqdm
from torch.utils.data import Dataset
import pathlib
from torchvision import transforms,datasets
import numpy as np
import  torch
from PIL import Image,ImageFile
import sys
from timm.data.constants import \
    IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import create_transform
ImageFile.LOAD_TRUNCATED_IMAGES = True
transform = transforms.Compose(
    [transforms.Resize([224, 224]),
     transforms.ToTensor(),
     transforms.Normalize([0.456, 0.485, 0.406], [0.224, 0.229, 0.225])])


import pathlib
import glob
from torch.utils.data import Dataset


from torchvision.datasets import ImageFolder



from pathlib import Path

import numpy as np
import pandas as pd

from torch.utils.data import Dataset
from copy import deepcopy
class Duplicate:
    """
    Duplicate an input and apply two different transforms. Used for SimCLR primarily.
    """

    def __init__(self, transforms1=None, transforms2= None):
        """
        Duplicates an input and applies the given transformations to each copy separately.

        Args:
            transforms1 (Optional[Callable]): _description_. Default is None.
            transforms2 (Optional[Callable]): _description_. Default is None.
        """
        # Wrapped into a list if it isn't one already to allow both a
        # list of transforms as well as `torchvision.transform.Compose` transforms.
        self.transforms1 = transforms1
        self.transforms2 = transforms2
    def __call__(self, input):
        """
        Args:
            input (torch.Tensor or any other type supported by the given transforms): Input.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple of two tensors.
        """
        input_array = np.array(input)
        out1, out2 = input_array, deepcopy(input_array)
        if self.transforms1 is not None:

            out1 = self.transforms1(Image.fromarray(out1))
        if self.transforms2 is not None:
            out2 = self.transforms2(Image.fromarray(out2))
        return (out1, out2)

class SSLRetinalDataset(Dataset):
    """
    Dataset class for SSL Radiomics dataset.

    Args:
        path (str): The path to the dataset.
        label (str, optional): The label column name in the dataset annotations. Default is None.
        radius (int, optional): The radius around the centroid for positive patch extraction. Default is 25.
        orient (bool, optional): Whether to orient the images to LPI orientation. Default is False.
        resample_spacing (float or tuple, optional): The desired spacing for resampling the images. Default is None.
        enable_negatives (bool, optional): Whether to include negative samples. Default is True.
        transform (callable, optional): A function/transform to apply on the images. Default is None.
    """

    def __init__(
        self,
        path,
        enable_negatives=True,
        transforms1=None,
        transforms2=None,
        input_is_target=False,
    ):
        """
        Creates an instance of the SSLRadiomicsDataset class with the given parameters.

        Args:
            path (str): The path to the dataset.
            enable_negatives (bool): True if negatives are enabled, False otherwise. Defaults to True.
            transform: The transformation to apply to the dataset. Defaults to None.
            input_is_target (bool): True if the input is the target, False otherwise. Defaults to False.
        Raises:
            None.

        Returns:
            None.
        """
        # super(SSLRetinalDataset, self).__init__()
        # self._path_11_train = [os.path.join(path,'training','images11',i) for i in os.listdir(os.path.join(path,'training','images11'))]
        # self._path_11_test =[os.path.join(path,'test','images11',i) for i in os.listdir(os.path.join(path,'test','images11'))]
        # self._path_11_train.extend(self._path_11_test)
        # self._path_00_train = [os.path.join(path,'training','images00',i) for i in os.listdir(os.path.join(path,'training','images00'))]
        # self._path_00_test = [os.path.join(path,'test','images00',i) for i in os.listdir(os.path.join(path,'test','images00'))]
        # self._path_00_train.extend(self._path_00_test)
        # self.enable_negatives = enable_negatives
        # self.transform = Duplicate(transforms1=transforms1,transforms2=transforms2)
        # self.input_is_target = input_is_target
        # self._neg_len = len(self._path_00_train)


        super(SSLRetinalDataset, self).__init__()
        self._path_11_train = [os.path.join(path,'training','images11',i) for i in os.listdir(os.path.join(path,'training','images11'))]
        self._path_11_test =[os.path.join(path,'test','images11',i) for i in os.listdir(os.path.join(path,'test','images11'))]
        self._path_11_train.extend(self._path_11_test)
        self._path_00_train = [os.path.join(path,'training','images00',i) for i in os.listdir(os.path.join(path,'training','images00'))]
        self._path_00_test = [os.path.join(path,'test','images00',i) for i in os.listdir(os.path.join(path,'test','images00'))]
        self._path_00_train.extend(self._path_00_test)
        self.enable_negatives = enable_negatives
        self.transform = Duplicate(transforms1=transforms1,transforms2=transforms2)
        self.input_is_target = input_is_target
        self._neg_len = len(self._path_00_train)

    def __len__(self):
        """
        Size of the dataset.
        """

        return len(self._path_11_train)

    def load_img(self,path):
        img = Image.open(path)
        return img.convert("RGB")

    def get_background_sample(self):


        random_patch_idx = ()

        # escape_count = 0
        # while is_overlapping(positive_patch_idx, random_patch_idx):
        #     if escape_count >= 3:
        #         logger.warning("Random patch has overlap with positive patch")
        #         return None

        #     random_patch_idx = get_random_patch()
        #     escape_count += 1
        ind = np.random.randint(0,self._neg_len)
        bg_patch_path = self._path_00_train[ind]
        bg_img = self.load_img(bg_patch_path)

        bg_tensor = self.transform(bg_img)
        return bg_tensor

    def __getitem__(self, index):
        """
        Implement how to load the data corresponding to the idx element in the dataset from your data source.
        """
        ra = np.random.randint(0,2)
        tr = np.random.randint(0,2)
        # Get a row from the CSV file
        img_path = self._path_11_train[index]
        image = self.load_img(img_path)


        image_tensor = self.transform(image)
        bg_tensor = self.get_background_sample()
        target=None
        if self.enable_negatives:
            if  ra==0:
                return {"positive": image_tensor, "negative": bg_tensor}, bg_tensor[tr],0
            else:
                return {"positive": image_tensor, "negative": bg_tensor}, image_tensor[tr],1

        return image_tensor, 1



def build_dataset(is_train, args):
    transform = build_transform(is_train, args)

    print("Transform = ")
    if isinstance(transform, tuple):
        for trans in transform:
            print(" - - - - - - - - - - ")
            for t in trans.transforms:
                print(t)
    else:
        for t in transform.transforms:
            print(t)
    print("---------------------------")
    root = args.data_path
    #if is_train:
    #    root = os.path.join(root,'training')
    #else:
    #    root = os.path.join(root,'test')
    dataset = SSLRetinalDataset(path=root, transforms1=transform,transforms2=transform)
    return dataset


def build_transform(is_train, args):
    resize_im = args.input_size > 32
    #imagenet_default_mean_and_std = args.imagenet_default_mean_and_std
    mean =  IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD

    if is_train:
        # this should always dispatch to transforms_imagenet_train
        # transform = create_transform(
        #     input_size=args.input_size,
        #     is_training=True,
        #     color_jitter=args.color_jitter,
        #     auto_augment=args.aa,
        #     interpolation=args.train_interpolation,
        #     re_prob=args.reprob,
        #     re_mode=args.remode,
        #     re_count=args.recount,
        #     mean=mean,
        #     std=std,
        # )

        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=0.0,
            hflip=0.5,  # 水平翻转的概率
            vflip=0.5,
            grayscale_prob=0.2,
            mean=mean,
            std=std,
        )


        if not resize_im:
            transform.transforms[0] = transforms.RandomCrop(
                args.input_size, padding=4)
        return transform

    t = []
    if resize_im:
        # warping (no cropping) when evaluated at 384 or larger
        if args.input_size > 384:  
            t.append(
            transforms.Resize((args.input_size, args.input_size), 
                            interpolation=transforms.InterpolationMode.BICUBIC), 
        )
            print(f"c {args.input_size} size input images...")
        else:
            if args.crop_pct is None:
                args.crop_pct = 224 / 256
            size = int(args.input_size / args.crop_pct)
            t.append(
                # to maintain same ratio w.r.t. 224 images
                transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),  
            )
            #t.append(transforms.CenterCrop(args.input_size)) pretrain no
    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)

if __name__ == '__main__':
    import argparse
    from network_multi_label_simclr import MultiAV_RIP_SimCLR
    from  loss_sim import NegativeMiningInfoNCECriterion
    parser = argparse.ArgumentParser('=========')
    parser.add_argument('-input_size',default=256)
    parser.add_argument('-crop_pct', default=None)
    parser.add_argument('-data_path', default=r'D:\su-lab\code\RIP-AV-main\AV\Preprocessing')
    args = parser.parse_args()
    #args = {'input_size':256,'crop_pct':None,'data_path':r'D:\su-lab\code\RIP-AV-main\AV\Preprocessing'}
    da = build_dataset(is_train=True,args=args)
    sampler_train = torch.utils.data.DistributedSampler(
        da, num_replicas=2, rank=0, shuffle=True, seed=0,
    )
    print(sampler_train)
    data_loader_train = torch.utils.data.DataLoader(
        da, sampler=sampler_train,
        batch_size=4,
        drop_last=True,

    )
    m = MultiAV_RIP_SimCLR()
    c = NegativeMiningInfoNCECriterion()
    for d,a in data_loader_train:
        out = m(d)

        print(c(out))
        #print(m(d))
        print(len(d['positive'][0]))
        print(d['positive'][0].shape)
        break
    #print(da.__getitem__(1))