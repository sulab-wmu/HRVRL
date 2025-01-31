# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Partly revised by YZ @UCL&Moorfields
# --------------------------------------------------------

import os
from torchvision import datasets, transforms
from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import numpy as np
from PIL import Image



def find_classes(directory):
    """Finds the class folders in a dataset.

    See :class:`DatasetFolder` for details.
    """
    #  bmilddr cmoderatedr  dseveredr eproliferativedr
    classes=['anodr','bmilddr','cmoderatedr','dseveredr','eproliferativedr']
    #classes=['anormal','bsuspectglaucoma','cglaucoma']
    #class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    class_to_idx = {'anodr':0,'bmilddr':1}
    #class_to_idx = {'anormal':0,'bsuspectglaucoma':1,'cglaucoma':1}
    print(class_to_idx)
    return classes, class_to_idx

class ImageFolderCustom(datasets.ImageFolder):

    def find_classes(self,directory):
        """Finds the class folders in a dataset.

        See :class:`DatasetFolder` for details.
        """
        
        return find_classes(directory)

class PadToSquare(object):
    def __call__(self, img):
        image = np.array(img)
        height, width = image.shape[:2]

        if height > width:
            # Padding on the sides (left and right) to make it square
            pad_left = (height - width) // 2
            pad_right = height - width - pad_left
            pad_top = 0
            pad_bottom = 0
        else:
            # Padding on the top and bottom to make it square
            pad_top = (width - height) // 2
            pad_bottom = width - height - pad_top
            pad_left = 0
            pad_right = 0

        # Add padding to the image
        padded_image = np.pad(
            image,
            ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
            mode='constant',
            constant_values=0
        )

        return Image.fromarray(padded_image)


class Crop2Rec(object):
	def __call__(self,img):
	    image = np.array(img)
	    height, width = image.shape[:2]
	    if height > width:
	        left = 0
	        right = width
	        top = (height - width) // 2
	        bottom = top + width
	    else:
	        top = 0
	        bottom = height
	        left = (width - height) // 2
	        right = left + height
	    image = image[top:bottom, left:right]
	    return Image.fromarray(image)





def build_dataset(is_train, args):
    if 'APT' in args.data_path:
        transform = build_transform_APT(is_train, args)
    else:
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
    root = os.path.join(args.data_path, is_train)
    dataset = datasets.ImageFolder(root, transform=transform)   
    #dataset = ImageFolderCustom(root,transform=transform)
    return dataset


def build_transform_APT(is_train, args):
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD
    # train transform
    if is_train=='train':
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            #interpolation='bicubic',
            color_jitter=None,
            auto_augment=args.aa,
            interpolation='bicubic',
            hflip=0.5,  # 水平翻转的概率
            vflip=0.5,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
        transform.transforms.insert(0,PadToSquare())
        return transform

    # eval transform
    t = []
    if args.input_size <= 384:
        crop_pct = 224 / 256
    else:
        crop_pct = 1.0
    size = int(args.input_size / crop_pct)
    size=args.input_size 
    t.append(PadToSquare())
    t.append(
        transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC), 
    )
    t.append(transforms.CenterCrop(args.input_size))
    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)


def build_transform(is_train, args):
    
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD
    # train transform
    if is_train=='train':
        # this should always dispatch to transforms_imagenet_train
        transform = transforms.Compose([
                Crop2Rec(),
                transforms.Resize((args.input_size,args.input_size),interpolation=transforms.InterpolationMode.BICUBIC),            
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomGrayscale(p=0.2),
                #transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.ColorJitter(),
                transforms.RandomRotation(degrees=(-180, 180)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ])
        #transform.transforms.insert(0,Crop2Rec())
        return transform

    # eval transform
    t = []
    if args.input_size <= 384:
        crop_pct = 224 / 256
    else:
        crop_pct = 1.0
    size = int(args.input_size / crop_pct)
    size=args.input_size 
    t.append(Crop2Rec())
    t.append(
        transforms.Resize((args.input_size,args.input_size), interpolation=transforms.InterpolationMode.BICUBIC), 
    )
    t.append(transforms.CenterCrop(args.input_size))
    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)