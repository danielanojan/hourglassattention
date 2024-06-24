from PIL import Image, ImageFile
import torchvision.transforms as transforms
import torch.utils.data as data
from .image_folder import make_dataset
from util import task
import random
import os
import math


class CreateDataset(data.Dataset):
    def __init__(self, opt):
        self.opt = opt

        #img size - num of images
        self.img_paths, self.img_size = make_dataset(opt.img_file)
        if opt.mask_type == 3:
            self.mask_paths, self.mask_size = make_dataset(opt.mask_file)

        # provides random file for training and testing
        #if opt.mask_file != 'none':
        #    self.mask_paths, self.mask_size = make_dataset(opt.mask_file)
        #    if not self.opt.isTrain:
        #       self.mask_paths = self.mask_paths * (max(1, math.ceil(self.img_size / self.mask_size)))
        self.transform = get_transform(opt)
        self.mask_transform = get_transform_mask(opt)

    def __getitem__(self, index):
        # load image
        img, img_path = self.load_img(index)
        # load mask
        #mask = self.load_mask(img, index)
        mask, mask_path = self.load_mask(index)
        return {'img': img, 'img_path': img_path, 'mask': mask}

    def __len__(self):
        return self.img_size

    def name(self):
        return "inpainting dataset"

    def load_img(self, index):
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        img_path = self.img_paths[index % self.img_size]
        img_path = os.path.join(self.opt.img_file, img_path)
        img_pil = Image.open(img_path).convert('RGB')

        img = self.transform(img_pil)
        img_pil.close()
        return img, img_path

    def load_mask(self, index):
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        mask_path = self.mask_paths[index % self.img_size]
        mask_path = os.path.join(self.opt.mask_file, mask_path)

        mask_pil = Image.open(mask_path).convert('L')
        mask = self.transform(mask_pil)
        mask_pil.close()
        return mask, mask_path



    def load_mask_ori(self, img, index):
        """Load different mask types for training and testing"""

        #finds a random num
        mask_type_index = random.randint(0, len(self.opt.mask_type) - 1)
        #loads that index in the input. If [2, 4] given as masks, randomly 2/4 is loaded.
        mask_type = self.opt.mask_type[mask_type_index]
        mask_type = 0
        # center mask
        if mask_type == 0:
            return task.center_mask(img)

        # random regular mask
        if mask_type == 1:
            return task.random_regular_mask(img)

        # random irregular mask
        if mask_type == 2:
            return task.random_irregular_mask(img)

        if mask_type == 4:
            return task.random_freefrom_mask(img)

        # external mask from "Image Inpainting for Irregular Holes Using Partial Convolutions (ECCV18)"
        if mask_type == 3:
            ImageFile.LOAD_TRUNCATED_IMAGES = True
            mask_path = self.mask_paths[index % self.img_size]
            mask_path = os.path.join(self.opt.mask_file, mask_path)

            mask_pil = Image.open(mask_path).convert('L')
            if self.opt.isTrain:
                #for training
                mask = self.mask_transform(mask_pil)
            else:
                #for testing
                mask_transform = transforms.Compose([
                    transforms.Resize(self.opt.fineSize),
                    transforms.ToTensor()
                ])
                mask = (mask_transform(mask_pil) == 0).float()
            mask_pil.close()
            return mask, mask_path

def dataloader(opt):
    datasets = CreateDataset(opt)
    dataset = data.DataLoader(datasets, batch_size=opt.batchSize, shuffle=not opt.no_shuffle, num_workers=int(opt.nThreads))

    return dataset


def get_transform(opt):
    """Basic process to transform PIL image to torch tensor"""
    transform_list = []
    osize = [opt.loadSize[0], opt.loadSize[1]]
    fsize = [opt.fineSize[0], opt.fineSize[1]]
    if opt.isTrain:
        if opt.resize_or_crop == 'resize_and_crop':
            transform_list.append(transforms.Resize(osize))
            transform_list.append(transforms.RandomCrop(fsize))
        elif opt.resize_or_crop == 'crop':
            transform_list.append(transforms.RandomCrop(fsize))
        elif opt.resize_or_crop == 'resize':
            transform_list.append(transforms.Resize(osize))
        #if not opt.no_augment:
            transform_list.append(transforms.ColorJitter(0.0, 0.0, 0.0, 0.0))
        if not opt.no_flip:
            transform_list.append(transforms.RandomHorizontalFlip())
        if not opt.no_rotation:
            transform_list.append(transforms.RandomRotation(3))
    else:
        transform_list.append(transforms.Resize(fsize))

    transform_list += [transforms.ToTensor()]

    return transforms.Compose(transform_list)

def get_transform_mask(opt):
    transform_list = []
    osize = [opt.loadSize[0], opt.loadSize[1]]
    fsize = [opt.fineSize[0], opt.fineSize[1]]
    if opt.isTrain:
        if opt.resize_or_crop == 'resize':
            transform_list.append(transforms.Resize(osize))
        if not opt.no_flip:
            transform_list.append(transforms.RandomHorizontalFlip())
        if not opt.no_rotation:
            transform_list.append(transforms.RandomRotation(3))
    else:
        transform_list.append(transforms.Resize(fsize))

    transform_list += [transforms.ToTensor()]

    return transforms.Compose(transform_list)
