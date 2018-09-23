# ---------------------------------------------------------
# Tensorflow DiscoGAN Implementation
# Licensed under The MIT License [see LICENSE for details]
# Written by Cheng-Bin Jin
# Email: sbkim0407@gmail.com
# ---------------------------------------------------------
import numpy as np
import utils as utils
import cv2


class Original(object):
    def __init__(self, flags):
        self.flags = flags
        self.dataset_name = flags.dataset
        self.image_size = (64, 64, 3)
        if self.flags.dataset == 'edges2shoes' or self.flags.dataset == 'edges2handbags' or \
                self.flags.dataset == 'cityscapes' or self.flags.dataset == 'facades':
            self.ori_image_size = (256, 512, 3)
        elif self.flags.dataset == 'maps':
            self.ori_image_size = (600, 1200, 3)

        self.train_path = '../../Data/{}/train'.format(self.dataset_name)
        self.val_path = '../../Data/{}/val'.format(self.dataset_name)
        self.data_x, self.data_y = None, None

    def __call__(self):
        if self.flags.is_train:
            return [self.train_path, self.train_path]
        else:
            self.read_val_data()
            return [self.val_path, self.val_path]

    def read_val_data(self):
        imgs_x, imgs_y = [], []
        val_path = utils.all_files_under(self.val_path)
        for path in val_path:
            x, y = utils.load_data(path, flip=False, is_test=True, is_gray_scale=False,
                                   transform_type='zero_center', img_size=self.ori_image_size)
            # scipy.misc.imresize reutrns uint8 type
            x = cv2.resize(x, dsize=None, fx=0.25, fy=0.25)  # (256, 256, 3) to (64, 64, 3)
            y = cv2.resize(y, dsize=None, fx=0.25, fy=0.25)  # (256, 256, 3) to (64, 64, 3)

            imgs_x.append(x)
            imgs_y.append(y)

        self.data_x = np.asarray(imgs_x).astype(np.float32)  # list to array
        self.data_y = np.asarray(imgs_y).astype(np.float32)  # list to array


class Bags2Shoes(object):
    def __init__(self, flags):
        self.flags = flags
        self.dataset_name = flags.dataset
        self.image_size = (64, 64, 3)
        self.ori_image_size = (256, 256, 3)

        self.bags_train_path = '../../Data/edges2handbags/train'
        self.shoes_train_path = '../../Data/edges2shoes/train'

        self.bags_val_path = '../../Data/edges2handbags/val'
        self.shoes_val_path = '../../Data/edges2shoes/val'

        self.data_x, self.data_y = None, None

    def __call__(self):
        if self.flags.is_train:
            return [self.bags_train_path, self.shoes_train_path]
        else:
            self.read_val_data()
            return [self.bags_val_path, self.shoes_val_path]

    def read_val_data(self):
        bags, shoeses = [], []
        bags_val_path = utils.all_files_under(self.bags_val_path)
        shoes_val_path = utils.all_files_under(self.shoes_val_path)

        # read bags data
        for bag_path in bags_val_path:
            _, bag = utils.load_data(bag_path, flip=False, is_test=True, is_gray_scale=False,
                                     transform_type='zero_center', img_size=self.ori_image_size)
            # scipy.misc.imresize reutrns uint8 type
            bag = cv2.resize(bag, dsize=None, fx=0.25, fy=0.25)  # (256, 256, 3) to (64, 64, 3)
            bags.append(bag)

        for shoes_path in shoes_val_path:
            _, shoes = utils.load_data(shoes_path, flip=False, is_test=True, is_gray_scale=False,
                                       transform_type='zero_center', img_size=self.ori_image_size)
            # scipy.misc.imresize reutrns uint8 type
            shoes = cv2.resize(shoes, dsize=None, fx=0.25, fy=0.25)  # (256, 256, 3) to (64, 64, 3)
            shoeses.append(shoes)

        self.data_x = np.asarray(bags).astype(np.float32)  # list to array
        self.data_y = np.asarray(shoeses).astype(np.float32)  # list to array


# noinspection PyPep8Naming
def Dataset(dataset_name, flags):
    if dataset_name == 'handbags2shoes':
        return Bags2Shoes(flags)
    elif dataset_name == 'edges2handbags' or dataset_name == 'edges2shoes' or dataset_name == 'maps' or \
            dataset_name == 'cityscapes' or dataset_name == 'facades':
        return Original(flags)
    else:
        raise NotImplementedError
