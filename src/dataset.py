# ---------------------------------------------------------
# Tensorflow DiscoGAN Implementation
# Licensed under The MIT License [see LICENSE for details]
# Written by Cheng-Bin Jin
# Email: sbkim0407@gmail.com
# ---------------------------------------------------------


class Original(object):
    def __init__(self, flags):
        self.flags = flags
        self.dataset_name = flags.dataset
        self.image_size = (64, 64, 3)

        self.train_path = '../../Data/{}/train'.format(self.dataset_name)
        self.val_path = '../../Data/{}/val'.format(self.dataset_name)

    def __call__(self):
        if self.flags.is_train:
            return [self.train_path, self.train_path]
        else:
            return [self.val_path, self.val_path]


class Bags2Shoes(object):
    def __init__(self, flags):
        self.flags = flags
        self.dataset_name = flags.dataset
        self.image_size = (64, 64, 3)

        self.bags_train_path = '../../Data/edges2handbags/train'
        self.shoes_train_path = '../../Data/edges2shoes/train'

        self.bags_val_path = '../../Data/edges2handbags/val'
        self.shoes_val_path = '../../Data/edges2shoes/val'

    def __cal__(self):
        if self.flags.is_train:
            return [self.bags_train_path, self.shoes_train_path]
        else:
            return [self.bags_val_path, self.shoes_val_path]


# noinspection PyPep8Naming
def Dataset(dataset_name, flags):
    if dataset_name == 'edges2handbags' or dataset_name == 'edges2shoes':
        return Original(flags)
    elif dataset_name == 'handbags2shoes':
        print('Hello handbags2shoes!')
    else:
        raise NotImplementedError
