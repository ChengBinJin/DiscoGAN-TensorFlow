# ---------------------------------------------------------
# Tensorflow DiscoGAN Implementation
# Licensed under The MIT License [see LICENSE for details]
# Written by vanhuyz
# ---------------------------------------------------------
import numpy as np
import tensorflow as tf
import time


class Reader(object):
    def __init__(self, file_path, image_size=(64, 64, 3), min_queue_examples=100, batch_size=1, num_threads=8,
                 side='left', ori_image_size=(256, 512, 3), name=None):
        self.file_path = file_path
        self.image_size = image_size
        self.factor = 1.05
        # (256, 512, 3) to (256, 256, 3)
        self.ori_image_size = (ori_image_size[0], ori_image_size[0], ori_image_size[2])
        self.bigger_size = [int(np.ceil(self.ori_image_size[0] * self.factor)),
                            int(np.ceil(self.ori_image_size[1] * self.factor))]
        self.min_queue_examples = min_queue_examples
        self.batch_size = batch_size
        self.num_threads = num_threads
        self.reader = tf.WholeFileReader()
        self.channel = self.image_size[2]
        self.side = side
        self.name = name

    def feed(self):
        with tf.name_scope(self.name):
            filename_queue = tf.train.string_input_producer(tf.train.match_filenames_once(self.file_path+'/*.jpg'),
                                                            capacity=2*self.min_queue_examples)
            _, serialized_example = self.reader.read(filename_queue)
            image = tf.image.decode_jpeg(serialized_example, channels=self.channel)
            image = self._preprocess(image)
            images = tf.train.shuffle_batch([image], batch_size=self.batch_size, num_threads=self.num_threads,
                                            capacity=self.min_queue_examples + 3 * self.batch_size,
                                            min_after_dequeue=self.min_queue_examples)
        return images

    def _preprocess(self, image):
        if self.side == 'left':
            print('self.ori_image_size: {}'.format(self.ori_image_size))
            image = tf.image.crop_to_bounding_box(image, offset_height=0, offset_width=0,
                                                  target_height=self.ori_image_size[0],
                                                  target_width=int(self.ori_image_size[1]))
        elif self.side == 'right':
            print('self.ori_image_size: {}'.format(self.ori_image_size))
            image = tf.image.crop_to_bounding_box(image, offset_height=0, offset_width=int(self.ori_image_size[1]),
                                                  target_height=self.ori_image_size[0],
                                                  target_width=self.ori_image_size[1])
        else:
            raise NotImplementedError

        random_seed = int(round(time.time()))
        # make image bigger
        image = tf.image.resize_images(image, size=(self.bigger_size[0], self.bigger_size[1]))
        # random crop
        image = tf.random_crop(image, size=self.ori_image_size, seed=random_seed)
        # random flip
        image = tf.image.random_flip_left_right(image, seed=random_seed)
        # resize to input image size
        image = tf.image.resize_images(image, size=(self.image_size[0], self.image_size[1]))
        # normalize to [-1., 1.]
        image = tf.image.convert_image_dtype(image, dtype=tf.float32) / 127.5 - 1.
        image.set_shape(self.image_size)
        return image

