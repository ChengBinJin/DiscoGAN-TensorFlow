import tensorflow as tf
import cv2

# hyperparameters
initializer = tf.truncated_normal_initializer(stddev=0.02)
learning_rate = 0.0002
batch_size = 8    # paper: 200, i think 256 is good
epoch = 100000
lambda_ = 10        # not mentioned in the paper
num_preprocess_threads = 8
min_queue_examples = 256

# Read image files
image_reader = tf.WholeFileReader()
shoes_filename_queue = tf.train.string_input_producer(
    tf.train.match_filenames_once("E:/BOBBY/TensorFlow/Data/edges2handbags/train/*.jpg"), capacity=200)
_, shoes_file = image_reader.read(shoes_filename_queue)
shoes_image = tf.image.decode_jpeg(shoes_file)
image = tf.image.crop_to_bounding_box(shoes_image, offset_height=0, offset_width=256, target_height=256, target_width=256)
image = tf.image.resize_images(image, size=(64, 64))

shoes_image = tf.cast(tf.reshape(image, shape=[64, 64, 3]), dtype=tf.float32) / 255.0
batch_shoes = tf.train.shuffle_batch([shoes_image], batch_size=batch_size, num_threads=num_preprocess_threads,
                                     capacity=min_queue_examples + 3 * batch_size, min_after_dequeue=min_queue_examples)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    try:
        x_imgs = sess.run(batch_shoes)
        for idx in range(x_imgs.shape[0]):
            x_img = x_imgs[idx, :, :, ::-1]
            cv2.imshow('img', x_img)
            cv2.waitKey(0)

    except KeyboardInterrupt:
        coord.request_stop()
    except Exception as e:
        coord.request_stop(e)
    finally:
        # when done, ask the threads to stop
        coord.request_stop()
        coord.join(threads)


