import os
import cv2
import numpy as np


def all_files_under(path, extension=None, special=None, append_path=True, sort=True):
    if append_path:
        if extension is None:
            filenames = [os.path.join(path, fname) for fname in os.listdir(path) if special in fname]
        else:
            filenames = [os.path.join(path, fname) for fname in os.listdir(path)
                         if (special in fname) and (fname.endswith(extension))]
    else:
        if extension is None:
            filenames = [os.path.basename(fname) for fname in os.listdir(path) if special in fname]
        else:
            filenames = [os.path.basename(fname) for fname in os.listdir(path)
                         if (special in fname) and (fname.endswith(extension))]

    if sort:
        filenames = sorted(filenames)

    return filenames


def main(paths_):
    gan_paths = all_files_under(paths_[0], extension='png', special='GAN_loss_AB_')
    gan_recon_paths = all_files_under(paths_[1], extension='png', special='GAN_rec_loss_AB_')
    discogan_AB_paths = all_files_under(paths_[2], extension='png', special='discoGAN_AB_')
    discogan_BA_paths = all_files_under(paths_[2], extension='png', special='discoGAN_BA_')

    frame_shape = cv2.imread(gan_paths[0]).shape
    print(frame_shape)

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_writer = cv2.VideoWriter('toy_experiments.avi', fourcc, 10.0, (frame_shape[1]*4, frame_shape[0]))

    for idx in range(len(gan_paths)):
        img_a = cv2.imread(gan_paths[idx])
        img_b = cv2.imread(gan_recon_paths[idx])
        img_c = cv2.imread(discogan_AB_paths[idx])
        img_d = cv2.imread(discogan_BA_paths[idx])

        frame = np.hstack([img_a, img_b, img_c, img_d])
        # write the frame
        video_writer.write(frame)

        cv2.imshow('Show', frame)
        cv2.waitKey(1)

    # Release everything if job is finished
    video_writer.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    paths = ['img/gan', 'img/gan_with_reconstruction_loss', 'img/discogan']
    main(paths)
