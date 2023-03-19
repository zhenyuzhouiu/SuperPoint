import os
os.environ['CUDA_VISIBLE_DEVICES'] = "1"
import cv2
import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


def show_path(images_path, points_path, mode=''):
    images_path = os.path.join(images_path, mode)
    points_path = os.path.join(points_path, mode)
    images = os.listdir(images_path)
    images.sort()
    gts = os.listdir(points_path)
    gts.sort()
    if len(images) == len(gts):
        for i in range(len(images)):
            image_file = os.path.join(images_path, images[i])
            gt_file = os.path.join(points_path, gts[i])
            img = cv2.imread(image_file)
            gt = np.load(gt_file)
            for j in range(gt.shape[0]):
                point = gt[j]
                point = (int(point[0]), int(point[1]))
                cv2.circle(img, point, radius=1, color=(0, 0, 255), thickness=1)
            cv2.namedWindow('show', cv2.WINDOW_NORMAL)
            cv2.imshow('show', img)
            cv2.waitKey(0)
            cv2.destroyWindow('show')
    else:
        print("The number of images is not equal to the number of ground truth.")


def show_file(images_path, points_path, resize=[1080, 1920]):
    """
    Args:
        images_file:
        points_file:
        resize: [height, width]

    Returns:

    """
    npz_files = os.listdir(points_path)
    npz_files.sort()
    for i in range(200):
        file_name = npz_files[i]
        # 读取图像的原始数据
        image_raw_data = tf.gfile.FastGFile(os.path.join(images_path, file_name.strip(".npz")+'.jpg'), 'rb').read()

        with tf.Session() as session:
            # 对图像进行jpeg的格式解码从而得到图像对应的三维矩阵
            img_data = tf.image.decode_jpeg(image_raw_data)

            # 首先将图片数据转化为实数类型。将0-255的像素值转化为0.0-1.0的实数
            img_data = tf.image.convert_image_dtype(img_data, dtype=tf.float32)
            target_size = tf.convert_to_tensor(resize)
            scales = tf.to_float(tf.divide(target_size, tf.shape(img_data)[:2]))
            new_size = tf.to_float(tf.shape(img_data)[:2]) * tf.reduce_max(scales)
            img_data = tf.image.resize_images(img_data, tf.to_int32(new_size),
                                              method=tf.image.ResizeMethod.BILINEAR)
            img_data = tf.image.resize_image_with_crop_or_pad(img_data, target_size[0], target_size[1])
            img_float = img_data.eval()

        npz = np.load(os.path.join(points_path, file_name))
        gt = npz['points']
        for j in range(gt.shape[0]):
            point = gt[j]
            point = (int(point[1]), int(point[0]))
            cv2.circle(img_float, point, radius=1, color=(1, 0, 0), thickness=1)
        plt.figure(1)
        plt.subplot(1, 1, 1)
        plt.imshow(img_float)
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--images_path', type=str,
                        default='../data_dir/fingernail/train', help='the source images path',
                        dest='images_path')
    parser.add_argument('--points_path', type=str,
                        default='../exper_dir/outputs/fingernail_point_fingernail_export1',
                        help='the points position path',
                        dest='points_path')
    parser.add_argument('--resize', type=int, default=[240, 320], help='resize the source image to the size [h, w]',
                        dest='resize')
    args = parser.parse_args()

    show_file(args.images_path, args.points_path, args.resize)
