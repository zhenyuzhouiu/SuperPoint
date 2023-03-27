import math
import os
import shutil

os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import cv2
import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm


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


def show_file(images_path, points_path, resize=[1080, 1920], visualization='minutiae'):
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
        image_raw_data = tf.gfile.FastGFile(os.path.join(images_path, file_name.strip(".npz") + '.jpg'), 'rb').read()

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
        if visualization == 'minutiae':
            gt = npz['points']
            gt_c = npz['classes']
            gt_a = npz['angles']
            for j in range(gt.shape[0]):
                point = gt[j]
                cls = gt_c[j]
                ang = gt_a[j]
                point = (int(point[1]), int(point[0]))
                dist = 6
                # math.cos() and math.sin() should give a radian input
                if 0 <= ang <= 90:
                    dx = dist * math.cos((ang/180)*math.pi)
                    dy = dist * math.sin((ang/180)*math.pi)
                    point2 = (int(point[0] + dx), int(point[1] + dy))
                else:
                    ang = 180 - ang

                    dx = dist * math.cos((ang/180)*math.pi)
                    dy = dist * math.sin((ang/180)*math.pi)
                    point2 = (int(point[0] - dx), int(point[1] + dy))
                if cls == 1:  # for bifurcation
                    # bifurcation with red color
                    cv2.arrowedLine(img_float, point, point2, color=[255, 0, 0], thickness=1, tipLength=0.5)
                else:
                    # ending with blue color
                    cv2.arrowedLine(img_float, point, point2, color=[0, 0, 255], thickness=1, tipLength=0.5)

            plt.figure(1)
            plt.subplot(1, 1, 1)
            plt.imshow(img_float)
            plt.show()
        else:
            gt = npz['points']
            for j in range(gt.shape[0]):
                point = gt[j]
                point = (int(point[1]), int(point[0]))
                cv2.circle(img_float, point, radius=1, color=(1, 0, 0), thickness=1)
            plt.figure(1)
            plt.subplot(1, 1, 1)
            plt.imshow(img_float)
            plt.show()


def npz_to_txt(points_path, txt_path):
    if os.path.exists(txt_path):
        shutil.rmtree(txt_path)
    os.mkdir(txt_path)

    npz_files = os.listdir(points_path)
    npz_files.sort()
    pbar = tqdm(npz_files)
    for n in pbar:
        file_name = n
        npz = np.load(os.path.join(points_path, file_name))
        prob = npz['prob']
        # prob_max, prob_min = np.max(prob), np.min(prob)
        # prob = (prob - prob_min) / (prob_max - prob_min)
        point = npz['points']
        cls = npz['classes']
        ang = npz['angles']
        for j in range(prob.shape[0]):
            prob_j = prob[j]
            point_j = point[j]
            cls_j = cls[j]
            ang_j = ang[j]
            ang_j = np.clip(ang_j, 1, 180)  # normalize angle range
            with open(os.path.join(txt_path, n.split('.')[0] + '.txt'), 'a+') as tf:
                # x y angle cls score
                if cls_j == 1:  # for bifurcation
                    line = str(int(point_j[1])) + str(' ') + str(int(point_j[0])) + str(' ') + \
                           str(int(ang_j)) + str(' ') + str(int(2)) + str(' ') + str(round(prob_j[0] * 100)) + str(
                        '\n')
                    tf.write(line)
                else:  # for ending
                    line = str(int(point_j[1])) + str(' ') + str(int(point_j[0])) + str(' ') + \
                           str(int(ang_j)) + str(' ') + str(int(1)) + str(' ') + str(round(prob_j[0] * 100)) + str(
                        '\n')
                    tf.write(line)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--images_path', type=str,
                        default='../data_dir/fingernail/train', help='the source images path',
                        dest='images_path')
    parser.add_argument('--points_path', type=str,
                        default='../exper_dir/outputs/fingernail-minutiae-multihead_fingernail_two_session/no_homograph_adaption',
                        help='the points position path',
                        dest='points_path')
    parser.add_argument('--txt_path', type=str,
                        default='../exper_dir/outputs/fingernail-minutiae-multihead_fingernail_two_session/no_homograph_adaption_txt')
    parser.add_argument('--resize', type=int, default=[240, 320], help='resize the source image to the size [h, w]',
                        dest='resize')
    parser.add_argument('--visualization', type=str, default='minutiae', dest='visualization')
    args = parser.parse_args()

    # show_file(args.images_path, args.points_path, args.resize, args.visualization)
    npz_to_txt(args.points_path, args.txt_path)
