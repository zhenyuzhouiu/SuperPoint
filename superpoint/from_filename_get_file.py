import math
import os
import shutil

import cv2
import numpy as np
from tqdm import tqdm


def copy_npz_by_filename(src_path, dst_path, list_filename):
    # list_filename contains image files
    for filename in list_filename:
        npz_filename = filename.split('.')[0] + '.npz'
        shutil.copy(os.path.join(src_path, npz_filename), os.path.join(dst_path, npz_filename))


def copy_image_by_filename(src_path, dst_path, list_filename):
    # list_filename contains npz files
    for filename in list_filename:
        image_filename = filename.split('.')[0] + '.jpg'

        subject_id, finger_id = image_filename.split('.')[0].split('_')[0], image_filename.split('.')[0].split('_')[1]
        if not os.path.exists(os.path.join(dst_path, image_filename)):
            shutil.copy(os.path.join(src_path, subject_id + '_' + finger_id, image_filename),
                        os.path.join(dst_path, image_filename))


def move_to_another_folder(src_path, dst_path):
    list_file = os.listdir(src_path)
    for file in list_file:
        shutil.copy(os.path.join(src_path, file), os.path.join(dst_path, file))


def resave_as_jpg(src_path, dst1_path, dst2_path):
    if not os.path.exists(dst1_path):
        os.mkdir(dst1_path)
    list_file = os.listdir(src_path)
    list_file.sort()
    list1_file = list_file[math.floor(len(list_file) / 2):]
    for file in tqdm(list1_file):
        try:
            image = cv2.imread(os.path.join(src_path, file))
            cv2.imwrite(os.path.join(dst1_path, file), image)
        except Exception as e:
            print(file)
            print(e)
            pass
        continue
    if not os.path.exists(dst2_path):
        os.mkdir(dst2_path)
    list2_file = list_file[:math.floor(len(list_file) / 2)]
    for file in tqdm(list2_file):
        try:
            image = cv2.imread(os.path.join(src_path, file))
            cv2.imwrite(os.path.join(dst2_path, file), image)
        except Exception as e:
            print(file)
            print(e)
            pass
        continue


def show_jpg(src_path):
    for file in tqdm(os.listdir(src_path)):
        image = cv2.imread(os.path.join(src_path, file))
        cv2.imshow('show', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def move_npz(src_path, dst_path):
    list_file = os.listdir(src_path)
    for file in tqdm(list_file):
        shutil.move(os.path.join(src_path, file), os.path.join(dst_path, file))
        # npz = np.load(os.path.join(src_path, file))
        # point = npz["points"]
        # np.savez(os.path.join(dst_path, file.split('.')[0]+'.npz'), npz)


def check_corresponding_label(src_path, dst_path):
    list_file = os.listdir(src_path)
    not_label = []
    for file in list_file:
        if not os.path.exists(os.path.join(dst_path, file.split('.')[0] + '.npz')):
            not_label.append(file)

    list_file = os.listdir(dst_path)
    not_image = []
    for file in list_file:
        if not os.path.exists(os.path.join(src_path, file.split('.')[0] + '.jpg')):
            not_image.append(file)

    print(not_label)
    print(not_image)


if __name__ == "__main__":
    list_filename = os.listdir("/mnt/Data/superpoint/exper/outputs/mp_synth-v11_ha1_trained-Left")
    src_path = "/mnt/Data/superpoint/exper/outputs/magic-point_fingerknuckleleftv2-left"
    dst1_path = "/mnt/Data/superpoint/exper/outputs/magic-point_fingerknuckleleftv2-left2"
    dst2_path = "/mnt/Data/superpoint/data/FINGERKNUCKLE/Left-2"
    move_npz(dst1_path, src_path)
