import tensorflow as tf
import cv2 as cv
import numpy as np

from superpoint.datasets.utils import photometric_augmentation as photaug
from superpoint.models.homographies import (sample_homography, compute_valid_mask,
                                            warp_points, filter_points)


def parse_primitives(names, all_primitives):
    p = all_primitives if (names == 'all') \
            else (names if isinstance(names, list) else [names])
    assert set(p) <= set(all_primitives)
    return p


def photometric_augmentation(data, **config):
    with tf.name_scope('photometric_augmentation'):
        primitives = parse_primitives(config['primitives'], photaug.augmentations)
        prim_configs = [config['params'].get(
                             p, {}) for p in primitives]

        indices = tf.range(len(primitives))
        if config['random_order']:
            indices = tf.random_shuffle(indices)

        def step(i, image):
            fn_pairs = [(tf.equal(indices[i], j),
                         lambda p=p, c=c: getattr(photaug, p)(image, **c))
                        for j, (p, c) in enumerate(zip(primitives, prim_configs))]
            image = tf.case(fn_pairs)
            return i + 1, image

        _, image = tf.while_loop(lambda i, image: tf.less(i, len(primitives)),
                                 step, [0, data['image']], parallel_iterations=1)

    return {**data, 'image': image}


def homographic_augmentation(data, add_homography=False, **config):
    with tf.name_scope('homographic_augmentation'):
        image_shape = tf.shape(data['image'])[:2]
        homography = sample_homography(image_shape, **config['params'])[0]
        warped_image = tf.contrib.image.transform(
                data['image'], homography, interpolation='BILINEAR')
        valid_mask = compute_valid_mask(image_shape, homography,
                                        config['valid_border_margin'])

        warped_points = warp_points(data['keypoints'], homography)
        warped_points = filter_points(warped_points, image_shape)

    ret = {**data, 'image': warped_image, 'keypoints': warped_points,
           'valid_mask': valid_mask}
    if add_homography:
        ret['homography'] = homography
    return ret


def add_dummy_valid_mask(data):
    """
    Add valid_mask which is a matrix with ones, and the valid_mask has the same shape of data['image']
    Args:
        data:

    Returns:
        {**data, 'valid_mask': valid_mask}

    """
    with tf.name_scope('dummy_valid_mask'):
        valid_mask = tf.ones(tf.shape(data['image'])[:2], dtype=tf.int32)
    return {**data, 'valid_mask': valid_mask}


def add_keypoint_map(data):
    """
    Add keypoint_map attribute based on data
    Args:
        data:

    Returns:
        data: {**data, 'keypoint_map': kmap}
        the h and w of kmap are same as data['image']

    """
    with tf.name_scope('add_keypoint_map'):
        image_shape = tf.shape(data['image'])[:2]
        kp = tf.minimum(tf.to_int32(tf.round(data['keypoints'])), image_shape-1)
        kmap = tf.scatter_nd(
                kp, tf.ones([tf.shape(kp)[0]], dtype=tf.int32), image_shape)
    return {**data, 'keypoint_map': kmap}


def add_keypoint_classe_angle_map(data):
    with tf.name_scope('add_keypoint_map'):
        image_shape = tf.shape(data['image'])[:2]
        kp = tf.minimum(tf.to_int32(tf.round(data['keypoints'])), image_shape-1)
        # tf.scatter_nd initially zero for numeric
        kmap = tf.scatter_nd(kp, tf.ones([tf.shape(kp)[0]], dtype=tf.int32), image_shape)
    with tf.name_scope('add_classes_map'):
        image_shape = tf.shape(data['image'])[:2]
        kp = tf.minimum(tf.to_int32(tf.round(data['keypoints'])), image_shape-1)
        # tf.shape(kp)[0] should equal to tf.shape(data['classes'])[0]
        cmap = tf.scatter_nd(kp, tf.to_int32(tf.squeeze(data['classes'], axis=1)), image_shape)
    # To do by Zhenyu ZHOU
    # At present, it cannot support CSL for angular loss
    with tf.name_scope('add_keypoint_map'):
        image_shape = tf.shape(data['image'])[:2]
        kp = tf.minimum(tf.to_int32(tf.round(data['keypoints'])), image_shape-1)
        amap = tf.scatter_nd(kp, tf.to_int32(tf.squeeze(data['angles'], axis=1)), image_shape)

    return {**data, 'keypoint_map': kmap, 'classes_map': cmap, 'angles_map': amap}


def downsample(image, coordinates, **config):
    with tf.name_scope('gaussian_blur'):
        k_size = config['blur_size']
        kernel = cv.getGaussianKernel(k_size, 0)[:, 0]
        kernel = np.outer(kernel, kernel).astype(np.float32)
        kernel = tf.reshape(tf.convert_to_tensor(kernel), [k_size]*2+[1, 1])
        pad_size = int(k_size/2)
        image = tf.pad(image, [[pad_size]*2, [pad_size]*2, [0, 0]], 'REFLECT')
        image = tf.expand_dims(image, axis=0)  # add batch dim
        image = tf.nn.depthwise_conv2d(image, kernel, [1, 1, 1, 1], 'VALID')[0]

    with tf.name_scope('downsample'):
        ratio = tf.divide(tf.convert_to_tensor(config['resize']), tf.shape(image)[0:2])
        coordinates = coordinates * tf.cast(ratio, tf.float32)
        image = tf.image.resize_images(image, config['resize'],
                                       method=tf.image.ResizeMethod.BILINEAR)

    return image, coordinates


def ratio_preserving_resize(image, **config):
    target_size = tf.convert_to_tensor(config['resize'])
    scales = tf.to_float(tf.divide(target_size, tf.shape(image)[:2]))
    new_size = tf.to_float(tf.shape(image)[:2]) * tf.reduce_max(scales)
    image = tf.image.resize_images(image, tf.to_int32(new_size),
                                   method=tf.image.ResizeMethod.BILINEAR)
    return tf.image.resize_image_with_crop_or_pad(image, target_size[0], target_size[1])
