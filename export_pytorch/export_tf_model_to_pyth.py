import warnings

warnings.simplefilter('ignore', category=FutureWarning)
warnings.simplefilter('ignore', category=DeprecationWarning)
warnings.simplefilter('ignore', category=Warning)
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import cv2
import yaml
import torch
import matplotlib.pyplot as plt
import superpoint_pytorch

import tensorflow.compat.v1 as tf

assert tf.__version__ < '2', 'Requires TF 1.x'

from superpoint.models import get_model
from pprint import pprint
from pathlib import Path
from notebooks.utils import plot_imgs


def init_sp_pyth(detection_thresh, nms_radius):
    sp_pyth = superpoint_pytorch.SuperPoint(detection_threshold=detection_thresh, nms_radius=nms_radius).eval()
    print('Config:', sp_pyth.conf)

    return sp_pyth


def ckpt_to_pth(ckpt_path, pth_path, sp_pyth):
    """
    Convert the Tensorflow weights into the PyTorch model
    Args:
        ckpt_path:
        pth_path:
        sp_pyth:

    Returns:
        sp_pyth:
    """
    path_ckpt_tf = Path(ckpt_path)

    reader = tf.train.NewCheckpointReader(str(path_ckpt_tf))
    name2shape = reader.get_variable_to_shape_map()
    keys = sorted(k for k in name2shape if not 'Adam' in k)
    keys = [k for k in keys if not any(s in k for s in ['beta1_power', 'beta2_power', 'global_step'])]

    tf_state_dict = {}
    for k in keys:
        k2 = k.replace('superpoint/', '').replace('vgg', 'backbone')
        k2 = k2.replace('gamma', 'weight').replace('beta', 'bias').replace('kernel', 'weight')
        k2 = k2.replace('moving_', 'running_').replace('variance', 'var')

        prefix, block, *remain = k2.split('/')
        if prefix in ('descriptor', 'detector'):
            idx = str(int(block.replace('conv', '')) - 1)
            k2 = (prefix, idx, *remain)
        else:
            i, j = block.replace('conv', '').split('_')
            k2 = (prefix, str(int(i) - 1), str(int(j) - 1), *remain)

        k2 = '.'.join(k2)
        assert k2 in sp_pyth.state_dict(), (k2, k)

        val = reader.get_tensor(k)
        if 'conv/kernel' in k:
            val = val.transpose(3, 2, 0, 1)  # or probably not 3,2,1,0
        expected_shape = sp_pyth.state_dict()[k2].shape
        assert val.shape == expected_shape, (val.shape, expected_shape)
        tf_state_dict[k2] = torch.Tensor(val)

    torch.save(tf_state_dict, pth_path)
    sp_pyth.load_state_dict(tf_state_dict)
    return sp_pyth


def load_image(image_path, image_size, rgb=True):
    """

    Args:
        image_path:
        image_size: w x h
        rgb: Ture: rgb; False: gray

    Returns:
        dst_img: [h, w, c]

    """
    image = cv2.imread(image_path, cv2.IMREAD_COLOR) if rgb else cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None
    if image.ndim == 2:
        image = np.expand_dims(image, -1)
    h, w, c = image.shape
    r = h / w
    dst_w, dst_h = image_size
    dst_r = dst_h / dst_w
    if r > dst_r:  # crop h
        crop_h = h - dst_r * w
        image = image[int(crop_h / 2):int(h - crop_h / 2), :, :]
    else:
        crop_w = w - h / dst_r
        image = image[:, int(crop_w / 2):int(w - crop_w / 2), :]

    dst_image = cv2.resize(image, dsize=(dst_w, dst_h))
    # dst_image = np.expand_dims(dst_image, -1) if dst_image.ndim == 2 else dst_image

    return dst_image


def sp_tf_vs_pyth_inference(ckpt_path, detection_thresh, nms_radius, image):
    # Run the inference with the PyTorch model
    with torch.no_grad():
        # np.array([None, :]) just like unsqueeze(0, 1)
        pred_th = sp_pyth({'image': torch.from_numpy(image[None, None]).float()})
    points_th = pred_th['keypoints'][0]
    plot_imgs([image], cmap='gray', titles=[f'PyTorch model, {len(points_th)} points'])
    plt.scatter(*points_th.T, lw=0, s=4, c='lime')
    plt.show()

    # Run the inference with the Tensorflow model
    with open(os.path.join(os.path.dirname(ckpt_path), 'config.yml'), 'r') as f:
        config_tf = yaml.load(f, Loader=yaml.Loader)
    config_tf['model'].update({
        'data_format': 'channels_last',
        'top_k': 0,
        'detection_threshold': detection_thresh,
        'nms': (nms_radius - 1) * 2,  # seems equivalent
    })
    pprint(config_tf['model'])

    tf.logging.set_verbosity(tf.logging.ERROR)

    SuperPointTF = get_model(config_tf['model']['name'])
    sp_tf = SuperPointTF(**config_tf['model'], n_gpus=1, data_shape={'image': [None, None, None, 1]})
    sp_tf.load(str(ckpt_path))

    pred_tf = sp_tf.predict({'image': image[:, :, None]}, keys='*')
    points_tf = np.stack(np.where(pred_tf['pred']), -1)[:, ::-1]
    plot_imgs([image], cmap='gray', titles=[f'TensorFlow model, {len(points_tf)} points'])
    plt.scatter(*points_tf.T, lw=0, s=4, c='lime')
    plt.show()

    # Compare the dense outputs
    image_tensor = torch.from_numpy(image[None, None]).float()
    with torch.no_grad():
        logits_dense = sp_pyth.detector(sp_pyth.backbone(image_tensor))
        desc_dense = sp_pyth.descriptor(sp_pyth.backbone(image_tensor))
    logits_dense = logits_dense.squeeze(0).permute(1, 2, 0).numpy()
    desc_dense = desc_dense.squeeze(0).permute(1, 2, 0).numpy()

    diff = np.abs(logits_dense - pred_tf['logits'])
    print('Diff logits:', diff.max(), diff.mean(), np.median(diff), 'max/mean/median')

    diff = np.abs(desc_dense - pred_tf['descriptors_raw'])
    print('Diff descriptors:', diff.max(), diff.mean(), np.median(diff), 'max/mean/median')
    return 0


def sp_pyth_batch(subject_path, sp_pyth, image_size):
    # Test the batched inference
    list_subject = os.listdir(subject_path)
    for subject in list_subject:
        images = []
        list_file = os.listdir(os.path.join(subject_path, subject))
        for file in list_file:
            images.append(load_image(os.path.join(subject_path, subject, file), image_size, rgb=False) / 255.)
        h, w = np.array([i.shape for i in images]).min(0)
        images = np.stack([i[:h, :w] for i in images])

        with torch.no_grad():
            # np.array([:, None]) just like unsqueeze(1)
            pred_th = sp_pyth({'image': torch.from_numpy(images[:, None]).float()})
        plot_imgs(images, cmap='gray')
        for p, ax in zip(pred_th['keypoints'], plt.gcf().axes):
            ax.scatter(*p.T, lw=0, s=4, c='lime')
        plt.show()

    return 0


if __name__ == "__main__":
    detection_thresh = 0.015
    nms_radius = 5
    ckpt_path = "/mnt/Data/superpoint/exper/superpoint_fingerknuckleleftv3/model.ckpt-2100000"
    pth_path = "/mnt/Data/superpoint/exper/superpoint_fingerknuckleleftv3/sp_fkleftv3-2100000.pth"
    image_path = "/mnt/Data/Project/Finger-Knuckle-Video/dataset_exp2/dataset/R2-10/0001_R2/0001_R2_1.jpg"
    subject_path = "/mnt/Data/Finger-Knuckle-Database/PolyUKnuckleV3/YOLOv5_Segment/184_208/Session_1/all/"
    image_size = [152, 200]

    # construct superpoint pytorch model
    sp_pyth = init_sp_pyth(detection_thresh, nms_radius)
    sp_pyth = ckpt_to_pth(ckpt_path, pth_path, sp_pyth)

    # Load an image to test the results
    image = load_image(image_path, image_size, rgb=False) / 255
    image = np.pad(image, [(0, int(np.ceil(s / 8)) * 8 - s) for s in image.shape[:2]])
    plot_imgs([image], cmap='gray')
    plt.show()

    sp_tf_vs_pyth_inference(ckpt_path, detection_thresh, nms_radius, image)

    sp_pyth_batch(subject_path, sp_pyth, image_size)
