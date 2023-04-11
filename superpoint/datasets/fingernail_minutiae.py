import numpy as np
import tensorflow as tf
from pathlib import Path

from .base_dataset import BaseDataset
from .utils import pipeline
from superpoint.settings import DATA_PATH, EXPER_PATH


class FingernailMinutiae(BaseDataset):
    default_config = {
        'labels': None,
        'cache_in_memory': False,
        'validation_size': 100,
        'truncate': None,
        'preprocessing': {
            'resize': [480, 640]
        },
        'num_parallel_calls': 10,
        'augmentation': {
            'photometric': {
                'enable': False,
                'primitives': 'all',
                'params': {},
                'random_order': True,
            },
            'homographic': {
                'enable': False,
                'params': {},
                'valid_border_margin': 0,
            },
        },
        'warped_pair': {
            'enable': False,
            'params': {},
            'valid_border_margin': 0,
        },
    }

    def _init_dataset(self, **config):
        """

        Args:
            **config:

        Returns:
            files = {'image_paths':,
                     'names':, p.stem for p in image_paths
                     'label_path':
                     }

        """
        base_path = Path(DATA_PATH, 'fingernail/train/')
        image_paths = list(base_path.iterdir())
        if config['truncate']:
            image_paths = image_paths[:config['truncate']]
        names = [p.stem for p in image_paths]
        image_paths = [str(p) for p in image_paths]
        files = {'image_paths': image_paths, 'names': names}

        if config['labels']:
            label_paths = []
            for n in names:
                p = Path(EXPER_PATH, config['labels'], '{}.npz'.format(n))
                assert p.exists(), 'Image {} has no corresponding label {}'.format(n, p)
                label_paths.append(str(p))
            files['label_paths'] = label_paths

        tf.data.Dataset.map_parallel = lambda self, fn: self.map(
                fn, num_parallel_calls=config['num_parallel_calls'])

        return files

    def _get_data(self, files, split_name, **config):
        """

        Args:
            files: dictionary
            split_name: ['training', 'validation', 'test']
            **config: **kwargs

        Returns:
            data: tf.data.Dataset.zip({'image', 'names', 'keypoints', 'valid_mask', 'keypoint_map'})

        """
        has_keypoints = 'label_paths' in files
        is_training = split_name == 'training'

        def _read_image(path):
            image = tf.read_file(path)
            # image = tf.image.decode_jpeg(image, channels=3)
            image = tf.image.decode_png(image, channels=3)
            return tf.cast(image, tf.float32)

        def _preprocess(image):
            image = tf.image.rgb_to_grayscale(image)
            if config['preprocessing']['resize']:
                image = pipeline.ratio_preserving_resize(image,
                                                         **config['preprocessing'])
            return image

        # Python function
        # Get the ground truth points
        def _read_points(filename):
            p_array = np.load(filename.decode('utf-8'))['points'].astype(np.float32)
            return p_array

        def _read_a_points(filename):
            a_p_array = np.load(filename.decode('utf-8'))['a_point'].astype(np.float32)
            return a_p_array

        def _read_classes(filename):
            c_array = np.load(filename.decode('utf-8'))['classes'].astype(np.float32)
            return c_array

        def _read_angles(filename):
            a_array = np.load(filename.decode('utf-8'))['angles'].astype(np.float32)
            return a_array

        def _read_labels(filename):
            """

            Args:
                filename:

            Returns:
                points, classes, angles

            """
            points = np.load(filename.decode('utf-8'))['points'].astype(np.float32)
            classes = np.load(filename.decode('utf-8'))['classes'].astype(np.float32)
            angles = np.load(filename.decode('utf-8'))['angles'].astype(np.float32)
            return {'points': points, 'classes': classes, 'angles': angles}

        names = tf.data.Dataset.from_tensor_slices(files['names'])
        images = tf.data.Dataset.from_tensor_slices(files['image_paths'])
        images = images.map(_read_image)
        images = images.map(_preprocess)
        data = tf.data.Dataset.zip({'image': images, 'name': names})

        # Add key points
        if has_keypoints:
            lb_path = tf.data.Dataset.from_tensor_slices(files['label_paths'])
            points = lb_path.map(lambda path: tf.py_func(_read_points, [path], tf.float32))
            a_points = lb_path.map(lambda path: tf.py_func(_read_a_points, [path], tf.float32))
            classes = lb_path.map(lambda path: tf.py_func(_read_classes, [path], tf.float32))
            angles = lb_path.map(lambda path: tf.py_func(_read_angles, [path], tf.float32))
            points = points.map(lambda p: tf.reshape(p, [-1, 2]))  # [num_points, 2]
            a_points = a_points.map(lambda p: tf.reshape(p, [-1, 2]))  # [num_points, 2]
            classes = classes.map(lambda c: tf.reshape(c, [-1, 1]))  # [num_points, 1]
            angles = angles.map(lambda a: tf.reshape(a, [-1, 1]))  # [num_points, 1]
            data = tf.data.Dataset.zip((data, points)).map(lambda d, p: {**d, 'keypoints': p})
            data = tf.data.Dataset.zip((data, a_points)).map(lambda d, ap: {**d, 'a_points': ap})
            data = tf.data.Dataset.zip((data, classes)).map(lambda d, c: {**d, 'classes': c})
            data = tf.data.Dataset.zip((data, angles)).map(lambda d, a: {**d, 'angles': a})
            data = data.map(pipeline.add_dummy_valid_mask)

        # Keep only the first elements for validation
        if split_name == 'validation':
            data = data.take(config['validation_size'])

        # Cache to avoid always reading from disk
        if config['cache_in_memory']:
            tf.logging.info('Caching data, fist access will take some time.')
            data = data.cache()

        # Generate the warped pair only during training super_point model
        if config['warped_pair']['enable']:
            assert has_keypoints
            warped = data.map_parallel(lambda d: pipeline.homographic_augmentation_minutiae(
                d, add_homography=True, **config['warped_pair']))
            if is_training and config['augmentation']['photometric']['enable']:
                warped = warped.map_parallel(lambda d: pipeline.photometric_augmentation(
                    d, **config['augmentation']['photometric']))
            warped = warped.map_parallel(pipeline.add_keypoint_map)
            # Merge with the original data
            data = tf.data.Dataset.zip((data, warped))
            data = data.map(lambda d, w: {**d, 'warped': w})

        # Data augmentation
        if has_keypoints and is_training:
            if config['augmentation']['photometric']['enable']:
                data = data.map_parallel(lambda d: pipeline.photometric_augmentation(
                    d, **config['augmentation']['photometric']))
            if config['augmentation']['homographic']['enable']:
                assert not config['warped_pair']['enable']  # doesn't support hom. aug.
                data = data.map_parallel(lambda d: pipeline.homographic_augmentation_minutiae(
                    d, **config['augmentation']['homographic']))

        # Generate the keypoint map, classes map and angles map
        if has_keypoints:
            # data = data.map_parallel(pipeline.add_keypoint_map)
            data = data.map_parallel(pipeline.add_keypoint_classe_angle_map)
        data = data.map_parallel(
            lambda d: {**d, 'image': tf.to_float(d['image']) / 255.})
        if config['warped_pair']['enable']:
            data = data.map_parallel(
                lambda d: {
                    **d, 'warped': {**d['warped'],
                                    'image': tf.to_float(d['warped']['image']) / 255.}})

        return data
