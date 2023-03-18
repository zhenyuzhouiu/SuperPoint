import tensorflow as tf

from .base_model import BaseModel, Mode
from .backbones.vgg import vgg_backbone
from .utils import detector_head, detector_loss, box_nms, detector_mse_loss,\
    classes_head, classes_loss, angle_head, angle_loss
from .homographies import homography_adaptation


class FingernailMinutiae(BaseModel):
    input_spec = {
            'image': {'shape': [None, None, None, 1], 'type': tf.float32}
    }
    required_config_keys = []
    default_config = {
            'data_format': 'channels_first',
            'kernel_reg': 0.,
            'grid_size': 8,
            'detection_threshold': 0.4,
            'homography_adaptation': {'num': 0},
            'nms': 0,
            'top_k': 0
    }

    def _model(self, inputs, mode, **config):
        config['training'] = (mode == Mode.TRAIN)
        image = inputs['image']

        def net(image):
            if config['data_format'] == 'channels_first':
                image = tf.transpose(image, [0, 3, 1, 2])
            features = vgg_backbone(image, **config)  # [N, H/8, W/8, F]
            out_points = detector_head(features, **config)
            out_classes = classes_head(features, **config)
            out_angles = angle_head(features, **config)

            return {**out_points, **out_classes, **out_angles}

        if (mode == Mode.PRED) and config['homography_adaptation']['num']:
            outputs = homography_adaptation(image, net, config['homography_adaptation'])
        else:
            outputs = net(image)

        prob = outputs['prob']
        if config['nms']:
            prob = tf.map_fn(lambda p: box_nms(p, config['nms'],
                                               min_prob=config['detection_threshold'],
                                               keep_top_k=config['top_k']), prob)
            outputs['prob_nms'] = prob
        pred = tf.to_int32(tf.greater_equal(prob, config['detection_threshold']))
        outputs['pred'] = pred

        return outputs

    def _loss(self, outputs, inputs, **config):
        if config['data_format'] == 'channels_first':
            outputs['logits'] = tf.transpose(outputs['logits'], [0, 2, 3, 1])
            outputs['classes_raw'] = tf.transpose(outputs['classes_raw'], [0, 2, 3, 1])
            outputs['angles_raw'] = tf.transpose(outputs['angles_raw'], [0, 2, 3, 1])
        loss_points = detector_loss(inputs['keypoint_map'], outputs['logits'],
                                    valid_mask=inputs['valid_mask'], **config)
        loss_classes = classes_loss(inputs['classes_map'], outputs['classes_raw'],
                                    valid_mask=inputs['keypoint_map'], **config)
        loss_angle = angle_loss(inputs['angles_map'], outputs['angles_raw'],
                                valid_mask=inputs['keypoint_map'], **config)
        loss = loss_points + loss_classes + loss_angle
        return loss

    def _metrics(self, outputs, inputs, **config):
        pred = inputs['valid_mask'] * outputs['pred']
        labels = inputs['keypoint_map']

        precision = tf.reduce_sum(pred * labels) / tf.reduce_sum(pred)
        recall = tf.reduce_sum(pred * labels) / tf.reduce_sum(labels)

        return {'precision': precision, 'recall': recall}
