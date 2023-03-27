import tensorflow as tf

from .base_model import BaseModel, Mode
from .backbones.vgg import vgg_backbone
from .utils import minutiae_head, minutiae_loss, box_nms
from .homographies import homography_adaptation, homography_adaptation_minutiae
from .utils import points_head, points_loss, classes_head, classes_loss, angle_head, angles_loss


class FingernailMinutiaeP(BaseModel):
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
            features = vgg_backbone(image, **config)  # [N, F, H/8, W/8]
            output_p = points_head(features, **config)
            # outputs = minutiae_head(features, **config)
            return output_p

        if (mode == Mode.PRED) and config['homography_adaptation']['num']:
            outputs = homography_adaptation_minutiae(image, net, config['homography_adaptation'])
        else:
            outputs = net(image)

        prob = outputs['prob']
        if config['nms']:
            prob = tf.map_fn(lambda p: box_nms(p, config['nms'],
                                               min_prob=config['detection_threshold'],
                                               keep_top_k=config['top_k']), prob)
            outputs['prob_nms'] = prob
        # pred = tf.to_int32(tf.greater_equal(prob, config['detection_threshold']))
        pred = tf.cast(tf.greater_equal(prob, config['detection_threshold']), tf.int32)

        outputs['pred'] = pred

        return outputs

    def _loss(self, outputs, inputs, **config):
        if config['data_format'] == 'channels_first':
            outputs['logits'] = tf.transpose(outputs['logits'], [0, 2, 3, 1])
        loss_p = points_loss(inputs['keypoint_map'], outputs['logits'], inputs['valid_mask'], **config)
        loss = config['p_loss']*loss_p
        return loss

    def _metrics(self, outputs, inputs, **config):

        # =========== for all points and for all classes
        pred = inputs['valid_mask'] * outputs['pred']
        labels = inputs['keypoint_map']

        precision = tf.reduce_sum(pred * labels) / tf.reduce_sum(pred)
        recall = tf.reduce_sum(pred * labels) / tf.reduce_sum(labels)


        return {'precision': precision, 'recall': recall}
