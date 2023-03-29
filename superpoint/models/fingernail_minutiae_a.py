import tensorflow as tf

from .base_model import BaseModel, Mode
from .backbones.vgg import vgg_backbone
from .utils import minutiae_head, minutiae_loss, box_nms
from .homographies import homography_adaptation, homography_adaptation_minutiae
from .utils import points_head, points_loss, classes_head, classes_loss, angle_head, angles_loss


class FingernailMinutiaeA(BaseModel):
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
            output_a = angle_head(features, **config)
            # outputs = minutiae_head(features, **config)
            return output_a

        if (mode == Mode.PRED) and config['homography_adaptation']['num']:
            outputs = homography_adaptation_minutiae(image, net, config['homography_adaptation'])
        else:
            outputs = net(image)

        return outputs

    def _loss(self, outputs, inputs, **config):
        if config['data_format'] == 'channels_first':
            outputs['angles_raw'] = tf.transpose(outputs['angles_raw'], [0, 2, 3, 1])
        loss_a = angles_loss(inputs['angles_map'], outputs['angles_raw'], inputs['keypoint_map'], **config)
        loss = config['a_loss']*loss_a
        return loss

    def _metrics(self, outputs, inputs, **config):
        labels = inputs['keypoint_map']

        # =========== for angle accuracy
        pred_ang = tf.cast(outputs['angles'], tf.int32)
        pred_ang = tf.cast(tf.equal(pred_ang, inputs['angles_map']), tf.int32)
        pred_ang = inputs['valid_mask'] * labels * pred_ang
        ang_acc = tf.reduce_sum(pred_ang) / tf.reduce_sum(labels)

        return {'ang_acc': ang_acc}
