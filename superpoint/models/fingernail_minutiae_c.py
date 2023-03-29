import tensorflow as tf

from .base_model import BaseModel, Mode
from .backbones.vgg import vgg_backbone
from .utils import minutiae_head, minutiae_loss, box_nms
from .homographies import homography_adaptation, homography_adaptation_minutiae
from .utils import points_head, points_loss, classes_head, classes_loss, angle_head, angles_loss


class FingernailMinutiaeC(BaseModel):
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
            output_c = classes_head(features, **config)
            # outputs = minutiae_head(features, **config)
            return output_c

        if (mode == Mode.PRED) and config['homography_adaptation']['num']:
            outputs = homography_adaptation_minutiae(image, net, config['homography_adaptation'])
        else:
            outputs = net(image)

        return outputs

    def _loss(self, outputs, inputs, **config):
        if config['data_format'] == 'channels_first':
            outputs['classes_raw'] = tf.transpose(outputs['classes_raw'], [0, 2, 3, 1])
        loss_c = classes_loss(inputs['classes_map'], outputs['classes_raw'], inputs['keypoint_map'], **config)
        loss = config['c_loss']*loss_c
        return loss

    def _metrics(self, outputs, inputs, **config):
        labels = inputs['keypoint_map']

        # =========== for classification accuracy
        pred_cls = tf.cast(outputs['classes'], tf.int32)
        pred_cls = tf.cast(tf.equal(pred_cls, inputs['classes_map']), tf.int32)
        pred_cls = inputs['valid_mask'] * labels * pred_cls
        cls_acc = tf.reduce_sum(pred_cls) / tf.reduce_sum(labels)

        return {'cls_acc': cls_acc}
