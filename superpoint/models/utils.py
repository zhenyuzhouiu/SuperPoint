import tensorflow as tf

from .homographies import warp_points
from .backbones.vgg import vgg_block


def detector_head(inputs, **config):
    """

    Args:
        inputs:
        **config:

    Returns:
        {
        'logits': x,  [N, 65, H/8, W/8]
        'prob': prob  [N, H, W] prob = tf.nn.softmax(x, axis=cindex)
        }

    """
    params_conv = {'padding': 'SAME', 'data_format': config['data_format'],
                   'batch_normalization': True,
                   'training': config['training'],
                   'kernel_reg': config.get('kernel_reg', 0.)}
    cfirst = config['data_format'] == 'channels_first'
    cindex = 1 if cfirst else -1  # index of the channel [N, C, H/8, W/8]

    with tf.variable_scope('detector', reuse=tf.AUTO_REUSE):
        x = vgg_block(inputs, 256, 3, 'conv1',
                      activation=tf.nn.relu, **params_conv)
        # grid_size: one pixel of feature maps represent grid size pixels on the input image
        x = vgg_block(x, 1+pow(config['grid_size'], 2), 1, 'conv2',
                      activation=None, **params_conv)  # [N, 65, H/8, W/8]

        prob = tf.nn.softmax(x, axis=cindex)
        # Strip the extra “no interest point” dustbin
        prob = prob[:, :-1, :, :] if cfirst else prob[:, :, :, :-1]
        # [N, 1, H, W], H*W is the original size of input images
        prob = tf.depth_to_space(
                prob, config['grid_size'], data_format='NCHW' if cfirst else 'NHWC')
        prob = tf.squeeze(prob, axis=cindex)  # [N, H, W]

    return {'logits': x, 'prob': prob}


def descriptor_head(inputs, **config):
    params_conv = {'padding': 'SAME', 'data_format': config['data_format'],
                   'batch_normalization': True,
                   'training': config['training'],
                   'kernel_reg': config.get('kernel_reg', 0.)}
    cfirst = config['data_format'] == 'channels_first'
    cindex = 1 if cfirst else -1  # index of the channel

    with tf.variable_scope('descriptor', reuse=tf.AUTO_REUSE):
        x = vgg_block(inputs, 256, 3, 'conv1',
                      activation=tf.nn.relu, **params_conv)
        x = vgg_block(x, config['descriptor_size'], 1, 'conv2',
                      activation=None, **params_conv)

        desc = tf.transpose(x, [0, 2, 3, 1]) if cfirst else x
        desc = tf.image.resize_bilinear(
            desc, config['grid_size'] * tf.shape(desc)[1:3])
        desc = tf.transpose(desc, [0, 3, 1, 2]) if cfirst else desc
        desc = tf.nn.l2_normalize(desc, cindex)

    return {'descriptors_raw': x, 'descriptors': desc}


def detector_loss(keypoint_map, logits, valid_mask=None, **config):
    # Convert the boolean labels to indices including the "no interest point" dustbin
    labels = tf.to_float(keypoint_map[..., tf.newaxis])  # for GPU  [N, H, W, 1]
    labels = tf.space_to_depth(labels, config['grid_size'])  # [N, H/8, W/8, 64]
    # tf.shape() return tf.Tensor
    shape = tf.concat([tf.shape(labels)[:3], [1]], axis=0)  # [N, H/8, W/8, 1]
    labels = tf.concat([2*labels, tf.ones(shape)], 3)  # [N, H/8/ W/8, 65]
    # Add a small random matrix to randomly break ties in argmax
    # If two ground truth corner positions land in the same bin,
    # then we randomly select one ground truth corner location
    labels = tf.argmax(labels + tf.random_uniform(tf.shape(labels), 0, 0.1),
                       axis=3)  # [N, H/8, W/8] with labels

    # Mask the pixels if bordering artifacts appear
    valid_mask = tf.ones_like(keypoint_map) if valid_mask is None else valid_mask
    valid_mask = tf.to_float(valid_mask[..., tf.newaxis])  # for GPU  # [N, H, W, 1]
    valid_mask = tf.space_to_depth(valid_mask, config['grid_size'])  # [N, H/8, W/8, 64]
    # computes tf.math.multiply of elements across dimensions of a tensor
    valid_mask = tf.reduce_prod(valid_mask, axis=3)  # AND along the channel dim [N, H/8, W/8]

    loss = tf.losses.sparse_softmax_cross_entropy(
            labels=labels, logits=logits, weights=valid_mask)
    return loss


def detector_mse_loss(keypoint_map, prob, valid_mask=None, **config):

    labels = tf.to_float(keypoint_map)  # [N, H, W]

    # Mask the pixels if bordering artifacts appear
    valid_mask = tf.ones_like(keypoint_map) if valid_mask is None else valid_mask  # [N, H, W]

    loss = tf.losses.mean_pairwise_squared_error(labels=labels, predictions=prob, weights=valid_mask)
    return loss


def descriptor_loss(descriptors, warped_descriptors, homographies, valid_mask=None, **config):
    # Compute the position of the center pixel of every cell in the image
    (batch_size, Hc, Wc) = tf.unstack(tf.to_int32(tf.shape(descriptors)[:3]))
    coord_cells = tf.stack(tf.meshgrid(
        tf.range(Hc), tf.range(Wc), indexing='ij'), axis=-1)
    coord_cells = coord_cells * config['grid_size'] + config['grid_size'] // 2  # (Hc, Wc, 2)
    # coord_cells is now a grid containing the coordinates of the Hc x Wc
    # center pixels of the 8x8 cells of the image

    # Compute the position of the warped center pixels
    warped_coord_cells = warp_points(tf.reshape(coord_cells, [-1, 2]), homographies)
    # warped_coord_cells is now a list of the warped coordinates of all the center
    # pixels of the 8x8 cells of the image, shape (N, Hc x Wc, 2)

    # Compute the pairwise distances and filter the ones less than a threshold
    # The distance is just the pairwise norm of the difference of the two grids
    # Using shape broadcasting, cell_distances has shape (N, Hc, Wc, Hc, Wc)
    coord_cells = tf.to_float(tf.reshape(coord_cells, [1, 1, 1, Hc, Wc, 2]))
    warped_coord_cells = tf.reshape(warped_coord_cells,
                                    [batch_size, Hc, Wc, 1, 1, 2])
    cell_distances = tf.norm(coord_cells - warped_coord_cells, axis=-1)
    s = tf.to_float(tf.less_equal(cell_distances, config['grid_size'] - 0.5))
    # s[id_batch, h, w, h', w'] == 1 if the point of coordinates (h, w) warped by the
    # homography is at a distance from (h', w') less than config['grid_size']
    # and 0 otherwise

    # Normalize the descriptors and
    # compute the pairwise dot product between descriptors: d^t * d'
    descriptors = tf.reshape(descriptors, [batch_size, Hc, Wc, 1, 1, -1])
    descriptors = tf.nn.l2_normalize(descriptors, -1)
    warped_descriptors = tf.reshape(warped_descriptors,
                                    [batch_size, 1, 1, Hc, Wc, -1])
    warped_descriptors = tf.nn.l2_normalize(warped_descriptors, -1)
    dot_product_desc = tf.reduce_sum(descriptors * warped_descriptors, -1)
    dot_product_desc = tf.nn.relu(dot_product_desc)
    dot_product_desc = tf.reshape(tf.nn.l2_normalize(
        tf.reshape(dot_product_desc, [batch_size, Hc, Wc, Hc * Wc]),
        3), [batch_size, Hc, Wc, Hc, Wc])
    dot_product_desc = tf.reshape(tf.nn.l2_normalize(
        tf.reshape(dot_product_desc, [batch_size, Hc * Wc, Hc, Wc]),
        1), [batch_size, Hc, Wc, Hc, Wc])
    # dot_product_desc[id_batch, h, w, h', w'] is the dot product between the
    # descriptor at position (h, w) in the original descriptors map and the
    # descriptor at position (h', w') in the warped image

    # Compute the loss
    positive_dist = tf.maximum(0., config['positive_margin'] - dot_product_desc)
    negative_dist = tf.maximum(0., dot_product_desc - config['negative_margin'])
    loss = config['lambda_d'] * s * positive_dist + (1 - s) * negative_dist

    # Mask the pixels if bordering artifacts appear
    valid_mask = tf.ones([batch_size,
                          Hc * config['grid_size'],
                          Wc * config['grid_size']], tf.float32)\
        if valid_mask is None else valid_mask
    valid_mask = tf.to_float(valid_mask[..., tf.newaxis])  # for GPU
    valid_mask = tf.space_to_depth(valid_mask, config['grid_size'])
    valid_mask = tf.reduce_prod(valid_mask, axis=3)  # AND along the channel dim
    valid_mask = tf.reshape(valid_mask, [batch_size, 1, 1, Hc, Wc])

    normalization = tf.reduce_sum(valid_mask) * tf.to_float(Hc * Wc)
    # Summaries for debugging
    # tf.summary.scalar('nb_positive', tf.reduce_sum(valid_mask * s) / normalization)
    # tf.summary.scalar('nb_negative', tf.reduce_sum(valid_mask * (1 - s)) / normalization)
    tf.summary.scalar('positive_dist', tf.reduce_sum(valid_mask * config['lambda_d'] *
                                                     s * positive_dist) / normalization)
    tf.summary.scalar('negative_dist', tf.reduce_sum(valid_mask * (1 - s) *
                                                     negative_dist) / normalization)
    loss = tf.reduce_sum(valid_mask * loss) / normalization
    return loss


def spatial_nms(prob, size):
    """Performs non maximum suppression on the heatmap using max-pooling. This method is
    faster than box_nms, but does not suppress contiguous that have the same probability
    value.

    Arguments:
        prob: the probability heatmap, with shape `[H, W]`.
        size: a scalar, the size of the pooling window.
    """

    with tf.name_scope('spatial_nms'):
        prob = tf.expand_dims(tf.expand_dims(prob, axis=0), axis=-1)
        pooled = tf.nn.max_pool(
                prob, ksize=[1, size, size, 1], strides=[1, 1, 1, 1], padding='SAME')
        prob = tf.where(tf.equal(prob, pooled), prob, tf.zeros_like(prob))
        return tf.squeeze(prob)


def box_nms(prob, size, iou=0.1, min_prob=0.01, keep_top_k=0):
    """Performs non-maximum suppression on the heatmap by considering hypothetical
    bounding boxes centered at each pixel's location (e.g. corresponding to the receptive
    field). Optionally only keeps the top k detections.

    Arguments:
        prob: the probability heatmap, with shape `[H, W]`.
        size: a scalar, the size of the bounding boxes.
        iou: a scalar, the IoU overlap threshold.
        min_prob: a threshold under which all probabilities are discarded before NMS.
        keep_top_k: an integer, the number of top scores to keep.
    """
    with tf.name_scope('box_nms'):
        # tf.where(tf.greater_equal()) return the index where is True
        # return the pts coordinates
        pts = tf.to_float(tf.where(tf.greater_equal(prob, min_prob)))  # [N, 2]
        size = tf.constant(size/2.)
        boxes = tf.concat([pts-size, pts+size], axis=1)  # left upper point & right bottom point
        scores = tf.gather_nd(prob, tf.to_int32(pts))  # get the prediction score
        with tf.device('/cpu:0'):
            indices = tf.image.non_max_suppression(
                    boxes, scores, tf.shape(boxes)[0], iou)
        pts = tf.gather(pts, indices)
        scores = tf.gather(scores, indices)
        if keep_top_k:
            k = tf.minimum(tf.shape(scores)[0], tf.constant(keep_top_k))  # when fewer
            scores, indices = tf.nn.top_k(scores, k)
            pts = tf.gather(pts, indices)
        prob = tf.scatter_nd(tf.to_int32(pts), scores, tf.shape(prob))
    return prob


def points_head(inputs, **config):
    params_conv = {'padding': 'SAME', 'data_format': config['data_format'],
                   'batch_normalization': True,
                   'training': config['training'],
                   'kernel_reg': config.get('kernel_reg', 0.)}
    cfirst = config['data_format'] == 'channels_first'
    cindex = 1 if cfirst else -1  # index of the channel [N, C, H/8, W/8]

    with tf.variable_scope('detector', reuse=tf.AUTO_REUSE):
        x = vgg_block(inputs, 256, 3, 'conv1',
                      activation=tf.nn.relu, **params_conv)
        # grid_size: one pixel of feature maps represent grid size pixels on the input image
        x = vgg_block(x, pow(config['grid_size'], 2), 3, 'conv2',
                      activation=None, **params_conv)  # [N, 65, H/8, W/8]

        # [N, 1, H, W], H*W is the original size of input images
        prob = tf.depth_to_space(
                x, config['grid_size'], data_format='NCHW' if cfirst else 'NHWC')
        prob = tf.nn.sigmoid(prob)
        prob = tf.squeeze(prob, axis=cindex)  # [N, H, W]

    return {'logits': x, 'prob': prob}


def points_loss(keypoint_map, logits, valid_mask=None, **config):
    logits = tf.depth_to_space(logits, config['grid_size'], data_format='NHWC')  # [N, H, W, 1]
    logits = tf.nn.sigmoid(logits)
    if logits.shape.ndims == 4:
        logits = tf.squeeze(logits, axis=-1)
    loss = tf.losses.log_loss(labels=keypoint_map, predictions=logits, weights=valid_mask)
    return loss


def classes_head(inputs, **config):
    params_conv = {'padding': 'SAME', 'data_format': config['data_format'],
                   'batch_normalization': True,
                   'training': config['training'],
                   'kernel_reg': config.get('kernel_reg', 0.)}
    cfirst = config['data_format'] == 'channels_first'
    cindex = 1 if cfirst else -1  # index of the channel [N, C, H/8, W/8]

    with tf.variable_scope('classes', reuse=tf.AUTO_REUSE):
        x = vgg_block(inputs, 256, 3, 'conv1',
                      activation=tf.nn.relu, **params_conv)
        # output_channel: objectiveness, bifurcation, ending
        x = vgg_block(x, 3*pow(config['grid_size'], 2), 3, 'conv2',
                      activation=None, **params_conv)  # [N, 192, H/8, W/8]

        # [N, 3, H, W], H*W is the original size of input images
        cls = tf.depth_to_space(x, config['grid_size'],
                                data_format='NCHW' if cfirst else 'NHWC')
        cls = tf.nn.softmax(cls, axis=cindex)
        # cls = cls[:, :-1, :, :] if cfirst else cls[:, :, :, :-1]
        cls = tf.argmax(cls, axis=cindex)  # [N, H, W]

    return {'classes_raw': x, 'classes': cls}


def classes_loss(classes_map, classes_raw, valid_maks=None, **config):
    logits = tf.depth_to_space(classes_raw, config['grid_size'], data_format='NHWC')
    loss = tf.losses.sparse_softmax_cross_entropy(labels=classes_map, logits=logits, weights=valid_maks)
    return loss


def angle_head(inputs, **config):
    params_conv = {'padding': 'SAME', 'data_format': config['data_format'],
                   'batch_normalization': True,
                   'training': config['training'],
                   'kernel_reg': config.get('kernel_reg', 0.)}
    cfirst = config['data_format'] == 'channels_first'
    cindex = 1 if cfirst else -1  # index of the channel [N, C, H/8, W/8]

    with tf.variable_scope('angles', reuse=tf.AUTO_REUSE):
        x = vgg_block(inputs, 256, 3, 'conv1',
                      activation=tf.nn.relu, **params_conv)
        # output_channel: objectiveness, bifurcation, ending
        x = vgg_block(x, pow(config['grid_size'], 2), 3, 'conv2',
                      activation=None, **params_conv)  # [N, 64, H/8, W/8]

        # [N, 181, H, W], H*W is the original size of input images
        ang = tf.depth_to_space(x, config['grid_size'],
                                data_format='NCHW' if cfirst else 'NHWC')
        # ang = tf.nn.softmax(ang, axis=cindex)
        # ang = tf.argmax(ang, axis=cindex)  # [N, H, W]
        ang = tf.nn.relu(ang)
        ang = tf.squeeze(ang, cindex)

    return {'angles_raw': x, 'angles': ang}


def angles_loss(angles_map, angle_raw, valid_maks=None, **config):
    logits = tf.depth_to_space(angle_raw, config['grid_size'], data_format='NHWC')  # [N, H, W, 1]
    logits = tf.nn.relu(logits)
    if logits.shape.ndims == 4:
        logits = tf.squeeze(logits, axis=-1)
    # logits = tf.squeeze(logits, axis=-1)
    loss = tf.losses.huber_loss(labels=angles_map, predictions=logits, weights=valid_maks)
    # loss = tf.losses.sparse_softmax_cross_entropy(labels=angles_map, logits=logits, weights=valid_maks)
    return loss


def minutiae_head_will_be_delete(inputs, **config):
    params_conv = {'padding': 'SAME', 'data_format': config['data_format'],
                   'batch_normalization': True,
                   'training': config['training'],
                   'kernel_reg': config.get('kernel_reg', 0.)}
    cfirst = config['data_format'] == 'channels_first'
    cindex = 1 if cfirst else -1  # index of the channel [N, C, H/8, W/8]

    with tf.variable_scope('detector', reuse=tf.AUTO_REUSE):
        x = vgg_block(inputs, 256, 3, 'conv1',
                      activation=tf.nn.relu, **params_conv)
        # grid_size: one pixel of feature maps represent grid size pixels on the input image
        x = vgg_block(x, 5 * pow(config['grid_size'], 2), 3, 'conv2',
                      activation=None, **params_conv)  # [N, 256, H/8, W/8]

        shuffle_x = tf.depth_to_space(x, config['grid_size'], data_format='NCHW' if cfirst else 'NHWC')

        prob = shuffle_x[:, 0, :, :] if cfirst else shuffle_x[:, :, :, 0]
        # prob = tf.squeeze(prob, axis=cindex)
        prob = tf.sigmoid(prob)
        c_prob = shuffle_x[:, 1:4, :, :] if cfirst else shuffle_x[:, :, :, 1:4]
        # c_prob = tf.sigmoid(c_prob)
        a_prob = shuffle_x[:, 4, :, :] if cfirst else shuffle_x[:, :, :, 4]
        # a_prob = tf.squeeze(a_prob, axis=cindex)  # [N, H, W]
        a_prob = tf.nn.relu(a_prob)

    return {'minutiae_raw': x, 'prob': prob, 'c_prob': c_prob, 'a_prob': a_prob}


def minutiae_head(inputs, **config):
    params_conv = {'padding': 'SAME', 'data_format': config['data_format'],
                   'batch_normalization': True,
                   'training': config['training'],
                   'kernel_reg': config.get('kernel_reg', 0.)}
    cfirst = config['data_format'] == 'channels_first'
    cindex = 1 if cfirst else -1  # index of the channel [N, C, H/8, W/8]

    with tf.variable_scope('detector', reuse=tf.AUTO_REUSE):
        x = vgg_block(inputs, 256, 3, 'conv1',
                      activation=tf.nn.relu, **params_conv)
        # grid_size: one pixel of feature maps represent grid size pixels on the input image
        x = vgg_block(x, 5 * pow(config['grid_size'], 2), 3, 'conv2',
                      activation=None, **params_conv)  # [N, 256, H/8, W/8]

        shuffle_x = tf.depth_to_space(x, config['grid_size'], data_format='NCHW' if cfirst else 'NHWC')

        prob = shuffle_x[:, 0, :, :] if cfirst else shuffle_x[:, :, :, 0]
        # prob = tf.squeeze(prob, axis=cindex)
        prob = tf.sigmoid(prob)
        cls = shuffle_x[:, 1:4, :, :] if cfirst else shuffle_x[:, :, :, 1:4]
        cls = tf.nn.softmax(cls, axis=cindex)
        # cls = cls[:, :-1, :, :] if cfirst else cls[:, :, :, :-1]
        cls = tf.argmax(cls, axis=cindex)  # [N, H, W]
        # c_prob = tf.sigmoid(c_prob)
        ang = shuffle_x[:, 4, :, :] if cfirst else shuffle_x[:, :, :, 4]
        # a_prob = tf.squeeze(a_prob, axis=cindex)  # [N, H, W]
        ang = tf.nn.relu(ang)

    return {'minutiae_raw': x, 'prob': prob, 'classes': cls, 'angles': ang}


def minutiae_loss_will_be_delete(outputs, inputs, **config):
    valid_mask = tf.to_float(inputs['valid_mask'])
    # ====== keypoint loss
    keypoint_map = tf.to_int32(inputs['keypoint_map'])  # [N, H, W]
    prob = outputs['prob']  # [N, H, W]
    p_loss = tf.losses.log_loss(labels=keypoint_map, predictions=prob, weights=valid_mask)

    # ====== classes loss
    classes_map = tf.to_int32(inputs['classes_map'])  # [N, H, W]
    c_prob = outputs['c_prob']  # [N, H, W, 3]
    print(classes_map)
    print(c_prob)
    c_loss = tf.losses.sparse_softmax_cross_entropy(labels=classes_map, logits=c_prob, weights=keypoint_map)
    # ====== angles loss
    angles_map = tf.to_int32(inputs['angles_map'])
    a_prob = outputs['a_prob']
    a_loss = tf.losses.mean_squared_error(labels=angles_map, predictions=a_prob, weights=keypoint_map)

    loss = config['p_loss']*p_loss + config['c_loss']*c_loss + config['a_loss']*a_loss
    # loss = p_loss
    return loss


def minutiae_loss(outputs, inputs, **config):

    cfirst = config['data_format'] == 'channels_first'
    cindex = 1 if cfirst else -1  # index of the channel [N, C, H/8, W/8]
    minutiae_raw = outputs['minutiae_raw']
    shuffle = tf.depth_to_space(minutiae_raw, config['grid_size'], data_format='NCHW' if cfirst else 'NHWC')

    prob = shuffle[:, 0, :, :] if cfirst else shuffle[:, :, :, 0]
    # prob = tf.squeeze(prob, axis=cindex)
    prob = tf.sigmoid(prob)
    cls = shuffle[:, 1:4, :, :] if cfirst else shuffle[:, :, :, 1:4]
    ang = shuffle[:, 4, :, :] if cfirst else shuffle[:, :, :, 4]
    # a_prob = tf.squeeze(a_prob, axis=cindex)  # [N, H, W]
    ang = tf.nn.relu(ang)

    if config['data_format'] == 'channels_first':
        cls = tf.transpose(cls, [0, 2, 3, 1])

    valid_mask = tf.to_float(inputs['valid_mask'])
    # ====== keypoint loss
    keypoint_map = tf.to_int32(inputs['keypoint_map'])  # [N, H, W]
    p_loss = tf.losses.log_loss(labels=keypoint_map, predictions=prob, weights=valid_mask)

    # ====== classes loss
    classes_map = tf.to_int32(inputs['classes_map'])  # [N, H, W]
    c_loss = tf.losses.sparse_softmax_cross_entropy(labels=classes_map, logits=cls, weights=keypoint_map)
    # ====== angles loss
    angles_map = tf.to_int32(inputs['angles_map'])
    a_loss = tf.losses.mean_squared_error(labels=angles_map, predictions=ang, weights=keypoint_map)

    loss = config['p_loss']*p_loss + config['c_loss']*c_loss + config['a_loss']*a_loss
    # loss = p_loss
    return loss
