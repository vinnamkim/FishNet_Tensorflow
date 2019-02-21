from modules import bottleneck, conv2d_fixed_padding, get_filters, batch_norm_relu, initial_conv, concat
from se_block import get_scale, squeeze
import tensorflow as tf


def bottleneck_block(inputs, filters, blocks, strides, training, mode, name, data_format, k, dilation):
    if mode == 'UP':
        outputs = bottleneck(
            inputs, filters, training, data_format,
            strides=1, mode=mode, k=k, dilation=dilation)
    else:
        outputs = bottleneck(
            inputs, filters, training, data_format,
            strides=1, mode='NORM', k=1, dilation=1)

    for _ in range(1, blocks):
        outputs = bottleneck(
            outputs, filters, training, data_format,
            strides=1, mode='NORM', k=1, dilation=dilation)

    return tf.identity(outputs, name)


def up_sample(inputs, data_format):
    if data_format == 'channels_first':
        inputs = tf.transpose(inputs, [0, 2, 3, 1])

    shape = inputs.get_shape().as_list()

    h, w = shape[1:3]

    size = [2 * h, 2 * w]

    outputs = tf.image.resize_nearest_neighbor(inputs, size)

    if data_format == 'channels_first':
        outputs = tf.transpose(outputs, [0, 3, 1, 2])

    return outputs


def down_sample(inputs, data_format):
    return tf.layers.max_pooling2d(
        inputs, pool_size=2, strides=2, padding='same', data_format=data_format)


class FishNet:
    def __init__(self, num_classes, network_planes=None, num_res_blks=None,
                 num_trans_blks=None, data_format='channels_first'):
        if not data_format:
            data_format = (
                'channels_first' if tf.test.is_built_with_cuda() else 'channels_last')

        self.num_classes = num_classes

        self.network_planes = network_planes[1:]
        self.num_trans_blks = num_trans_blks
        self.num_res_blks = num_res_blks

        self.data_format = data_format
        self.inplanes = network_planes[0]

        self.channels = {
            'tail': network_planes[1:4],
            'se': network_planes[4],
            'body': network_planes[5:8],
            'head': network_planes[8:],
        }
        self.res_blocks = {
            'tail': num_res_blks[0:3],
            'se': num_res_blks[3],
            'body': num_res_blks[4:7],
            'head': num_res_blks[7:],
        }
        self.num_trans_blocks = {
            'tail': None,
            'body': num_trans_blks[:3],
            'head': num_trans_blks[3:]
        }

        self.se_ratio = 16

    def _get_structure_info(self, structure):
        channels = self.channels[structure]
        res_blocks = self.res_blocks[structure]
        num_trans_blocks = self.num_trans_blocks[structure]

        if len(channels) != len(res_blocks):
            raise RuntimeError('parameters error')

        if num_trans_blocks is not None:
            if len(channels) != len(num_trans_blocks):
                raise RuntimeError('parameters error')

        return channels, res_blocks, num_trans_blocks

    def _make_res_block(self, inputs, n_stages, filters, training, stride, k):
        outputs = inputs
        for _ in range(n_stages):
            outputs = bottleneck(outputs, filters, training,
                                 self.data_format, stride, k)
        return outputs

    def _make_stage(self, inputs, channel, res_block, k, dilation):
        return NotImplemented

    def _make_tail(self, inputs, training):
        # channels 128 256 512
        # res_blocks 4 8 4
        channels, res_blocks, _ = self._get_structure_info('tail')

        output_blocks = [inputs]

        for idx, params in enumerate(zip(channels, res_blocks)):
            filters, blocks = params

            inputs = bottleneck_block(
                inputs, filters, blocks, 1, training,
                'NORM', 'tail_block_' + str(idx), self.data_format, k=1, dilation=1)

            inputs = down_sample(inputs, self.data_format)

            output_blocks.append(inputs)

        return output_blocks

    def _make_se_score_feats(self, inputs, out_filters, training):
        in_filters = get_filters(inputs, self.data_format)

        inputs = batch_norm_relu(inputs, training, self.data_format)
        inputs = conv2d_fixed_padding(inputs, in_filters // 2,
                                      kernel_size=1, strides=1, data_format=self.data_format)
        inputs = batch_norm_relu(inputs, training, self.data_format)
        inputs = conv2d_fixed_padding(inputs, out_filters,
                                      kernel_size=1, strides=1, data_format=self.data_format)
        return inputs

    def _make_se_block(self, inputs, training):
        in_filters = get_filters(inputs, self.data_format)
        blocks = self.res_blocks['se']
        scale_channels = self.channels['se']
        se_ratio = self.se_ratio

        score_feats = self._make_se_score_feats(
            inputs, in_filters * 2, training)

        scales = get_scale(score_feats, se_ratio,
                           scale_channels, training, self.data_format)

        outputs = bottleneck_block(
            inputs, in_filters, blocks, strides=1,
            training=training, mode='NORM', name=None,
            data_format=self.data_format, k=1, dilation=1) * scales + scales

        return tf.identity(outputs, 'se_block')

    def _make_body(self, inputs, in_transfers, training):
        channels, res_blocks, num_trans_blocks = self._get_structure_info(
            'body')

        output_blocks = [inputs]

        for idx, objects in enumerate(zip(channels, res_blocks, num_trans_blocks, in_transfers)):
            filters, blocks, trans_blocks, in_transfer = objects

            k = get_filters(inputs, self.data_format) // filters
            dilation = 2 ** idx

            inputs = bottleneck_block(
                inputs, filters, blocks, 1, training,
                'UP', 'body_bottleneck_block_' + str(idx), self.data_format, k=k, dilation=dilation)

            inputs = up_sample(inputs, self.data_format)

            trans_filters = get_filters(in_transfer, self.data_format)

            in_transfer = bottleneck_block(
                in_transfer, trans_filters, trans_blocks, 1, training,
                'NORM', 'body_in_transfer_' + str(idx), self.data_format, k=1, dilation=1)

            if self.data_format == 'channels_first':
                inputs = tf.concat(
                    values=[inputs, in_transfer], axis=1, name='body_block_' + str(idx))
            else:
                inputs = tf.concat(
                    values=[inputs, in_transfer], axis=3, name='body_block_' + str(idx))

            output_blocks.append(inputs)

        return output_blocks

    def _make_head(self, inputs, in_transfers, training):
        channels, res_blocks, num_trans_blocks = self._get_structure_info(
            'head')

        output_blocks = [inputs]

        for idx, objects in enumerate(zip(channels, res_blocks, num_trans_blocks, in_transfers)):
            filters, blocks, trans_blocks, in_transfer = objects

            inputs = bottleneck_block(
                inputs, filters, blocks, 1, training,
                'NORM', 'body_bottleneck_block_' + str(idx), self.data_format, k=1, dilation=1)

            inputs = down_sample(inputs, self.data_format)

            trans_filters = get_filters(in_transfer, self.data_format)

            in_transfer = bottleneck_block(
                in_transfer, trans_filters, trans_blocks, 1, training,
                'NORM', 'body_in_transfer_' + str(idx), self.data_format, k=1, dilation=1)

            if self.data_format == 'channels_first':
                inputs = tf.concat(
                    values=[inputs, in_transfer], axis=1, name='body_block_' + str(idx))
            else:
                inputs = tf.concat(
                    values=[inputs, in_transfer], axis=3, name='body_block_' + str(idx))

            output_blocks.append(inputs)

        return output_blocks

    def _make_score_feats(self, inputs, training):
        in_filters = get_filters(inputs, self.data_format)

        inputs = batch_norm_relu(inputs, training, self.data_format)
        inputs = conv2d_fixed_padding(inputs, in_filters // 2,
                                      kernel_size=1, strides=1, data_format=self.data_format)
        inputs = batch_norm_relu(inputs, training, self.data_format)

        inputs = squeeze(inputs, self.data_format)

        inputs = tf.layers.dense(inputs, units=self.num_classes, use_bias=True)

        return tf.identity(inputs, 'final_dense')

    def __call__(self, inputs, training):
        if self.data_format == 'channels_first':
            inputs = tf.transpose(inputs, [0, 3, 1, 2])
        
        with tf.name_scope("tail"):
            before_tail = initial_conv(
                inputs, self.inplanes, training, self.data_format)

            tail_outputs = self._make_tail(before_tail, training)

            after_tail, in_transfers = tail_outputs[-1], reversed(tail_outputs[:3])

            se_output = self._make_se_block(after_tail, training)

        with tf.name_scope("body"):
            body_outputs = self._make_body(se_output, in_transfers, training)

            after_body, in_transfers = body_outputs[-1], reversed(body_outputs[:3])

        with tf.name_scope("head"):
            head_outputs = self._make_head(after_body, in_transfers, training)

            after_head = head_outputs[-1]

        with tf.name_scope("logits"):
            outputs = self._make_score_feats(after_head, training)

        return outputs
