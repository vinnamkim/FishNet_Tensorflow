# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

_BATCH_NORM_DECAY = 0.997
_BATCH_NORM_EPSILON = 1e-5
_BATCH_NORM_FUSED = True


def concat(values, data_format):
    if data_format == 'channels_first':
        return tf.concat(values, axis=1)
    else:
        return tf.concat(values, axis=-1)


def get_filters(inputs, data_format):
    shape = inputs.get_shape().as_list()

    if data_format == 'channels_first':
        return shape[1]
    else:
        return shape[-1]

def batch_norm_relu(inputs, training, data_format):
    outputs = tf.layers.batch_normalization(
        inputs=inputs, axis=1 if data_format == 'channels_first' else 3,
        momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, training=training,
        fused=_BATCH_NORM_FUSED
    )
    outputs = tf.nn.relu(outputs)

    return outputs


def _fixed_padding(inputs, kernel_size, data_format):
    """Pads the input along the spatial dimensions independently of input size.

    Args:
      inputs: A tensor of size [batch, channels, height_in, width_in] or
        [batch, height_in, width_in, channels] depending on data_format.
      kernel_size: The kernel to be used in the conv2d or max_pool2d operation.
                   Should be a positive integer.
      data_format: The input format ('channels_last' or 'channels_first').

    Returns:
      A tensor with the same format as the input with the data either intact
      (if kernel_size == 1) or padded (if kernel_size > 1).
    """
    pad_total = kernel_size - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg

    if data_format == 'channels_first':
        padded_inputs = tf.pad(inputs, [[0, 0], [0, 0],
                                        [pad_beg, pad_end], [pad_beg, pad_end]])
    else:
        padded_inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end],
                                        [pad_beg, pad_end], [0, 0]])
    return padded_inputs


def conv2d_fixed_padding(inputs, filters, kernel_size, strides, data_format):
    """Strided 2-D convolution with explicit padding."""
    # The padding is consistent and is based only on `kernel_size`, not on the
    # dimensions of `inputs` (as opposed to using `tf.layers.conv2d` alone).
    if strides > 1:
        inputs = _fixed_padding(inputs, kernel_size, data_format)

    return tf.layers.conv2d(
        inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides,
        padding=('SAME' if strides == 1 else 'VALID'), use_bias=False,
        kernel_initializer=tf.variance_scaling_initializer(),
        data_format=data_format)

def _squeeze_idt(inputs, k, data_format):
    shape = inputs.get_shape()

    if data_format == 'channels_first':
        _, c, h, w = shape

        outputs = tf.reduce_sum(
            tf.reshape(inputs, [-1, c // k, k, h, w]),
            axis=2
        )
    else:
        _, h, w, c = shape

        outputs = tf.reduce_sum(
            tf.reshape(inputs, [-1, h, w, c // k, k]),
            axis=4
        )

    return outputs

def bottleneck(inputs, filters, training, data_format, strides=1, mode='NORM', k=1, dilation=1):
    in_filters = get_filters(inputs, data_format)

    residuals = inputs

    outputs = batch_norm_relu(inputs, training, data_format)
    outputs = conv2d_fixed_padding(inputs, filters // 4, 1, 1, data_format)

    outputs = batch_norm_relu(outputs, training, data_format)
    outputs = conv2d_fixed_padding(
        inputs, filters // 4, 3, strides, data_format)

    outputs = batch_norm_relu(outputs, training, data_format)
    outputs = conv2d_fixed_padding(inputs, filters, 1, 1, data_format)

    if mode == 'UP':
        residuals = _squeeze_idt(inputs, k, data_format)
    elif in_filters != filters or strides > 1:
        residuals = batch_norm_relu(residuals, training, data_format)
        residuals = conv2d_fixed_padding(residuals, filters, 1, strides, data_format)

    outputs = tf.add_n([outputs, residuals])

    return outputs


def initial_conv(inputs, filters, training, data_format):
    outputs = conv2d_fixed_padding(inputs, filters // 2,
                                   kernel_size=3, strides=2, data_format=data_format)
    outputs = batch_norm_relu(outputs, training, data_format)

    outputs = conv2d_fixed_padding(outputs, filters // 2,
                                   kernel_size=3, strides=1, data_format=data_format)
    outputs = batch_norm_relu(outputs, training, data_format)

    outputs = conv2d_fixed_padding(outputs, filters,
                                   kernel_size=3, strides=1, data_format=data_format)
    outputs = batch_norm_relu(outputs, training, data_format)

    outputs = tf.layers.max_pooling2d(outputs, pool_size=3, strides=2,
                                      padding='same', data_format=data_format)
    return outputs

