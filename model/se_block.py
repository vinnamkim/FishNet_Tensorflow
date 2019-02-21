import tensorflow as tf
from modules import get_filters, batch_norm_relu

def squeeze(inputs, data_format):
    if data_format == 'channels_first':
        _, _, h, w = inputs.get_shape().as_list()    
    else:
        _, h, w, _ = inputs.get_shape().as_list()

    outputs = tf.layers.average_pooling2d(inputs, [h, w], strides=1, data_format=data_format)
    
    if data_format == 'channels_first':
        outputs = tf.squeeze(outputs, axis=[2, 3])
    else:
        outputs = tf.squeeze(outputs, axis=[1, 2])
    
    return outputs

def get_scale(inputs, ratio, channels, training, data_format):
    inputs = batch_norm_relu(inputs, training, data_format)
    squeezed = squeeze(inputs, data_format)

    scale = tf.layers.dense(squeezed, units=channels // ratio, activation=tf.nn.relu)
    scale = tf.layers.dense(scale, units=channels, activation=tf.nn.sigmoid)

    if data_format == 'channels_first':
        scale = tf.expand_dims(scale, axis=-1)
        scale = tf.expand_dims(scale, axis=-1)
    else:
        scale = tf.expand_dims(scale, axis=1)
        scale = tf.expand_dims(scale, axis=1)

    return scale
