import tensorflow as tf
import load_flags
from model.fishnet import FishNet

if __name__ == '__main__':
    inputs = tf.placeholder(
        tf.float32,
        [4, 3, 224, 224],
        'inputs'
    )
    data_format = 'channels_first'

    #outputs = squeeze(inputs, data_format)

    network_planes = [64, 128, 256, 512, 512, 512, 384, 256, 320, 832, 1600]
    num_res_blks = [2, 2, 6, 2, 1, 1, 1, 1, 2, 2]
    num_trans_blks = [1, 1, 1, 1, 1, 4]
    num_classes = 1000

    model = FishNet(num_classes, network_planes, num_res_blks, num_trans_blks)
    outputs = model(inputs, training=True)

    print 'end'