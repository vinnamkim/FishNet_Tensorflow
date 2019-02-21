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
"""Runs a FishNet model on the ImageNet dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from absl import app as absl_app
from absl import flags
import tensorflow as tf  # pylint: disable=g-bad-import-order

from official.utils.flags import core as flags_core
from official.utils.logs import logger
from official.resnet import imagenet_preprocessing
from official.resnet import resnet_run_loop
import fishnet_run_loop

_DEFAULT_IMAGE_SIZE = 224
_NUM_CHANNELS = 3
_NUM_CLASSES = 1001

_NUM_IMAGES = {
    'train': 1281167,
    'validation': 50000,
}

_NUM_TRAIN_FILES = 1024
_SHUFFLE_BUFFER = 10000

DATASET_NAME = 'ImageNet'


def define_fishnet_flags():
    """Add flags and validators for ResNet."""
    flags_core.define_base()
    flags_core.define_performance(num_parallel_calls=False,
                                  tf_gpu_thread_mode=True,
                                  datasets_num_private_threads=True,
                                  datasets_num_parallel_batches=True)
    flags_core.define_image()
    flags_core.define_benchmark()
    flags.adopt_module_key_flags(flags_core)

    flags.DEFINE_bool(
        name='fine_tune', short_name='ft', default=False,
        help=flags_core.help_wrap(
            'If True do not train any parameters except for the final layer.'))
    flags.DEFINE_string(
        name='pretrained_model_checkpoint_path', short_name='pmcp', default=None,
        help=flags_core.help_wrap(
            'If not None initialize all the network except the final layer with '
            'these values'))
    flags.DEFINE_boolean(
        name='eval_only', default=False,
        help=flags_core.help_wrap('Skip training and only perform evaluation on '
                                  'the latest checkpoint.'))
    flags.DEFINE_boolean(
        name='image_bytes_as_serving_input', default=False,
        help=flags_core.help_wrap(
            'If True exports savedmodel with serving signature that accepts '
            'JPEG image bytes instead of a fixed size [HxWxC] tensor that '
            'represents the image. The former is easier to use for serving at '
            'the expense of image resize/cropping being done as part of model '
            'inference. Note, this flag only applies to ImageNet and cannot '
            'be used for CIFAR.'))

    choice_kwargs = dict(
        name='fishnet_size', short_name='rs', default='99',
        help=flags_core.help_wrap('The size of the FishNet model to use.'))

    fishnet_size_choices = ['99', '150']
    flags.DEFINE_enum(enum_values=fishnet_size_choices, **choice_kwargs)
    flags.adopt_module_key_flags(fishnet_run_loop)
    flags_core.set_defaults(train_epochs=90)

###############################################################################
# Data processing
###############################################################################


def get_filenames(is_training, data_dir):
    """Return filenames for dataset."""
    if is_training:
        return [
            os.path.join(data_dir, 'train-%05d-of-01024' % i)
            for i in range(_NUM_TRAIN_FILES)]
    else:
        return [
            os.path.join(data_dir, 'validation-%05d-of-00128' % i)
            for i in range(128)]


def _parse_example_proto(example_serialized):
    """Parses an Example proto containing a training example of an image.

    The output of the build_image_data.py image preprocessing script is a dataset
    containing serialized Example protocol buffers. Each Example proto contains
    the following fields (values are included as examples):

      image/height: 462
      image/width: 581
      image/colorspace: 'RGB'
      image/channels: 3
      image/class/label: 615
      image/class/synset: 'n03623198'
      image/class/text: 'knee pad'
      image/object/bbox/xmin: 0.1
      image/object/bbox/xmax: 0.9
      image/object/bbox/ymin: 0.2
      image/object/bbox/ymax: 0.6
      image/object/bbox/label: 615
      image/format: 'JPEG'
      image/filename: 'ILSVRC2012_val_00041207.JPEG'
      image/encoded: <JPEG encoded string>

    Args:
      example_serialized: scalar Tensor tf.string containing a serialized
        Example protocol buffer.

    Returns:
      image_buffer: Tensor tf.string containing the contents of a JPEG file.
      label: Tensor tf.int32 containing the label.
      bbox: 3-D float Tensor of bounding boxes arranged [1, num_boxes, coords]
        where each coordinate is [0, 1) and the coordinates are arranged as
        [ymin, xmin, ymax, xmax].
    """
    # Dense features in Example proto.
    feature_map = {
        'image/encoded': tf.FixedLenFeature([], dtype=tf.string,
                                            default_value=''),
        'image/class/label': tf.FixedLenFeature([], dtype=tf.int64,
                                                default_value=-1),
        'image/class/text': tf.FixedLenFeature([], dtype=tf.string,
                                               default_value=''),
    }
    sparse_float32 = tf.VarLenFeature(dtype=tf.float32)
    # Sparse features in Example proto.
    feature_map.update(
        {k: sparse_float32 for k in ['image/object/bbox/xmin',
                                     'image/object/bbox/ymin',
                                     'image/object/bbox/xmax',
                                     'image/object/bbox/ymax']})

    features = tf.parse_single_example(example_serialized, feature_map)
    label = tf.cast(features['image/class/label'], dtype=tf.int32)

    xmin = tf.expand_dims(features['image/object/bbox/xmin'].values, 0)
    ymin = tf.expand_dims(features['image/object/bbox/ymin'].values, 0)
    xmax = tf.expand_dims(features['image/object/bbox/xmax'].values, 0)
    ymax = tf.expand_dims(features['image/object/bbox/ymax'].values, 0)

    # Note that we impose an ordering of (y, x) just to make life difficult.
    bbox = tf.concat([ymin, xmin, ymax, xmax], 0)

    # Force the variable number of bounding boxes into the shape
    # [1, num_boxes, coords].
    bbox = tf.expand_dims(bbox, 0)
    bbox = tf.transpose(bbox, [0, 2, 1])

    return features['image/encoded'], label, bbox


def parse_record(raw_record, is_training, dtype):
    """Parses a record containing a training example of an image.

    The input record is parsed into a label and image, and the image is passed
    through preprocessing steps (cropping, flipping, and so on).

    Args:
      raw_record: scalar Tensor tf.string containing a serialized
        Example protocol buffer.
      is_training: A boolean denoting whether the input is for training.
      dtype: data type to use for images/features.

    Returns:
      Tuple with processed image tensor and one-hot-encoded label tensor.
    """
    image_buffer, label, bbox = _parse_example_proto(raw_record)

    image = imagenet_preprocessing.preprocess_image(
        image_buffer=image_buffer,
        bbox=bbox,
        output_height=_DEFAULT_IMAGE_SIZE,
        output_width=_DEFAULT_IMAGE_SIZE,
        num_channels=_NUM_CHANNELS,
        is_training=is_training)
    image = tf.cast(image, dtype)

    return image, label


def input_fn(is_training, data_dir, batch_size, num_epochs=1,
             dtype=tf.float32, datasets_num_private_threads=None,
             num_parallel_batches=1):
    """Input function which provides batches for train or eval.

    Args:
      is_training: A boolean denoting whether the input is for training.
      data_dir: The directory containing the input data.
      batch_size: The number of samples per batch.
      num_epochs: The number of epochs to repeat the dataset.
      dtype: Data type to use for images/features
      datasets_num_private_threads: Number of private threads for tf.data.
      num_parallel_batches: Number of parallel batches for tf.data.

    Returns:
      A dataset that can be used for iteration.
    """
    filenames = get_filenames(is_training, data_dir)
    dataset = tf.data.Dataset.from_tensor_slices(filenames)

    if is_training:
        # Shuffle the input files
        dataset = dataset.shuffle(buffer_size=_NUM_TRAIN_FILES)

    # Convert to individual records.
    # cycle_length = 10 means 10 files will be read and deserialized in parallel.
    # This number is low enough to not cause too much contention on small systems
    # but high enough to provide the benefits of parallelization. You may want
    # to increase this number if you have a large number of CPU cores.
    dataset = dataset.apply(tf.contrib.data.parallel_interleave(
        tf.data.TFRecordDataset, cycle_length=10))

    return resnet_run_loop.process_record_dataset(
        dataset=dataset,
        is_training=is_training,
        batch_size=batch_size,
        shuffle_buffer=_SHUFFLE_BUFFER,
        parse_record_fn=parse_record,
        num_epochs=num_epochs,
        dtype=dtype,
        datasets_num_private_threads=datasets_num_private_threads,
        num_parallel_batches=num_parallel_batches
    )


def get_synth_input_fn(dtype):
    return resnet_run_loop.get_synth_input_fn(
        _DEFAULT_IMAGE_SIZE, _DEFAULT_IMAGE_SIZE, _NUM_CHANNELS, _NUM_CLASSES,
        dtype=dtype)


def _get_network_params(fishnet_size):
    choices = {
        150: {
            #  input size:   [224, 56, 28,  14 | 7,   14,  28 | 56,   28,  14]
            # output size:   [56,  28, 14,   7 | 14,  28,  56 | 28,   14,   7]
            #                  |    |    |   |    |    |    |    |     |    |
            'network_planes': [64, 128, 256, 512, 512, 512, 384, 256, 320, 832, 1600],
            'num_res_blks': [2, 4, 8, 4, 2, 2, 2, 2, 2, 4],
            'num_trans_blks': [2, 2, 2, 2, 2, 4],
        },
        99: {
            #  input size:   [224, 56, 28,  14 | 7,   14,  28 | 56,   28,  14]
            # output size:   [56,  28, 14,   7 | 14,  28,  56 | 28,   14,   7]
            #                  |    |    |   |    |    |    |    |     |    |
            'network_planes': [64, 128, 256, 512, 512, 512, 384, 256, 320, 832, 1600],
            'num_res_blks': [2, 2, 6, 2, 1, 1, 1, 1, 2, 2],
            'num_trans_blks': [1, 1, 1, 1, 1, 4],
        }
    }

    try:
        return choices[fishnet_size]
    except KeyError:
        err = ('Could not find layers for selected FishNet size.\n'
               'Size received: {}; sizes allowed: {}.'.format(
                   fishnet_size, choices.keys()))
        raise ValueError(err)


def imagenet_model_fn(features, labels, mode, params):
    """Our model_fn for ResNet to be used with our Estimator."""

    # Warmup and higher lr may not be valid for fine tuning with small batches
    # and smaller numbers of training images.
    if params['fine_tune']:
        warmup = False
        base_lr = .1
    else:
        warmup = True
        base_lr = .128

    learning_rate_fn = resnet_run_loop.learning_rate_with_decay(
        batch_size=params['batch_size'], batch_denom=256,
        num_images=_NUM_IMAGES['train'], boundary_epochs=[30, 60, 80, 90],
        decay_rates=[1, 0.1, 0.01, 0.001, 1e-4], warmup=warmup, base_lr=base_lr)
    
    network_params = _get_network_params(params['fishnet_size'])

    return fishnet_run_loop.fishnet_model_fn(
        features=features,
        labels=labels,
        mode=mode,
        num_classes=_NUM_CLASSES,
        network_planes=network_params['network_planes'],
        num_res_blks=network_params['num_res_blks'],
        num_trans_blks=network_params['num_trans_blks'],
        weight_decay=1e-4,
        learning_rate_fn=learning_rate_fn,
        momentum=0.9,
        data_format=params['data_format'],
        loss_scale=params['loss_scale'],
        loss_filter_fn=None,
        fine_tune=params['fine_tune']
    )


def run_fishnet(flags_obj):
    """Run FishNet ImageNet training and eval loop.

    Args:
      flags_obj: An object containing parsed flag values.
    """
    input_function = (flags_obj.use_synthetic_data and
                      get_synth_input_fn(flags_core.get_tf_dtype(flags_obj)) or
                      input_fn)

    fishnet_run_loop.fishnet_main(
        flags_obj, imagenet_model_fn, input_function, DATASET_NAME,
        shape=[_DEFAULT_IMAGE_SIZE, _DEFAULT_IMAGE_SIZE, _NUM_CHANNELS])


def main(_):
    with logger.benchmark_context(flags.FLAGS):
        run_fishnet(flags.FLAGS)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    define_fishnet_flags()
    absl_app.run(main)
