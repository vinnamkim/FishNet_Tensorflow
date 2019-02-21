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
"""Contains utility and supporting functions for FishNet.

  This module contains FishNet code which does not directly build layers. This
includes dataset management, hyperparameter and optimizer code, and argument
parsing. Code for defining the FishNet layers can be found in model/fishnet.py.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import math
import multiprocessing
import os

# pylint: disable=g-bad-import-order
from absl import flags
import tensorflow as tf

from official.resnet import resnet_model
from official.utils.flags import core as flags_core
from official.utils.export import export
from official.utils.logs import hooks_helper
from official.utils.logs import logger
from official.utils.misc import distribution_utils
from official.utils.misc import model_helpers
from official.resnet.resnet_run_loop import override_flags_and_set_envars_for_gpu_thread_pool, image_bytes_serving_input_fn
from model.fishnet import FishNet

def fishnet_model_fn(features, labels, mode, num_classes, network_planes, num_res_blks,
                     num_trans_blks, weight_decay, learning_rate_fn, momentum,
                     data_format, loss_scale, loss_filter_fn=None, dtype=resnet_model.DEFAULT_DTYPE,
                     fine_tune=False):
    """Shared functionality for different fishnet model_fns.

    Initializes the FishNetModel representing the model layers
    and uses that model to build the necessary EstimatorSpecs for
    the `mode` in question. For training, this means building losses,
    the optimizer, and the train op that get passed into the EstimatorSpec.
    For evaluation and prediction, the EstimatorSpec is returned without
    a train op, but with the necessary parameters for the given mode.

    Args:
      features: tensor representing input images
      labels: tensor representing class labels for all input images
      mode: current estimator mode; should be one of
        `tf.estimator.ModeKeys.TRAIN`, `EVALUATE`, `PREDICT`
      num_classes:
      network_planes:
      num_res_blks:
      num_trans_blks:
      weight_decay: weight decay loss rate used to regularize learned variables.
      learning_rate_fn: function that returns the current learning rate given
        the current global_step
      momentum: momentum term used for optimization
      data_format: Input format ('channels_last', 'channels_first', or None).
        If set to None, the format is dependent on whether a GPU is available.
      resnet_version: Integer representing which version of the ResNet network to
        use. See README for details. Valid values: [1, 2]
      loss_scale: The factor to scale the loss for numerical stability. A detailed
        summary is present in the arg parser help text.
      loss_filter_fn: function that takes a string variable name and returns
        True if the var should be included in loss calculation, and False
        otherwise. If None, batch_normalization variables will be excluded
        from the loss.
      dtype: the TensorFlow dtype to use for calculations.
      fine_tune: If True only train the dense layers(final layers).

    Returns:
      EstimatorSpec parameterized according to the input params and the
      current mode.
    """

    # Generate a summary node for the images
    tf.summary.image('images', features, max_outputs=6)
    # Checks that features/images have same data type being used for calculations.
    assert features.dtype == dtype

    model = FishNet(num_classes, network_planes,
                    num_res_blks, num_trans_blks, data_format)

    logits = model(features, mode == tf.estimator.ModeKeys.TRAIN)

    # This acts as a no-op if the logits are already in fp32 (provided logits are
    # not a SparseTensor). If dtype is is low precision, logits must be cast to
    # fp32 for numerical stability.
    logits = tf.cast(logits, tf.float32)

    predictions = {
        'classes': tf.argmax(logits, axis=1),
        'probabilities': tf.nn.softmax(logits, name='softmax_tensor')
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        # Return the predictions and the specification for serving a SavedModel
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions,
            export_outputs={
                'predict': tf.estimator.export.PredictOutput(predictions)
            })

    # Calculate loss, which includes softmax cross entropy and L2 regularization.
    cross_entropy = tf.losses.sparse_softmax_cross_entropy(
        logits=logits, labels=labels)

    # Create a tensor named cross_entropy for logging purposes.
    tf.identity(cross_entropy, name='cross_entropy')
    tf.summary.scalar('cross_entropy', cross_entropy)

    # If no loss_filter_fn is passed, assume we want the default behavior,
    # which is that batch_normalization variables are excluded from loss.
    def exclude_batch_norm(name):
        return 'batch_normalization' not in name
    loss_filter_fn = loss_filter_fn or exclude_batch_norm

    # Add weight decay to the loss.
    with tf.name_scope("weight_decay"):
        l2_loss = weight_decay * tf.add_n(
            # loss is computed using fp32 for numerical stability.
            [tf.nn.l2_loss(tf.cast(v, tf.float32)) for v in tf.trainable_variables()
            if loss_filter_fn(v.name)])
        tf.summary.scalar('l2_loss', l2_loss)
    loss = cross_entropy + l2_loss

    if mode == tf.estimator.ModeKeys.TRAIN:
        global_step = tf.train.get_or_create_global_step()

        learning_rate = learning_rate_fn(global_step)

        # Create a tensor named learning_rate for logging purposes
        tf.identity(learning_rate, name='learning_rate')
        tf.summary.scalar('learning_rate', learning_rate)

        optimizer = tf.train.MomentumOptimizer(
            learning_rate=learning_rate,
            momentum=momentum
        )

        def _dense_grad_filter(gvs):
            """Only apply gradient updates to the final layer.

            This function is used for fine tuning.

            Args:
              gvs: list of tuples with gradients and variable info
            Returns:
              filtered gradients so that only the dense layer remains
            """
            return [(g, v) for g, v in gvs if 'dense' in v.name]

        if loss_scale != 1:
            # When computing fp16 gradients, often intermediate tensor values are
            # so small, they underflow to 0. To avoid this, we multiply the loss by
            # loss_scale to make these tensor values loss_scale times bigger.
            scaled_grad_vars = optimizer.compute_gradients(loss * loss_scale)

            if fine_tune:
                scaled_grad_vars = _dense_grad_filter(scaled_grad_vars)

            # Once the gradient computation is complete we can scale the gradients
            # back to the correct scale before passing them to the optimizer.
            unscaled_grad_vars = [(grad / loss_scale, var)
                                  for grad, var in scaled_grad_vars]
            minimize_op = optimizer.apply_gradients(
                unscaled_grad_vars, global_step)
        else:
            grad_vars = optimizer.compute_gradients(loss)
            if fine_tune:
                grad_vars = _dense_grad_filter(grad_vars)
            minimize_op = optimizer.apply_gradients(grad_vars, global_step)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        train_op = tf.group(minimize_op, update_ops)
    else:
        train_op = None

    accuracy = tf.metrics.accuracy(labels, predictions['classes'])
    accuracy_top_5 = tf.metrics.mean(tf.nn.in_top_k(predictions=logits,
                                                    targets=labels,
                                                    k=5,
                                                    name='top_5_op'))
    metrics = {'accuracy': accuracy,
               'accuracy_top_5': accuracy_top_5}

    # Create a tensor named train_accuracy for logging purposes
    tf.identity(accuracy[1], name='train_accuracy')
    tf.identity(accuracy_top_5[1], name='train_accuracy_top_5')
    tf.summary.scalar('train_accuracy', accuracy[1])
    tf.summary.scalar('train_accuracy_top_5', accuracy_top_5[1])

    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions,
        loss=loss,
        train_op=train_op,
        eval_metric_ops=metrics)


def fishnet_main(flags_obj, model_function, input_function, dataset_name, shape=None):
    """Shared main loop for ResNet Models.

    Args:
      flags_obj: An object containing parsed flags. See define_resnet_flags()
        for details.
      model_function: the function that instantiates the Model and builds the
        ops for train/eval. This will be passed directly into the estimator.
      input_function: the function that processes the dataset and returns a
        dataset that the estimator can train on. This will be wrapped with
        all the relevant flags for running and passed to estimator.
      dataset_name: the name of the dataset for training and evaluation. This is
        used for logging purpose.
      shape: list of ints representing the shape of the images used for training.
        This is only used if flags_obj.export_dir is passed.
    """

    model_helpers.apply_clean(flags.FLAGS)

    # Ensures flag override logic is only executed if explicitly triggered.
    if flags_obj.tf_gpu_thread_mode:
        override_flags_and_set_envars_for_gpu_thread_pool(flags_obj)

    # Creates session config. allow_soft_placement = True, is required for
    # multi-GPU and is not harmful for other modes.
    session_config = tf.ConfigProto(
        inter_op_parallelism_threads=flags_obj.inter_op_parallelism_threads,
        intra_op_parallelism_threads=flags_obj.intra_op_parallelism_threads,
        allow_soft_placement=True)

    distribution_strategy = distribution_utils.get_distribution_strategy(
        flags_core.get_num_gpus(flags_obj), flags_obj.all_reduce_alg)

    # Creates a `RunConfig` that checkpoints every 24 hours which essentially
    # results in checkpoints determined only by `epochs_between_evals`.
    run_config = tf.estimator.RunConfig(
        train_distribute=distribution_strategy,
        session_config=session_config,
        save_checkpoints_secs=60*60*24)

    # Initializes model with all but the dense layer from pretrained ResNet.
    if flags_obj.pretrained_model_checkpoint_path is not None:
        warm_start_settings = tf.estimator.WarmStartSettings(
            flags_obj.pretrained_model_checkpoint_path,
            vars_to_warm_start='^(?!.*dense)')
    else:
        warm_start_settings = None

    classifier = tf.estimator.Estimator(
        model_fn=model_function, model_dir=flags_obj.model_dir, config=run_config,
        warm_start_from=warm_start_settings, params={
            'data_format': flags_obj.data_format,
            'batch_size': flags_obj.batch_size,
            'fishnet_size': int(flags_obj.fishnet_size),
            'loss_scale': flags_core.get_loss_scale(flags_obj),
            'dtype': flags_core.get_tf_dtype(flags_obj),
            'fine_tune': flags_obj.fine_tune
        })

    run_params = {
        'batch_size': flags_obj.batch_size,
        'dtype': flags_core.get_tf_dtype(flags_obj),
        'fishnet_size': flags_obj.fishnet_size,
        'synthetic_data': flags_obj.use_synthetic_data,
        'train_epochs': flags_obj.train_epochs,
    }
    if flags_obj.use_synthetic_data:
        dataset_name = dataset_name + '-synthetic'

    benchmark_logger = logger.get_benchmark_logger()
    benchmark_logger.log_run_info('fishnet', dataset_name, run_params,
                                  test_id=flags_obj.benchmark_test_id)

    train_hooks = hooks_helper.get_train_hooks(
        flags_obj.hooks,
        model_dir=flags_obj.model_dir,
        batch_size=flags_obj.batch_size)

    def input_fn_train(num_epochs):
        return input_function(
            is_training=True,
            data_dir=flags_obj.data_dir,
            batch_size=distribution_utils.per_device_batch_size(
                flags_obj.batch_size, flags_core.get_num_gpus(flags_obj)),
            num_epochs=num_epochs,
            dtype=flags_core.get_tf_dtype(flags_obj),
            datasets_num_private_threads=flags_obj.datasets_num_private_threads,
            num_parallel_batches=flags_obj.datasets_num_parallel_batches)

    def input_fn_eval():
        return input_function(
            is_training=False,
            data_dir=flags_obj.data_dir,
            batch_size=distribution_utils.per_device_batch_size(
                flags_obj.batch_size, flags_core.get_num_gpus(flags_obj)),
            num_epochs=1,
            dtype=flags_core.get_tf_dtype(flags_obj))

    if flags_obj.eval_only or not flags_obj.train_epochs:
        # If --eval_only is set, perform a single loop with zero train epochs.
        schedule, n_loops = [0], 1
    else:
        # Compute the number of times to loop while training. All but the last
        # pass will train for `epochs_between_evals` epochs, while the last will
        # train for the number needed to reach `training_epochs`. For instance if
        #   train_epochs = 25 and epochs_between_evals = 10
        # schedule will be set to [10, 10, 5]. That is to say, the loop will:
        #   Train for 10 epochs and then evaluate.
        #   Train for another 10 epochs and then evaluate.
        #   Train for a final 5 epochs (to reach 25 epochs) and then evaluate.
        n_loops = math.ceil(flags_obj.train_epochs /
                            flags_obj.epochs_between_evals)
        schedule = [
            flags_obj.epochs_between_evals for _ in range(int(n_loops))]
        # over counting.
        schedule[-1] = flags_obj.train_epochs - sum(schedule[:-1])

    for cycle_index, num_train_epochs in enumerate(schedule):
        tf.logging.info('Starting cycle: %d/%d', cycle_index, int(n_loops))

        if num_train_epochs:
            classifier.train(input_fn=lambda: input_fn_train(num_train_epochs),
                             hooks=train_hooks, max_steps=flags_obj.max_train_steps)

        tf.logging.info('Starting to evaluate.')

        # flags_obj.max_train_steps is generally associated with testing and
        # profiling. As a result it is frequently called with synthetic data, which
        # will iterate forever. Passing steps=flags_obj.max_train_steps allows the
        # eval (which is generally unimportant in those circumstances) to terminate.
        # Note that eval will run for max_train_steps each loop, regardless of the
        # global_step count.
        eval_results = classifier.evaluate(input_fn=input_fn_eval,
                                           steps=flags_obj.max_train_steps)

        benchmark_logger.log_evaluation_result(eval_results)

        if model_helpers.past_stop_threshold(
                flags_obj.stop_threshold, eval_results['accuracy']):
            break

    if flags_obj.export_dir is not None:
        # Exports a saved model for the given classifier.
        export_dtype = flags_core.get_tf_dtype(flags_obj)
        if flags_obj.image_bytes_as_serving_input:
            input_receiver_fn = functools.partial(
                image_bytes_serving_input_fn, shape, dtype=export_dtype)
        else:
            input_receiver_fn = export.build_tensor_serving_input_receiver_fn(
                shape, batch_size=flags_obj.batch_size, dtype=export_dtype)
        classifier.export_savedmodel(flags_obj.export_dir, input_receiver_fn,
                                     strip_default_attrs=True)
