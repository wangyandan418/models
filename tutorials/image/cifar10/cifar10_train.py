# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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

"""A binary to train CIFAR-10 using a single GPU.

Accuracy:
cifar10_train.py achieves ~86% accuracy after 100K steps (256 epochs of
data) as judged by cifar10_eval.py.

Speed: With batch_size 128.

System        | Step Time (sec/batch)  |     Accuracy
------------------------------------------------------------------
1 Tesla K20m  | 0.35-0.60              | ~86% at 60K steps  (5 hours)
1 Tesla K40m  | 0.25-0.35              | ~86% at 100K steps (4 hours)

Usage:
Please see the tutorial and website for how to download the CIFAR-10
data set, compile the program and train the model.

http://tensorflow.org/tutorials/deep_cnn/
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import time
import re

import tensorflow as tf

import cifar10

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', '/tmp/cifar10_train',
                           """Directory where to write event logs """
                           """and checkpoint.""")
# tf.app.flags.DEFINE_integer('max_steps', 1000000,
#                             """Number of batches to run.""")
tf.app.flags.DEFINE_integer('max_steps', 10,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
tf.app.flags.DEFINE_integer('log_frequency', 1,
                            """How often to log results to the console.""")


def train():
  """Train CIFAR-10 for a number of steps."""
  with tf.Graph().as_default():
    global_step = tf.train.get_or_create_global_step()

    # Get images and labels for CIFAR-10.
    # Force input pipeline to CPU:0 to avoid operations sometimes ending up on
    # GPU and resulting in a slow down.
    with tf.device('/cpu:0'):
      images, labels = cifar10.distorted_inputs()

    # Build a Graph that computes the logits predictions from the
    # inference model.
    logits = cifar10.inference(images)

    # Calculate loss.
    loss = cifar10.loss(logits, labels)

    n = tf.constant(2.5)
    conv1_std_co = tf.constant(0.326984)
    conv2_std_co = tf.constant(0.099911)
    local3_std_co = tf.constant(0.010653)
    local4_std_co = tf.constant(0.015261)
    softmax_linear_std_co = tf.constant(0.222937)

    conv1_quan = tf.multiply(n, conv1_std_co)
    conv2_quan = tf.multiply(n, conv2_std_co)
    local3_quan = tf.multiply(n, local3_std_co)
    local4_quan = tf.multiply(n, local4_std_co)
    softmax_linear_quan = tf.multiply(n, softmax_linear_std_co)

    f1_conv1 = tf.sign(conv1_weights + conv1_quan) * (conv1_weights + conv1_quan)
    f2_conv1 = tf.sign(conv1_weights) * conv1_weights
    f3_conv1 = tf.sign(conv1_weights - conv1_quan) * (conv1_weights - conv1_quan)

    f1_conv2 = tf.sign(conv2_weights + conv2_quan) * (conv2_weights + conv2_quan)
    f2_conv2 = tf.sign(conv2_weights) * conv2_weights
    f3_conv2 = tf.sign(conv2_weights - conv2_quan) * (conv2_weights - conv2_quan)

    f1_local3 = tf.sign(local3_weights + local3_quan) * (local3_weights + local3_quan)
    f2_local3 = tf.sign(local3_weights) * local3_weights
    f3_local3 = tf.sign(local3_weights - local3_quan) * (local3_weights - local3_quan)

    f1_local4 = tf.sign(local4_weights + local4_quan) * (local4_weights + local4_quan)
    f2_local4 = tf.sign(local4_weights) * local4_weights
    f3_local4 = tf.sign(local4_weights - local4_quan) * (local4_weights - local4_quan)

    f1_softmax_linear = tf.sign(softmax_linear_weights + softmax_linear_quan) * (softmax_linear_weights + softmax_linear_quan)
    f2_softmax_linear = tf.sign(softmax_linear_weights) * softmax_linear_weights
    f3_softmax_linear = tf.sign(softmax_linear_weights - softmax_linear_quan) * (softmax_linear_weights - softmax_linear_quan)

    conv1_regularizers = tf.where(tf.less(conv1_weights, -tf.divide(conv1_quan, 2.0)), f1_conv1,
                                  tf.where(tf.less(conv1_weights, tf.divide(conv1_quan, 2.0)), f2_conv1, f3_conv1))
    conv2_regularizers = tf.where(tf.less(conv2_weights, -tf.divide(conv2_quan, 2.0)), f1_conv2,
                                  tf.where(tf.less(conv2_weights, tf.divide(conv2_quan, 2.0)), f2_conv2, f3_conv2))
    local3_regularizers = tf.where(tf.less(local3_weights, -tf.divide(local3_quan, 2.0)), f1_local3,
                                tf.where(tf.less(fc1_weights, tf.divide(local3_quan, 2.0)), f2_local3, f3_local3))
    local4_regularizers = tf.where(tf.less(local4_weights, -tf.divide(local4_quan, 2.0)), f1_local4,
                                tf.where(tf.less(local4_weights, tf.divide(local4_quan, 2.0)), f2_local4, f3_local4))
    softmax_linear_regularizers = tf.where(tf.less(softmax_linear_weights, -tf.divide(softmax_linear_quan, 2.0)), f1_softmax_linear,
                                   tf.where(tf.less(softmax_linear_weights, tf.divide(softmax_linear_quan, 2.0)), f2_softmax_linear, f3_softmax_linear))

    quantify_regularizers = (tf.reduce_sum(conv1_regularizers) +
                             tf.reduce_sum(conv2_regularizers) +
                             tf.reduce_sum(local3_regularizers) +
                             tf.reduce_sum(local4_regularizers)+
                             tf.reduce_sum(softmax_linear_regularizers))

    # Build a Graph that trains the model with one batch of examples and
    # updates the model parameters.
    train_op = cifar10.train(loss, global_step)

    class _LoggerHook(tf.train.SessionRunHook):
      """Logs loss and runtime."""

      def begin(self):
        self._step = -1
        self._start_time = time.time()

      def before_run(self, run_context):
        self._step += 1
        return tf.train.SessionRunArgs(loss)  # Asks for loss value.

      def after_run(self, run_context, run_values):
        if self._step % FLAGS.log_frequency == 0:
          current_time = time.time()
          duration = current_time - self._start_time
          self._start_time = current_time

          loss_value = run_values.results
          examples_per_sec = FLAGS.log_frequency * FLAGS.batch_size / duration
          sec_per_batch = float(duration / FLAGS.log_frequency)

          format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                        'sec/batch)')
          print (format_str % (datetime.now(), self._step, loss_value,
                               examples_per_sec, sec_per_batch))

    # for var in tf.trainable_variables():
    #     tf.summary.histogram(var.op.name, var)


    config = tf.ConfigProto(log_device_placement=FLAGS.log_device_placement)
    config.gpu_options.allow_growth = True
    print_op = tf.no_op()
    # for var in tf.trainable_variables():
    #     weights_pattern = ".*weights.*"
    #     if re.compile(weights_pattern).match(var.op.name):
    #         ini_mean, ini_variance = tf.nn.moments(tf.reshape(var, [-1]), [0])
    #         ini_std = tf.sqrt(ini_variance)
    #         print_var = tf.Print(var.op.name, [var.op.name, ini_std], var.op.name)
    #         # print_ini_op0 = tf.Print(ini_std, [ini_std], 'ini_std')
    #         print_op = tf.group(print_op, print_var)
    # with tf.control_dependencies([print_op]):
    #     train_op = tf.group(train_op, tf.no_op())


    with tf.train.MonitoredTrainingSession(
        checkpoint_dir=FLAGS.train_dir,
        hooks=[tf.train.StopAtStepHook(last_step=FLAGS.max_steps),
               tf.train.NanTensorHook(loss),
               _LoggerHook()],
        config=config) as mon_sess:
      while not mon_sess.should_stop():
        mon_sess.run(train_op)

def main(argv=None):  # pylint: disable=unused-argument
  cifar10.maybe_download_and_extract()
  if tf.gfile.Exists(FLAGS.train_dir):
    tf.gfile.DeleteRecursively(FLAGS.train_dir)
  tf.gfile.MakeDirs(FLAGS.train_dir)
  train()



if __name__ == '__main__':
  tf.app.run()
