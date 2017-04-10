"""Trainer."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from constants import GioConstants

import argparse
import os
import sys

import tensorflow as tf

FLAGS = None

RNN_HIDDEN = 128
CONV_HIDDEN = 128

BATCH_SIZE = 16
LEARNING_RATE = 0.01


def read_and_decode(filename_queue):
  reader = tf.TFRecordReader(options=GioConstants.tf_record_options)
  _, serialized_example = reader.read(filename_queue)
  context_features_def = {
    'width': tf.FixedLenFeature([], dtype=tf.int64),
    'height': tf.FixedLenFeature([], dtype=tf.int64),
    'num_players': tf.FixedLenFeature([], dtype=tf.int64),
    'num_turns': tf.FixedLenFeature([], dtype=tf.int64),
    'label': tf.FixedLenFeature([], dtype=tf.int64)
  }
  board_size = GioConstants.max_width * \
      GioConstants.max_height * GioConstants.num_channels
  sequence_features_def = {
    'board': tf.FixedLenSequenceFeature([board_size], dtype=tf.float32),
    'army_count': tf.FixedLenSequenceFeature([GioConstants.max_players],
                                             dtype=tf.float32),
    'fort_count': tf.FixedLenSequenceFeature([GioConstants.max_players],
                                             dtype=tf.float32),
    'land_count': tf.FixedLenSequenceFeature([GioConstants.max_players],
                                             dtype=tf.float32),
  }
  context_features, sequence_features = tf.parse_single_sequence_example(
      serialized=serialized_example,
      context_features=context_features_def,
      sequence_features=sequence_features_def)
  width = tf.cast(context_features['width'], tf.float32)
  height = tf.cast(context_features['height'], tf.float32)
  num_players = tf.cast(context_features['num_players'], tf.float32)
  num_turns = tf.cast(context_features['num_turns'], tf.int32)
  board = tf.cast(sequence_features['board'], tf.float32)
  army_count = tf.cast(sequence_features['army_count'], tf.float32)
  land_count = tf.cast(sequence_features['land_count'], tf.float32)
  label = tf.one_hot(tf.cast(context_features['label'], tf.int32),
                     depth=GioConstants.max_players)
  return (width,
          height,
          board,
          num_players,
          num_turns,
          army_count,
          land_count,
          label)


def get_inputs(batch_size):
  with tf.name_scope('input'):
    filename_queue = tf.train.string_input_producer(
        [os.path.join(FLAGS.examples_dir, 'train.tfrecords')])

    features = read_and_decode(filename_queue)

    return tf.train.batch(
        features,
        batch_size=batch_size,
        num_threads=2,
        capacity=500 + 3 * batch_size,
        dynamic_pad=True)


def main(unused_argv):
  ps_hosts = FLAGS.ps_hosts.split(",")
  worker_hosts = FLAGS.worker_hosts.split(",")

  # Create a cluster from the parameter server and worker hosts.
  cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})

  # Create and start a server for the local task.
  server = tf.train.Server(cluster,
                           job_name=FLAGS.job_name,
                           task_index=FLAGS.task_index)
  if FLAGS.job_name == "ps":
    server.join()
  elif FLAGS.job_name == "worker":

    # Assigns ops to the local worker by default.
    with tf.device(
        tf.train.replica_device_setter(
            worker_device="/job:worker/task:%d" % FLAGS.task_index,
            cluster=cluster)):

      (
        width,
        height,
        board,
        num_players,
        num_turns,
        army_count,
        land_count,
        labels
      ) = get_inputs(BATCH_SIZE)
      num_turns_float = tf.cast(num_turns, tf.float32)
      net = tf.stack([width, height, num_players, num_turns_float], 1)

      # Expand channel fields to one-hot vectors, increasing channels from 3 to
      # 1 (Army) + 9 (Owners, none + max_players) + 4 (Types)
      board = tf.reshape(board, shape=[BATCH_SIZE,
                                       -1,
                                       GioConstants.max_width,
                                       GioConstants.max_height,
                                       GioConstants.num_channels])
      army = tf.reshape(board[:, :, :, :, 0], shape=[BATCH_SIZE,
                                                     -1,
                                                     GioConstants.max_width,
                                                     GioConstants.max_height,
                                                     1])
      # Owners starts at -1. Shift to start at 0 by adding 1.
      owners_one_hot = tf.one_hot(tf.cast(board[:, :, :, :, 1] + 1, tf.int32),
                                  depth=GioConstants.max_players + 1)
      # Types starts at 1 and has depth 4.
      types_one_hot = tf.one_hot(tf.cast(board[:, :, :, :, 2] - 1, tf.int32),
                                 depth=4)
      board = tf.concat([army, owners_one_hot, types_one_hot], 4)

      # Apply convolutions. Since the number of turns is dynamic, we need
      # to use TensorFlow's control flow methods for stepping through time.
      def _step(time, conv_outs):
        conv_inputs = board[:, time, :, :, :]  # Shape (batch, 32, 32, 14)

        def apply_conv(reuse=True):
          conv1 = tf.layers.conv2d(
              inputs=conv_inputs,
              filters=16,
              kernel_size=3,
              padding="same",
              activation=tf.nn.elu)
          pool1 = tf.layers.max_pooling2d(
              inputs=conv1,
              pool_size=[2, 2],
              strides=2)
          conv2 = tf.layers.conv2d(
              inputs=pool1,
              filters=32,
              kernel_size=3,
              padding="same",
              activation=tf.nn.elu)
          pool2 = tf.layers.max_pooling2d(
              inputs=conv2,
              pool_size=[2, 2],
              strides=2)
          final_width = int(GioConstants.max_width / 4)
          final_height = int(GioConstants.max_height / 4)
          pool2_flat = tf.reshape(pool2, [-1, final_width * final_height * 32])
          ret = tf.contrib.layers.fully_connected(inputs=pool2_flat,
                                                  num_outputs=CONV_HIDDEN)
          # ret shape is [batch_size, CONV_HIDDEN], reshape to
          # [batch_size, 1, CONV_HIDDEN] since we concatenate on time axis (1).
          return tf.reshape(ret, [BATCH_SIZE, 1, CONV_HIDDEN])

        # # Need to ensure vars are created before setting reuse.
        conv_outs = tf.concat(
          [
              tf.cond(tf.equal(time, 0), lambda: apply_conv(False), apply_conv),
              conv_outs
          ], 1)

        return (time + 1, conv_outs)

      max_seq_length = tf.cast(tf.reduce_max(num_turns), tf.int32)
      empty_conv_outs = tf.zeros([BATCH_SIZE, 0, CONV_HIDDEN], dtype=tf.float32)
      _, conv_outs = tf.while_loop(
          cond=lambda time, *_: tf.less(time, max_seq_length),
          body=_step,
          loop_vars=(tf.constant(0), empty_conv_outs),
          # We need to set the time axis as one that changes (as we concat).
          shape_invariants=(tf.TensorShape([]),
                            tf.TensorShape([BATCH_SIZE, None, CONV_HIDDEN])))

      with tf.variable_scope('board_lstm'):
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(RNN_HIDDEN)
        output, state = tf.nn.dynamic_rnn(lstm_cell,
                                          conv_outs,
                                          sequence_length=num_turns,
                                          dtype=tf.float32)
        net = tf.concat([output[:, -1, :], net], 1)

      for idx, seq_feature in enumerate((army_count, land_count)):
        reuse_vars = (idx > 0)
        with tf.variable_scope('score_lstm', reuse=reuse_vars):
          lstm_cell = tf.contrib.rnn.BasicLSTMCell(RNN_HIDDEN,
                                                   reuse=reuse_vars)
          output, state = tf.nn.dynamic_rnn(lstm_cell,
                                            seq_feature,
                                            sequence_length=num_turns,
                                            dtype=tf.float32)
          net = tf.concat([output[:, -1, :], net], 1)

      logits = tf.contrib.layers.fully_connected(
          inputs=net,
          num_outputs=GioConstants.max_players)
      loss = tf.nn.softmax_cross_entropy_with_logits(labels=labels,
                                                     logits=logits)
      tf.summary.scalar("loss", tf.reduce_mean(loss))

      predicted_value = tf.argmax(logits, 1)
      label_value = tf.argmax(labels, 1)
      tf.summary.scalar(
          "accuracy",
          tf.reduce_mean(
              tf.cast(tf.equal(predicted_value, label_value), tf.float32)))
      tf.summary.scalar(
          "l1_distance",
          tf.reduce_mean(tf.abs(tf.subtract(predicted_value, label_value)))
      )

      for v in tf.trainable_variables():
        tf.summary.histogram("Vars/" + v.name, v)

      global_step = tf.contrib.framework.get_or_create_global_step()

      train_op = tf.train.AdagradOptimizer(LEARNING_RATE).minimize(
          loss, global_step=global_step)

    # The StopAtStepHook handles stopping after running given steps.
    hooks = [tf.train.StopAtStepHook(last_step=FLAGS.num_steps)]

    # The MonitoredTrainingSession takes care of session initialization,
    # restoring from a checkpoint, saving to a checkpoint, and closing when done
    # or an error occurs.
    with tf.train.MonitoredTrainingSession(master=server.target,
                                           is_chief=(FLAGS.task_index == 0),
                                           checkpoint_dir=FLAGS.model_dir,
                                           hooks=hooks) as mon_sess:
      while not mon_sess.should_stop():
        # Run a training step asynchronously.
        # See `tf.train.SyncReplicasOptimizer` for additional details on how to
        # perform *synchronous* training.
        # mon_sess.run handles AbortedError in case of preempted PS.
        mon_sess.run(train_op)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--examples_dir',
      type=str,
      default='/Users/abel/data/gio/examples',
      help='Directory that contains examples.'
  )
  parser.add_argument(
      '--model_dir',
      type=str,
      default='/Users/abel/data/gio/model',
      help='Directory to write the model checkpoints and summaries to.'
  )
  # Flags for defining the tf.train.ClusterSpec
  parser.add_argument(
      "--ps_hosts",
      type=str,
      default="",
      help="Comma-separated list of hostname:port pairs"
  )
  parser.add_argument(
      "--worker_hosts",
      type=str,
      default="",
      help="Comma-separated list of hostname:port pairs"
  )
  parser.add_argument(
      "--job_name",
      type=str,
      default="",
      help="One of 'ps', 'worker'"
  )
  # Flags for defining the tf.train.Server
  parser.add_argument(
      "--task_index",
      type=int,
      default=0,
      help="Index of task within the job"
  )
  parser.add_argument(
    "--num_steps",
      type=int,
      default=200000,
      help="Number of steps to train model for."
  )
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
