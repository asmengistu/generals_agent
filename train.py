"""Trainer."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from constants import GioConstants
import model

import argparse
import os
import sys

import tensorflow as tf

FLAGS = None

LEARNING_RATE = 0.0002


def read_and_decode(filename_queue):
  reader = tf.TFRecordReader(options=GioConstants.tf_record_options)
  _, serialized_example = reader.read(filename_queue)
  context_features_def = {
    "width": tf.FixedLenFeature([], dtype=tf.int64),
    "height": tf.FixedLenFeature([], dtype=tf.int64),
    "num_players": tf.FixedLenFeature([], dtype=tf.int64),
    "num_turns": tf.FixedLenFeature([], dtype=tf.int64),
    "label": tf.FixedLenFeature([], dtype=tf.int64)
  }
  board_size = GioConstants.max_width * \
      GioConstants.max_height * GioConstants.num_channels
  sequence_features_def = {
    "board": tf.FixedLenSequenceFeature([board_size], dtype=tf.float32),
    "army_count": tf.FixedLenSequenceFeature([GioConstants.max_players],
                                             dtype=tf.float32),
    "fort_count": tf.FixedLenSequenceFeature([GioConstants.max_players],
                                             dtype=tf.float32),
    "land_count": tf.FixedLenSequenceFeature([GioConstants.max_players],
                                             dtype=tf.float32),
  }
  context_features, sequence_features = tf.parse_single_sequence_example(
      serialized=serialized_example,
      context_features=context_features_def,
      sequence_features=sequence_features_def)
  return {
    "width": tf.cast(context_features["width"], tf.float32),
    "height": tf.cast(context_features["height"], tf.float32),
    "num_players": tf.cast(context_features["num_players"], tf.float32),
    "num_turns": tf.cast(context_features["num_turns"], tf.int32),
    "board": tf.cast(sequence_features["board"], tf.float32),
    "army_count": tf.cast(sequence_features["army_count"], tf.float32),
    "land_count": tf.cast(sequence_features["land_count"], tf.float32),
    "label": tf.cast(context_features["label"], tf.float32)
  }


def get_inputs(batch_size):
  with tf.name_scope("input"):
    filename_queue = tf.train.string_input_producer(
        [os.path.join(FLAGS.examples_dir, "train.tfrecords")])

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
      loss = model.get_loss(get_inputs(GioConstants.batch_size))

      # for v in tf.trainable_variables():
      #   tf.summary.histogram("Vars/" + v.name, v)

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


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "--examples_dir",
      type=str,
      default="/Users/abel/data/gio/examples",
      help="Directory that contains examples."
  )
  parser.add_argument(
      "--model_dir",
      type=str,
      default="/Users/abel/data/gio/model",
      help="Directory to write the model checkpoints and summaries to."
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
