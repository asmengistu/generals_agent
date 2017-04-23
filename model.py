from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from constants import GioConstants

import tensorflow as tf

RNN_HIDDEN = 512
RNN_STACK_SIZE = 2
CONV_HIDDEN = 1024
BOARD_TIMESTEP = 16
SCORE_TIMESTEP = 16


def get_last_relevant_layer(tensor, seq_lengths):
  """Returns the last relevant (non zero padded) layer from a tensor."""
  batch_size = tf.shape(tensor)[0]
  max_length = tf.shape(tensor)[1]
  # numpy equivalent: tensor[: seq_lengths - 1], where seq_lengths is an array.
  # We need to flatten to [batch_size * max_time_step, layer_size], and then
  # select the indices that correspond to the last relevant layer within the
  # sequence examples.
  indices = tf.range(0, batch_size) * max_length + (seq_lengths - 1)
  layer_size = int(tensor.get_shape()[2])
  flat = tf.reshape(tensor, [-1, layer_size])
  return tf.gather(flat, indices)


def get_rnn_cell(reuse=False):
  cell = tf.contrib.rnn.BasicLSTMCell(RNN_HIDDEN, reuse=reuse)
  return tf.contrib.rnn.DropoutWrapper(cell, 0.5)


def get_rnn_stack(reuse=False):
  cells = []
  for i in range(RNN_STACK_SIZE):
    cells.append(get_rnn_cell())
  return tf.contrib.rnn.MultiRNNCell(cells)


def conv_and_pool(inputs, filters):
  conv = tf.layers.conv2d(inputs=inputs,
                          filters=filters,
                          kernel_size=3,
                          padding="same",
                          activation=tf.nn.relu)
  return tf.layers.max_pooling2d(inputs=conv,
                                 pool_size=[2, 2],
                                 strides=2)


def get_loss(features):
  label = features["label"]
  num_turns = features["num_turns"]
  num_turns_float = tf.cast(num_turns, tf.float32)
  net = tf.stack([
                   features["width"],
                   features["height"],
                   features["num_players"],
                   num_turns_float
                 ], 1)

  board = tf.reshape(features["board"], shape=[GioConstants.batch_size,
                                               -1,  # time/turn axis.
                                               GioConstants.max_width,
                                               GioConstants.max_height,
                                               GioConstants.num_channels])
  army = tf.reshape(board[:, :, :, :, 0], shape=[GioConstants.batch_size,
                                                 -1,  # time/turn axis.
                                                 GioConstants.max_width,
                                                 GioConstants.max_height,
                                                 1])

  # Expand channel fields to one-hot vectors, increasing channels from 3 to 16.
  # 1 (Army) + 9 (Owners: none + max_players) + 7 (Types, incl. fog and void)
  # Owners starts at -1 (no owner).
  owners_one_hot = tf.one_hot(tf.cast(board[:, :, :, :, 1] + 1, tf.int32),
                              depth=GioConstants.max_players + 1)
  # -1: void, 0: fog, 1: empty, 2: mountain/fort, 3: fort, 4: general
  types_one_hot = tf.one_hot(tf.cast(board[:, :, :, :, 2] + 1, tf.int32),
                             depth=7)
  board = tf.concat([army, owners_one_hot, types_one_hot], 4)

  # Apply convolutions. Since the number of turns is dynamic, we need
  # to use TensorFlow's control flow methods for stepping through time.
  def step(time, conv_outs):
    conv_inputs = board[:, time, :, :, :]  # Shape (batch, 32, 32, 16)

    def apply_conv(reuse=True):
      with tf.variable_scope("board_conv", reuse=reuse):
        pool_1 = conv_and_pool(conv_inputs, 16)
        pool_2 = conv_and_pool(pool_1, 32)
        final_width = int(GioConstants.max_width / 4)
        # Every pool layer decreases size by a factor of 2.
        final_height = int(GioConstants.max_height / 4)
        pool_2_flat = tf.reshape(pool_2,
                                 [-1, final_width * final_height * 32])
        ret = tf.contrib.layers.fully_connected(inputs=pool_2_flat,
                                                num_outputs=CONV_HIDDEN)
      # ret shape is [batch_size, CONV_HIDDEN], reshape to
      # [batch_size, 1, CONV_HIDDEN] since we concatenate on time axis (1).
      return tf.reshape(ret, [GioConstants.batch_size, 1, CONV_HIDDEN])

    # Need to ensure vars are created before setting reuse.
    conv_outs = tf.concat(
      [
          tf.cond(tf.equal(time, 0), lambda: apply_conv(False), apply_conv),
          conv_outs
      ], 1)

    return (time + BOARD_TIMESTEP, conv_outs)

  max_seq_length = tf.cast(tf.reduce_max(num_turns), tf.int32)
  empty_conv_outs = tf.zeros([GioConstants.batch_size, 0, CONV_HIDDEN],
                             dtype=tf.float32)
  _, conv_outs = tf.while_loop(
      cond=lambda time, *_: tf.less(time, max_seq_length),
      body=step,
      loop_vars=(tf.constant(0), empty_conv_outs),
      # We need to set the time axis as one that changes (as we concat).
      shape_invariants=(tf.TensorShape([]),
                        tf.TensorShape([GioConstants.batch_size,
                                        None,  # Time axis is not constant.
                                        CONV_HIDDEN])))

  with tf.variable_scope("board_lstm"):
    stacked_lstms = get_rnn_stack()
    conv_seq_lengths = tf.cast(tf.ceil(num_turns / BOARD_TIMESTEP), tf.int32)
    output, _ = tf.nn.dynamic_rnn(stacked_lstms,
                                  conv_outs,
                                  sequence_length=conv_seq_lengths,
                                  dtype=tf.float32)
    relevant_outputs = get_last_relevant_layer(output, conv_seq_lengths)
    net = tf.concat([relevant_outputs, net], 1)

  score_features = (features["army_count"], features["land_count"])
  for idx, score_feature in enumerate(score_features):
    with tf.variable_scope("score_lstm_" + str(idx)):
      stacked_lstms = get_rnn_stack()
      # Go every SCORE_TIMESTEP turns on the time axis.
      score_inputs = score_feature[:, ::SCORE_TIMESTEP, :]
      score_seq_lengths = tf.cast(tf.ceil(num_turns / SCORE_TIMESTEP), tf.int32)
      output, _ = tf.nn.dynamic_rnn(stacked_lstms,
                                    score_inputs,
                                    sequence_length=score_seq_lengths,
                                    dtype=tf.float32)
      relevant_outputs = get_last_relevant_layer(output, score_seq_lengths)
      net = tf.concat([relevant_outputs, net], 1)

  rank_preds = tf.contrib.layers.fully_connected(
      inputs=net,
      num_outputs=1)
  label = tf.reshape(label, [-1, 1])
  loss = tf.losses.absolute_difference(labels=label,
                                       predictions=rank_preds)
  tf.summary.scalar("loss", tf.reduce_mean(loss))

  tf.summary.scalar(
      "accuracy",
      tf.reduce_mean(
          tf.cast(tf.equal(tf.round(rank_preds), label), tf.float32)))
  tf.summary.scalar(
      "l1_distance",
      tf.reduce_mean(tf.abs(tf.subtract(rank_preds, label)))
  )
  return loss
