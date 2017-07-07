### DEPRECATED. Was used to test model where examples were built with multiple
### turns. [0, num_turns]

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from constants import GioConstants

import tensorflow as tf

RNN_HIDDEN = 256
RNN_STACK_SIZE = 2
GRID_RNN_SIZE = 16
BOARD_RNN_SIZE = 64
CONV_FC_SIZES = [512]
TIME_STEP = 16


def get_last_relevant_layer(tensor, seq_lengths):
  """Returns the last relevant (non zero padded) layer from a tensor."""
  batch_size = tf.shape(tensor)[0]
  max_length = tf.shape(tensor)[1]
  # numpy equivalent: tensor[: seq_lengths - 1], where seq_lengths is an array.
  # We need to flatten to [batch_size * max_time_step, layer_size], and then
  # select the indices that correspond to the last relevant layer within the
  # sequence examples.
  layer_size = int(tensor.get_shape()[2])
  flat = tf.reshape(tensor, [-1, layer_size])
  indices = tf.range(0, batch_size) * max_length + (seq_lengths - 1)
  return tf.gather(flat, indices)


def get_rnn_cell(reuse=False, cell_size=RNN_HIDDEN):
  cell = tf.contrib.rnn.BasicLSTMCell(cell_size, reuse=reuse)
  return tf.contrib.rnn.DropoutWrapper(cell, 0.5)


def get_rnn_stack(reuse=False):
  cells = []
  for i in range(RNN_STACK_SIZE):
    cells.append(get_rnn_cell())
  return tf.contrib.rnn.MultiRNNCell(cells)


def conv(inputs, filters):
  return tf.layers.conv2d(inputs=inputs,
                          filters=filters,
                          kernel_size=3,
                          padding="same",
                          activation=tf.nn.relu)


def pool(inputs, pool_size=[2, 2]):
  return tf.layers.max_pooling2d(inputs=inputs,
                                 pool_size=pool_size,
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
  with tf.device("/cpu:0"):
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
    # -1: void, 0: fog, 1: empty, 2: mountain, 3: fort, 4: general, 5: mtn/fort
    types_one_hot = tf.one_hot(tf.cast(board[:, :, :, :, 2] + 1, tf.int32),
                               depth=7)
    board = tf.concat([army, owners_one_hot, types_one_hot], 4)
    board = board[:, ::TIME_STEP, :, :, :]
    score_features = (
        features["army_count"][:, ::TIME_STEP, :],
        features["land_count"][:, ::TIME_STEP, :]
    )

  # Apply convolutions. Since the number of turns is dynamic, we need
  # to use TensorFlow's control flow methods for stepping through time.
  def step(time, loop_output):
    board_inputs = board[:, time, :, :, :]  # Shape (batch, 32, 32, 17)

    def apply_conv(reuse=True):
      # Will be the final layer from the conv stack.
      layer = None
      with tf.variable_scope("board_conv_stack", reuse=reuse):
        conv_1 = conv(board_inputs, 32)
        pool_1 = pool(conv_1)
        layer = tf.reshape(pool_1, [-1, 16 * 16 * 32])

      with tf.variable_scope("board_fc", reuse=reuse):
        for idx, size in enumerate(CONV_FC_SIZES):
          layer = tf.contrib.layers.fully_connected(inputs=layer,
                                                    num_outputs=size)
      # ret shape is [batch_size, CONV_FC_SIZES[-1]], reshape to
      # [batch_size, 1, CONV_FC_SIZES[-1]] since we concat on time axis (1).
      return tf.reshape(layer, [GioConstants.batch_size, 1, CONV_FC_SIZES[-1]])

    def apply_2d_lstm(reuse=True):
      row_outs = tf.zeros([GioConstants.batch_size, 0, GRID_RNN_SIZE])
      col_outs = tf.zeros([GioConstants.batch_size, 0, GRID_RNN_SIZE])
      # Simple LSTM cells per row where we collect the final output.
      # Same for columns. We will combine each with a bidirectional LSTM later.
      for idx in range(32):
        with tf.variable_scope("board_2d_lstm_" + str(idx), reuse=reuse):
          with tf.variable_scope("2d_cell") as cell_scope:
            cell = get_rnn_cell(reuse=reuse, cell_size=GRID_RNN_SIZE)
          rows = board_inputs[:, idx, :, :]  # Shape (batch, 32, 17)
          cols = board_inputs[:, :, idx, :]  # Shape (batch, 32, 17)
          with tf.variable_scope(cell_scope, reuse=False):
            row_outputs, _ = tf.contrib.rnn.static_rnn(cell,
                                                       tf.unstack(rows, axis=1),
                                                       dtype=tf.float32)
          # Share weights between row-wise/col-wise cells.
          with tf.variable_scope(cell_scope, reuse=True):
            col_outputs, _ = tf.contrib.rnn.static_rnn(cell,
                                                       tf.unstack(cols, axis=1),
                                                       dtype=tf.float32)
          last_out_row = tf.reshape(
              row_outputs[-1],
              [GioConstants.batch_size, 1, GRID_RNN_SIZE])
          last_out_col = tf.reshape(
              col_outputs[-1],
              [GioConstants.batch_size, 1, GRID_RNN_SIZE])
          row_outs = tf.concat([last_out_row, row_outs], 1)
          col_outs = tf.concat([last_out_col, col_outs], 1)

      # Shape (batch, 32, GRID_RNN_SIZE*2). Run bidirectional RNN on these.
      grid_out = tf.concat([row_outs, col_outs], 2)
      board_rnn_out = None
      with tf.variable_scope("board_summary_lstm", reuse=reuse):
        cell_fw = get_rnn_cell(reuse=reuse, cell_size=BOARD_RNN_SIZE)
        cell_bw = get_rnn_cell(reuse=reuse, cell_size=BOARD_RNN_SIZE)
        outputs, _, _ = tf.contrib.rnn.static_bidirectional_rnn(
            cell_fw, cell_bw, tf.unstack(grid_out, axis=1), dtype=tf.float32)
        board_rnn_out = outputs[-1]

      return tf.reshape(board_rnn_out,
                        [GioConstants.batch_size, 1, BOARD_RNN_SIZE * 2])

    # Need to ensure vars are created before setting reuse.
    loop_output = tf.concat(
      [
          tf.cond(tf.equal(time, 0),
                  lambda: apply_2d_lstm(False),
                  apply_2d_lstm),
          loop_output
      ], 1)

    return (time + 1, loop_output)

  max_seq_length = tf.cast(tf.reduce_max(num_turns) / TIME_STEP, tf.int32)
  empty_loop_outputs = tf.zeros([GioConstants.batch_size, 0, BOARD_RNN_SIZE*2],
                                dtype=tf.float32)
  # with tf.variable_scope("board_loop"):
  _, conv_outs = tf.while_loop(
      cond=lambda time, *_: tf.less(time, max_seq_length),
      body=step,
      loop_vars=(tf.constant(0), empty_loop_outputs),
      # We need to set the time axis as one that changes (as we concat).
      shape_invariants=(tf.TensorShape([]),
                        tf.TensorShape([GioConstants.batch_size,
                                        None,  # Time axis is not constant.
                                        BOARD_RNN_SIZE*2])))

  with tf.variable_scope("board_lstm"):
    stacked_lstms = get_rnn_stack()
    conv_seq_lengths = tf.cast(tf.ceil(num_turns / TIME_STEP), tf.int32)
    output, _ = tf.nn.dynamic_rnn(stacked_lstms,
                                  conv_outs,
                                  sequence_length=conv_seq_lengths,
                                  dtype=tf.float32)
    relevant_outputs = get_last_relevant_layer(output, conv_seq_lengths)
    net = tf.concat([relevant_outputs, net], 1)

  score_features = (features["army_count"], features["land_count"])
  for idx, score_inputs in enumerate(score_features):
    with tf.variable_scope("score_lstm_" + str(idx)):
      stacked_lstms = get_rnn_stack()
      score_seq_lengths = tf.cast(tf.ceil(num_turns / TIME_STEP), tf.int32)
      output, _ = tf.nn.dynamic_rnn(stacked_lstms,
                                    score_inputs,
                                    sequence_length=score_seq_lengths,
                                    dtype=tf.float32)
      relevant_outputs = get_last_relevant_layer(output, score_seq_lengths)
      net = tf.concat([relevant_outputs, net], 1)

  rank_preds = tf.contrib.layers.fully_connected(inputs=net, num_outputs=1)
  label = tf.reshape(label, [-1, 1])
  loss = tf.losses.mean_squared_error(labels=label,
                                      predictions=rank_preds)
  tf.summary.scalar("loss", tf.reduce_mean(loss))

  tf.summary.scalar("info/num_players_avg",
                    tf.reduce_mean(features["num_players"]))
  tf.summary.scalar("info/num_turns_avg",
                    tf.reduce_mean(num_turns))

  tf.summary.scalar(
      "accuracy",
      tf.reduce_mean(
          tf.cast(tf.equal(tf.round(rank_preds), label), tf.float32)))
  tf.summary.scalar(
      "l1_distance",
      tf.reduce_mean(tf.abs(tf.subtract(rank_preds, label)))
  )
  return loss
