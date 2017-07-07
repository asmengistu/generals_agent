from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from constants import GioConstants

import tensorflow as tf

RNN_HIDDEN = 256
RNN_STACK_SIZE = 2
GRID_RNN_SIZE = 32
BOARD_RNN_SIZE = 128
CONV_FC_SIZES = [512, 256]
FINAL_FC_SIZES = [512, 256]


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
  # Add 0-d features.
  net = tf.stack([
                   features["width"],
                   features["height"],
                   features["num_players"],
                   num_turns_float
                 ], 1)
  # Add score features
  net = tf.concat([net, features["army_count"], features["land_count"]], 1)

  board = tf.reshape(features["board"], shape=[GioConstants.batch_size,
                                               GioConstants.max_width,
                                               GioConstants.max_height,
                                               GioConstants.num_channels])
  army = tf.reshape(board[:, :, :, 0], shape=[GioConstants.batch_size,
                                              GioConstants.max_width,
                                              GioConstants.max_height,
                                              1])

  # Expand channel fields to one-hot vectors, increasing channels from 3 to 16.
  # 1 (Army) + 9 (Owners: none + max_players) + 7 (Types, incl. fog and void)
  # Owners starts at -1 (no owner).
  owners_one_hot = tf.one_hot(tf.cast(board[:, :, :, 1] + 1, tf.int32),
                              depth=GioConstants.max_players + 1)
  # -1: void, 0: fog, 1: empty, 2: mountain, 3: fort, 4: general, 5: mtn/fort
  types_one_hot = tf.one_hot(tf.cast(board[:, :, :, 2] + 1, tf.int32),
                             depth=7)
  board = tf.concat([army, owners_one_hot, types_one_hot], 3)

  def apply_conv(reuse=True):
    # Will be the final layer from the conv stack.
    layer = None
    with tf.variable_scope("board_conv_stack", reuse=reuse):
      conv_1 = conv(board, 32)
      pool_1 = pool(conv_1)
      conv_2 = conv(pool_1, 64)
      pool_2 = pool(conv_2)
      conv_3 = conv(pool_2, 128)
      pool_3 = pool(conv_3)
      layer = tf.reshape(pool_3, [-1, 4 * 4 * 128])

    with tf.variable_scope("board_fc", reuse=reuse):
      for idx, size in enumerate(CONV_FC_SIZES):
        layer = tf.contrib.layers.fully_connected(inputs=layer,
                                                  num_outputs=size)
    # ret shape is [batch_size, CONV_FC_SIZES[-1]]
    return layer

  def apply_2d_lstm(reuse=True):
    row_outs = tf.zeros([GioConstants.batch_size, 0, GRID_RNN_SIZE])
    col_outs = tf.zeros([GioConstants.batch_size, 0, GRID_RNN_SIZE])
    # Simple LSTM cells per row where we collect the final output.
    # Same for columns. We will combine each with a bidirectional LSTM later.
    for idx in range(32):
      with tf.variable_scope("board_2d_lstm_" + str(idx), reuse=reuse):
        with tf.variable_scope("2d_cell") as cell_scope:
          cell = get_rnn_cell(reuse=reuse, cell_size=GRID_RNN_SIZE)
        rows = board[:, idx, :, :]  # Shape (batch, 32, 17)
        cols = board[:, :, idx, :]  # Shape (batch, 32, 17)
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
    return board_rnn_out

  processed_board = apply_conv(False)
  net = tf.concat([net, processed_board], 1)

  # Final FC layers.
  with tf.variable_scope("final_fc", reuse=False):
    for idx, size in enumerate(FINAL_FC_SIZES):
      net = tf.contrib.layers.fully_connected(inputs=net, num_outputs=size)

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
