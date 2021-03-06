"""Converts models to TFRecords file format with Example protos."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import defaultdict
from gio_model import GioModel
from constants import GioConstants

import argparse
import os
import sys

import numpy as np
import tensorflow as tf

FLAGS = None

# Probability of selecting viewer from the rank array. Biased to selecting
# players that performed well.
VIEW_SELECT_PROBS = np.array([0.45, 0.25, 0.1, 0.05, 0.05, 0.05, 0.025, 0.025])
counters = defaultdict(int)


def _int64_feature(value):
  if type(value) == int or type(value) == np.int64:
    value = [value]
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _float_feature(value):
  if type(value) == float or type(value) == np.float64:
    value = [value]
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _feature_list(value):
  return tf.train.FeatureList(feature=value)


def valid_game(gio_model):
  if gio_model.num_turns < GioConstants.min_time:
    counters['valid under min turns'] += 1
    return False
  return True


def minMaxNorm(values):
  norm = (np.max(values) - np.min(values))
  if norm == 0:
    return values
  return values / norm


def normalize(board):
  """Normalizes the board. Min-Max normalization for the army channel."""
  board[:, :, 0] = minMaxNorm(board[:, :, 0])


def switchView(score_vector, hero_index):
  """Ensures the hero's score is at index 0."""
  if hero_index == 0:
    return score_vector
  ret_vector = np.copy(score_vector)
  tmp = ret_vector[0]
  ret_vector[0] = ret_vector[hero_index]
  ret_vector[hero_index] = tmp
  return ret_vector


def widenScoreboard(scoreboard):
  num_players = len(scoreboard)
  if num_players == GioConstants.max_players:
    return scoreboard
  new_board = np.zeros(GioConstants.max_players)
  new_board[:num_players] = scoreboard
  return new_board


def build_examples(model, hero_probs, examples_per_game, only_live=True):
  """Builds a TensorFlow.Example message from the model."""
  valid_turns_range = range(
      GioConstants.min_time,
      int(min(GioConstants.max_time, 0.95 * model.num_turns)))
  turn_snapshots = np.random.choice(valid_turns_range,
                                    examples_per_game,
                                    replace=False)
  examples = []
  for turn_snapshot in turn_snapshots:
    scoreboard = model.getScoreBoardAtTurn(turn_snapshot)
    if only_live:
      # Player-vector. 1 if player at index is alive.
      live_players = (scoreboard['army'][-1] > 0).astype(int)
      # Update hero probabilities to ensure we always select a live player.
      hero_probs = hero_probs * live_players
    hero_probs /= np.sum(hero_probs)
    # Point of view (chosen from rank with hero_probs).
    hero = np.random.choice(model.num_players, p=hero_probs)

    board = model.getBoardViewAtTurn(hero, turn_snapshot)
    army = widenScoreboard(switchView(scoreboard['army'], hero))
    forts = widenScoreboard(switchView(scoreboard['forts'], hero))
    land = widenScoreboard(switchView(scoreboard['land'], hero))
    label = np.where(np.array(model.ranks) == hero)[0]

    normalize(board)
    # Normalize army/land
    army, land = map(minMaxNorm, (army, land))
    width, height = model.getMapSize()
    # Set-up empty board with max-size.
    max_board = np.zeros((GioConstants.max_width,
                          GioConstants.max_height,
                          GioConstants.num_channels))

    # Set owner and type channel of non-game squares to -1.
    max_board[:, :, 1:] = -1

    # Boards only differ in width and height. We fill the new empty max board
    # with the values from the current board.
    max_board[:board.shape[0], :board.shape[1], :] = board
    # Flatten board on non-time dimensions
    flattened_board = max_board.reshape([-1])
    examples.append(tf.train.Example(
        features=tf.train.Features(feature={
          'width': _int64_feature(width),
          'height': _int64_feature(height),
          'num_players': _int64_feature(model.num_players),
          'num_turns': _int64_feature(turn_snapshot),
          'board': _float_feature(flattened_board),
          'army_count': _float_feature(army),
          'fort_count': _float_feature(forts),
          'land_count': _float_feature(land),
          # Label is the index in the rank vector in 'hero' mode, or the index
          # of the winning player in 'god' mode.
          'label': _int64_feature(label)
        })))
  return examples


def write_examples(game_ids, name, bias_hero=False):
  """Converts a dataset to tfrecords."""
  filename = os.path.join(FLAGS.out_dir, name + '.tfrecords')
  print('Writing', filename)
  import traceback
  with tf.python_io.TFRecordWriter(
        filename, options=GioConstants.tf_record_options) as writer:
    for game_id in game_ids:
      model = get_model(game_id)
      if not valid_game(model):
        continue
      probs = np.ones(model.num_players)
      if bias_hero:
        probs = VIEW_SELECT_PROBS[:model.num_players]
      # Normalize probabilities.
      probs /= np.sum(probs)
      try:
        examples = build_examples(model, probs, FLAGS.examples_per_game)
      except Exception as err:
        print('Error trying to write {}: {}'.format(model.id, err))
        print(traceback.format_exc())
        counters['error'] += 1
        continue
      for example in examples:
        writer.write(example.SerializeToString())
        counters['written'] += 1
        if counters['written'] % 10 == 0:
          print('Written so far:', counters['written'])


def get_model(game_id):
    return GioModel.fromFile(FLAGS.in_dir + '/%s.giomodel.gz' % (game_id))


def create_dirs():
  if not os.path.isdir(FLAGS.out_dir):
    os.makedirs(FLAGS.out_dir)


def main(unused_argv):
  get_game_id_fn = lambda path: path[:path.find('.')]
  all_game_ids = map(get_game_id_fn, os.listdir(FLAGS.in_dir))
  all_game_ids = all_game_ids[:FLAGS.num_games]
  num_games = len(all_game_ids)
  validation_end_idx = int(FLAGS.validation_ratio * num_games)
  test_end_idx = validation_end_idx + int(FLAGS.test_ratio * num_games)
  validation_games = all_game_ids[:validation_end_idx]
  test_games = all_game_ids[validation_end_idx:test_end_idx]
  train_games = all_game_ids[test_end_idx:]

  create_dirs()

  write_examples(test_games, 'test')
  write_examples(validation_games, 'validation')
  write_examples(train_games, 'train')

  print('Counters: ')
  for counter in counters:
    print('{}: {}'.format(counter, counters[counter]))


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--in_dir',
      type=str,
      default='/Users/abel/data/gio/models',
      help='Directory that contains the input giomodel files.'
  )
  parser.add_argument(
      '--out_dir',
      type=str,
      default='/Users/abel/data/gio/examples',
      help='Directory to write the converted results to.'
  )
  parser.add_argument(
      '--validation_ratio',
      type=float,
      default=0.1,
      help='Ratio of data to write as validation data.'
  )
  parser.add_argument(
      '--test_ratio',
      type=float,
      default=0.1,
      help='Ratio of data to write as test data.'
  )
  parser.add_argument(
      '--num_games',
      type=int,
      default=100,
      help='Number of games to generate examples from.'
  )
  parser.add_argument(
      '--examples_per_game',
      type=int,
      default=1,
      help="Number of examples per game."
  )
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
