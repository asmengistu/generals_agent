"""Converts models to TFRecords file format with Example protos."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import defaultdict
from gio_model import GioModel

import argparse
import os
import sys

import numpy as np
import tensorflow as tf

FLAGS = None

MAX_WIDTH, MAX_HEIGHT, MAX_TIME = 32, 32, 300
MIN_TURNS = 150
# Probability of selecting viewer from the rank array. Biased to selecting
# players that performed well.
VIEW_SELECT_PROBS = np.array([0.45, 0.25, 0.1, 0.05, 0.05, 0.05, 0.025, 0.025])
counters = defaultdict(int)


def _int64_feature(value):
  if type(value) == int:
    value = [value]
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _float_feature(value):
  if type(value) == float:
    value = [value]
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def valid_game(gio_model):
  if gio_model.num_turns < MIN_TURNS:
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
  board[:, :, :, 0] = minMaxNorm(board[:, :, :, 0])


def build_example(model, hero_probs):
  """Builds a TensorFlow.Example message from the model."""
  # Point of view will be from hero (chosen from rank with hero_probs).
  hero = np.random.choice(model.ranks, 1, p=hero_probs)
  # Clip the number of turns
  turn_clip = np.random.randint(min(MAX_TIME, model.num_turns))
  model.clipBoard(turn_clip)
  board = model.getBoardView(hero)
  normalize(board)
  scoreboard = model.getScoreBoard()
  army = scoreboard['army']
  forts = scoreboard['forts']
  land = scoreboard['land']
  army, land = map(minMaxNorm, (army, land))

  width, height = model.getMapSize()
  return tf.train.Example(features=tf.train.Features(feature={
      'width': _int64_feature(width),
      'height': _int64_feature(height),
      'num_players': _int64_feature(model.num_players),
      'board': _float_feature(board.flatten()),
      'army_count': _float_feature(army.flatten()),
      'fort_count': _float_feature(forts.flatten()),
      'land_count': _float_feature(land.flatten()),
      # Label is the index in the rank vector.
      'label': _int64_feature(map(int, np.array(model.ranks) == hero))
    }))


def write_examples(game_ids, name, bias_hero=False):
  """Converts a dataset to tfrecords."""
  filename = os.path.join(FLAGS.out_dir, name + '.tfrecords')
  print('Writing', filename)
  with tf.python_io.TFRecordWriter(filename) as writer:
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
        example = build_example(model, probs)
      except Exception as err:
        print('Error trying to write {}: {}'.format(model.id, err))
        counters['error'] += 1
        continue
      writer.write(example.SerializeToString())
      counters['written'] += 1
      if counters['written'] % 10 == 0:
        print('Written so far:', counters['written'])


def get_model(game_id):
    return GioModel.fromFile(FLAGS.in_dir + '/%s.giomodel.gz' % (game_id))


def main(unused_argv):
  get_game_id_fn = lambda path: path[:path.find('.')]
  all_game_ids = map(get_game_id_fn, os.listdir(FLAGS.in_dir))
  # Reduce to 100 games for testing
  all_game_ids = all_game_ids[:10000]
  num_games = len(all_game_ids)
  validation_end_idx = int(FLAGS.validation_ratio * num_games)
  test_end_idx = validation_end_idx + int(FLAGS.test_ratio * num_games)
  validation_games = all_game_ids[:validation_end_idx]
  test_games = all_game_ids[validation_end_idx:test_end_idx]
  train_games = all_game_ids[test_end_idx:]

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
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
