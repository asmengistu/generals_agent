"""Project-wide constants."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

TFRecordOptions = tf.python_io.TFRecordOptions
TFRecordCompressionType = tf.python_io.TFRecordCompressionType


class GioConstants(object):
  """Contains constants used across packages."""
  max_width = 32
  max_height = 32
  max_time = 300
  min_time = 50
  max_players = 8
  min_turns = 150
  num_channels = 3
  tf_record_options = TFRecordOptions(TFRecordCompressionType.GZIP)
