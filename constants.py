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
  max_time = 600
  min_time = 0
  max_players = 8
  num_channels = 3
  batch_size = 64
  tf_record_options = TFRecordOptions(TFRecordCompressionType.GZIP)
