"""Trainer."""
from __future__ import absolute_import
from __future__ import print_function

from subprocess import Popen
from time import time

import argparse
import os

FLAGS = None

BASE_PORT = 2222


def create_dirs():
  model_dir = os.path.join(FLAGS.model_dir, FLAGS.run_name)
  if not os.path.isdir(model_dir):
    os.makedirs(model_dir)
  log_dir = os.path.join(FLAGS.log_dir, FLAGS.run_name)
  if not os.path.isdir(log_dir):
    os.makedirs(log_dir)


def get_train_cmd(args):
  string_args_list = ["--{}={}".format(k, v) for k, v in args.items()]
  return "python train.py {}".format(" ".join(string_args_list))


def main():
  print("Starting training...")
  ps_start = BASE_PORT
  worker_start = BASE_PORT + FLAGS.num_ps
  ps_hosts = [
      'localhost:%d' % (ps_start + i,) for i in range(FLAGS.num_ps)
  ]
  worker_hosts = [
      'localhost:%d' % (worker_start + i,) for i in range(FLAGS.num_workers)
  ]
  shared_flags = {
    'examples_dir': FLAGS.examples_dir,
    'model_dir': os.path.join(FLAGS.model_dir, FLAGS.run_name),
    'num_steps': FLAGS.num_steps
  }
  if ps_hosts:
    shared_flags.update({'ps_hosts': ','.join(ps_hosts)})
  shared_flags.update({'worker_hosts': ','.join(worker_hosts)})

  commands = []

  for ps_task in range(FLAGS.num_ps):
    task_flags = shared_flags.copy()
    task_flags.update({
      'task_index': ps_task,
      'job_name': 'ps'
    })
    commands.append(get_train_cmd(task_flags))

  for worker_task in range(FLAGS.num_workers):
    task_flags = shared_flags.copy()
    task_flags.update({
      'task_index': worker_task,
      'job_name': 'worker'
    })
    commands.append(get_train_cmd(task_flags))

  print("Running the following commands:")
  for command in commands:
    print(command)

  create_dirs()

  processes = []
  log_files = []
  try:
    for idx, cmd in enumerate(commands):
      log_file = open(os.path.join(FLAGS.log_dir,
                                   FLAGS.run_name,
                                   "{}.log".format(idx)),
                      'a')
      log_file.write("\n# {}\n".format(time()))
      proc = Popen([cmd],
                   shell=True,
                   stdin=None,
                   stdout=None,
                   stderr=log_file)
      processes.append(proc)
      log_files.append(log_file)
    # Wait for last worker job
    processes[-1].wait()
  except KeyboardInterrupt:
    print("Keyboard interrupt, stopping training.")
  finally:
    # If last worker stops for some reason, stop all workers.
    for process in processes:
      if process.poll() is None:
        # Still running...
        process.kill()
    for log_file in log_files:
      log_file.close()

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "--run_name",
      type=str,
      default="test",
      help="Run name for job."
  )
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
  parser.add_argument(
      "--log_dir",
      type=str,
      default="/Users/abel/data/gio/logs",
      help="Directory for logs."
  )
  # Flags for defining the tf.train.ClusterSpec
  parser.add_argument(
      "--num_ps",
      type=int,
      default=1,
      help="Number of parameter servers."
  )
  parser.add_argument(
      "--num_workers",
      type=int,
      default=2,
      help="Number of workers."
  )
  parser.add_argument(
    "--num_steps",
      type=int,
      default=200000,
      help="Number of steps to train model for."
  )
  parser.add_argument
  FLAGS, _ = parser.parse_known_args()
  main()
