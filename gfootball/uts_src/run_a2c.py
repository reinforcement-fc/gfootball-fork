# coding=utf-8
# Copyright 2019 Google LLC
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Runs football_env on OpenAI's A2C."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import multiprocessing
import os
import sys
from absl import app
from absl import flags
from baselines import logger
from baselines.bench import monitor
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.a2c import a2c
import gfootball.env as football_env
from gfootball.examples import models  

FLAGS = flags.FLAGS

flags.DEFINE_string('level', 'academy_empty_goal_close',
                    'Defines type of problem being solved')
flags.DEFINE_enum('state', 'extracted_stacked', ['extracted',
                                                 'extracted_stacked'],
                  'Observation to be used for training.')
flags.DEFINE_enum('reward_experiment', 'scoring',
                  ['scoring', 'scoring,checkpoints'],
                  'Reward to be used for training.')
flags.DEFINE_enum('policy', 'cnn', ['cnn', 'lstm', 'mlp', 'impala_cnn',
                                    'gfootball_impala_cnn'],
                  'Policy architecture')
flags.DEFINE_integer('num_timesteps', int(2e6),
                     'Number of timesteps to run for.')
flags.DEFINE_integer('num_envs', 8,
                     'Number of environments to run in parallel.')
flags.DEFINE_integer('nsteps', 5, 'Number of environment steps per epoch for A2C.')
flags.DEFINE_integer('save_interval', 100,
                     'How frequently checkpoints are saved.')
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_float('lr', 0.0007, 'Learning rate')
flags.DEFINE_float('ent_coef', 0.01, 'Entropy coefficient')
flags.DEFINE_float('vf_coef', 0.5, 'Value function coefficient')
flags.DEFINE_float('gamma', 0.99, 'Discount factor')
flags.DEFINE_float('max_grad_norm', 0.5, 'Max gradient norm (clipping)')
flags.DEFINE_bool('render', False, 'If True, environment rendering is enabled.')
flags.DEFINE_bool('dump_full_episodes', False,
                  'If True, trace is dumped after every episode.')
flags.DEFINE_bool('dump_scores', False,
                  'If True, sampled traces after scoring are dumped.')
flags.DEFINE_string('load_path', None, 'Path to load initial checkpoint from.')
flags.DEFINE_string('save_path', None, 'Path to save the model.')
flags.DEFINE_bool('write_video', False, 'If True, writing a video.')

log_dir = "/opt/ml/model/logs"
model_dir = "/opt/ml/model"

logger.configure(dir=log_dir, format_strs=['stdout', 'tensorboard'])

print(f"log_dir:{logger.get_dir()}")

if not os.path.exists(model_dir):
  os.makedirs(model_dir)

def create_single_football_env(iprocess):
  """Creates gfootball environment."""
  env = football_env.create_environment(
      env_name=FLAGS.level, stacked=('stacked' in FLAGS.state),
      rewards=FLAGS.reward_experiment,
      logdir=logger.get_dir(),
      write_goal_dumps=FLAGS.dump_scores and (iprocess == 0),
      write_full_episode_dumps=FLAGS.dump_full_episodes and (iprocess == 0),
      render=FLAGS.render and (iprocess == 0),
      dump_frequency=50 if FLAGS.render and iprocess == 0 else 0)
  env = monitor.Monitor(env, logger.get_dir() and os.path.join(logger.get_dir(),
                                                               str(iprocess)))
  return env


def train(_):
  """Trains an A2C policy."""
  vec_env = SubprocVecEnv([
      (lambda _i=i: create_single_football_env(_i))
      for i in range(FLAGS.num_envs)
  ], context=None)

  # Import tensorflow after we create environments. TF is not fork sake, and
  # we could be using TF as part of environment if one of the players is
  # controlled by an already trained model.
  import tensorflow.compat.v1 as tf
  ncpu = multiprocessing.cpu_count()
  config = tf.ConfigProto(allow_soft_placement=True,
                          intra_op_parallelism_threads=ncpu,
                          inter_op_parallelism_threads=ncpu)
  config.gpu_options.allow_growth = True
  tf.Session(config=config).__enter__()

  model = a2c.learn(network=FLAGS.policy,
             env=vec_env,
             seed=FLAGS.seed,
             nsteps=FLAGS.nsteps,
             total_timesteps=FLAGS.num_timesteps,
             vf_coef=FLAGS.vf_coef,
             ent_coef=FLAGS.ent_coef,
             max_grad_norm=FLAGS.max_grad_norm,
             lr=FLAGS.lr,
             gamma=FLAGS.gamma,
             log_interval=1,
             load_path=FLAGS.load_path)

  if FLAGS.save_path:
    os.makedirs(FLAGS.save_path, exist_ok=True)
    model.save(f"{FLAGS.save_path}/a2c_{FLAGS.num_timesteps}_model")
    print(f"Model saved at {FLAGS.save_path}/a2c_{FLAGS.num_timesteps}_model")

if len(sys.argv) == 1:
    hyperparameters_file = os.path.join(os.path.dirname(__file__), 'hyperparameters.txt')
    if os.path.exists(hyperparameters_file):
        with open(hyperparameters_file, 'r') as f:
            additional_args = [
                    f"--{line.strip()}" for line in f 
                    if line.strip() and not line.strip().startswith("#")
            ]
            sys.argv.extend(additional_args)
    else:
        print(f"{hyperparameters_file} does NOT exist")

    print("add hyperparameters.txt to sys.argv")
    print("sys.argv:", sys.argv)

if __name__ == '__main__':
  app.run(train)
