# coding=utf-8
"""Runs football_env on OpenAI's deepq (DQN)."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import multiprocessing
import os
import sys
import cv2
import gym  # Ensure gym is imported
import numpy as np
import subprocess
from absl import app, flags
from baselines import logger
from baselines.bench import monitor
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.deepq import deepq
import gfootball.env as football_env
import tensorflow.compat.v1 as tf
from gfootball.examples import models  
#from gym.wrappers import RecordVideo
from datetime import datetime
import matplotlib.pyplot as plt

log_dir = "/opt/ml/model/logs"
model_dir = "/opt/ml/model"
video_dir = "/opt/ml/model/videos"
logger.configure(dir=log_dir)
print(f"log_dir:{logger.get_dir()}")

# Create directories if they don't exist
os.makedirs(model_dir, exist_ok=True)
os.makedirs(video_dir, exist_ok=True)

FLAGS = flags.FLAGS

# Define flags
flags.DEFINE_string('level', 'academy_empty_goal_close', 'Defines the environment level.')
flags.DEFINE_enum('state', 'extracted', ['extracted', 'extracted_stacked'], 'Observation type.')
flags.DEFINE_enum('reward_experiment', 'scoring', ['scoring', 'scoring,checkpoints'], 'Reward type.')
flags.DEFINE_integer('num_timesteps', int(2e6), 'Total training steps.')
flags.DEFINE_float('lr', 0.00025, 'Learning rate for DQN.')
flags.DEFINE_float('gamma', 0.99, 'Discount factor.')
flags.DEFINE_integer('buffer_size', 50000, 'Replay buffer size.')
flags.DEFINE_float('exploration_fraction', 0.1, 'Fraction of training time for exploration.')
flags.DEFINE_float('exploration_final_eps', 0.02, 'Final value of epsilon for epsilon-greedy exploration.')
flags.DEFINE_integer('train_freq', 4, 'Frequency of training steps.')
flags.DEFINE_integer('learning_starts', 1000, 'Steps before training starts.')
flags.DEFINE_integer('target_network_update_freq', 500, 'Frequency to update the target network.')
flags.DEFINE_integer('num_envs', 8, 'Number of environments to run in parallel.')
flags.DEFINE_integer('save_interval', 300_000, 'Save model every this many steps.')
flags.DEFINE_bool('dump_scores', False, 'Dump Scores'),
flags.DEFINE_string('save_path', model_dir, 'Path to save the model checkpoints.')
flags.DEFINE_bool('render', True, 'Enable rendering.')

def print_flags():
    print("Starting with the following flag values:")
    for flag_name in FLAGS:
        print(f"{flag_name}: {FLAGS[flag_name].value}")

        
def create_single_football_env(iprocess = 0):
    """Creates a single football environment."""
    env = football_env.create_environment(
        env_name=FLAGS.level,
        stacked=('stacked' in FLAGS.state),
        rewards=FLAGS.reward_experiment,
        logdir=logger.get_dir(),
        render=FLAGS.render)
    env = monitor.Monitor(env, logger.get_dir() and os.path.join(logger.get_dir(), str(iprocess)), allow_early_resets=True)
    
    # Set up manual video recording
    #if iprocess == 0:
    env = VideoCaptureWrapper(env, video_dir=video_dir, record_frequency=50)

    return env

class VideoCaptureWrapper(gym.Wrapper):
    """Wrapper to capture video for specific episodes."""
    def __init__(self, env, video_dir, record_frequency=50):
        super(VideoCaptureWrapper, self).__init__(env)
        self.video_dir = video_dir
        self.record_frequency = record_frequency
        self.episode_id = 0
        self.recording = False
        self.video_writer = None

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self.episode_id += 1

        # Start recording for specific episodes
        if self.episode_id % self.record_frequency == 0:
            self.recording = True
            video_path = os.path.join(self.video_dir, f"episode_{self.episode_id}.mp4")
            frame_shape = self.env.observation_space.shape[:2]
            self.video_writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (frame_shape[1], frame_shape[0]))
        else:
            self.recording = False

        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        # Record the current frame if recording
        if self.recording:
            frame = self.env.render(mode='rgb_array')
            self.video_writer.write(frame)

        # Stop recording at the end of the episode
        if done and self.recording:
            self.video_writer.release()
            self.recording = False

        return obs, reward, done, info
    
import subprocess

def check_gpu():
    try:
        result = subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)


        if result.returncode == 0:
            print("GPU is available. Detected GPUs:")
            print(result.stdout)  # GPU 정보 출력
            return True
        else:
            print("No GPU available, running on CPU.")
            print(result.stderr)  # 오류 메시지 출력 (필요 시)
            return False
    except FileNotFoundError:
        # nvidia-smi가 없는 경우
        print("No GPU available or nvidia-smi command not found.")
        return False


def train(_):
    """Trains a DQN policy."""
    print("RUN_DQN start.")
    print_flags()

    # GPU 상태를 체크하고 결과를 로그로 출력
    gpu_available = check_gpu()

    #vec_env = SubprocVecEnv([
    #  (lambda _i=i: create_single_football_env(_i))
    #  for i in range(FLAGS.num_envs)
    #  ], context=None)
    # Use a single environment wrapped in DummyVecEnv
    
    env = DummyVecEnv([lambda: create_single_football_env()])
        
    # Store training rewards for visualization
    reward_log = []
    step_log = []
    
    # Create the environment
    #env = DummyVecEnv([lambda: create_single_football_env()])

    # Configure and train the model
    model = deepq.learn(
        env=env,
        
        network='cnn',  # DQN typically uses CNN for image-based input
        lr=FLAGS.lr,
        total_timesteps=FLAGS.num_timesteps,
        buffer_size=FLAGS.buffer_size,
        exploration_fraction=FLAGS.exploration_fraction,
        exploration_final_eps=FLAGS.exploration_final_eps,
        train_freq=FLAGS.train_freq,
        learning_starts=FLAGS.learning_starts,
        target_network_update_freq=FLAGS.target_network_update_freq,
        gamma=FLAGS.gamma,
        print_freq=10,
        checkpoint_freq=FLAGS.save_interval,  # Set checkpointing frequency
        callback=lambda lcl, glb: save_callback(lcl, glb, reward_log, step_log)
    )
                
    # Save the model
    #if FLAGS.save_path:    
    #    os.makedirs(FLAGS.save_path, exist_ok=True)
    #    model.save(f"{FLAGS.save_path}/dqn_{FLAGS.num_timesteps}_model.pkl")
    #    print(f"Model saved at {FLAGS.save_path}/dqn_{FLAGS.num_timesteps}_model.pkl")
        # Save the final model
    if FLAGS.save_path:    
        model.save(f"{FLAGS.save_path}/dqn_final_model.pkl")
        print(f"Final model saved at {FLAGS.save_path}/dqn_final_model.pkl")

    # Plot training rewards
    plot_rewards(step_log, reward_log)

def save_callback(local_vars, global_vars, reward_log, step_log):
    """Callback function to save model checkpoints and log rewards."""
    if 't' in local_vars and 'episode_rewards' in local_vars:
        step_log.append(local_vars['t'])
        reward_log.append(np.mean(local_vars['episode_rewards'][-10:]))  # Log the average reward over the last 10 episodes

    # Checkpointing
    if FLAGS.save_path and local_vars['t'] % FLAGS.save_interval == 0:
        checkpoint_path = os.path.join(FLAGS.save_path, f"dqn_checkpoint_{local_vars['t']}.pkl")
        local_vars['act'].save(checkpoint_path)
        print(f"Checkpoint saved at {checkpoint_path}")

def plot_rewards(steps, rewards):
    """Plot rewards over training steps."""
    plt.figure(figsize=(10, 6))
    plt.plot(steps, rewards, label='Average Reward (last 10 episodes)')
    plt.xlabel('Steps')
    plt.ylabel('Reward')
    plt.title('Training Rewards over Time')
    plt.legend()
    plt.grid()
    plt_path = os.path.join(log_dir, f"training_rewards_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    plt.savefig(plt_path)
    plt.show()
    print(f"Reward plot saved at {plt_path}")
    
print("sys.argv:", sys.argv)

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