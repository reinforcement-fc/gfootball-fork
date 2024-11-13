"""Runs football_env on OpenAI's ACKTR."""

# Import compatibility for Python 2/3 features (division, print function, and absolute imports)
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Import required libraries
import os
import gym
import cv2
import csv
import sys
import subprocess
import multiprocessing
import numpy as np
from absl import app, flags
from baselines import logger
from baselines.bench import monitor
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.acktr import acktr
import gfootball.env as football_env
from gfootball.examples import models  
from datetime import datetime
import matplotlib.pyplot as plt

# Set directories for logging, model checkpoints, and videos
log_dir = "/opt/ml/model/logs"
model_dir = "/opt/ml/model"
video_dir = log_dir # "/opt/ml/model/videos"
checkpoint_dir = "/opt/ml/model/checkpoints"  # New directory for checkpoints

logger.configure(dir=log_dir)

print(f"log_dir:{logger.get_dir()}")  # Confirm the log directory location

# Ensure necessary directories exist
os.makedirs(model_dir, exist_ok=True)
os.makedirs(video_dir, exist_ok=True)
os.makedirs(checkpoint_dir, exist_ok=True)  # Create checkpoint directory if not exists

# Initialize flag system for custom arguments
FLAGS = flags.FLAGS

# Define command-line flags
flags.DEFINE_string('level', 'academy_empty_goal_close', 'Defines the environment level.')
flags.DEFINE_enum('state', 'extracted', ['extracted', 'extracted_stacked'], 'Observation type.')
flags.DEFINE_enum('reward_experiment', 'scoring', ['scoring', 'scoring,checkpoints'], 'Reward type.')
flags.DEFINE_integer('num_timesteps', int(2e6), 'Total training steps.')
flags.DEFINE_float('lr', 0.00025, 'Learning rate for ACKTR.')
flags.DEFINE_integer('num_envs', 8, 'Number of environments to run in parallel.')
flags.DEFINE_integer('save_interval', 100_000, 'Save model every this many steps.')
flags.DEFINE_string('save_path', model_dir, 'Path to save the model checkpoints.')
flags.DEFINE_bool('render', True, 'Enable rendering.')
flags.DEFINE_bool('dump_full_episodes', True,
                  'If True, trace is dumped after every episode.')
flags.DEFINE_bool('dump_scores', True,
                  'If True, sampled traces after scoring are dumped.')
flags.DEFINE_string('load_path', None, 'Path to load initial checkpoint from.')
flags.DEFINE_integer('checkpoint_interval', 100_000, 'Save model every this many steps.')

# Print flag values to confirm configuration
def print_flags():
    print("Starting with the following flag values:")
    for flag_name in FLAGS:
        print(f"{flag_name}: {FLAGS[flag_name].value}")

# Create a single football environment
def create_single_football_env(iprocess=0):
    """Creates a single football environment."""
    env = football_env.create_environment(
        env_name=FLAGS.level,
        stacked=('stacked' in FLAGS.state),
        rewards=FLAGS.reward_experiment,
        logdir=logger.get_dir(),
        render=FLAGS.render,
        write_full_episode_dumps=FLAGS.dump_full_episodes and (iprocess == 0),
        write_goal_dumps=FLAGS.dump_scores and (iprocess == 0),
        dump_frequency=50 if FLAGS.render else 0
    )
        # Add a monitor wrapper to log environment progress and allow video recording
    env = monitor.Monitor(env, logger.get_dir() and os.path.join(logger.get_dir(), str(iprocess)), allow_early_resets=True)
    env = VideoCaptureWrapper(env, video_dir=video_dir, record_frequency=50)  # Video recording wrapper, record_frequency is by the number of episodes
    return env

# Define a wrapper class to record video for specific episodes
class VideoCaptureWrapper(gym.Wrapper):
    """Wrapper to capture video for specific episodes."""
    def __init__(self, env, video_dir, record_frequency=50):
        super(VideoCaptureWrapper, self).__init__(env)
        self.video_dir = video_dir
        self.record_frequency = record_frequency
        self.episode_id = 0
        self.recording = False
        self.video_writer = None
        
        # Ensure the video directory exists
        print(f"Video dir: {self.video_dir}")
        os.makedirs(self.video_dir, exist_ok=True)

    # Override reset to set up video recording for specified episodes
    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self.episode_id += 1

        if self.episode_id % self.record_frequency == 0:
            self.recording = True
            video_path = os.path.join(self.video_dir, f"episode_{self.episode_id}.mp4")
            print(f"Preparing to save video at {video_path}")
            
            frame = self.env.render(mode='rgb_array')
            if frame is not None:
                frame_shape = frame.shape[:2]
                self.video_writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (frame_shape[1], frame_shape[0]))
                print(f"Started recording video for episode {self.episode_id} at {video_path}")
            else:
                print("Warning: No frame captured for video recording. Ensure `render` mode is enabled.")
                self.recording = False
        else:
            self.recording = False
        return obs

    # Override step to write each frame if recording is active
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        if self.recording:
            frame = self.env.render(mode='rgb_array')
            if frame is not None:
                self.video_writer.write(frame)
            else:
                print("Warning: Failed to capture frame during recording.")
        if done and self.recording:
            if self.video_writer is not None:
                self.video_writer.release()
                print(f"Stopped recording for episode {self.episode_id}. Video saved successfully.")
            else:
                print("Error: Video writer was not initialized properly.")
            self.recording = False
            self.video_writer = None
            
        return obs, reward, done, info

# Check if a GPU is available using nvidia-smi
def check_gpu():
    try:
        result = subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
        if result.returncode == 0:
            print("GPU is available. Detected GPUs:")
            print(result.stdout)
            return True
        else:
            print("No GPU available, running on CPU.")
            return False
    except FileNotFoundError:
        print("No GPU available or nvidia-smi command not found.")
        return False

# Define the training function
def train(_):
    """Trains an ACKTR policy."""
    print("RUN_ACKTR start.")
    print_flags()

    gpu_available = check_gpu()  # Confirm GPU availability
    
    # Initialize a dummy vectorized environment for parallelism
    env = DummyVecEnv([lambda: create_single_football_env()])

    # Configure TensorFlow session for ACKTR
    import tensorflow.compat.v1 as tf
    ncpu = multiprocessing.cpu_count()
    config = tf.ConfigProto(allow_soft_placement=True, intra_op_parallelism_threads=ncpu, inter_op_parallelism_threads=ncpu)
    config.gpu_options.allow_growth = True
    tf.Session(config=config).__enter__()

    # Configure and train the model with ACKTR
    '''
    model = acktr.learn(
        network='cnn',
        env=env,
        seed=0,
        total_timesteps=FLAGS.num_timesteps,
        lr=FLAGS.lr,
        gamma=0.99,
        log_interval=10,
        callback=save_callback
    )
    '''
    # Initialize training parameters
    total_timesteps = FLAGS.num_timesteps
    checkpoint_interval = FLAGS.checkpoint_interval
    timesteps_per_loop = checkpoint_interval  # How many steps each training loop will run

    # Load the model if needed (optional)
    model = None
    current_timesteps = 0

    # Custom training loop for manual checkpointing
    while current_timesteps < total_timesteps:
        # Train for checkpoint interval steps or the remaining steps if close to the total
        steps_to_train = min(timesteps_per_loop, total_timesteps - current_timesteps)
        print(f"Training for {steps_to_train} steps...")
        
        # Apply variable scope with auto reuse to avoid variable duplication issues
        with tf.variable_scope('acktr_model', reuse=tf.AUTO_REUSE):
            model = acktr.learn(
                network='cnn',
                env=env,
                seed=0,
                total_timesteps=steps_to_train,
                lr=FLAGS.lr,
                gamma=0.99,
                log_interval=30
            )
            
        

        current_timesteps += steps_to_train

        # Save checkpoint after each interval
        checkpoint_path = os.path.join(checkpoint_dir, f"acktr_checkpoint_{current_timesteps}.pkl")
        #model.summary()
        model.save(checkpoint_path)
        print(f"Checkpoint saved at {checkpoint_path}")
    
    # Save the final model
    if FLAGS.save_path:
        model.save(f"{FLAGS.save_path}/acktr_final_model.pkl")
        print(f"Final model saved at {FLAGS.save_path}/acktr_final_model.pkl")
        
    plot_rewards()  # Plot training rewards after training

# Callback for saving checkpoints and logging rewards
def save_callback(local_vars, global_vars):
    """Callback function to save model checkpoints and log rewards."""
    step_count = local_vars.get('t')
    if step_count and step_count % FLAGS.save_interval == 0:
        checkpoint_path = os.path.join(checkpoint_dir, f"acktr_checkpoint_{step_count}.pkl")
        local_vars['act'].save(checkpoint_path)
        print(f"Checkpoint saved at {checkpoint_path}")

# Function to plot training rewards
def plot_rewards():
    monitor_file = os.path.join(log_dir, "progress.csv")
    episodes, rewards = [], []
    with open(monitor_file, mode='r') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)  # Skip metadata row
        for i, row in enumerate(csv_reader):
            if row[1] != 'NaN' and row[1]:
                episodes.append(i)
                rewards.append(float(row[1]))

    plt.figure(figsize=(10, 6))
    plt.plot(episodes, rewards, label='Episode Reward')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Training Rewards over Episodes')
    plt.legend()
    plt.grid()
    plt_path = os.path.join(log_dir, f"training_rewards_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    plt.savefig(plt_path)
    plt.show()
    print(f"Reward plot saved at {plt_path}")

# Load additional hyperparameters from file if present
print("sys.argv:", sys.argv)
if len(sys.argv) == 1:
    hyperparameters_file = os.path.join(os.path.dirname(__file__), 'hyperparameters.txt')
    if os.path.exists(hyperparameters_file):
        with open(hyperparameters_file, 'r') as f:
            additional_args = [f"--{line.strip()}" for line in f if line.strip() and not line.strip().startswith("#")]
            sys.argv.extend(additional_args)
    else:
        print(f"{hyperparameters_file} does NOT exist")
    print("add hyperparameters.txt to sys.argv")
    print("sys.argv:", sys.argv)

# Run training
if __name__ == '__main__':
    app.run(train)