# ----------------------------
# ✅ Imports and global config
# ----------------------------
import os
import gymnasium as gym
import gymnasium_robotics
import numpy as np
from datetime import datetime

from sb3_contrib import TQC
from stable_baselines3 import HerReplayBuffer
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import EvalCallback

# 🔧 Global constants
ENV_ID = "HandManipulateBlockRotateXYZ_ContinuousTouchSensors-v1"
NUM_ENVS = 16
SEED = 42
TOTAL_TIMESTEPS = 15_000_000

# Logging path
log_dir = f"./logs/tqc_her_shadowhand_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
os.makedirs(log_dir, exist_ok=True)

# ----------------------------
# ✅ Main training logic
# ----------------------------
def make_env(rank):
    def _init():
        env = gym.make(ENV_ID, reward_type="sparse")
        env.reset(seed=SEED + rank)
        return Monitor(env)
    return _init


if __name__ == "__main__":
    # Vectorized training environment
    train_env = SubprocVecEnv([make_env(i) for i in range(NUM_ENVS)])

    # Evaluation environment (single env)
    eval_env = make_env(999)()

    # Action noise
    n_actions = train_env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.2 * np.ones(n_actions))

    # Define the TQC + HER model
    model = TQC(
        policy="MultiInputPolicy",
        env=train_env,
        learning_rate=1e-3,
        buffer_size=1_000_000,
        learning_starts=50_000,
        batch_size=256,
        gamma=0.98,
        tau=0.05,
        train_freq=1,
        gradient_steps=1,
        action_noise=action_noise,
        replay_buffer_class=HerReplayBuffer,
        replay_buffer_kwargs=dict(
            n_sampled_goal=4,
            goal_selection_strategy="future",
        ),
        verbose=1,
        tensorboard_log=log_dir,
        policy_kwargs=dict(
            net_arch=[256, 256, 256],
        ),
        seed=SEED,
    )

    # Evaluation callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=log_dir,
        log_path=log_dir,
        eval_freq=10_000,
        n_eval_episodes=10,
        deterministic=True,
        render=False,
    )

    # Start training
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=eval_callback)

    # Save the final model
    model.save(os.path.join(log_dir, "tqc_her_shadowhand_final"))
