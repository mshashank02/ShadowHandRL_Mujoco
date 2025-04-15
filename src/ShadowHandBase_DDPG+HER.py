import gymnasium as gym
import gymnasium_robotics
gym.register_envs(gymnasium_robotics)

import numpy as np
from stable_baselines3 import DDPG, HerReplayBuffer
from stable_baselines3.common.noise import NormalActionNoise
from torch import nn

# Load Shadow Hand env
env_id = "HandManipulateBlockRotateXYZ-v1"
env = gym.make(env_id)

# Action noise (Gaussian std = 0.2)
n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.2 * np.ones(n_actions))

# Define model
model = DDPG(
    policy="MultiInputPolicy",
    env=env,
    replay_buffer_class=HerReplayBuffer,
    replay_buffer_kwargs=dict(
        n_sampled_goal=4,
        goal_selection_strategy="future"
    ),
    buffer_size=int(1e6),              # ✅ Passed here (not inside kwargs!)
    action_noise=action_noise,
    learning_rate=1e-3,
    batch_size=256,
    gamma=0.98,
    tau=0.05,
    verbose=1,
    tensorboard_log="./logs/shadowhand_ddpg_her/",
    policy_kwargs=dict(
        net_arch=[256, 256, 256],
        activation_fn=nn.ReLU
    )
)

# Train
model.learn(total_timesteps=1_000_000)
model.save("ShadowHandTouchSensors_RL/src/model/ddpg_her_shadowhand")

