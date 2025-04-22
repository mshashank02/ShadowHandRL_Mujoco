import gymnasium as gym
import gymnasium_robotics
import numpy as np
from stable_baselines3 import DDPG, HerReplayBuffer
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.monitor import Monitor
from torch import nn

# === Register robotics environments ===
gym.register_envs(gymnasium_robotics)

# === Debug Test Script: HER + DDPG on FetchReach ===

# Setup a simple goal-based env
env_id = "FetchReach-v3"  # much easier than ShadowHand

# Create and wrap env
env = gym.make(env_id, reward_type="sparse")
env = Monitor(env)
train_env = DummyVecEnv([lambda: env])

# Check observation space structure
obs, _ = env.reset()
print("\n✅ Observation Keys:", list(obs.keys()))

# Confirm the action and goal space
print("Action space:", env.action_space)
print("Observation['observation'] sample:", obs['observation'][:5])
print("Observation['desired_goal'] sample:", obs['desired_goal'])

# Check random policy baseline
successes = []
obs, _ = env.reset()
done = False
while not done:
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    successes.append(info.get("is_success", 0.0))
print("\n🔍 Random Policy Success:", np.mean(successes))

# Define action noise
n_actions = train_env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.2 * np.ones(n_actions))

# Setup DDPG + HER model
model = DDPG(
    policy="MultiInputPolicy",
    env=train_env,
    replay_buffer_class=HerReplayBuffer,
    replay_buffer_kwargs=dict(
        n_sampled_goal=4,
        goal_selection_strategy="future"
    ),
    buffer_size=100_000,
    batch_size=256,
    gamma=0.98,
    tau=0.05,
    action_noise=action_noise,
    learning_rate=1e-3,
    learning_starts=1000,
    verbose=1,
    policy_kwargs=dict(
        net_arch=[256, 256],
        activation_fn=nn.ReLU
    )
)

# Train a little and log
model.learn(total_timesteps=100000)

# Evaluate success rate
successes = []
for ep in range(5):
    obs, _ = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        successes.append(info.get("is_success", 0.0))
print("\n📊 Success rate after 5k steps:", np.mean(successes))
