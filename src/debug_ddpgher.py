import os
import gymnasium as gym
import gymnasium_robotics
import numpy as np
import pandas as pd
import torch.nn as nn
from stable_baselines3 import DDPG, HerReplayBuffer
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from gymnasium import ObservationWrapper, ActionWrapper
from gymnasium.spaces import Box, Dict
from datetime import datetime

# === Register robotics environments ===
gym.register_envs(gymnasium_robotics)

# === Observation and Action Clipping Wrappers ===
class ClipObservation(ObservationWrapper):
    def __init__(self, env, low=-200, high=200):
        super().__init__(env)
        self.low = low
        self.high = high
        assert isinstance(env.observation_space, Dict), "HER requires Dict observation space"
        self.observation_space = env.observation_space

    def observation(self, observation):
        observation["observation"] = np.clip(observation["observation"], self.low, self.high)
        return observation

class ClipAction(ActionWrapper):
    def __init__(self, env, low=-5.0, high=5.0):
        super().__init__(env)
        self.low = low
        self.high = high
        self.action_space = Box(low, high, shape=env.action_space.shape, dtype=np.float32)

    def action(self, action):
        return np.clip(action, self.low, self.high)

# === Environment Factory ===
def make_env():
    def _init():
        env = gym.make("FetchSlide-v3")
        env = ClipObservation(env)
        env = ClipAction(env)
        env = Monitor(env)
        return env
    return _init

# === Success Rate Evaluation Callback ===
class SuccessEvalCallback(BaseCallback):
    def __init__(self, eval_env, eval_freq, log_path, n_eval_episodes=10, deterministic=True, save_freq=None, model_prefix="ddpg_her_shadowhand", verbose=0):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.log_path = log_path
        self.n_eval_episodes = n_eval_episodes
        self.deterministic = deterministic
        self.success_rates = []
        self.save_freq = save_freq
        self.model_prefix = model_prefix

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            successes = []
            for _ in range(self.n_eval_episodes):
                obs, _ = self.eval_env.reset()
                done = False
                while not done:
                    action, _states = self.model.predict(obs, deterministic=self.deterministic)
                    obs, _, terminated, truncated, info = self.eval_env.step(action)
                    done = terminated or truncated
                    if isinstance(info, (list, tuple)):
                        info = info[0]
                    successes.append(info.get("is_success", 0.0))

            mean_success = np.mean(successes)
            self.success_rates.append((self.num_timesteps, mean_success))
            print(f"\n✅ [Eval] Step {self.num_timesteps}: Success Rate = {mean_success:.3f}")

            if self.save_freq and self.n_calls % self.save_freq == 0:
                save_path = os.path.join(self.log_path, f"{self.model_prefix}_step{self.num_timesteps}.zip")
                self.model.save(save_path)

        return True

    def _on_training_end(self):
        os.makedirs(self.log_path, exist_ok=True)
        df = pd.DataFrame(self.success_rates, columns=["step", "success_rate"])
        df.to_csv(os.path.join(self.log_path, "success_rates_debug.csv"), index=False)

# === Main Training Loop ===
if __name__ == "__main__":
    num_envs = 1  # Single environment for non-parallelized version
    total_timesteps = 57_000_000
    eval_freq = 190_000
    save_freq = 950_000  # every 5 epochs

    # Create training env (non-parallelized)
    train_env = DummyVecEnv([make_env()])

    # Evaluation env (not vectorized)
    eval_env = make_env()()

    # Action noise
    n_actions = train_env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.2 * np.ones(n_actions))

    # Timestamp for log folder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"./logs/ddpg_her_shadowhand_{timestamp}"
    os.makedirs(log_dir, exist_ok=True)

    # Model
    model = DDPG(
        policy="MultiInputPolicy",
        env=train_env,
        replay_buffer_class=HerReplayBuffer,
        replay_buffer_kwargs=dict(
            n_sampled_goal=4,
            goal_selection_strategy="future"
        ),
        buffer_size=int(1e6),
        action_noise=action_noise,
        learning_rate=1e-3,
        batch_size=256,
        gamma=0.98,
        tau=0.05,
        learning_starts=32_000,
        verbose=1,
        tensorboard_log=log_dir,
        policy_kwargs=dict(
            net_arch=[256, 256, 256],
            activation_fn=nn.ReLU
        )
    )

    # Eval callback
    eval_callback = SuccessEvalCallback(
        eval_env=eval_env,
        eval_freq=eval_freq,
        log_path=log_dir,
        n_eval_episodes=10,
        deterministic=True,
        save_freq=save_freq,
        model_prefix="ddpg_her_shadowhand"
    )
    eval_callback.model = model
    _ = eval_env.reset()  # ✅ Reset so that it's ready to be stepped
    print("🔍 Running pre-training evaluation...")
    eval_callback._on_step()

    # Train
    model.learn(total_timesteps=total_timesteps, callback=eval_callback)

    # Save final model
    model.save(os.path.join(log_dir, "ddpg_her_shadowhand_final"))
