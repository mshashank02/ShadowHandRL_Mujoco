import os
import gymnasium as gym
import gymnasium_robotics
import numpy as np
import pandas as pd
from stable_baselines3 import DDPG, HerReplayBuffer
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
from torch import nn
from gymnasium import ObservationWrapper, ActionWrapper
from gymnasium.spaces import Box, Dict

# === Register robotics environments ===
gym.register_envs(gymnasium_robotics)

# === Custom Wrappers ===
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

# === Training Environment Factory ===
def make_env():
    def _init():
        env = gym.make("HandManipulateBlockRotateXYZ_ContinuousTouchSensors-v1")
        env = ClipObservation(env)
        env = ClipAction(env)
        return env
    return _init

# === Evaluation Environment ===
def make_eval_env():
    env = gym.make("HandManipulateBlockRotateXYZ_ContinuousTouchSensors-v1")
    env = ClipObservation(env)
    env = ClipAction(env)
    env = Monitor(env)
    return env

# === Custom Evaluation Callback with Success Logging ===
from stable_baselines3.common.callbacks import BaseCallback

class SuccessEvalCallback(BaseCallback):
    def __init__(self, eval_env, eval_freq, log_path, n_eval_episodes=10, deterministic=True, verbose=0):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.log_path = log_path
        self.n_eval_episodes = n_eval_episodes
        self.deterministic = deterministic
        self.success_rates = []

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            successes = []
            for _ in range(self.n_eval_episodes):
                obs = self.eval_env.reset()
                done = False
                while not done:
                    action, _ = self.model.predict(obs, deterministic=self.deterministic)
                    obs, _, done, info = self.eval_env.step(action)
                    if isinstance(info, list):
                        info = info[0]
                    if "is_success" in info:
                        successes.append(info["is_success"])
            mean_success = np.mean(successes)
            self.success_rates.append(mean_success)
            print(f"✅ [Eval] Success rate at step {self.n_calls}: {mean_success:.3f}")
        return True

    def _on_training_end(self):
        df = pd.DataFrame({
            "step": np.arange(len(self.success_rates)) * self.eval_freq,
            "success_rate": self.success_rates
        })
        os.makedirs(self.log_path, exist_ok=True)
        df.to_csv(os.path.join(self.log_path, "success_rates.csv"), index=False)

# === Main ===
if __name__ == "__main__":
    import multiprocessing
    multiprocessing.set_start_method("fork", force=True)

    num_envs = 2
    train_env = SubprocVecEnv([make_env() for _ in range(num_envs)])
    eval_env = make_eval_env()

    n_actions = train_env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.2 * np.ones(n_actions))

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
        learning_starts=10_000,
        verbose=1,
        tensorboard_log="./logs/shadowhand_ddpg_her/",
        policy_kwargs=dict(
            net_arch=[256, 256, 256],
            activation_fn=nn.ReLU
        )
    )

    eval_callback = SuccessEvalCallback(
        eval_env=eval_env,
        eval_freq=190_000,
        log_path="./logs/eval_success",
        n_eval_episodes=10,
        deterministic=True,
        verbose=1
    )

    model.learn(total_timesteps=60_000_000, callback=eval_callback)
    model.save("ShadowHandTouchSensors_RL/src/model/ddpg_her_shadowhand_vec")
