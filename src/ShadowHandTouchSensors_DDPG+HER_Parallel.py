import os
import gymnasium as gym
import gymnasium_robotics
import numpy as np
import pandas as pd
from stable_baselines3 import DDPG, HerReplayBuffer
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.callbacks import BaseCallback
from torch import nn
from gymnasium import ObservationWrapper, ActionWrapper
from gymnasium.spaces import Box, Dict


# Register robotics environments
gym.register_envs(gymnasium_robotics)


# === Observation and Action Wrappers ===
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
        env = gym.make("HandManipulateBlockRotateXYZ-v1")
        assert isinstance(env.observation_space, Dict), "Environment must use Dict observation space for HER."
        env = ClipObservation(env)
        env = ClipAction(env)
        return env
    return _init


# === Success Rate Evaluation Callback ===
class SuccessEvalCallback(EvalCallback):
    def __init__(self, eval_env, eval_freq, log_path, **kwargs):
        super().__init__(eval_env, eval_freq=eval_freq, log_path=log_path, **kwargs)
        self.success_rates = []

    def _on_step(self) -> bool:
        result = super()._on_step()

        if self.n_calls % self.eval_freq == 0:
            successes = self.last_eval_info.get("is_success", [])
            if isinstance(successes, (list, np.ndarray)):
                success_rate = np.mean(successes)
                self.success_rates.append(success_rate)
                print(f"✅ Success Rate at step {self.num_timesteps}: {success_rate:.3f}")

        return result

    def _on_training_end(self):
        df = pd.DataFrame({
            "step": np.arange(len(self.success_rates)) * self.eval_freq,
            "success_rate": self.success_rates
        })
        os.makedirs(self.log_path, exist_ok=True)
        df.to_csv(os.path.join(self.log_path, "success_rates.csv"), index=False)


# === Main Script ===
if __name__ == "__main__":
    import multiprocessing
    multiprocessing.set_start_method("fork", force=True)

    num_envs = 32
    train_env = SubprocVecEnv([make_env() for _ in range(num_envs)])
    eval_env = SubprocVecEnv([make_env()])
    
    # Action noise
    n_actions = train_env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.2 * np.ones(n_actions))

    # DDPG + HER
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
        learning_starts=10000,  # ✅ Prevent HER crash at start
        verbose=1,
        tensorboard_log="./logs/shadowhand_ddpg_her/",
        policy_kwargs=dict(
            net_arch=[256, 256, 256],
            activation_fn=nn.ReLU
        )
    )

    # Eval callback (per epoch = 190k steps)
    eval_callback = SuccessEvalCallback(
        eval_env=eval_env,
        eval_freq=190_000,
        log_path="./logs/eval_success",
        n_eval_episodes=10,
        deterministic=True,
        render=False
    )

    # Train for 57 million timesteps = 300 epochs
    model.learn(total_timesteps=57_000_000, callback=eval_callback)

    # Save final model
    model.save("ShadowHandTouchSensors_RL/src/model/ddpg_her_shadowhand_vec")
