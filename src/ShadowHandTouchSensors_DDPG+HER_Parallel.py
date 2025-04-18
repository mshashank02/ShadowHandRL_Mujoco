import os
import gymnasium as gym
import gymnasium_robotics
import numpy as np
import pandas as pd
from stable_baselines3 import DDPG, HerReplayBuffer
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
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


# === Environment Factory ===
def make_env():
    def _init():
        env = gym.make("HandManipulateBlockRotateXYZ_ContinuousTouchSensors-v1")
        env = ClipObservation(env)
        env = ClipAction(env)
        return env
    return _init

def make_eval_env():
    env = gym.make("HandManipulateBlockRotateXYZ_ContinuousTouchSensors-v1")
    env = ClipObservation(env)
    env = ClipAction(env)
    env = Monitor(env)  # ✅ Needed for success tracking
    return env


# === Success Evaluation Callback ===
class SuccessEvalCallback(EvalCallback):
    def __init__(self, eval_env, eval_freq, log_path, n_eval_episodes=10, **kwargs):
        eval_env = Monitor(eval_env)
        super().__init__(eval_env, eval_freq=eval_freq, log_path=log_path,
                         n_eval_episodes=n_eval_episodes, **kwargs)
        self.success_rates = []

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            _, _, ep_infos = evaluate_policy(
                self.model,
                self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                render=False,
                deterministic=self.deterministic,
                return_episode_rewards=True
            )

            success_values = [ep_info.get("is_success", 0.0) for ep_info in ep_infos]
            success_rate = np.mean(success_values)
            self.success_rates.append(success_rate)
            print(f"✅ [Eval] Success rate at step {self.n_calls}: {success_rate:.3f}")
        return True

    def _on_training_end(self):
        df = pd.DataFrame({
            "step": np.arange(len(self.success_rates)) * self.eval_freq,
            "success_rate": self.success_rates
        })
        os.makedirs(self.log_path, exist_ok=True)
        df.to_csv(os.path.join(self.log_path, "success_rates.csv"), index=False)


# === Main Training Script ===
if __name__ == "__main__":
    import multiprocessing
    multiprocessing.set_start_method("fork", force=True)

    # Parallel training envs
    num_envs = 2
    train_env = SubprocVecEnv([make_env() for _ in range(num_envs)])

    # Evaluation environment (single, wrapped with Monitor)
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
        learning_starts=10000,
        verbose=1,
        tensorboard_log="./logs/shadowhand_ddpg_her/",
        policy_kwargs=dict(
            net_arch=[256, 256, 256],
            activation_fn=nn.ReLU
        )
    )

    eval_callback = SuccessEvalCallback(
        eval_env=eval_env,
        eval_freq=190_000,  # One epoch equivalent
        log_path="./logs/eval_success",
        n_eval_episodes=10,
        deterministic=True,
        render=False
    )

    model.learn(total_timesteps=60_000_000, callback=eval_callback)
    model.save("ShadowHandTouchSensors_RL/src/model/ddpg_her_shadowhand_vec")
