
import os
import gymnasium as gym
import gymnasium_robotics
import numpy as np
import pandas as pd
import torch.nn as nn
from stable_baselines3 import DDPG, HerReplayBuffer
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize, VecEnvWrapper, DummyVecEnv
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from datetime import datetime

gym.register_envs(gymnasium_robotics)

def make_env(rank):
    def _init():
        env = gym.make("HandManipulateBlockRotateXYZ_ContinuousTouchSensors-v1", reward_type="sparse")
        env.reset(seed=rank + int(datetime.now().timestamp()) % 10000)
        env = Monitor(env)
        return env
    return _init

class DiverseNoiseWrapper(VecEnvWrapper):
    def __init__(self, venv, base_sigma=0.2):
        super().__init__(venv)
        self.env_count = venv.num_envs
        self.noises = [
            NormalActionNoise(
                mean=np.zeros(self.action_space.shape[-1]),
                sigma=np.random.uniform(0.8, 1.2) * base_sigma * np.ones(self.action_space.shape[-1])
            )
            for _ in range(self.env_count)
        ]
        self.actions = None

    def step_async(self, actions):
        self.actions = actions
        self.venv.step_async(actions)

    def step_wait(self):
        obs, rewards, dones, infos = self.venv.step_wait()
        if self.actions is not None:
            noisy_actions = np.stack([
                np.clip(self.actions[i] + self.noises[i](), self.action_space.low, self.action_space.high)
                for i in range(self.env_count)
            ])
            self.actions = noisy_actions
        return obs, rewards, dones, infos

    def reset(self, **kwargs):
        return self.venv.reset(**kwargs)

class SuccessEvalCallback(BaseCallback):
    def __init__(self, eval_env, eval_freq, log_path, n_eval_episodes=50, deterministic=True, save_freq=None, model_prefix="ddpg_her_shadowhand", verbose=0):
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
                obs = self.eval_env.reset()
                done = False
                while not done:
                    action, _ = self.model.predict(obs, deterministic=self.deterministic)
                    obs, _, dones, infos = self.eval_env.step(action)
                    done = dones[0]
                    info = infos[0]
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
        df.to_csv(os.path.join(self.log_path, "success_rates_touch_{timestamp}.csv"), index=False)

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.set_start_method("forkserver", force=True)

    num_envs = 2
    total_timesteps = 57_000_000
    eval_freq = 10_000
    save_freq = 950_000

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"./logs/ddpg_her_shadowhand_touch_{timestamp}"
    os.makedirs(log_dir, exist_ok=True)

    # Training env
    raw_env = SubprocVecEnv([make_env(i) for i in range(num_envs)])
    diverse_env = DiverseNoiseWrapper(raw_env)
    train_env = VecNormalize(diverse_env, norm_obs=True, norm_reward=False, clip_obs=200.0)

    # Eval env
    eval_raw_env = make_env(9999)()
    eval_env = DummyVecEnv([lambda: eval_raw_env])
    eval_env = VecNormalize(eval_env, training=False, norm_obs=True, norm_reward=False)
    eval_env.reset()

    n_actions = train_env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.2 * np.ones(n_actions))

    model = DDPG(
        policy="MultiInputPolicy",
        env=train_env,
        replay_buffer_class=HerReplayBuffer,
        replay_buffer_kwargs=dict(n_sampled_goal=4, goal_selection_strategy="future"),
        buffer_size=int(1e6),
        action_noise=action_noise,
        learning_rate=1e-4,
        batch_size=256,
        gamma=0.98,
        tau=0.05,
        learning_starts=50_000,
        verbose=1,
        tensorboard_log=log_dir,
        policy_kwargs=dict(net_arch=[256, 256, 256], activation_fn=nn.ReLU)
    )

    eval_callback = SuccessEvalCallback(
        eval_env=eval_env,
        eval_freq=eval_freq,
        log_path=log_dir,
        n_eval_episodes=50,
        deterministic=True,
        save_freq=save_freq,
        model_prefix="ddpg_her_shadowhand_touch"
    )
    eval_callback.model = model
    eval_callback._on_step()

    model.learn(total_timesteps=total_timesteps, callback=eval_callback)
    model.save(f"./model/ddpg_her_shadowhand_touch_final")
    train_env.save(f"./models/vecnormalize_train_touch.pkl")
