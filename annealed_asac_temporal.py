
from contextlib import ExitStack

from stable_baselines3.sac import SAC
from typing import Any, TypeVar, Union

import numpy as np
from configs import ASACTorchConfig
import torch as th
import gymnasium as gym
import gymnasium_robotics
gym.register_envs(gymnasium_robotics)
from stable_baselines3.common.monitor import Monitor

from stable_baselines3.common.vec_env import VecVideoRecorder

from torch.nn import functional as F

from stable_baselines3.common.buffers import ReplayBuffer, DictReplayBuffer
from stable_baselines3.common.utils import polyak_update, update_learning_rate
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv

from sb3_replay_buffers import CustomReplayBuffer, NStepReplayBuffer
from dict_replay_buffer import DictCustomReplayBuffer
SelfASAC = TypeVar("SelfASAC", bound="ASAC")
from copy import deepcopy
import os


from logger import Logger
USE_STANDARD_TRANSITION=False
METRICS = Logger()
RHO0 = 0.0
N_STEP = 1
HISTORY_STEPS = 50000
MIN_ZETA = 1e-3
def just_min(x, dim=1, keepdim=True):
    return th.min(x, dim=dim, keepdim=keepdim)[0]
import numpy as np
import torch as th
from stable_baselines3.common.buffers import ReplayBuffer
from typing import Optional, List, Union

import numpy as np
import torch as th
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.type_aliases import ReplayBufferSamples, DictReplayBufferSamples
from gymnasium import spaces

import numpy as np
import torch as th
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.type_aliases import ReplayBufferSamples

class ThermodynamicReplayBuffer(ReplayBuffer):
    def __init__(self, *args, n_steps: int = 1, gamma: float = 0.99, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_steps = n_steps
        self.gamma = gamma

    def _get_samples(self, batch_inds: np.ndarray, env=None) -> ReplayBufferSamples:
        return super()._get_samples(batch_inds, env) 

    # ----------------------------------------------------------------
    # NEW: Simple Continuous History Sampler
    # ----------------------------------------------------------------
    def sample_continuous_history(self, n_steps: int = 5000) -> th.Tensor:
        """
        Grabs the last 'n_steps' of rewards from the buffer as a single 
        continuous trajectory, ignoring episode boundaries.
        Returns: (n_envs, n_steps) tensor
        """
        # 1. Handle "Not enough data" case
        current_size = self.buffer_size if self.full else self.pos
        if current_size == 0:
            return th.tensor([], device=self.device)
            
        # Clip n_steps to available data
        lookback = min(n_steps, current_size)

        # 2. Compute Indices (Circular Buffer Logic)
        # np.arange creates indices [pos-lookback, ..., pos-1]
        # Modulo (%) handles the wrap-around automatically
        indices = (np.arange(self.pos - lookback, self.pos)) % self.buffer_size

        # 3. Fetch Rewards
        # self.rewards shape is (buffer_size, n_envs)
        history = self.rewards[indices] # shape: (lookback, n_envs)

        # 4. Convert to Tensor and Transpose
        # FFT expects (Batch/Envs, Time), so we transpose (T, B) -> (B, T)
        trajectory_tensor = th.as_tensor(history, device=self.device).T.float()

        return trajectory_tensor

class ASAC(SAC):
    """
    ASAC (Average-Reward Soft Actor-Critic) is a variant of the SAC algorithm.
    It inherits from the SAC class in stable_baselines3.
    """
    def __init__(self, 
                 *args,
                 eval_env=None,
                 long_eval_env=None,
                 rho_learning_rate=0.1,
                 value_based_centering=True,
                 rho_unbiased_step_size=True,
                 use_min_aggregation=True,
                 reward_centering=True,
                 recompute_td=False,
                 max_grad_norm=None,
                 eval_freq,
                 **kwargs):
        self.eval_env = eval_env
        self.long_eval_env = long_eval_env

        self.value_based_centering = value_based_centering
        self.rho_unbiased_step_size = rho_unbiased_step_size
        self.custom_dones_replay_buffer = True
        self.use_min_aggregation = use_min_aggregation
        self.aggregation = just_min if use_min_aggregation else th.mean
        self.reward_centering = reward_centering
        self.recompute_td = recompute_td
        self.max_grad_norm = max_grad_norm

        # Determine replay buffer class based on observation space
        # if kwargs.get('env') and hasattr(kwargs['env'], 'observation_space'):
        #     from gymnasium import spaces
        #     if isinstance(kwargs['env'].observation_space, spaces.Dict):
        #         if USE_STANDARD_TRANSITION:
        #             kwargs['replay_buffer_class'] = DictReplayBuffer
        #             print("Using the sac replay buffer for dict obs")
        #         else:
        #             kwargs['replay_buffer_class'] = DictCustomReplayBuffer
        #     else:
        #         kwargs['replay_buffer_class'] = CustomReplayBuffer
        # else:
        #     kwargs['replay_buffer_class'] = CustomReplayBuffer
        if N_STEP > 1:
            kwargs['replay_buffer_kwargs'] = {'n_steps': N_STEP}
        self.eval_freq = eval_freq 
        self.env_steps = 0
        super(ASAC, self).__init__(*args, replay_buffer_class=ThermodynamicReplayBuffer, **kwargs)

        if self.use_sde:
            raise NotImplementedError
        
        self.rho_learning_rate = rho_learning_rate
        # self.rho = th.tensor([0.0, 0.0], requires_grad=True, device=self.device)
        self.rho = th.tensor(RHO0, requires_grad=True, device=self.device)
        self.rho_optimizer = th.optim.SGD([self.rho], lr=self.rho_learning_rate)

        critic_params = list(self.critic.parameters())
        self.critic.optimizer = self.critic.optimizer_class(critic_params, lr=self.learning_rate, **self.critic.optimizer_kwargs)  # type: ignore[call-arg]
        self.o_n = 0


    def custom_critic(self, obs: th.Tensor, actions: th.Tensor, dones: th.Tensor) -> tuple[th.Tensor, ...]:
        return self.critic(obs, actions)


    def custom_target_critic(self, obs: th.Tensor, actions: th.Tensor, dones: th.Tensor) -> tuple[th.Tensor, ...]:
        return self.critic_target(obs, actions)

    def compute_friction(self, trajectories, current_beta):
        """
        Calculates thermodynamic friction (zeta) using the full two-point correlation sum.
        
        Args:
            trajectories (th.Tensor): Tensor of shape (Batch_Size, Time_Steps) containing rewards.
            current_beta (float): The current inverse temperature.
            
        Returns:
            zeta (float): The estimated thermodynamic friction.
        """
        # 0. Safety Check: If empty, return a default safe value
        if trajectories.numel() == 0:
            print("Warning: No valid trajectories for friction computation. Returning default zeta=1.0")
            return th.tensor(1.0, device=self.device)

        # 1. Ensure 2D Shape: If input is 1D (Time,), convert to (1, Time)
        if trajectories.ndim == 1:
            trajectories = trajectories.unsqueeze(0)
        # 1. Center the rewards (subtract mean per trajectory)
        # shape: (B, T)
        means = trajectories.mean(dim=1, keepdim=True)
        centered = trajectories - means
        
        # 2. Compute Autocovariance using FFT for efficiency
        # Pad with zeros to avoid circular correlation issues
        n = centered.shape[-1]
        padded = th.nn.functional.pad(centered, (0, n))
        
        # FFT -> Power Spectrum -> IFFT
        fft_vals = th.fft.rfft(padded, dim=-1)
        # Multiply by conjugate to get power spectrum
        power_spectrum = fft_vals * th.conj(fft_vals)
        # Inverse FFT to get autocorrelation
        autocorr = th.fft.irfft(power_spectrum, dim=-1)
        
        # Take only the valid part (first n elements) and normalize by sequence length
        # Note: Unbiased estimator would divide by (n - lag), but for large n, n is fine.
        autocorr = autocorr[:, :n] / th.arange(n, 0, -1, device=trajectories.device)
        
        # 3. Sum the Autocovariance (Integrate)
        # We assume the correlation decays. You might want to clip the sum 
        # at the first negative value or a specific lag window to reduce noise.
        # Here we sum the whole valid window for simplicity.
        integrated_autocov = autocorr.sum(dim=1).mean() 
        # compare to the error of summing without the last half:
        integrated_autocov_lho = autocorr[:, :n//2].sum(dim=1).mean()
        # and without the last value:
        integrated_autocov_loo = autocorr[:, :-1].sum(dim=1).mean()
        diff_loo = th.abs(integrated_autocov - integrated_autocov_loo)
        diff_lho = th.abs(integrated_autocov - integrated_autocov_lho)
        # log the errors:
        self.logger.record("thermo/integrated_autocov_loo", float(diff_loo.item()))
        self.logger.record("thermo/integrated_autocov_lho", float(diff_lho.item()))
        
        # 4. Apply Thermodynamic Definition: zeta = beta * Sum[Cov(t)]
        zeta = current_beta * integrated_autocov
        
        # Clamp to avoid numerical instability
        return th.max(zeta, th.tensor(MIN_ZETA, device=self.device))

    def thermodynamic_alpha_update(self, recent_trajectories, step_size=0.01):
        """
        Updates temperature based on constant thermodynamic speed with full friction tensor.
        d(beta) = step_size / sqrt(zeta)
        """
        with th.no_grad():
            # 1. Get current Beta
            current_alpha = th.exp(self.log_ent_coef)
            current_beta = 1.0 / (current_alpha + 1e-8)

            # 2. Calculate Full Thermodynamic Friction (Zeta)
            # Input must be (Batch, Time) sequences, NOT random samples.
            zeta = self.compute_friction(recent_trajectories, current_beta)

            # 3. Calculate Delta Beta
            # The generalized constant speed rule is d_lambda = step_size / sqrt(metric)
            # Here lambda = beta, metric = zeta.
            d_beta = step_size / th.sqrt(zeta)

            # 4. Apply Update
            new_beta = current_beta + d_beta
            new_alpha = 1.0 / new_beta
            self.log_ent_coef.copy_(th.log(new_alpha))

            return new_alpha
    
        
    def _store_transition(
        self,
        replay_buffer: ReplayBuffer,
        buffer_action: np.ndarray,
        new_obs: Union[np.ndarray, dict[str, np.ndarray]],
        reward: np.ndarray,
        dones: np.ndarray,
        infos: list[dict[str, Any]],
    ) -> None:
        """
        Store transition in the replay buffer.
        We store the normalized action and the unnormalized observation.
        It also handles terminal observations (because VecEnv resets automatically).

        :param replay_buffer: Replay buffer object where to store the transition.
        :param buffer_action: normalized action
        :param new_obs: next observation in the current episode
            or first observation of the episode (when dones is True)
        :param reward: reward for the current transition
        :param dones: Termination signal
        :param infos: List of additional information about the transition.
            It may contain the terminal observations and information about timeout.
        """
        if USE_STANDARD_TRANSITION:
            return super()._store_transition(
                replay_buffer,
                buffer_action,
                new_obs,
                reward,
                dones,
                infos,
            )
        
        # Store only the unnormalized version
        if self._vec_normalize_env is not None:
            new_obs_ = self._vec_normalize_env.get_original_obs()
            reward_ = self._vec_normalize_env.get_original_reward()
        else:
            # Avoid changing the original ones
            self._last_original_obs, new_obs_, reward_ = self._last_obs, new_obs, reward
        truncated = infos[0]['TimeLimit.truncated']
        terminated = dones and not truncated
        # Avoid modification by reference
        next_obs = deepcopy(new_obs_)
        # As the VecEnv resets automatically, new_obs is already the
        # first observation of the next episode
        for i, done in enumerate(dones):
            if done and infos[i].get("terminal_observation") is not None:
                if isinstance(next_obs, dict):
                    next_obs_ = infos[i]["terminal_observation"]
                    # VecNormalize normalizes the terminal observation
                    if self._vec_normalize_env is not None:
                        next_obs_ = self._vec_normalize_env.unnormalize_obs(next_obs_)
                    # Replace next obs for the correct envs
                    for key in next_obs.keys():
                        next_obs[key][i] = next_obs_[key]
                else:
                    next_obs[i] = infos[i]["terminal_observation"]
                    # VecNormalize normalizes the terminal observation
                    if self._vec_normalize_env is not None:
                        next_obs[i] = self._vec_normalize_env.unnormalize_obs(next_obs[i, :])  # type: ignore[assignment]
        # 'nft' defined below
        if terminated:
            # adds to the stack, so "knows" about prev new_obs being s_f
            # zero out rewards, lazy way to keep same type
            # "new_obs is already the first observation of the next episode"

            replay_buffer.add(
                self._last_original_obs,  # type: ignore[arg-type]
                new_obs,  # type: ignore[arg-type]
                buffer_action, # actions are arbitrary here
                reward_,
                np.array([False]), # in pseudo transition back to start, never done
                infos,
            )

        else:
            replay_buffer.add(
                self._last_original_obs,  # type: ignore[arg-type]
                next_obs,  # type: ignore[arg-type]
                buffer_action,
                reward_,
                dones, # the dones being trunc'd or term'd is checked after sampling
                infos,
            )
        assert dones.shape == (1,), "Not yet implemented for vectorized envs"
        

        self._last_obs = new_obs
        # Save the unnormalized observation
        if self._vec_normalize_env is not None:
            self._last_original_obs = new_obs_


    # def optimize_alpha(self, log_prob):
    #     ent_coef_losses, ent_coefs = [], []
    #     ent_coef_loss = None
    #     if self.ent_coef_optimizer is not None and self.log_ent_coef is not None:
    #         # Important: detach the variable from the graph
    #         # so we don't change it with other losses
    #         # see https://github.com/rail-berkeley/softlearning/issues/60
    #         ent_coef = th.exp(self.log_ent_coef.detach())
    #         assert isinstance(self.target_entropy, float)
    #         ent_coef_loss = -(self.log_ent_coef * (log_prob + self.target_entropy).detach()).mean()
    #         ent_coef_losses.append(ent_coef_loss.item())
    #     else:
    #         ent_coef = self.ent_coef_tensor
    #     ent_coefs.append(ent_coef.item())

    #     # Optimize entropy coefficient, also called
    #     # entropy temperature or alpha in the paper
    #     if ent_coef_loss is not None and self.ent_coef_optimizer is not None:
    #         self.ent_coef_optimizer.zero_grad()
    #         ent_coef_loss.backward()
    #         self.ent_coef_optimizer.step()
        
    #     return ent_coef, ent_coefs, ent_coef_losses

            
    def optimize_rho(self, replay_data, target_q_values, ent_coef, log_prob):
        rho_grads = th.stack([(q.detach() - target_q_values.detach())/N_STEP for q in 
                                self.custom_target_critic(replay_data.observations, replay_data.actions, replay_data.dones)])#.mean(axis=0)
        rho_grad = th.mean(rho_grads)
        # rho_grad_idx = th.min(th.abs(rho_grads), dim=0)[1]
        # rho_grad = rho_grads[rho_grad_idx].squeeze()
        self.rho_optimizer.zero_grad()
        self.rho.backward(gradient=rho_grad)
        self.rho_optimizer.step()


    def optimize_critic(self, loss):
        self.critic.optimizer.zero_grad()
        loss.backward()
        if self.max_grad_norm is not None:
            th.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        self.critic.optimizer.step()


    def get_target_q(self, replay_data, ent_coef, bootstrap_with_target_net=True) -> th.Tensor:
        try:
            bootstrap_dones = replay_data.next_dones
        except:
            bootstrap_dones = replay_data.dones

        with th.no_grad():
            # Select action according to policy
            next_actions, next_log_prob = self.actor.action_log_prob(replay_data.next_observations)
            # Compute the next Q values: min over all critics targets
            if bootstrap_with_target_net:
                next_q_values = th.cat(self.custom_target_critic(replay_data.next_observations, next_actions, bootstrap_dones), dim=1)
            else:
                next_q_values = th.cat(self.custom_critic(replay_data.next_observations, next_actions, bootstrap_dones), dim=1).detach()
            next_q_values = self.aggregation(next_q_values, dim=1, keepdim=True)

            curr_phi = th.cat(self.custom_critic(replay_data.observations, replay_data.actions, replay_data.dones), dim=1)
            curr_phi = self.aggregation(curr_phi, dim=1, keepdim=True) #- ent_coef * now_log_prob.reshape(-1, 1)

            
            next_q_values = next_q_values - ent_coef * next_log_prob.reshape(-1, 1) # theres no "free action"
            target_q_values = replay_data.rewards - self.reward_centering * N_STEP * self.rho.detach() + self.gamma * next_q_values

        return target_q_values
    
    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        # Switch to train mode
        self.policy.set_training_mode(True)
        
        # [Standard Scheduler Logic for Actor/Critic/Rho LRs...]
        optimizers = [self.actor.optimizer, self.critic.optimizer]
        self._update_learning_rate(optimizers)
        
        self.o_n = self.o_n + self.rho_learning_rate * (1 - self.o_n)
        rho_lr = self.rho_learning_rate / self.o_n
        if self.rho_unbiased_step_size:
            update_learning_rate(self.rho_optimizer, rho_lr)

        actor_losses, critic_losses = [], []
        
        for gradient_step in range(gradient_steps):
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            rewards = replay_data.rewards

            actions_pi, log_prob = self.actor.action_log_prob(replay_data.observations)
            log_prob = log_prob.reshape(-1, 1)

            # --- MODIFIED: THERMODYNAMIC TEMPERATURE UPDATE ---
            # Instead of optimizing alpha w.r.t target entropy, we enforce the schedule.
            if self.ent_coef_optimizer is None:
                current_ent_coef = self.ent_coef_tensor
            else:

                recent_history = self.replay_buffer.sample_continuous_history(n_steps=HISTORY_STEPS)

                # 2. Check if we have enough data (e.g., at least 100 steps)
                if recent_history.shape[-1] > 100 and self.env_steps > HISTORY_STEPS:
                    current_ent_coef = self.thermodynamic_alpha_update(recent_history, step_size=1e-6)
                else:
                    # Skip update if not enough data yet
                    current_ent_coef = th.exp(self.log_ent_coef).detach()

            # For logging compatibility
            ent_coefs = [current_ent_coef.item()] 
            ent_coef = current_ent_coef

            # --------------------------------------------------

            # Target Q calculation
            target_q_values = self.get_target_q(replay_data, ent_coef, bootstrap_with_target_net=True)
            
            # Current Q calculation
            current_q_values = self.custom_critic(replay_data.observations, replay_data.actions, replay_data.dones)

            # Optimize Rho (Average Reward Dual Variable)
            if self.reward_centering:
                old_rho = self.rho.detach().clone()
                self.optimize_rho(replay_data, target_q_values, ent_coef, log_prob)
                if self.recompute_td:
                    correction = -self.rho.detach() + old_rho
                    target_q_values = target_q_values + correction

            # Critic Update
            critic_loss = 0.5*th.mean(th.stack([(current_q - target_q_values)**2                                 
                                        for current_q in current_q_values]))
            self.optimize_critic(critic_loss)
            critic_losses.append(critic_loss.item())

            # Actor Update
            q_values_pi = th.cat(self.custom_critic(replay_data.observations, actions_pi, replay_data.dones), dim=1)
            min_qf_pi = self.aggregation(q_values_pi, dim=1, keepdim=True)
            
            # Alpha acts as the weight for the entropy term
            actor_loss = (ent_coef * log_prob - min_qf_pi).mean()

            actor_losses.append(actor_loss.item())

            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()

            # Polyak Update
            if gradient_step % self.target_update_interval == 0:
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)

        self._n_updates += gradient_steps


        metric_dict = {
                "env_steps": self.env_steps,
                "n_updates": self._n_updates,
                "rho": float(self.rho.item()),
                "ent_coef": float(np.mean(ent_coefs)),
                "actor_loss": float(np.mean(actor_losses)),
                "critic_loss": float(np.mean(critic_losses)),
                "scaled_entropy": float(np.mean((ent_coef * log_prob).detach().cpu().numpy())),
                "targetQ": float(np.mean(target_q_values.detach().cpu().numpy())),
                "currentQ": float(np.mean(th.stack(current_q_values).detach().cpu().numpy())),
                }
        if self._n_updates % 1000*gradient_steps == 0:
            METRICS.write(
                metrics=metric_dict
            )
            self.logger.record("rho", float(self.rho.item()))

            # Also log to SB3 logger:
            for key, value in metric_dict.items():
                self.logger.record(f"train/{key}", value)
            # Explicitly flush the logger to ensure metrics are written
            self.logger.dump(self.env_steps)

        if self._n_updates % self.eval_freq == 0:
            print("Evaluating policy...")
            # use a video recorder for the eval env:
            video_folder = f"videos/update_{self._n_updates}_step_{self.env_steps}"
            os.makedirs(video_folder, exist_ok=True)
            
            rewards_list, lengths_list = evaluate_policy(self.policy, self.eval_env, return_episode_rewards=True, n_eval_episodes=10)
            mean_return = np.mean([tot_rwd for tot_rwd, ep_len in zip(rewards_list, lengths_list)])
            mean_length = np.mean(lengths_list)
            eval_reward_rate = np.mean([tot_rwd / ep_len for tot_rwd, ep_len in zip(rewards_list, lengths_list)])

            # Log with METRICS:
            METRICS.write(
                metrics={
                    "env_steps": self.env_steps,
                    "n_updates": self._n_updates,
                    "eval_reward": mean_return,
                    "eval_reward_rate": eval_reward_rate,
                    "eval_ep_length": mean_length,
                    "raw_eval_rewards": rewards_list,
                    "raw_eval_lengths": lengths_list,
               }
           )
            self.logger.record("eval/reward", mean_return)
            self.logger.record("eval/reward_rate", eval_reward_rate)
            self.logger.record("eval/ep_length", mean_length)

            self.logger.dump(self.env_steps)


    def _on_step(self):
        super()._on_step()
        self.env_steps += 1

def train_agent(config: ASACTorchConfig, log_name=''):
    log_name = 'ASAC_' + log_name
    env_id = config.env_id_str
    gamma = config.gamma
    ent_coef = config.ent_coef
    batch_size = config.batch_size
    buffer_size = config.buffer_size
    gradient_steps = config.gradient_steps
    train_freq = config.train_freq
    learning_rate = config.learning_rate
    learning_starts = config.learning_starts
    target_update_interval = config.target_update_interval
    tau = config.tau
    rho_learning_rate = config.rho_learning_rate
    eval_freq = config.eval_freq
    
    env = gym.make(env_id)
    eval_env = gym.make(env_id, render_mode='rgb_array')
    long_eval_env = gym.make(env_id, max_episode_steps=10_000, render_mode='rgb_array')

    # Determine policy based on observation space
    policy_type = 'MultiInputPolicy' if isinstance(env.observation_space, gym.spaces.Dict) else 'MlpPolicy'
    
    # Wrap dict observation envs with DummyVecEnv to avoid recursion issues
    if isinstance(env.observation_space, gym.spaces.Dict):
        # Apply camera wrapper for AntMaze environments
        if 'ntMaze' in env_id:
            env = DummyVecEnv([lambda: Monitor(gym.make(env_id))])
            eval_env = DummyVecEnv([lambda: gym.make(env_id, render_mode='rgb_array')])
            long_eval_env = DummyVecEnv([lambda: gym.make(env_id, max_episode_steps=10_000, render_mode='rgb_array')])
        else:
            env = DummyVecEnv([lambda: gym.make(env_id)])
            eval_env = DummyVecEnv([lambda: gym.make(env_id, render_mode='rgb_array')])
            long_eval_env = DummyVecEnv([lambda: gym.make(env_id, max_episode_steps=10_000, render_mode='rgb_array')])

    agent = ASAC(env=env,
                 eval_env=eval_env,
                 long_eval_env=long_eval_env,
                 policy=policy_type,
                 gamma=gamma,
                 ent_coef=ent_coef,
                 batch_size=batch_size,
                 buffer_size=buffer_size,
                 gradient_steps=gradient_steps,
                 train_freq=train_freq,
                 learning_rate=learning_rate,
                 learning_starts=learning_starts,
                 target_update_interval=target_update_interval,
                 tau=tau,
                 rho_learning_rate=rho_learning_rate,
                 eval_freq=eval_freq,
                 device='cuda',
                 tensorboard_log=f'local-asac-{env_id}/{log_name}/',
                 value_based_centering=config.value_based_centering,
                 rho_unbiased_step_size=config.rho_unbiased_step_size,
                 use_min_aggregation=config.use_min_aggregation,
                 reward_centering=config.reward_centering,
                 recompute_td=config.recompute_td,
                 max_grad_norm=config.max_grad_norm,
    )

    agent.learn(total_timesteps=config.total_timesteps, tb_log_name=log_name)



def main() -> None:
    use_sac = {
        'gamma': 0.99,
        'value_based_centering': False,
        'reward_centering': False,
        'max_grad_norm': None,
    }
    use_asac = {
        'gamma': 1.0,
        'value_based_centering': True,
        'reward_centering': True,
        'max_grad_norm': None,
    }
    alg_config = use_sac
    gamma=alg_config['gamma']
    rhoLR=0.1
    ent_coef='auto_0.2' # required to setup logentcoef
    config = ASACTorchConfig(
        env_id_str='Humanoid-v5',
        gamma=gamma,
        ent_coef=ent_coef,
        batch_size=256,
        buffer_size=1_000_000,
        gradient_steps=1,
        train_freq=1,
        learning_rate=3e-4,
        learning_starts=10_000,
        target_update_interval=1,
        tau=0.005,
        rho_learning_rate=rhoLR,
        eval_freq=30_000,
        total_timesteps=10_000_000,
        value_based_centering=alg_config['value_based_centering'],
        rho_unbiased_step_size=False,
        use_min_aggregation=True,
        reward_centering=alg_config['reward_centering'], ###
        recompute_td=False,
        max_grad_norm=alg_config['max_grad_norm'], ###
    )
    with ExitStack() as stack:
        train_agent(config, log_name=f'temporal_nstep{N_STEP}_{gamma}_{rhoLR}_ent{ent_coef}_rho0={RHO0}_maxgrad={alg_config["max_grad_norm"]}_hist{HISTORY_STEPS}')

if __name__ == "__main__":
    for _ in range(5):
        main()

# todo: change replay buffer for sac