import torch as th
from torch import nn
import configparser

config_dir = 'configs/configs_ppo.ini'
config = configparser.ConfigParser()
config.read(config_dir)
torch_seed = config.getint('MODEL_CONFIG', 'torch_seed')
th.manual_seed(torch_seed)
th.backends.cudnn.benchmark = False
th.backends.cudnn.deterministic = True

from torch.optim import Adam, RMSprop

import numpy as np
import os, logging
from copy import deepcopy
from single_agent.Memory_common import OnPolicyReplayMemory, OnPolicyReplayMemoryClip
from single_agent.Model_common import ActorNetwork, CriticNetwork
from common.utils import index_to_one_hot, to_tensor_var, VideoRecorder


from cma import CMAEvolutionStrategy
from MAPPO import MAPPO
class PPOCMA(MAPPO):
    """
    PPO-CMA: Combining Proximal Policy Optimization with Covariance Matrix Adaptation.
    Extends MAPPO by adding CMA-ES optimization.
    """
    def __init__(self, env, state_dim, action_dim,
                 memory_capacity=10000, max_steps=None,
                 roll_out_n_steps=1, target_tau=1.,
                 target_update_steps=5, clip_param=0.2,
                 reward_gamma=0.99, reward_scale=20,
                 actor_hidden_size=128, critic_hidden_size=128,
                 actor_output_act=nn.functional.log_softmax, critic_loss="mse",
                 actor_lr=0.0001, critic_lr=0.0001, test_seeds=0,
                 optimizer_type="rmsprop", entropy_reg=0.01,
                 max_grad_norm=0.5, batch_size=100, episodes_before_train=100,
                 use_cuda=True, traffic_density=1, reward_type="global_R",
                 cma_population_size=4, cma_sigma=0.5):
        """
        Initialize PPO-CMA by extending MAPPO.
        Additional parameters for CMA-ES:
        - cma_population_size: Number of samples per CMA-ES generation.
        - cma_sigma: Initial standard deviation for CMA-ES sampling.
        """
        # Initialize MAPPO (PPO)
        super().__init__(env, state_dim, action_dim, memory_capacity, max_steps,
                         roll_out_n_steps, target_tau, target_update_steps,
                         clip_param, reward_gamma, reward_scale, actor_hidden_size,
                         critic_hidden_size, actor_output_act, critic_loss, actor_lr,
                         critic_lr, test_seeds, optimizer_type, entropy_reg,
                         max_grad_norm, batch_size, episodes_before_train, use_cuda,
                         traffic_density, reward_type)
        
        # CMA-ES parameters
        self.cma_population_size = cma_population_size
        self.cma_sigma = cma_sigma
        self.n_agents = len(self.env.controlled_vehicles)
        # Flattened actor parameters for CMA-ES
        actor_params = self.get_parameters().astype(np.float32) 
        self.cma_es = CMAEvolutionStrategy(actor_params, self.cma_sigma, {
            "popsize": self.cma_population_size
        })
        if self.use_cuda:
            self.actor = self.actor.float().cuda()
            self.critic = self.critic.float().cuda()
            self.actor_target = self.actor_target.float().cuda()
            self.critic_target = self.critic_target.float().cuda()
        else:
            self.actor = self.actor.float()
            self.critic = self.critic.float()
            self.actor_target = self.actor_target.float()
            self.critic_target = self.critic_target.float()

    def _softmax_action(self, state, n_agents):
        state_var = to_tensor_var([state], self.use_cuda, dtype="float")  # 强制使用 float32

        softmax_action = []
        for agent_id in range(n_agents):
            # 确保输入数据的类型和模型的权重类型一致
            state_input = state_var[:, agent_id, :].float()  # 强制转换为 float32
            softmax_action_var = th.exp(self.actor(state_input).float())

            if self.use_cuda:
                softmax_action.append(softmax_action_var.data.cpu().numpy()[0])
            else:
                softmax_action.append(softmax_action_var.data.numpy()[0])
        return softmax_action
    def exploration_action(self, state, n_agents):
        softmax_actions = self._softmax_action(state, n_agents)
        actions = []
        for pi in softmax_actions:
            # 检查 softmax 输出是否为概率分布
            assert np.isclose(np.sum(pi), 1.0), "Softmax output is not a valid probability distribution."
            actions.append(np.random.choice(np.arange(len(pi)), p=pi))
        return actions
    def action(self, state, n_agents):
        softmax_actions = self._softmax_action(state, n_agents)
        actions = []
        for pi in softmax_actions:
            # 检查 softmax 输出是否为概率分布
            assert np.isclose(np.sum(pi), 1.0), "Softmax output is not a valid probability distribution."
            actions.append(np.random.choice(np.arange(len(pi)), p=pi))
        return actions


    def get_parameters(self):
        """
        Flatten and return the parameters of the actor network for CMA-ES.
        Ensures the output type matches the network's dtype.
        """
        return th.cat([param.view(-1) for param in self.actor.parameters()]).detach().cpu().numpy().astype(np.float32)

    def set_parameters(self, params):
        """
        Set the parameters of the actor network from a flattened array.
        Ensures the input type matches the network's dtype.
        """
        idx = 0
        for param in self.actor.parameters():
            param_size = param.numel()
            param.data = th.tensor(params[idx:idx + param_size], dtype=param.dtype).view(param.size()).to(param.device)
            idx += param_size

    def train(self):
        """
        Combines PPO's gradient-based update and CMA-ES's evolutionary optimization.
        """
        if self.n_episodes <= self.episodes_before_train:
            return

        # 1. PPO Gradient Update
        batch = self.memory.sample(self.batch_size)
        states_var = to_tensor_var(batch.states, self.use_cuda).view(-1, self.n_agents, self.state_dim)
        actions_var = to_tensor_var(batch.actions, self.use_cuda).view(-1, self.n_agents, self.action_dim)
        rewards_var = to_tensor_var(batch.rewards, self.use_cuda).view(-1, self.n_agents, 1)

        for agent_id in range(self.n_agents):
            # Compute advantages
            values = self.critic(states_var[:, agent_id, :], actions_var[:, agent_id, :]).detach()
            advantages = rewards_var[:, agent_id, :] - values

            # Actor update (PPO gradient step)
            self.actor_optimizer.zero_grad()
            action_log_probs = self.actor(states_var[:, agent_id, :])
            action_log_probs = th.sum(action_log_probs * actions_var[:, agent_id, :], dim=1)
            old_action_log_probs = self.actor_target(states_var[:, agent_id, :]).detach()
            old_action_log_probs = th.sum(old_action_log_probs * actions_var[:, agent_id, :], dim=1)
            ratio = th.exp(action_log_probs - old_action_log_probs)
            surr1 = ratio * advantages
            surr2 = th.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantages
            actor_loss = -th.mean(th.min(surr1, surr2))
            actor_loss.backward()
            if self.max_grad_norm is not None:
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            self.actor_optimizer.step()

            # Critic update
            self.critic_optimizer.zero_grad()
            target_values = rewards_var[:, agent_id, :]
            values = self.critic(states_var[:, agent_id, :], actions_var[:, agent_id, :])
            critic_loss = nn.MSELoss()(values, target_values)
            critic_loss.backward()
            if self.max_grad_norm is not None:
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.critic_optimizer.step()

        # 2. CMA-ES Update
        actor_params = self.get_parameters()  # Flattened parameters
        solutions = self.cma_es.ask()  # Generate candidate solutions
        solution_rewards = []

        for solution in solutions:
            # Apply the solution to the actor
            self.set_parameters(solution)
            
            # Evaluate the solution's performance
            rewards = self.evaluate_actor()  # Evaluate using a rollout
            solution_rewards.append(rewards)

        # Update CMA-ES with the evaluated rewards
        self.cma_es.tell(solutions, [-r for r in solution_rewards])  # CMA-ES minimizes, so negate rewards

        # Restore the best solution from CMA-ES
        best_solution = solutions[np.argmax(solution_rewards)]
        self.set_parameters(best_solution)

        # 3. Target Network Update
        if self.n_episodes % self.target_update_steps == 0 and self.n_episodes > 0:
            self._soft_update_target(self.actor_target, self.actor)
            self._soft_update_target(self.critic_target, self.critic)

    
    def evaluate_actor(self):
        """
        Evaluate the current actor by performing a full rollout in the environment.
        Returns the total reward achieved during the rollout.
        """
        self.env_state, _ = self.env.reset()
        self.n_agents = len(self.env.controlled_vehicles)  # 动态获取代理数量
        done = False
        total_reward = 0
        step = 0
        while not done:
            # 动作生成
            action = self.action(self.env_state, self.n_agents)
            # print(f"生成的动作: {action}, 类型: {type(action)}")
            # 确保动作为元组格式
            action = tuple(action)

            # 环境交互
            self.env_state, reward, done, info = self.env.step(action)
            # print(f"环境返回 - 奖励: {reward}, 额外信息: {info}")

            # 根据奖励类型解析奖励
            if self.reward_type == "regionalR":
                total_reward += sum(info["regional_rewards"])
            elif self.reward_type == "global_R":
                total_reward += reward
            step += 1
        return total_reward

