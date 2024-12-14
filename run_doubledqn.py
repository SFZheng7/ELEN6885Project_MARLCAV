from MADoubleDQN import MADoubleDQN
from single_agent.utils_common import agg_double_list

import sys
sys.path.append("../highway-env")
import os
import gym
import numpy as np
import matplotlib.pyplot as plt
import highway_env

MAX_EPISODES = 20000
EPISODES_BEFORE_TRAIN = 10
EVAL_EPISODES = 3
EVAL_INTERVAL = 200

# max steps in each episode, prevent from running too long
MAX_STEPS = 100

MEMORY_CAPACITY = 1000000
BATCH_SIZE = 128
CRITIC_LOSS = "mse"
MAX_GRAD_NORM = None

REWARD_DISCOUNTED_GAMMA = 0.99
EPSILON_START = 0.99
EPSILON_END = 0.05
EPSILON_DECAY = 50000

def run_doubledqn(output_dir="./results/doubledqn"):
    # Create output directories
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    models_dir = os.path.join(output_dir, "models")
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    plots_dir = os.path.join(output_dir, "plots")
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)

    env = gym.make('merge-multi-agent-v0')
    env_eval = gym.make('merge-multi-agent-v0')
    state_dim = env.n_s
    action_dim = env.n_a

    madouble_dqn = MADoubleDQN(env=env, memory_capacity=MEMORY_CAPACITY,
                                state_dim=state_dim, action_dim=action_dim,
                                batch_size=BATCH_SIZE, max_steps=MAX_STEPS,
                                reward_gamma=REWARD_DISCOUNTED_GAMMA,
                                epsilon_start=EPSILON_START, epsilon_end=EPSILON_END,
                                epsilon_decay=EPSILON_DECAY, max_grad_norm=MAX_GRAD_NORM,
                                episodes_before_train=EPISODES_BEFORE_TRAIN)

    episodes = []
    eval_rewards = []
    while madouble_dqn.n_episodes < MAX_EPISODES:
        madouble_dqn.interact()
        if madouble_dqn.n_episodes >= EPISODES_BEFORE_TRAIN:
            madouble_dqn.train()
        if madouble_dqn.episode_done and ((madouble_dqn.n_episodes + 1) % EVAL_INTERVAL == 0):
            rewards, _ = madouble_dqn.evaluation(env_eval, EVAL_EPISODES)
            rewards_mu, rewards_std = agg_double_list(rewards)
            print("Episode %d, Average Reward %.2f" % (madouble_dqn.n_episodes + 1, rewards_mu))
            episodes.append(madouble_dqn.n_episodes + 1)
            eval_rewards.append(rewards_mu)

            # Save the model
            model_path = os.path.join(models_dir, f"doubledqn_episode_{madouble_dqn.n_episodes + 1}.pth")
            madouble_dqn.save(model_path, madouble_dqn.n_episodes + 1)

    # Final save
    final_model_path = os.path.join(models_dir, f"doubledqn_final.pth")
    madouble_dqn.save(final_model_path, MAX_EPISODES + 2)

    episodes = np.array(episodes)
    eval_rewards = np.array(eval_rewards)

    plt.figure()
    plt.plot(episodes, eval_rewards)
    plt.xlabel("Episode")
    plt.ylabel("Average Reward")
    plt.legend(["Double DQN"])
    plt_path = os.path.join(plots_dir, "reward_plot.png")
    plt.savefig(plt_path)
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) >= 2:
        run_doubledqn(sys.argv[1])
    else:
        run_doubledqn()
