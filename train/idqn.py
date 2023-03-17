# from rl.ethereum_env import EthereumEnv
import collections
# from ma_gym.wrappers import Monitor
# from ..ethereum_env import EthereumEnv
import importlib
import random
import sys
import pickle
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import importlib
from tqdm import tqdm

sys.path.append('/Users/xinyutian/Desktop/dku_sw/')
rl = importlib.import_module('rl.ethereum_env')
EthereumEnv = rl.EthereumEnv

USE_WANDB = False  # if enabled, logs data on wandb server


class ReplayBuffer:
    def __init__(self, buffer_limit):
        self.buffer = collections.deque(maxlen=buffer_limit)

    def put(self, transition):
        self.buffer.append(transition)

    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []

        for transition in mini_batch:
            s, a, r, s_prime, done = transition
            s_lst.append(s)
            a_lst.append(a)
            r_lst.append(r)
            s_prime_lst.append(s_prime)
            done_mask_lst.append(done)

        return torch.tensor(s_lst, dtype=torch.float), \
            torch.tensor(a_lst, dtype=torch.float), \
            torch.tensor(r_lst, dtype=torch.float), \
            torch.tensor(s_prime_lst, dtype=torch.float), \
            torch.tensor(done_mask_lst, dtype=torch.bool)

    def size(self):
        return len(self.buffer)


class QNet(nn.Module):
    def __init__(self, observation_space, action_space):
        super(QNet, self).__init__()
        self.num_agents = len(observation_space)
        for agent_i in range(self.num_agents):
            n_obs = observation_space[agent_i].shape[0]
            setattr(self, 'agent_{}'.format(agent_i), nn.Sequential(nn.Linear(n_obs, 128),
                                                                    nn.ReLU(),
                                                                    nn.Linear(
                                                                        128, 64),
                                                                    nn.ReLU(),
                                                                    nn.Linear(64, action_space[agent_i].n)))

    def forward(self, obs):
        q_values = [torch.empty(obs.shape[0], )] * self.num_agents
        for agent_i in range(self.num_agents):
            q_values[agent_i] = getattr(self, 'agent_{}'.format(
                agent_i))(obs[:, agent_i, :]).unsqueeze(1)

        return torch.cat(q_values, dim=1)

    def sample_action(self, obs, epsilon):
        out = self.forward(obs)
        mask = (torch.rand((out.shape[0],)) <= epsilon)
        action = torch.empty((out.shape[0], out.shape[1],))
        action[mask] = torch.randint(
            0, out.shape[2], action[mask].shape).float()
        action[~mask] = out[~mask].argmax(dim=2).float()
        return action


def train(q, q_target, memory, optimizer, gamma, batch_size, update_iter=10):
    loss_list = []
    reward_list = []
    
    for epoch in range(update_iter):
        # print("epoch: ", epoch)
        s, a, r, s_prime, done_mask = memory.sample(batch_size)

        q_out = q(s)
        q_a = q_out.gather(2, a.unsqueeze(-1).long()).squeeze(-1)
        max_q_prime = q_target(s_prime).max(dim=2)[0]
        target = r + gamma * max_q_prime * done_mask
        loss = F.smooth_l1_loss(q_a, target.detach())

        loss_list.append(loss.abs().mean().item())
        reward_list.append(r)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return loss_list, reward_list


def test(env, num_episodes, q):
    information = []

    score = np.zeros(env.validator_size)
    for episode_i in range(num_episodes):
        print("episode_i: ", episode_i)
        state = env.reset()
        done = False
        information_at_i = []
        while not done:
            action = q.sample_action(torch.Tensor(state).unsqueeze(0), epsilon=0)[
                0].data.cpu().numpy().tolist()
            next_state, reward, done, info = env.step(action)
            score += np.array(reward)
            state = next_state
            information_at_i.append(info)
        information.append(information_at_i)


    return sum(score / num_episodes), information

# learning


def main(lr, gamma, batch_size, buffer_limit, log_interval, max_episodes,
         max_epsilon, min_epsilon, test_episodes, warm_up_steps, update_iter):
    env = EthereumEnv(validator_size=101)
    # print("env created")
    test_env = EthereumEnv(validator_size=101)
    # print("test_env created")
    memory = ReplayBuffer(buffer_limit)
    # print("memory created")

    # Initialize networks
    q = QNet(env.observation_space, env.action_space)
    # print("q created")
    q_target = QNet(env.observation_space, env.action_space)
    # print("q_target created")
    q_target.load_state_dict(q.state_dict())
    # print("q_target loaded")

    # Initialize optimizer
    optimizer = optim.Adam(q.parameters(), lr=lr)
    # print("optimizer created")

    score = np.zeros(env.validator_size)
    # print("score created")

    loss_lists = []
    reward_lists = []
    information = []

    for episode_i in tqdm(range(max_episodes)):
        # print("episode_i: ", episode_i)
        epsilon = max(min_epsilon, max_epsilon - (max_epsilon -
                      min_epsilon) * (episode_i / (0.4 * max_episodes)))
        state = env.reset()
        done = False
        while not done:
            action = q.sample_action(torch.Tensor(state).unsqueeze(0), epsilon)[
                0].data.cpu().numpy().tolist()
            next_state, reward, done, info = env.step(action)
            memory.put(
                (state, action, (np.array(reward)).tolist(), next_state, done))
            score += np.array(reward)
            # print(memory.size())
            state = next_state

        if memory.size() > warm_up_steps:
            loss_list, reward_list = train(q, q_target, memory, optimizer,
                  gamma, batch_size, update_iter)
            loss_lists.append(loss_list)
            reward_lists.append(reward_list)

        if episode_i % log_interval == 0 and episode_i != 0:
            q_target.load_state_dict(q.state_dict())
            print("q_target loaded")
            test_score, information = test(test_env, test_episodes, q)
            information.append(information)
            print("test_score: ", test_score)
            print("#{:<10}/{} episodes , avg train score : {:.1f}, test score: {:.1f} n_buffer : {}, eps : {:.1f}"
                  .format(episode_i, max_episodes, sum(score / log_interval), test_score, memory.size(), epsilon))
            score = np.zeros(env.validator_size)
            print("score reset")

    env.close()
    test_env.close()
    torch.save(q.state_dict(), 'model.pth')

    # test the model   
    q.load_state_dict(torch.load('model.pth'))
    test(test_env, test_episodes, q)

    return loss_lists, reward_lists, information


if __name__ == '__main__':
    kwargs = {
        'lr': 0.0005,
        'batch_size': 101,
        'gamma': 0.99,
        'buffer_limit': 5000,
        # 'buffer_limit': 64 * 101,
        'log_interval': 20,
        'max_episodes': 64,
        'max_epsilon': 0.9,
        'min_epsilon': 0.1,
        'test_episodes': 5,
        'warm_up_steps': 200,
        # 'warm_up_steps': 16,
        'update_iter': 10}
    
    loss_lists, reward_lists, information = main(**kwargs)
    # save to pickle
    with open('loss_lists.pkl', 'wb') as f:
        pickle.dump(loss_lists, f)
    with open('reward_lists.pkl', 'wb') as f:
        pickle.dump(reward_lists, f)
    with open('information.pkl', 'wb') as f:
        pickle.dump(information, f)

