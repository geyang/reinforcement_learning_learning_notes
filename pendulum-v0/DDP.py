'''
Deep Deterministic Policy Gradient
'''
import random
from collections import deque
from torch.autograd import Variable
from torch import Tensor
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data_utils
import numpy as np


class ReplyMemory(object):
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.buffer = deque(maxlen=self.buffer_size)

    def add(self, item):
        self.buffer.append(item)

    def size(self):
        return len(self.buffer)

    def sample_batch(self, batch_size):
        size = self.size()
        if size < batch_size:
            batch = random.sample(self.buffer, size)
        else:
            batch = random.sample(self.buffer, batch_size)
        return batch


class ActorNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, act_mul, act_bias):
        super(ActorNetwork, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.act_mul = Variable(act_mul)
        self.act_bias = Variable(act_bias)
        self.fc1 = nn.Linear(self.input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, self.output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.tanh(self.fc4(x))
        return x


class CriticNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(CriticNetwork, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.fc1 = nn.Linear(self.input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, self.output_dim)

    def forward(self, x, y):
        x = torch.cat((x, y), 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


class DDPGAgent(object):
    def __init__(self, state_dim, action_dim, gamma=0.98, use_cuda=False):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.use_cuda = use_cuda
        self.actor_net = ActorNetwork(self.state_dim, self.action_dim, Tensor(np.ones((1, action_dim))),
                                      Tensor(np.ones((1, action_dim))))
        self.target_actor_net = ActorNetwork(self.state_dim, self.action_dim, Tensor(np.ones((1, action_dim))),
                                             Tensor(np.ones((1, action_dim))))
        self.critic_net = CriticNetwork(self.state_dim + self.action_dim, 1)
        self.target_critic_net = CriticNetwork(self.state_dim + self.action_dim, 1)
        if self.use_cuda:
            self.actor_net = self.actor_net.cuda()
            self.critic_net = self.critic_net.cuda()
            self.target_actor_net = self.target_actor_net.cuda()
            self.target_critic_net = self.target_critic_net.cuda()
        self.criterion = nn.MSELoss()
        self.actor_net_optimizer = optim.Adam(self.actor_net.parameters(), lr=5e-5)
        self.critic_net_optimizer = optim.Adam(self.critic_net.parameters(), lr=5e-4)

    def get_action_output(self, states: torch.Tensor):
        if self.use_cuda:
            states = states.cuda()
        return self.actor_net(Variable(states)).data

    def train_on_batch(self, states: torch.Tensor, actions: torch.Tensor, train_target: torch.Tensor):
        if self.use_cuda:
            states = states.cuda()
            actions = actions.cuda()
            train_target = train_target.cuda()
        states = Variable(states)
        actions = Variable(actions)
        train_target = Variable(train_target)

        self.critic_net_optimizer.zero_grad()
        q_value0 = self.critic_net(states, actions)
        loss0 = self.criterion(q_value0, train_target)
        loss0.backward()
        self.critic_net_optimizer.step()

        self.actor_net_optimizer.zero_grad()
        self.critic_net_optimizer.zero_grad()
        actions0 = self.actor_net(states)
        q_value0 = self.critic_net(states, actions0)
        loss0 = -torch.mean(q_value0)
        loss0.backward()
        self.actor_net_optimizer.step()

    def train_batch_from_rpm(self, batch: list):
        size = len(batch)
        states = np.zeros((size, self.state_dim))
        new_states = np.zeros((size, self.state_dim))
        q_target = np.zeros((size, 1))
        actions = np.zeros((size, self.action_dim))

        for i in range(size):
            states[i, :] = batch[i][0][0, :]
            new_states[i, :] = batch[i][2][0, :]
            actions[i, :] = batch[i][1][:]

        if self.use_cuda:
            actions1 = self.target_actor_net(Variable(torch.Tensor(new_states).cuda()))
            t_q_value1 = self.target_critic_net(Variable(torch.Tensor(new_states).cuda()), actions1).data.cpu().numpy()

        else:
            actions1 = self.target_actor_net(Variable(torch.Tensor(new_states)))
            t_q_value1 = self.target_critic_net(Variable(torch.Tensor(new_states)), actions1).data.cpu().numpy()
        for i in range(size):
            if batch[i][4]:
                q_target[i] = batch[i][3]
            else:
                q_target[i] = batch[i][3] + self.gamma * t_q_value1[i]

        return Tensor(states), Tensor(actions), Tensor(q_target)

    def replace_weights(self, tau=0.001):
        '''
        update_target_net_params
        '''
        net_state = self.actor_net.state_dict()
        t_net_state = self.target_actor_net.state_dict()
        for name in t_net_state.keys():
            params = (1 - tau) * t_net_state[name] + tau * net_state[name]
            t_net_state[name] = params
        self.target_actor_net.load_state_dict(t_net_state)

        net_state = self.critic_net.state_dict()
        t_net_state = self.target_critic_net.state_dict()
        for name in t_net_state.keys():
            params = (1 - tau) * t_net_state[name] + tau * net_state[name]
            t_net_state[name] = params
        self.target_critic_net.load_state_dict(t_net_state)


MAX_STEPS = 200
DEBUG = False


def main():
    global DEBUG
    env = gym.make('Pendulum-v0')
    env._max_episode_steps = MAX_STEPS
    # env = wrappers.Monitor(env, "monitors", force=True)
    agent = DDPGAgent(3, 1, gamma=0.99, use_cuda=torch.cuda.is_available())
    rpm = ReplyMemory(buffer_size=5000)
    nb_epoch = 2000
    up_load = False
    up_load_threshold = -125
    sigma = 2.0  # standard deviation of noise
    sigma_decay = 0.9997
    final_sigma = 1e-3
    total_rewards = []
    max_mean_reward = float("-inf")
    for epoch in range(nb_epoch):
        states = env.reset()
        states = states.reshape((1, states.shape[0]))
        total_reward = 0
        while 1:
            action = agent.get_action_output(Tensor(states)).cpu().numpy() * 2
            noise = np.random.normal() * sigma
            action[0] += noise
            if action[0] < -2:  # bound action
                action[0] = -2
            elif action[0] > 2:
                action[0] = 2
            if sigma > final_sigma:
                sigma *= sigma_decay
            new_states, reward, done, _ = env.step(action[0])

            if DEBUG:
                env.render()
            if np.mean(reward) > -1.5:
                DEBUG = True

            new_states = new_states.reshape((1, new_states.shape[0]))
            rpm.add([np.copy(states), action[0], np.copy(new_states), reward, done])
            states = new_states
            total_reward += reward
            if done:
                total_rewards.append(total_reward)
                mean_reward = np.mean(total_rewards[-min(len(total_rewards), 100):])
                max_mean_reward = max(max_mean_reward, mean_reward)
                print("epoch:{}, total_reward:{:.2f}, mean_reward:{:.2f}, max_mean_reward:{:.2f}"
                      .format(epoch, total_reward / MAX_STEPS, mean_reward / MAX_STEPS, max_mean_reward / MAX_STEPS))
                if max_mean_reward > up_load_threshold:
                    up_load = True
                break
            state, actions, train_target = agent.train_batch_from_rpm(rpm.sample_batch(64))
            agent.train_on_batch(state, actions, train_target)
            agent.replace_weights()
    env.close()
    if up_load:
        gym.upload("/tmp/gym-results2", api_key="")


if __name__ == "__main__":
    import gym
    from gym import wrappers

    main()
