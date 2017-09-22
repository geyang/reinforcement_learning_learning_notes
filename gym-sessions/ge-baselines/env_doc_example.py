# the purpose of this document is to show the quirks and details of the OpenAI environments.
import gym, time
from moleskin import Moleskin
m = Moleskin(file='./env_docs.txt')
m.print("# to view this file with color, do `less -R ${m.log_filename}`".format(m=m))

# ## action space definitions
envs = ['FrozenLake-v0', 'CartPole-v0', 'Pendulum-v0', 'InvertedPendulum-v1', 'Reacher-v1']

for name in envs:
    env = gym.make(name)
    m.print(name)
    m.print('type of action_space: ', end='\t')
    m.green(env.action_space, end='\t')
    m.green(type(env.action_space))
    m.print('type of observation: ', end='\t')
    m.green(env.observation_space, end='\t')
    m.green(type(env.observation_space))
    env.close()
    time.sleep(0.25)

# import torch
# from torch.autograd import Variable
# a = torch.LongTensor(4, 2).zero_()
# b = Variable(torch.LongTensor(4, 2).zero_())
# c = torch.mul(Variable(a), b)
# print(c)
