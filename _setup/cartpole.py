import gym, time

env = gym.make('CartPole-v0')
env.reset()
done = False
while not done:
    env.render()
    action = env.action_space.sample()
    ob, r, done, _ = env.step(action)  # take a random action
    time.sleep(0.05)
