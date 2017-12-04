# Reinforcement Learning Notes

My notes on reinforcement learning. I have moved some of the newer algorithms to different repos so the list here is incomplete.

## Plans (2017-12-04)

- [ ] C51, distributional Q-learning
- [ ] Solve Montezuma with re-weighted sampling
- [ ] Move PPO into this repo

### Done

- [x] DQN
    - [x] prioritized replay
    - [x] double Q-learning (or half Q-learning)
    - [x] dueling networks
    - [x] $\epsilon$-greedy with linear scheduling
- [x] Gradients, and REINFORCE algorithm
- [x] policy gradients
- [x] Setups
    - [x] Get MuJoCo
    - [x] setup OpenAI Gym on AWS (yay!:confetti_ball:)
    - [x] install `MuJoCo` :confetti_ball:
    - [x] install `mujoco-py` (need to upgrade to 1.50 now supports python 3.6)
- [x] make a list of concepts to keep track of

### Backlog

- [ ] TRPO
- [ ] A3C
- [ ] Behavior Cloning
- [ ] DAgger


### On How to Ask for Help
I found textbook to be the most reliable source but it's easy to get lost in the chapters. So the best way to ask for guidance seem to be:
> I'm reading Chapter xx and topic xx atm, what are the key things I should pay attention to?

### Reference Readings

- [ ] David Silver's RL course [index](david%20silver%20RL%20course/course%20index.md)
- [ ] Berkeley RL course [http://rll.berkeley.edu/deeprlcourse/](http://rll.berkeley.edu/deeprlcourse/) 
- [x] [http://blog.shakirm.com/2015/11/machine-learning-trick-of-the-day-5-log-derivative-trick/](http://blog.shakirm.com/2015/11/machine-learning-trick-of-the-day-5-log-derivative-trick/)
- [ ] [https://arxiv.org/pdf/1506.05254.pdf](https://arxiv.org/pdf/1506.05254.pdf) is a longer explanation of different viewpoints for taking derivatives. 
- [x] Contextual bandits: 
    - http://hunch.net/?p=298
    - https://getstream.io/blog/introduction-contextual-bandits/

## Research Ideas

- Curiosity as reward
- Finding answers as reward
- inferring intention
- Learning to predict (lots of prior art. self-supervision)
- Auxiliary supervision and Auxiliary modalities.
- inverse reinforcement learning != imitation learning
