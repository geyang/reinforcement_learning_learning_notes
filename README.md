# Reinforcement Learning Notes

My notes on reinforcement learning

## Todo

- [ ] make a list of concepts to keep track of
- [ ] TRPO
- [ ] A3C
- [ ] DQN
- [ ] Behavior Cloning
- [ ] DAgger

### In Progress

- [ ] Finish David Silver's RL course [index](david%20silver%20RL%20course/course%20index.md)
- [ ] Berkeley RL course [http://rll.berkeley.edu/deeprlcourse/](http://rll.berkeley.edu/deeprlcourse/) 

- [x] gredients
> Since almost all algs are just computing gradients, it's important to know what that means. 
> [http://blog.shakirm.com/2015/11/machine-learning-trick-of-the-day-5-log-derivative-trick/](http://blog.shakirm.com/2015/11/machine-learning-trick-of-the-day-5-log-derivative-trick/)
>
> The above is a derivation of reinforce. It's quite good. I'd focus on internalizing the problem here. That we want to compute a derivative of an expectation with respect to a distribution (the policy), and we need a trick to do so. 
>
> [https://arxiv.org/pdf/1506.05254.pdf](https://arxiv.org/pdf/1506.05254.pdf) is a longer explanation of different viewpoints for taking derivatives. 

### Done

- [x] Gradients, and REINFORCE algorithm
- [x] Get MuJoCo
- [x] setup OpenAI Gym on AWS (yay!:confetti_ball:)
- [x] install `MuJoCo` :confetti_ball:
- [x] install `mujoco-py` (why is this so hard...)


### On How to Ask for Help
I found textbook to be the most reliable source, but there is usually too much material. So the best way to ask for guidance seem to be:
> I'm reading Chapter xx and topic xx atm, what are the key things I should pay attention to?
    

## Reading List
- [ ] 

### Done
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
