# Reinforcement Learning with CartPole

This report contains the following explorations:
1. Using a random guess algorithm
2. use hill-climbing by adding small perturbation to the parameters
3. Policy-gradient algorithm
4. Making the neural network deep

## TODOs
- [ ] generate expert trajectories $\tau_{expert}$
- [ ] supervised learning from these expert trajectories

### Done
- [x] Vanilla policy gradient


## The experiment

### Random Guess

This example is implemented in [./randomly_sampled_policy.py](./randomly_sampled_policy.py). 

### Hill Climbing

This example is implemented in [./hill_climbing.py](./hill_climbing.py). 

### Simple Gradient

This example is implemented in [./simple_gradient.py](./simple_gradient.py). 

### Vanilla Policy Gradient

This example is implemented in [./vanilla_policy_gradient.py](./vanilla_policy_gradient.py). We want to minimize 
$$
\nabla_\theta \eta = E[\sum \gamma^t r_t \sum \nabla_\theta \log P(a_t|\theta)]
$$

The outer integral is the reward. The inner integral is *the score*. 

#### Reducing the Bias

Now causality gives us a better bound on the gradient estimate:

$$
\nabla_\theta \eta = E[\sum_{t'=t}^T \gamma^t r_t \sum_{t=0}^T \nabla_\theta \log P(a_t|\theta)]
$$

This bound is better because only the rewards **after** the current time step should be attributed to the policy at step $t$. In other words, policy at each time $t$ should be eligible only for the reward that occurs **afterward**. 

To compute this graph, only the score term need to have a gradient. The reward term is 
given by the environment.

The expectation can be taken either over a single episode (trajectory) or a mini-batch. 
In the most simple case, we can even flatten all of the episodes together.

#### references

0. http://www.scholarpedia.org/article/Policy_gradient_methods
1. https://openai.com/requests-for-research/#cartpole
2. https://www.youtube.com/watch?v=oPGVsoBonLM
3. http://www0.cs.ucl.ac.uk/staff/D.Silver/web/Teaching_files/pg.pdf
4. Actor-critic vs REINFORCE: https://www.quora.com/Whats-the-difference-between-Reinforce-and-Actor-Critic/answer/Ishan-Durugkar?srid=XCIK

### Going Deep

A deeper version of this policy is implemented in [./vanilla_policy_gradient_deep.py
](./vanilla_policy_gradient_deep.py). In this example, I used a multi-layer 
perceptron network to model the policy. Going beyond 3 layers makes the network harder to train.
 Batch normalization did not help during the experiment.

## Next steps

- Natural Q-Learning: https://openai.com/requests-for-research/#natural-q-learning
- Description to Code
- Multi-task RL with Continuous Actions.
- TRPO


## References:

0. http://www.scholarpedia.org/article/Policy_gradient_methods
1. https://openai.com/requests-for-research/#cartpole
2. https://www.youtube.com/watch?v=oPGVsoBonLM
3. http://www0.cs.ucl.ac.uk/staff/D.Silver/web/Teaching_files/pg.pdf
4. Actor-critic vs REINFORCE: https://www.quora.com/Whats-the-difference-between-Reinforce-and-Actor-Critic/answer/Ishan-Durugkar?srid=XCIK