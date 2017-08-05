# baseline for gym tasks

## Tasks
- [ ] get VPG to work with CartPole
- [ ] how is baseline used during inference/sampling?
    **Answer**: baseline is not used during inference. It is only used during REINFORCE to reduce the variance.
- [ ] somehow CartPole-v0 performance with this new VPG in pyTorch is a bit mediocre. How to solve?
    solved in 1300 steps. After which catastrophically forgot and re-learned (multiple times)

## Algorithms
- [ ] VPG

### Backlog
- [ ] PPO
- [ ] TRPO


1. log-barier method 

    http://www.stat.cmu.edu/~ryantibs/convexopt-S15/scribes/15-barr-method-scribed.pdf 

    constrained optimization problem< replace constraint with a loss term

2. REPS

    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.180.3977&rep=rep1&type=pdf
    
    information lost by policy is minimized : constrain minimize information loss

3. TRPO
      KL constraint magnitude of the policy stays small minimize KL between policy update
      Theoreticed be bounded by the max KL , but instead use mean for no good reason
      you can’t use first order optimization methods, can’t use ADAM have to use conjugate gradient 
       
4. PPO uses log barrier surrogate form of the objective, Proximal Policy Optimization


5. DDPG

    doesn’t work well unless the reward is suitably scaled. 

5. Hanset Experience Replay

    worked well

6. NACQ

7. MAML


Curiosity 
- safe extension