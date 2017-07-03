# MDP, MAP and POMDP

current *state* completely characterises the process, and process is *fully* observable.

Almost all RL are MDP:

- Optimal control deals with continuous MDP


- Partially observable problem can be converted into MDPs




Formulation:

A Markov Process is a tuple $\langle S,\,P\rangle.$

A state St is Markov if and only if

$$ P[ S_{t+1} | S_t ] = P[ S_{t+1} | S_1, …, S_t ] $$

$P_{i,j}$ forms a matrix.



| Terminal Space | state with no exit      |
| -------------- | ----------------------- |
| sample episode | sequence $S_1, S_2, …,$ |



## Markov Reward Process (MAP)



Markov process with value judgements.

> Markov Reward Process is a tuple $\langle S, P, R, \gamma \rangle.$

| Markov Reward Process |                                |
| --------------------- | ------------------------------ |
| $\gamma$              | discount factor for the reward |



### Why is this $\gamma$ useful?

- Mathematically stable
- avoids infinity in cyclic markov processes
- Uncertainty about the future
- financial interest for money atm
- animal/human prefer immediate reward
- **undiscouted MAP** is useful if all sequences terminate.



### Value Function

> The *state value function $v(s)$* is the expected return starting from state s:
> $$
> v(s) = \langle G_t \vert S_t = s\rangle
> $$
>



### Bellman Equation for MRPs

- immediate reward $R_{t+1}$ 
- discuouted successor rewards $\gamma \,V(S_{t+1})$ 

$$
V(s) = \langle R_{t+1} + \gamma v(S_{t+1}) \vert _{S_t = s} \rangle
$$

> the index $t+1$ is arbitrary. 

$$
V(s) = R{t} + \gamma \sum_{s' \in S} P_{s,s'} V(s')
$$

Matrix Formulation:
$$
\vec v = \vec R + \gamma P\cdot \vec v
$$
Linear solution to Bellman's equation: 
$$
V = (I - \gamma P)^{-1} \vec R
$$

- computation complexity is $O(n^3)$ for $n$ states
- Iterative methods include
  - Dynamic Programming
  - Monte-Carlo evaluation
  - Temporal-Difference Learning

## MDP

> A Markov Reward 