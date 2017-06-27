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



> The state value function $v(s)$ is the expected return starting from state s:
> $$
>
> v(s) = \langle G_t \vert S_t = s\rangle
>
> $$
>

