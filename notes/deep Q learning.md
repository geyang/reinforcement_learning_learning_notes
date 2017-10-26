# Deep Q Learning

The approximate optimal value-action function is $$
Q*(s,\,a) = \max_\pi \mathcal E[r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \dots \vert s_t=s_0, a_t=a_0, \pi].
$$

This is the maximum sum of rewards $r_t$ discounted by $\gamma$ at each time-step $t$, achievable by policy $\pi = P(a|s)$.

Problems:
- Reinforcement learning is known to be unstable or divergent when neural net is used as the function approximator.

This instability means small updates to Q may significantly change the policy, therefore the distrubution. Correlations between the action-values($Q$) and the target values $r + \gamma \max_{d'} Q(s', a')$.



