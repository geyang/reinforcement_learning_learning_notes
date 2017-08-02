# Generative Adversarial Imitation Learning

JoHo:

## Overview

Two ways to learn expert behavior:

1. recover expert cost function with inverse RL.
    indirect and slow
2. GAIL extracts policy directly from data.
    similar to imitation learning and GAN. 

Derive a model-free imitation learning algo with significant perf gain over existing MF methods, in complex high-D env.

## Problem Setup

1. Data limitation: 
    - trajectory sample only
    - no query allowed
    - no reinforcement signal.

Behavior cloning learns policy, fits single time-step decision.
suffers from co-variate shifts.

IRL learns cost function prioritize entire trajectory over others. no such problem.

IRL is hard to run b/c require running RL in inner loop.

Methods to Scale IRL is hence needed.

Out-performs in high-dimension physics based control tasks. Why p