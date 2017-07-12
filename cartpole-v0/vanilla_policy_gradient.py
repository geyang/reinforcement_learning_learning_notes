import tensorflow as tf
import numpy as np
import time
import random
import gym
import math
import matplotlib.pyplot as plt


def softmax(x):
    e_x = np.exp(x - np.max(x))
    out = e_x / e_x.sum()
    return out


def policy_gradient():
    """usage:

    when running episode: (inside run_episode)
        sess.run(action_p, feed_dict={state: [s]})

    when running optimization:
        sess.run(optimizer, feed_dict={state: states, actions: [actions]})
    """
    with tf.variable_scope("policy"):
        # Placeholders
        state = tf.placeholder("float", [None, 4])
        actions_placeholder = tf.placeholder("int32", [None])
        reward_placeholder = tf.placeholder("float", [None, 1])

        # parameters
        params = tf.get_variable("policy_parameters", [4, 2],
                                 initializer=tf.random_normal_initializer(mean=0, stddev=0.1))
        linear = tf.matmul(state, params)
        prob_action = tf.nn.softmax(linear)

        # calculate eligibility
        oh = tf.transpose(tf.one_hot(actions_placeholder, 2, axis=0))
        action_likelihood = tf.reduce_sum(prob_action * oh, axis=1)
        eligibility = tf.log(action_likelihood) - 1e-3
        # loss = - tf.reduce_sum(eligibility) * tf.reduce_sum(reward_placeholder)
        integrant = eligibility * tf.reduce_sum(reward_placeholder, reduction_indices=1)
        loss = - tf.reduce_sum(integrant)

        optimizer = tf.train.AdamOptimizer(0.01).minimize(loss)
        return prob_action, state, actions_placeholder, \
               reward_placeholder, optimizer, \
               [action_likelihood, loss]


def run_episode(env, sess, state_ph, p_action, sleep=0):
    s = env.reset()
    states = [s]
    actions = []
    rewards = []

    for _ in range(200):
        a_p, *_ = sess.run(p_action, feed_dict={state_ph: [s]})
        rand = np.random.random()
        a = 0 if rand <= a_p[0] else 1
        # print(rand, a_p, a)
        s, r, done, _ = env.step(a)
        if DEBUG:
            env.render()
            if sleep:
                time.sleep(sleep)
            if done:
                time.sleep(0.1)
        if done:
            break
        states.append(s)
        actions.append(a)
        rewards.append(r)
    return states[:-1], actions, rewards


env = gym.make('CartPole-v0')

DEBUG = False

gamma = 0.99

p_action, state_ph, actions_pl, rewards_pl, adam, etc = policy_gradient()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(100000):
        states, actions, rewards = run_episode(env, sess, state_ph, p_action, sleep=0.01)
        # Now run optimizer
        *_, loss_val, = \
            sess.run([p_action, adam, *etc],
                     feed_dict={state_ph: states,
                                actions_pl: actions,
                                # rewards_pl: np.array([rewards]).T})
                                rewards_pl: np.array([
                                    list(map(lambda kv: (gamma ** (len(rewards) - kv[0]) - 1)/(gamma - 1), enumerate(rewards)))
                                ]).T})
        print(len(rewards), loss_val)
        if len(rewards) > 20 and np.mean(rewards[-20:]) >= 199:
            print("finished after {} steps".format(i))
            break
