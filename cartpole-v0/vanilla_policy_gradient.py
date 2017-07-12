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
        actions_placeholder = tf.placeholder("int32", [None, 1])
        reward_placeholder = tf.placeholder("float", [None, 1])

        # parameters
        params = tf.get_variable("policy_parameters", [4, 2], initializer=tf.random_normal_initializer)
        linear = tf.matmul(state, params)
        prob_action = tf.nn.softmax(linear)

        # calculate eligibility
        action_log_likelihood = tf.reduce_sum(prob_action * tf.one_hot(actions_placeholder, 2, axis=-1),
                                              reduction_indices=[1])
        eligibility = tf.log(action_log_likelihood)
        loss = - tf.reduce_sum(eligibility) * tf.reduce_sum(reward_placeholder)

        optimizer = tf.train.AdamOptimizer(0.000001).minimize(loss)
        return prob_action, state, actions_placeholder, reward_placeholder, optimizer


def run_episode(env, sess, state_ph, p_action, sleep=0):
    states = []
    actions = []
    rewards = []

    s = env.reset()
    for _ in range(200):
        a_p, *_ = sess.run(p_action, feed_dict={state_ph: [s]})
        a = 0 if np.random.rand(1)[0] <= a_p[0] else 1
        print(np.random.rand(1)[0], a_p[0], np.sum(a_p), a)
        s, r, done, _ = env.step(a)
        states.append(s)
        actions.append(a)
        rewards.append(r)
        if DEBUG:
            env.render()
        if sleep:
            time.sleep(sleep)
        if done:
            time.sleep(0.1)
            break
    return states, actions, rewards


env = gym.make('CartPole-v0')
DEBUG = True

p_action, state_ph, actions_pl, rewards_pl, adam = policy_gradient()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(20000):
        states, actions, rewards = run_episode(env, sess, state_ph, p_action, sleep=0.05)
        # Now run optimizer
        test = np.array([states]).T
        sess.run(adam, feed_dict={state_ph: states,
                                  actions_pl: np.array([actions]).T,
                                  rewards_pl: np.array([rewards]).T})
        if np.sum(rewards) >= 200:
            print("finished after {} steps".format(i))
            break
