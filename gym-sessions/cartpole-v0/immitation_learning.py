import tensorflow as tf
import numpy as np
import time
import gym
import helpers

# DONE: produce learned trajectory from expert
# DONE: learn from expert trajectory
# DONE: visualization and demos (in jupyter notebook)

# Try to load the data
try:
    expert_data = helpers.load('./imitation-data/cartpole_vpc_expert.tau')
    print('expert data is loaded!')
except:
    raise Exception('Can not load expert data.')


def policy_gradient():
    """usage:

    when running episode: (inside run_episode)
        sess_2.run(action_p, feed_dict={state: [s]})

    when running optimization:
        sess_2.run(optimizer, feed_dict={state: states, actions: [actions]})
    """
    with tf.variable_scope("policy"):
        # Placeholders
        state_placeholder = tf.placeholder("float", [None, 4])
        actions_placeholder = tf.placeholder("int32", [None])
        reward_placeholder = tf.placeholder("float", [None, 1])

        # one-hot actions
        oh = tf.transpose(tf.one_hot(actions_placeholder, 2, axis=0))

        # parameters
        params = tf.get_variable("policy_parameters", [4, 2],
                                 initializer=tf.random_normal_initializer(mean=0, stddev=0.1))
        linear = tf.matmul(state_placeholder, params)
        prob_action = tf.nn.softmax(linear)

        # Teacher-Forcing Loss
        TF_loss = tf.nn.softmax_cross_entropy_with_logits(labels=oh, logits=linear)
        imitation_optimizer = tf.train.AdamOptimizer(0.05).minimize(TF_loss)

        # calculate eligibility
        action_likelihood = tf.reduce_sum(prob_action * oh, axis=1)
        eligibility = tf.log(action_likelihood) - 1e-3
        # loss = - tf.reduce_sum(eligibility) * tf.reduce_sum(reward_placeholder)
        integrant = eligibility * tf.reduce_sum(reward_placeholder, reduction_indices=1)
        loss = - tf.reduce_sum(integrant)

        policy_gradient_optimizer = tf.train.AdamOptimizer(0.05).minimize(loss)
        return state_placeholder, actions_placeholder, reward_placeholder, \
               TF_loss, imitation_optimizer, prob_action, policy_gradient_optimizer, \
               [action_likelihood, loss]


def teacher_train(sess, state_ph, action_ph):
    pass


def run_episode(env, sess, state_ph, p_action, sleep=0):
    s = env.reset()
    states = [s]
    actions = []
    rewards = []

    for _ in range(MAX_STEPS - 1):
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


MAX_STEPS = 5000
env = gym.make('CartPole-v0')
env._max_episode_steps = MAX_STEPS

DEBUG = False

gamma = 0.99

state_ph, actions_ph, rewards_ph, \
tf_loss, imitate, \
p_action, adam, etc = policy_gradient()

episode_lens = []
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(10):
        print('Epoch-{ind}'.format(ind=i))
        for i, episode in enumerate(expert_data):
            sess.run(imitate, feed_dict={state_ph: episode['states'], actions_ph: episode['actions']})

        # test the learned result
        for i in range(5):
            states, actions, rewards = run_episode(env, sess, state_ph, p_action, sleep=0.01)
            # Now run optimizer
            print(i, '. ', len(rewards), sep='', end='\t')

    # generate analysis data, four traces for each starting state.
    print('* Now run students with same trajectory as expert.')

    # Make True to see on VNC.
    DEBUG = False
    student_data = []
    for i, episode in enumerate(expert_data):
        seed = episode['seed']
        teacher_states = episode['states']
        env.seed(seed)
        states, actions, rewards = run_episode(env, sess, state_ph, p_action, sleep=0.01)
        print("#{index}\tseed: {seed}\trewards: {rewards}"
              .format(index=i, seed=seed, rewards=len(rewards)))
        student_data.append(dict(seed=seed, states=states, actions=actions, rewards=rewards))

    helpers.save('./imitation-data/cartpole_vpc_student.tau', student_data)
    print('cartpole_vpc_student.tau data is saved.')


### Results:
#
# The performance quickly converges, giving very good behavior cloning.
# This is expected because the domain is very small and there are only 8 parameters.
#
# 0. 46	1. 42	2. 56	3. 73	4. 49
# 0. 132	1. 151	2. 175	3. 199	4. 140
# 0. 1277	1. 471	2. 176	3. 691	4. 382
# 0. 211	1. 2389	2. 3074	3. 88	4. 87
# 0. 4737	1. 1324	2. 4820	3. 4999	4. 227
# 0. 4999	1. 4999	2. 4999	3. 4999	4. 4999
# 0. 4999	1. 1176	2. 4999	3. 4999	4. 4999
# 0. 4936	1. 4999	2. 4999	3. 4999	4. 4999
# 0. 4999	1. 4999	2. 4999	3. 4999	4. 4999
# 0. 4999	1. 4999	2. 4999	3. 4999	4. 4999
