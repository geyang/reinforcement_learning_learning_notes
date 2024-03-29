import tensorflow as tf
import tf_helpers
import numpy as np


class VPG:
    def __init__(self, ob_size, ac_size, ac_is_discrete, RUN_ID, ENV):
        self.ac_size = ac_size
        self.discrete_features = 2
        self.ac_is_discrete = ac_is_discrete

        self.sess = tf.Session()
        self.writer = tf.summary.FileWriter(
            '/tmp/tensorflow/{ENV}/advantage/{RUN_ID}/'.format(RUN_ID=RUN_ID, ENV=ENV))
        self.writer.add_graph(self.sess.graph)
        self.logs = tf_helpers.Log(self.writer)

        with tf.name_scope("vanilla_policy_gradient"):
            # Placeholders
            self.lr_placeholder = tf.placeholder('float', [], name='Policy_Learning_Rate')
            self.value_lr_placeholder = tf.placeholder('float', [], name='Value_Learning_Rate')
            self.state_placeholder = tf.placeholder("float", [None, ob_size], name='IN_State')
            self.actions_placeholder = tf \
                .placeholder(tf.int32 if ac_is_discrete else "float", [None, ac_size], name='IN_Action')
            self.reward_placeholder = tf.placeholder("float", [None, 1], name='IN_Reward')

            with tf.name_scope('mu_NET'):
                x = self.state_placeholder
                # x = tf_helpers.dense(2, x, scope_name='layer_1', nonlin='sigmoid')
                # x = tf_helpers.dense(128, x, scope_name='layer_2', nonlin='relu')
                # x = tf_helpers.dense(128, x, scope_name='layer_3', nonlin='relu')
                mu = tf_helpers.dense(self.discrete_features or ac_size, x, scope_name='layer_4', nonlin=None)

            if ac_is_discrete:
                self.action, self.action_probs = self.discrete_action(mu)
                # calculate eligibility
                self.eligibility = self.discrete_eligibility_fn(self.actions_placeholder, self.action_probs)
            else:
                with tf.name_scope('log_stddev_NET'):
                    x = self.state_placeholder
                    x = tf_helpers.dense(128, x, scope_name='layer_1', nonlin='relu')
                    x = tf_helpers.dense(128, x, scope_name='layer_2', nonlin='relu')
                    x = tf_helpers.dense(128, x, scope_name='layer_3', nonlin='relu')
                    log_stddev = tf_helpers.dense(1, x, scope_name='layer_4', nonlin='tanh')

                self.action = self.sample_action(mu, log_stddev)
                # calculate eligibility
                self.eligibility = self.eligibility_fn(self.actions_placeholder, mu, log_stddev)

            self.value, self.value_optimizer = self.value_function(self.state_placeholder, self.reward_placeholder,
                                                                   self.value_lr_placeholder)

            with tf.name_scope("advantage"):
                self.advantage = self.reward_placeholder - self.value * 0

            self.surrogate_loss = self.surrogate_fn(self.eligibility, self.advantage)

            with tf.name_scope("optimizer"):
                self.optimizer = tf.train.AdamOptimizer(self.lr_placeholder).minimize(self.surrogate_loss)

            self.sess.run(tf.global_variables_initializer())

    def act(self, ob):
        acs, vpreds = self.sess.run(
            [self.action, self.value],
            feed_dict={self.state_placeholder: ob}
        )
        return acs, vpreds

    def learn_value_function(self, obs, rs, acs, lr):
        self.sess.run(
            [self.value_optimizer],
            feed_dict={
                self.value_lr_placeholder: lr,
                self.state_placeholder: obs,
                self.reward_placeholder: rs
            })

    def learn(self, obs, rs, acs, lr):
        self.sess.run(
            [self.optimizer],
            feed_dict={
                self.lr_placeholder: lr,
                self.state_placeholder: obs,
                self.reward_placeholder: rs,
                self.actions_placeholder: acs
            })

    def __enter__(self):
        return self

    def __exit__(self, *err):
        self.sess.close()

    @staticmethod
    def discrete_action(mu, scope_name="sample_action"):
        """Only supports bernoulli distribution. Does not support categorical distribution."""
        with tf.name_scope(scope_name):
            # probs is the bernoulli or categorical distribution
            probs = mu / tf.expand_dims(tf.reduce_sum(mu, axis=1), axis=1)
            SHAPE = tf.shape(probs[:, 0])
            # this is a bernoulli distribution. Use tf.int32 for the output.
            action = tf.where((tf.random_uniform(SHAPE) - probs[:, 0]) < 0,
                              tf.ones(SHAPE, tf.int32),
                              tf.zeros(SHAPE, tf.int32))
            # action = tf.contrib.distributions.Bernoulli(probs=probs)
        return action, probs

    @staticmethod
    def sample_action(mu, log_stddev, scope_name="sample_action"):
        with tf.name_scope(scope_name):
            SHAPE = tf.shape(mu)
            action = (tf.random_normal(SHAPE) + mu) / (2 * tf.exp(log_stddev * 2))
        return action

    @staticmethod
    def discrete_eligibility_fn(actions_ph, action_prob, scope_name="eligibility"):
        with tf.name_scope(scope_name):
            oh = tf.transpose(tf.one_hot(actions_ph, 2, axis=0))
            action_likelihood = tf.reduce_sum(action_prob * oh, axis=1)
            eligibility = tf.log(action_likelihood) - 1e-6
        return eligibility

    @staticmethod
    def eligibility_fn(actions_ph, mu, log_stddev=None, scope_name="eligibility"):
        with tf.name_scope(scope_name):
            elgb = - 0.5 * (np.log(2) + np.log(np.pi)) - log_stddev \
                   - (actions_ph - mu) ** 2 / (2 * tf.exp(log_stddev * 2)) \
                   - 1e-6
        return elgb

    @staticmethod
    def surrogate_fn(elgb, advantage, scope_name="surrogate_loss"):
        with tf.name_scope(scope_name):
            surrogate_loss = tf.multiply(elgb, advantage)
            loss = - tf.reduce_mean(surrogate_loss)
            return loss

    @staticmethod
    def value_function(state_ph, rewards_ph, value_lr, scope_name="value_function"):
        '''predicts the value of the current state, not sum of discounted future rewards.'''
        with tf.name_scope(scope_name):
            x = state_ph
            x = tf_helpers.dense(128, x, scope_name="layer_1", nonlin='relu')
            # x = tf_helpers.dense(128, x, scope_name="layer_2", nonlin='relu')
            # x = tf_helpers.dense(128, x, scope_name="layer_3", nonlin='relu')
            x = tf_helpers.dense(1, x, scope_name="layer_4")

            with tf.name_scope('optimizer'):
                error = rewards_ph - x
                loss = error ** 2
                optimizer = tf.train.AdamOptimizer(value_lr).minimize(loss)

            # variable_summaries(rewards_ph, 'sampled_reward_value')
            # variable_summaries(x, 'value_prediction')
            # variable_summaries(error, 'error')
            # variable_summaries(loss, 'loss')

            return x, optimizer
