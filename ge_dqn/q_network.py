import contextlib
import baselines.common.tf_util as U
import tensorflow as tf
import tensorflow.contrib.layers as layers
from config import G


def fully_connected(*, inputs, num_outputs, scope="fc", activation_fn=None, layer_norm=False, **rest):
    with tf.variable_scope(scope, reuse=False) if scope else contextlib.ExitStack:
        if not layer_norm:
            return layers.fully_connected(inputs, num_outputs=num_outputs, activation_fn=activation_fn, **rest)
        else:
            _ = layers.fully_connected(inputs, num_outputs=num_outputs, activation_fn=None, **rest)
            _ = layers.layer_norm(_, center=True, scale=True)
            return activation_fn(_) if callable(activation_fn) else _


def _center(*, inputs):
    """:param inputs: assumes axis[0] is the batch dimension."""
    with tf.variable_scope('center'):
        mean = tf.reduce_mean(inputs, axis=1)
        return inputs - tf.expand_dims(mean, 1)


# note: let's stick with discrete policies for now. No dueling. No TD weighted priority replay.
class q_policy:
    # noinspection PyInitNewSignature
    def __init__(self, *, obs, epsilon: tf.placeholder = None, action_space):
        act_size = action_space.n
        _ = obs
        if G.conv_params is not None:
            with tf.variable_scope('obs_conv_net'):
                for n_out, kernel_n, stride in G.conv_params:
                    _ = layers.convolution2d(_, num_outputs=n_out, kernel_size=kernel_n, stride=stride,
                                             activation_fn=tf.nn.relu)
                _ = layers.flatten(_)
        obs_hidden = _

        with tf.variable_scope('action_value'):
            for ind, hidden_size in enumerate(G.value_params):
                _ = fully_connected(inputs=_, num_outputs=hidden_size, scope='fc_{}'.format(ind),
                                    activation_fn=tf.nn.relu, layer_norm=G.use_layer_norm)
            self.action_values = fully_connected(inputs=_, num_outputs=act_size, activation_fn=None)
            self.q_values = self.action_values

        if G.use_dueling:
            # todo: implement alternative dueling (multiplicative dueling)
            with tf.variable_scope('state_value'):
                _ = obs_hidden
                for ind, hidden_size in enumerate(G.value_params):
                    _ = fully_connected(inputs=_, num_outputs=hidden_size, scope='fc_{}'.format(ind),
                                        activation_fn=tf.nn.relu, layer_norm=G.use_layer_norm)
                self.state_value = fully_connected(inputs=_, num_outputs=1)
                self.q_values = self.state_value + _center(inputs=self.action_values)

        with tf.variable_scope('act'):
            self.act_argmax = tf.argmax(self.q_values, axis=1)

            if G.stochastic_action:  # act randomly \epsilon of times
                assert epsilon is not None, "need epsilon placeholder for stochastic actions."
                batch_size = tf.shape(obs)[0]
                chose_random = tf.random_uniform(tf.stack([batch_size]), minval=0, maxval=1, dtype=tf.float32) < epsilon
                random_ind = tf.random_uniform(tf.stack([batch_size]), minval=0, maxval=act_size, dtype=tf.int64)
                self.action = tf.where(chose_random, random_ind, self.act_argmax)
            else:
                self.action = self.act_argmax

        self.act = U.function(inputs=[obs, epsilon], outputs=[self.action, self.action_values, self.q_values])

        current_scope = tf.get_variable_scope()
        self.trainables = U.scope_vars(current_scope, trainable_only=True)

    # note: not really used.
    # def __call__(self, *, action, state):  # todo: add extra feed_dict entries
    #     return U.get_session().run(self.action_scores, feed_dict={self.act: action, self.obs: state})
