import tensorflow as tf
import numpy as np
from moleskin import Moleskin

m = Moleskin()


gamma = 0.9

def get_value(rs):
    discounted_total_reward = 0
    values = []
    for r in rs[::-1]:
        discounted_total_reward = r + discounted_total_reward * gamma
        values.insert(0, discounted_total_reward)
    return values

m.debug(get_value(range(10)))

m.debug(get_value([1] * 10))

np.expand_dims(range(10), axis=1)

torque = tf.placeholder('float', shape=[None, 1], name='torque')
inv_L = tf.placeholder('float', shape=[None, 1], name="inv_L")
prod = tf.multiply(torque, inv_L)
m.debug(prod)

ob = list(range(3))
m.debug(ob)

ob_vec = np.array([ob])
m.debug(ob_vec.shape)

with tf.Session() as sess:
    res = sess.run(prod, feed_dict={torque: [[3]] * 5, inv_L: [[0.3]] * 5})
    m.debug(res.shape)

    rewards = list(range(100))
    mean = tf.reduce_mean(rewards)
    mean_value = sess.run(mean)
    m.debug(mean_value)

