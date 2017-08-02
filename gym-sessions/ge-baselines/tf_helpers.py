import tensorflow as tf


def dense(out_n, x, scope_name='dense', nonlin=False):
    with tf.name_scope(scope_name):
        shape = x.get_shape().as_list()[-1:] + [out_n]
        # initializer = tf.random_normal_initializer(mean=0, stddev=1)
        initializer = tf.contrib.layers.xavier_initializer()
        params = tf.Variable(initializer(shape), name='Weights')
        out = tf.matmul(x, params)
        if nonlin:
            out = getattr(tf.nn, nonlin)(out)
        return out
