import tensorflow as tf
import baselines.common.tf_util as U


def get(name):
    """this should be used during debug only."""
    *scope, var_name = name.split('/')
    with tf.variable_scope('/'.join(scope), reuse=True):
        v_op = tf.get_variable(var_name)
        sess = U.get_session()
        return sess.run(v_op)
