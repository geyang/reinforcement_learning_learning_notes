import tensorflow as tf
from moleskin import moleskin as M

import ge_tf_utils
import baselines.common.tf_util as U

if __name__ == "__main__":
    with U.single_threaded_session() as sess:
        with tf.variable_scope('scope'):
            with tf.variable_scope('child', reuse=False):
                original = tf.get_variable('x', shape=[2, 3], dtype=tf.float32)

        with tf.variable_scope('scope', reuse=True):
            with tf.variable_scope('child', reuse=False):
                assert tf.get_variable_scope()._reuse == True, "child scope of reusable parent is always True"
                reused = tf.get_variable('x', shape=[2, 3], dtype=tf.float32)

        U.initialize()

        M.debug(sess.run(original))
        M.debug(sess.run(reused))
        M.debug(ge_tf_utils.get('scope/child/x'))

        assert (sess.run(original) == sess.run(reused)).all(), 'original, reused, and one from scope are identical'
