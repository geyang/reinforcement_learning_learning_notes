import tensorflow as tf
import numpy as np


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


class Log:
    def __init__(self, writer=None):
        if writer:
            self.summary_writer = writer
        else:
            self.summary_writer = tf.summary.FileWriter('/tmp/tensorflow/'.format())

    # def log(self, variable, name_scope='summary'):
    #     with tf.name_scope(name_scope):
    #         self.summaries = tf.summary.merge([
    #             tf.summary.scalar('scaler', variable)
    #         ])
    #     self.summary_writer.add_summary(self.summaries)
    #     # tf.summary.scalar("loss", self.loss),
    #     # tf.summary.histogram("loss_hist", self.losses),
    #     # tf.summary.histogram("q_data_hist", self.predictions),
    #     # tf.summary.scalar("max_q_value", tf.reduce_max(self.predictions))

    def scaler(self, data, global_step, tag='summary'):
        summary = tf.Summary()
        summary.value.add(tag=tag, simple_value=float(data))
        self.summary_writer.add_summary(summary, global_step)

        summary = tf.Summary()
        summary.value.add(tag=tag, simple_value=float(data))
        self.summary_writer.add_summary(summary, global_step)

    def histogram(self, data, global_step, bins=1000, tag='histogram'):
        """Logs the histogram of a list/vector of data."""

        data = np.array(data)

        # Create histogram using numpy
        counts, bin_edges = np.histogram(data, bins=bins)

        # Fill fields of histogram proto
        hist = tf.HistogramProto()
        hist.min = float(np.min(data))
        hist.max = float(np.max(data))
        hist.num = int(np.prod(data.shape))
        hist.sum = float(np.sum(data))
        hist.sum_squares = float(np.sum(data ** 2))

        # Requires equal number as bins, where the first goes from -DBL_MAX to bin_edges[1]
        # See https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/summary.proto#L30
        # Thus, we drop the start of the first bin
        bin_edges = bin_edges[1:]

        # Add bin edges and counts
        for edge in bin_edges:
            hist.bucket_limit.append(edge)
        for c in counts:
            hist.bucket.append(c)

        # Create and write Summary
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, histo=hist)])
        self.summary_writer.add_summary(summary, global_step)
        self.summary_writer.flush()
