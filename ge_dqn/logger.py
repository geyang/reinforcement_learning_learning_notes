from typing import Union, Callable, Any
from collections import OrderedDict, deque
from termcolor import colored as c
import tensorflow as tf
import numpy as np
import baselines.common.tf_util as U

from moleskin import moleskin as M


class MovingAverage:
    def __init__(self, len=100):
        self.d = deque(maxlen=len)

    def append(self, d):
        self.d.append(d)

    @property
    def latest(self):
        return self.d[-1]

    @property
    def mean(self):
        try:
            return np.mean(self.d)
        except ValueError:
            return None

    @property
    def max(self):
        try:
            return np.max(self.d)
        except ValueError:
            return None

    @property
    def min(self):
        try:
            return np.min(self.d)
        except ValueError:
            return None


class Color:
    # noinspection PyInitNewSignature
    def __init__(self, value, color, formatter: Union[Callable[[Any], Any], None] = lambda v: v):
        self.value = value
        self.color = color
        self.formatter = formatter

    def __str__(self):
        return str(self.formatter(self.value)) if callable(self.formatter) else str(self.value)

    def __len__(self):
        return len(str(self.value))

    def __format__(self, format_spec):
        if self.color == 'default':
            return self.formatter(self.value).__format__(format_spec)
        else:
            return c(self.formatter(self.value).__format__(format_spec), self.color)


def percent(v):
    return f'{round(v * 100):.1f}%'


def ms(v):
    return f'{v*1000:.1f}ms'


def sec(v):
    return f'{v:.3f}s'


def default(value, *args, **kwargs):
    return Color(value, 'default', *args, **kwargs)


def red(value, *args, **kwargs):
    return Color(value, 'red', *args, **kwargs)


def green(value, *args, **kwargs):
    return Color(value, 'green', *args, **kwargs)


def gray(value, *args, **kwargs):
    return Color(value, 'gray', *args, **kwargs)


def grey(value, *args, **kwargs):
    return Color(value, 'gray', *args, **kwargs)


def yellow(value, *args, **kwargs):
    return Color(value, 'yellow', *args, **kwargs)


def brown(value, *args, **kwargs):
    return Color(value, 'brown', *args, **kwargs)


class Logger:
    # noinspection PyInitNewSignature
    def __init__(self, log_directory):
        self.summary_writer = tf.summary.FileWriter(log_directory)
        self.index = None
        self.data = OrderedDict()
        self.do_not_print_list = set()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.summary_writer.close()

    def log_params(self, **kwargs):
        # ======================================
        #        ALTERNATIVE IMPLEMENTATION
        # --------------------------------------
        # value = "Random text"
        # text_tensor = tf.make_tensor_proto(value, dtype=tf.string)
        # meta = tf.SummaryMetadata()
        # meta.plugin_data.plugin_name = "text"
        # summary = tf.Summary()
        # summary.value.add(tag="whatever", metadata=meta, tensor=text_tensor)
        # summary_writer.add_summary(summary)
        # --------------------------------------

        key_width = 30
        value_width = 20

        table = []
        for n, (title, section_data) in enumerate(kwargs.items()):
            table.append((title, ""))
            print('═' * (key_width + 1) + f"{'═' if n == 0 else '╧'}" + '═' * (value_width + 1))
            print(c(f'{title:^{key_width}}', 'yellow'))
            print('─' * (key_width + 1) + "┬" + '─' * (value_width + 1))
            for key, value in section_data.items():
                value_string = str(value)
                table.append((key, value_string))
                print(c(f'{key:^{key_width}}', 'white'), "│", f'{value_string:<{value_width}}')

        if n > 0:
            print('═' * (key_width + 1) + f"{'═' if n == 0 else '╧'}" + '═' * (value_width + 1))

        table_tensor = tf.convert_to_tensor(table, dtype=tf.string)
        summary_op = tf.summary.text('experiment_parameters', table_tensor)
        self.summary_writer.add_summary(U.get_session().run(summary_op), 0)

    def log(self, index: Union[int, Color], *dicts, silent=False, **kwargs) -> None:
        """

        :param index: the global index, be it the global timesteps or the epoch index
        :param dicts: a dictionary of key/value pairs, allowing more flexible key name with '/' etc.
        :param silent: Bool, log but do not print. To keep the standard out silent.
        :param kwargs: key/value arguments.
        :return:
        """
        if self.index != index and self.index is not None:
            self.flush()
        self.index = index

        data_dict = {}
        for d in dicts:
            data_dict.update(d)
        data_dict.update(kwargs)

        if silent:
            self.do_not_print_list.update(data_dict.keys())

        summary = tf.Summary()
        for key, v in data_dict.items():
            try:
                summary.value.add(tag=key, simple_value=v.value if type(v) is Color else v)
            except TypeError as e:
                M.debug(key, v)
                raise e
            self.data[key] = v
        self.summary_writer.add_summary(summary, index)

    def flush(self, min_key_width=20, min_value_width=20):
        if not self.data:
            return

        keys = [k for k in self.data.keys() if k not in self.do_not_print_list]

        if len(keys) > 0:
            max_key_len = max([min_key_width] + [len(k) for k in keys])
            max_value_len = max([min_value_width] + [len(str(self.data[k])) for k in keys])
            output = None
            for k in keys:
                v = self.data[k]
                if output is None:
                    output = "╒" + "═" * max_key_len + "╤" + "═" * max_value_len + "╕\n"
                else:
                    output += "├" + "─" * max_key_len + "┼" + "─" * max_value_len + "┤\n"
                if k not in self.do_not_print_list:
                    k = k.replace('_', " ")
                    v = "None" if v is None else v  # for NoneTypes which doesn't have __format__ method
                    output += "│{k:^{max_key_len}}│{v:^{max_value_len}}│\n".format(**locals())
            output += "╘" + "═" * max_key_len + "╧" + "═" * max_value_len + "╛\n"
            print(output, end="")

        self.data.clear()
        self.do_not_print_list.clear()

    def log_histogram(self, index, *dicts, bins="auto", **kwargs):
        """Logs the histogram of a list/vector of values."""
        if self.index != index and self.index is not None:
            self.flush()
        self.index = index

        data_dict = {}
        for d in dicts:
            data_dict.update(d)
        data_dict.update(kwargs)

        # Create and write Summary
        summary = tf.Summary()
        for key, v in data_dict.items():

            values = np.array(v)
            # Create histogram using numpy
            counts, bin_edges = np.histogram(values, bins=bins)

            # Fill fields of histogram proto
            hist = tf.HistogramProto()
            hist.min = float(np.min(values))
            hist.max = float(np.max(values))
            hist.num = int(np.prod(values.shape))
            hist.sum = float(np.sum(values))
            hist.sum_squares = float(np.sum(values ** 2))

            # Requires equal number as bins, where the first goes from -DBL_MAX to bin_edges[1]
            # See https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/summary.proto#L30
            # Thus, we drop the start of the first bin
            bin_edges = bin_edges[1:]

            # Add bin edges and counts
            for edge in bin_edges:
                hist.bucket_limit.append(edge)
            for c in counts:
                hist.bucket.append(c)

            summary.value.add(tag=key, histo=hist)
            # note: do not add to `self.data[key] = v`

        self.summary_writer.add_summary(summary, index)


if __name__ == "__main__":
    d = Color(3.1415926, 'red')
    s = "{:.1}".format(d)
    print(s)

    logger = Logger('/mnt/slab/krypton/unitest')
    logger.log(0, some=Color(0.1, 'yellow'))
    logger.log(1, some=Color(0.28571, 'yellow', lambda v: f"{v * 100:.5f}%"))
    logger.log(2, some=Color(0.85, 'yellow', percent))
    logger.log(3, {"some_var/smooth": 10}, some=Color(0.85, 'yellow', percent))
    logger.log(4, some=Color(10, 'yellow'))
