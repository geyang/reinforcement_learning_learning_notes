from typing import Callable, Any, Union


class Slicer(object):
    def __init__(self, size, getter, setter=None):
        """size: used to determine the proper range for the indexing"""
        self._size = size
        self.getter_fn = getter
        self.setter_fn = setter

    def __setitem__(self, key, value):
        return self.setter_fn(key, value)

    def __getitem__(self, item):
        """emulate python's exactly list indexing behavior. Not trivial."""
        if isinstance(item, slice):
            start, stop, step = item.start, item.stop, item.step
            if start is None:
                start = 0
            elif start < 0:
                start = max(0, start + self._size)
            else:
                start = min(self._size, start)
            if stop is None:
                stop = self._size
            elif stop < 0:
                stop = max(0, stop + self._size)
            else:
                stop = min(self._size, stop)
            if step is None:
                step = 1
        else:  # simple indexing
            assert -self._size <= item < self._size, f"item need to be between [-{self._size}, {self._size})"
            start = item if item >= 0 else item + self._size
            stop = start + 1
            step = 1
        return self.getter_fn(start, stop, step)

    def __call__(self):
        return self[:]


class SegmentTree(object):
    def __init__(self, size: int, operation: Callable, neutral_element: Any) -> None:
        assert size > 0, "size has to be greater or equal to 1"
        self._size = 2 ** (size - 1).bit_length()  # size
        self._neutral_element = neutral_element
        self._values = [neutral_element for _ in range(2 * self._size)]
        self._op = operation

        def reduce(start, stop, step) -> Any:
            """closed start open end. step = 1 by default"""
            assert step is None or step == 1, NotImplementedError("steps in slicing are not supported")
            return self._reduce_helper(1, start, stop)  # node starts from 1 instead of 0

        self.reduce = Slicer(self._size, reduce)

    def __len__(self):
        return self._size

    def _node_range(self, i: int):
        assert i > 0, "node index `i` need to be larger or equal to 1."
        assert i < 2 * self._size, f"node index `i` ({i}) need to be less than or equal to the size of the entire " \
                                   f"array ({self._size}). When i >= self._size, it overflows to the actual data cells."
        range = 2 ** (i.bit_length() - 1)
        seg_length = self._size / range
        seg_index = i - range
        return seg_length * seg_index, seg_length * (seg_index + 1)

    def _reduce_helper(self, i, start, stop):
        """range is open on the right the same way as list ranges"""
        if stop is None:
            stop = self._size
        s, e = self._node_range(i)

        if start <= s < e <= stop:
            return self._values[i]
        elif s < stop and start < e:
            return self._reduce_helper(i * 2, start, stop) + self._reduce_helper(i * 2 + 1, start, stop)
        else:
            return self._neutral_element

    def __setitem__(self, idx, val):
        # index of the leaf
        assert (-self._size - 1) < idx < self._size, \
            f"index need to be within than the size ({-self._size - 1}, {self._size})"
        idx %= self._size
        idx += self._size
        self._values[idx] = val
        while idx >= 2:
            idx //= 2
            self._values[idx] = self._op(self._values[2 * idx], self._values[2 * idx + 1])

    def __getitem__(self, idx):
        assert 0 <= idx < self._size
        return self._values[self._size + idx]


class SumTree(SegmentTree):
    def __init__(self, length: int):
        from operator import add
        super(SumTree, self).__init__(length, add, 0)
        self.sum = self.reduce

    def __setitem__(self, key, value):
        assert value >= 0, f"value {value} has to be semi-definite (greater than 0)."
        return super().__setitem__(key, value)

    def argsumle(self, target, start=None, stop=None):
        start = start or 0
        stop = stop or self._size
        i = (start + stop) // 2
        result = self.reduce[:i]
        if result == target or start == stop - 1:
            return i
        elif result > target:
            return self.argsumle(target, start, i)
        elif result < target:
            return self.argsumle(target, i, stop)


class MinTree(SegmentTree):
    def __init__(self, length: int):
        from math import inf
        super(MinTree, self).__init__(length, min, inf)
        self.min = self.reduce
