from operator import add
from moleskin import moleskin as M

from segment_trees import SegmentTree, SumTree


@M.timeit
def test_node_range():
    size = 15
    capacity = 2 ** size.bit_length()
    s = SegmentTree(size, add, 0)
    assert s._node_range(1) == (0, capacity), ""
    assert s._node_range(2) == (0, capacity // 2), ""
    assert s._node_range(3) == (capacity // 2, capacity), ""
    assert s._node_range(4) == (0, capacity // 4), ""
    assert s._node_range(5) == (capacity // 4, capacity // 2), ""


@M.timeit
def test_sum():
    size = 200
    s = SegmentTree(size, add, 0)
    for i in range(200):
        s[i] = i
        # note: low-level check on the sum value in each node
        assert s._values[256 + i] == i
        assert s._values[1] == sum(range(i + 1))
        assert s._values[2] == sum(range(min(i + 1, 128)))
    assert s.reduce[:] == sum(range(size)), f"total sum {s.reduce[:]} != {sum(range(size))}"
    for i in range(190):
        assert s.reduce[i:i + 10] == sum(range(i, i + 10)), f"sum {s.reduce[i: i+10]} should be {sum(range(i, i+10))}"
    for i in range(200):
        assert s.reduce[i] == i, f"sum {s.reduce[i]} should be {i}"
    for i in range(200):
        assert s.reduce[:i] == sum(range(i)), f"sum {s.reduce[:i]} should be {sum(range(i))}"


@M.timeit
def test_sum_tree():
    s = SumTree(50000)
    for i in range(50000):
        s[i] = 0.5
    index = s.argsumle(1001)
