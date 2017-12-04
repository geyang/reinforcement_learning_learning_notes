import random


class ReplayBuffer(object):
    def __init__(self, *, size: int, seed: int = 42):
        from collections import deque
        self._q = deque(maxlen=size)
        # self.rng = np.random.RandomState(seed=seed)
        self.rng = random.Random(x=seed)

    def add(self, *, s0, action, reward, s1, done):
        """
        append new sample to the replay buffer
        :param s0: the initial state
        :param action: the action at time t0
        :param s1: the state at t1
        :param reward: the reward from action and state pair at t0
        :param done: boolean for whether the episode is over.
        :return:
        """
        data = s0, action, reward, s1, done
        self._q.append(data)

    def sample(self, batch_size, **kwargs):
        return self.rng.choices(self._q, k=batch_size, **kwargs)


class PrioritizedReplayBuffer:
    def __init__(self, *, size: int, alpha: float, seed: int = 42):
        self._q = [0] * size
        # self._q = np.empty((size,))
        self._insertion_index = 0
        self.rng = random.Random(x=seed)

        assert alpha > 0
        self._alpha = alpha  # needed for adding samples.

        from segment_trees import SumTree, MinTree
        self._sum_tree = SumTree(size)
        self._min_tree = MinTree(size)
        self._max_priority = 1.0  # set the max priority to 1.

    def __len__(self):
        return len(self._q)

    def add(self, *, s0, action, reward, s1, done):
        """You don't drop the least important ones. Instead, you drop the least recent added ones. As a result, some
        of the episodes never got played."""
        i = self._insertion_index  # index of entry to be over-written
        self._q[i] = s0, action, reward, s1, done
        priority = self._max_priority ** self._alpha  # note: this is not very scientific. There could be a better way.
        self._sum_tree[i] = priority
        self._min_tree[i] = priority
        self._insertion_index = (self._insertion_index + 1) % len(self)

    def sample(self, batch_size, beta):
        assert beta > 0
        Z = self._sum_tree.sum[:]
        n = len(self)
        # sample weights and get sample indices
        indices, weights, experiences = [], [], []
        for _ in range(batch_size):
            r = self.rng.random() * Z
            idx = self._sum_tree.argsumle(r)
            indices.append(idx)
        p_min = self._min_tree.min[:] / Z
        # fixme: when p_min is zero, following statement throws divide-by-zero warning
        max_weight = (p_min * n) ** -beta

        for i in indices:
            p = self._sum_tree[i] / Z
            weights.append((p * n) ** -beta / max_weight)
            experiences.append(self._q[i])

        return experiences, weights, indices

    def update_priorities(self, indices, priorities):
        assert len(indices) == len(priorities), \
            f"indices and priorities have to have the same length: {len(indices)} != {len(priorities)}"
        for i, priority in zip(indices, priorities):
            assert priority > 0  # when priority is zero, the max_weight goes to infinity. So no zero allowed.
            assert 0 <= i < len(self)
            weight = priority ** self._alpha
            self._sum_tree[i] = weight
            self._min_tree[i] = weight

            self._max_priority = max(self._max_priority, priority)
