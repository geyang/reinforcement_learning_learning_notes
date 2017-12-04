from abc import abstractmethod

from typing import Union


class Schedule:
    @abstractmethod
    def value(self, t):
        pass

    @abstractmethod
    def __getitem__(self, item):
        pass


class Linear(Schedule):
    def __init__(self, n: int, start: float, stop: float) -> None:
        """Linear interpolation between initial_p and final_p over
        schedule_timesteps. After this many timesteps pass final_p is
        returned.
        Parameters
        ----------
        n: int
            Number of timesteps for which to linearly anneal initial_p
            to final_p
        start: float
            initial output value
        stop: float
            final output value
        """
        self._n = n
        self._start = start
        self._stop = stop

    def value(self, t: Union[int, float]) -> float:
        """See Schedule.value"""
        t = min(max(t, 0), self._n)
        fraction = t / self._n
        return self._start + fraction * (self._stop - self._start)

    def __getitem__(self, item):
        return self.value(item)


if __name__ == "__main__":
    l = Linear(100, 13, -2)
    assert l.value(-10) == 13
    assert l.value(0) == 13
    assert l.value(10) == 11.5
    assert l.value(20) == 10.0
    assert l.value(100) == -2.0
    assert l.value(1000) == -2.0
