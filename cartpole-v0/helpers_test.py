import os
from helpers import save, load


def test():
    path = './test-data/example.pkl'
    directory, _ = os.path.split(path)
    save(path, {"some": list(range(100))})
    ob = load(path)
    assert ob["some"] == list(range(100)), 'object should be correct'
    # save again
    save(path, {"some": list(range(200))})
    ob = load(path)
    assert ob["some"] == list(range(200)), 'object should be correct'
