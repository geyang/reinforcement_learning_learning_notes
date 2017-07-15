import os
import pickle


def save(path, data, no_overwrite=False):
    directory, fn = os.path.split(path)
    # make a new path
    try:
        os.makedirs(directory)
    except:
        pass
    if no_overwrite and os.path.isfile(path):
        raise Exception('data file already exists')
    with open(path, 'wb+') as f:
        pickle.dump(data, f)


def load(path):
    # make a new path
    with open(path, 'rb') as f:
        data = pickle.load(f)
        return data

# from ramuel.yaml import YAML
#
#
# def save(path, data):
#     yaml = YAML(type="safe")
#     yaml.load()
