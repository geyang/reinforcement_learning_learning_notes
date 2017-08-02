import os
import shutil

from helpers import save, load


def test():
    try:
        path = "./test-data/example.pkl"
        directory, _ = os.path.split(path)
        save(path, {"some": list(range(100))})
        ob = load(path)
        assert ob["some"] == list(range(100)), 'object should be correct'
        # save again
        save(path, {"some": list(range(200))})
        ob = load(path)
        assert ob["some"] == list(range(200)), 'object should be correct'

        # Now turn on `no_overwrite` flag
        try:
            has_erred = False
            save(path, {"some": list(range(200))}, no_overwrite=True)
        except:
            has_erred = True
        if not has_erred:
            raise Exception('Should raise file already exist error.')
    except Exception as e:
        raise e
    finally:
        # clean up afterward.
        shutil.rmtree("./test-data")





