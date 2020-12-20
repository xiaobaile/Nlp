import time


def cal_time(fn):
    """ test the function running time.
    """
    def wrapper(*args, **kwargs):
        start_time = time.time()
        f = fn(*args, **kwargs)
        end_time = time.time()
        print("%s() runtime: %s ms" % (fn.__name__, 1000*(end_time - start_time)))
        return f
    return wrapper
