import time
import functools
import os
import inspect

def timeit_decorator(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        file_name = inspect.getfile(func)
        file_name = os.path.basename(file_name)
        print(f"A função {func.__name__} no arquivo {file_name} levou {elapsed_time:.4f} segundos para ser completada")
        return result
    return wrapper