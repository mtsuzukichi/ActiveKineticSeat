# shared_memory_manager.py

import numpy as np
from multiprocessing import shared_memory

def create_shared_memory(shape, dtype):
    n_bytes = int(np.prod(shape) * np.dtype(dtype).itemsize)
    shm = shared_memory.SharedMemory(create=True, size=n_bytes)
    array = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
    array.fill(0)
    return shm , array

def open_shared_memory(name):
    return shared_memory.SharedMemory(name=name)
