import os
from contextlib import contextmanager
from functools import lru_cache
from typing import Callable, Sequence, Union

import numpy as np
import pycuda.autoinit
from pycuda.compiler import DynamicModule
from pycuda.driver import (
    Context,
    DeviceAllocation,
    Event,
    from_device,
    jit_input_type,
    to_device,
)

KERNEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernels")


@lru_cache(maxsize=8)
def compile_function(code_filename: str, function_name: str) -> Callable:
    module = DynamicModule()
    with open(os.path.join(KERNEL_DIR, code_filename), "rb") as f:
        module.add_data(f.read(), jit_input_type.PTX, name="kernel.ptx")
    module.link()
    return module.get_function(function_name)


def numpy_to_gpu(arr: np.ndarray) -> DeviceAllocation:
    return to_device(arr)


def gpu_to_numpy(
    allocation: DeviceAllocation, shape: Sequence[int], dtype: Union[np.dtype, str]
) -> np.ndarray:
    return from_device(allocation, shape, dtype)


def sync():
    Context.synchronize()


@contextmanager
def measure_time() -> Callable[[], float]:
    start = Event()
    end = Event()

    start.record()
    start.synchronize()

    def delay_fn() -> float:
        return start.time_till(end) / 1000

    yield delay_fn

    end.record()
    end.synchronize()
