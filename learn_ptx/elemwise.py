from math import ceil

import numpy as np

from .context import compile_function, gpu_to_numpy, numpy_to_gpu, sync


def sqrt_example():
    fn = compile_function("elemwise_sqrt.ptx", "sqrtElements")
    inputs = np.abs(np.random.normal(size=[10000]).astype(np.float32))
    input_buf = numpy_to_gpu(inputs)
    output_buf = numpy_to_gpu(inputs)
    block_size = 1024
    fn(
        input_buf,
        output_buf,
        np.int32(len(inputs) - 10),
        grid=(ceil(len(inputs) / block_size), 1, 1),
        block=(block_size, 1, 1),
    )
    sync()
    results = gpu_to_numpy(output_buf, inputs.shape, inputs.dtype)
    expected = np.sqrt(inputs)
    print(
        f"maximum absolute error of sqrt is {np.abs(results[:-10] - expected[:-10]).max()}"
    )
    print(
        f"maximum absolute error of masked is {np.abs(results[-10:] - inputs[-10:]).max()}"
    )


if __name__ == "__main__":
    sqrt_example()
