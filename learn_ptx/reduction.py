from math import ceil

import numpy as np

from .context import compile_function, gpu_to_numpy, measure_time, numpy_to_gpu, sync


def reduction_bool_naive():
    fn = compile_function("reduction_bool_naive.ptx", "reductionBoolNaive")
    inputs = np.random.normal(size=[16384, 16384]).astype(np.float32)
    threshold = np.median(inputs.min(axis=-1))
    outputs = np.zeros([inputs.shape[0]], dtype=np.uint8)
    input_buf = numpy_to_gpu(inputs)
    output_buf = numpy_to_gpu(outputs)
    with measure_time() as timer:
        fn(
            input_buf,
            output_buf,
            np.float32(threshold),
            np.int64(inputs.shape[1]),
            grid=(inputs.shape[0], 1, 1),
            block=(1, 1, 1),
        )
    sync()
    results = gpu_to_numpy(output_buf, outputs.shape, outputs.dtype)
    expected = (inputs < threshold).any(axis=-1)
    print(f"took {timer()} seconds")
    print(f"disagreement frac {np.mean((expected != results).astype(np.float32))}")
    print(f"true frac {results.astype(np.float32).mean()}")


if __name__ == "__main__":
    reduction_bool_naive()
