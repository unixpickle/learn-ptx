from math import ceil

import numpy as np

from .context import compile_function, gpu_to_numpy, measure_time, numpy_to_gpu, sync


def sort_bitonic_warp():
    fn = compile_function("sort_bitonic_warp.ptx", "sortBitonicWarp")
    inputs = np.random.normal(size=[16384, 32]).astype(np.float32)
    input_buf = numpy_to_gpu(inputs)
    with measure_time() as timer:
        fn(
            input_buf,
            grid=(inputs.shape[0], 1, 1),
            block=(32, 1, 1),
        )
    sync()
    results = gpu_to_numpy(input_buf, inputs.shape, inputs.dtype)
    expected = np.sort(inputs, axis=-1)
    print(f"took {timer()} seconds")
    assert np.allclose(results, expected), f"\n{results=}\n{expected=}"


if __name__ == "__main__":
    sort_bitonic_warp()
