from math import ceil

import numpy as np

from .context import compile_function, gpu_to_numpy, measure_time, numpy_to_gpu, sync


def sort_bitonic_warp():
    fn = compile_function("sort_bitonic_warp.ptx", "sortBitonicWarp")
    inputs = np.random.normal(size=[16384 * 32, 32]).astype(np.float32)
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


def sort_bitonic_warp_v2():
    fn = compile_function("sort_bitonic_warp_v2.ptx", "sortBitonicWarpV2")
    inputs = np.random.normal(size=[16384 * 32, 32]).astype(np.float32)
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


def sort_bitonic_warp_v3():
    fn = compile_function("sort_bitonic_warp_v3.ptx", "sortBitonicWarpV3")
    inputs = np.random.normal(size=[16384 * 32, 32]).astype(np.float32)
    input_buf = numpy_to_gpu(inputs)
    loop_count = 8  # more work per thread
    loop_stride = (int(np.prod(inputs.shape)) * 4) // loop_count
    with measure_time() as timer:
        fn(
            input_buf,
            np.int64(loop_count),
            np.int64(loop_stride),
            grid=((inputs.shape[0] // 8) // loop_count, 1, 1),
            block=(32, 8, 1),
        )
    sync()
    results = gpu_to_numpy(input_buf, inputs.shape, inputs.dtype)
    expected = np.sort(inputs, axis=-1)
    print(f"took {timer()} seconds")
    assert np.allclose(results, expected), f"\n{results=}\n{expected=}"


def sort_bitonic_block():
    fn = compile_function("sort_bitonic_block.ptx", "sortBitonicBlock")
    inputs = np.random.normal(size=[16384, 1024]).astype(np.float32)
    input_buf = numpy_to_gpu(inputs)
    with measure_time() as timer:
        fn(
            input_buf,
            grid=(inputs.shape[0], 1, 1),
            block=(32, 32, 1),
        )
    sync()
    results = gpu_to_numpy(input_buf, inputs.shape, inputs.dtype)
    expected = np.sort(inputs, axis=-1)
    print(f"took {timer()} seconds")
    assert np.allclose(results, expected), f"\n{results=}\n{expected=}"


if __name__ == "__main__":
    sort_bitonic_warp_v3()
