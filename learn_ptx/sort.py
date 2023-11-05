import time
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


def sort_bitonic_block_v2():
    fn = compile_function("sort_bitonic_block_v2.ptx", "sortBitonicBlockV2")
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


def sort_merge_global():
    warp_fn = compile_function("sort_bitonic_warp_v2.ptx", "sortBitonicWarpV2")
    global_fn = compile_function("sort_merge_global.ptx", "sortMergeGlobal")
    inputs = np.random.normal(size=[2**28]).astype(np.float32)
    tmp = np.zeros_like(inputs)
    input_buf = numpy_to_gpu(inputs)
    tmp_buf = numpy_to_gpu(tmp)
    num_el = int(np.prod(inputs.shape))
    print("sorting on GPU...")
    with measure_time() as timer:
        # Sort per warp before merging.
        warp_fn(
            input_buf,
            grid=(num_el // 32, 1, 1),
            block=(32, 1, 1),
        )
        n_sorted = 32
        while n_sorted < num_el:
            # Maximum of 8 warps per block, to maximize occupancy when possible.
            concurrency = min(num_el // (2 * n_sorted), 8)
            grid_size = num_el // (2 * n_sorted * concurrency)
            global_fn(
                input_buf,
                tmp_buf,
                np.int64(n_sorted),
                grid=(grid_size, 1, 1),
                block=(32, concurrency, 1),
            )
            input_buf, tmp_buf = tmp_buf, input_buf
            n_sorted *= 2
    sync()
    results = gpu_to_numpy(input_buf, inputs.shape, inputs.dtype)
    print("sorting on CPU...")
    t1 = time.time()
    expected = np.sort(inputs, axis=-1)
    t2 = time.time()
    print(f"took {timer()} seconds on GPU and {t2 - t1} seconds on CPU")
    assert np.allclose(results, expected), f"\n{results=}\n{expected=}"


if __name__ == "__main__":
    sort_merge_global()
