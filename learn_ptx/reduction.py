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


def reduction_bool_warp():
    fn = compile_function("reduction_bool_warp.ptx", "reductionBoolWarp")
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
            block=(32, 1, 1),
        )
    sync()
    results = gpu_to_numpy(output_buf, outputs.shape, outputs.dtype)
    expected = (inputs < threshold).any(axis=-1)
    print(f"took {timer()} seconds")
    print(f"disagreement frac {np.mean((expected != results).astype(np.float32))}")
    print(f"true frac {results.astype(np.float32).mean()}")


def reduction_bool_warp_vec():
    fn = compile_function("reduction_bool_warp_vec.ptx", "reductionBoolWarpVec")
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
            block=(32, 1, 1),
        )
    sync()
    results = gpu_to_numpy(output_buf, outputs.shape, outputs.dtype)
    expected = (inputs < threshold).any(axis=-1)
    print(f"took {timer()} seconds")
    print(f"disagreement frac {np.mean((expected != results).astype(np.float32))}")
    print(f"true frac {results.astype(np.float32).mean()}")


def reduction_trans_bool_naive():
    fn = compile_function("reduction_trans_bool_naive.ptx", "reductionTransBoolNaive")
    inputs = np.random.uniform(size=[16384, 16384]).astype(np.float32)
    threshold = np.median(inputs.min(axis=-1))
    outputs = np.zeros([inputs.shape[1]], dtype=np.uint8)
    input_buf = numpy_to_gpu(inputs)
    output_buf = numpy_to_gpu(outputs)
    with measure_time() as timer:
        fn(
            input_buf,
            output_buf,
            np.float32(threshold),
            np.int64(inputs.shape[0]),
            grid=(inputs.shape[1], 1, 1),
            block=(1, 1, 1),
        )
    sync()
    results = gpu_to_numpy(output_buf, outputs.shape, outputs.dtype)
    expected = (inputs < threshold).any(axis=0)
    print(f"took {timer()} seconds")
    print(f"disagreement frac {np.mean((expected != results).astype(np.float32))}")
    print(f"true frac {results.astype(np.float32).mean()}")


def reduction_trans_bool_blocked():
    fn = compile_function(
        "reduction_trans_bool_blocked.ptx", "reductionTransBoolBlocked"
    )
    inputs = np.random.uniform(size=[16384, 16384]).astype(np.float32)
    threshold = np.median(inputs.min(axis=-1))
    outputs = np.zeros([inputs.shape[1]], dtype=np.uint8)
    input_buf = numpy_to_gpu(inputs)
    output_buf = numpy_to_gpu(outputs)
    with measure_time() as timer:
        fn(
            input_buf,
            output_buf,
            np.float32(threshold),
            np.int64(inputs.shape[0]),
            grid=(inputs.shape[1] // 256, 1, 1),
            block=(256, 1, 1),
        )
    sync()
    results = gpu_to_numpy(output_buf, outputs.shape, outputs.dtype)
    expected = (inputs < threshold).any(axis=0)
    print(f"took {timer()} seconds")
    print(f"disagreement frac {np.mean((expected != results).astype(np.float32))}")
    print(f"true frac {results.astype(np.float32).mean()}")


def reduction_all_max_naive():
    fn = compile_function("reduction_all_max_naive.ptx", "reductionAllMaxNaive")
    inputs = np.random.uniform(size=[16384**2]).astype(np.float32)
    outputs = np.zeros([1], dtype=np.float32)
    input_buf = numpy_to_gpu(inputs)
    output_buf = numpy_to_gpu(outputs)
    with measure_time() as timer:
        fn(
            input_buf,
            output_buf,
            np.int64(len(inputs) // 1024),
            grid=(1, 1, 1),
            block=(1024, 1, 1),
        )
    sync()
    results = gpu_to_numpy(output_buf, outputs.shape, outputs.dtype)
    expected = np.max(inputs)
    print(f"took {timer()} seconds")
    assert results[0] == expected, f"{results[0]=} {expected=}"


def reduction_all_max_naive_opt():
    fn = compile_function("reduction_all_max_naive_opt.ptx", "reductionAllMaxNaiveOpt")
    inputs = np.random.uniform(size=[16384**2]).astype(np.float32)
    outputs = np.zeros([1], dtype=np.float32)
    input_buf = numpy_to_gpu(inputs)
    output_buf = numpy_to_gpu(outputs)
    with measure_time() as timer:
        fn(
            input_buf,
            output_buf,
            np.int64(len(inputs) // 1024),
            grid=(1, 1, 1),
            block=(1024, 1, 1),
        )
    sync()
    results = gpu_to_numpy(output_buf, outputs.shape, outputs.dtype)
    expected = np.max(inputs)
    print(f"took {timer()} seconds")
    assert results[0] == expected, f"{results[0]=} {expected=}"


def reduction_all_max_multistep():
    opt_reduce = compile_function(
        "reduction_all_max_naive_opt.ptx", "reductionAllMaxNaiveOpt"
    )
    small_reduce = compile_function(
        "reduction_all_max_naive.ptx", "reductionAllMaxNaive"
    )
    inputs = np.random.uniform(size=[16384**2]).astype(np.float32)
    inputs[16384 * 8192 + 1337] = 1.5  # hide a needle in the haystack
    outputs = np.zeros([1024], dtype=np.float32)
    input_buf = numpy_to_gpu(inputs)
    output_buf = numpy_to_gpu(outputs)
    with measure_time() as timer:
        opt_reduce(
            input_buf,
            output_buf,
            np.int64((len(inputs) // 1024) // 1024),
            grid=(1024, 1, 1),
            block=(1024, 1, 1),
        )
        small_reduce(
            output_buf,
            output_buf,
            np.int64(1),
            grid=(1, 1, 1),
            block=(1024, 1, 1),
        )
    sync()
    results = gpu_to_numpy(output_buf, outputs.shape, outputs.dtype)
    expected = np.max(inputs)
    print(f"took {timer()} seconds")
    assert results[0] == expected, f"{results[0]=} {expected=}"


if __name__ == "__main__":
    reduction_all_max_multistep()
