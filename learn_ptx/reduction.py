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


def reduction_all_max_naive_opt_novec():
    fn = compile_function(
        "reduction_all_max_naive_opt_novec.ptx", "reductionAllMaxNaiveOptNoVec"
    )
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


def reduction_all_max_flexible_multistep():
    # opt_reduce = compile_function(
    #     "reduction_all_max_naive_opt_flexible.ptx", "reductionAllMaxNaiveOptFlexible"
    # )
    # opt_reduce = compile_function(
    #     "reduction_all_max_naive_opt_flexible_novec.ptx",
    #     "reductionAllMaxNaiveOptFlexibleNovec",
    # )
    opt_reduce = compile_function(
        "reduction_all_max_naive_opt_flexible_widevec.ptx",
        "reductionAllMaxNaiveOptFlexibleWidevec",
    )
    small_reduce = compile_function(
        "reduction_all_max_naive.ptx", "reductionAllMaxNaive"
    )
    inputs = np.random.uniform(size=[16384**2]).astype(np.float32)
    inputs[16384 * 8192 + 1337] = 1.5  # hide a needle in the haystack
    expected = np.max(inputs)
    outputs = np.zeros([1024], dtype=np.float32)
    input_buf = numpy_to_gpu(inputs)
    warp_values = [1, 2, 4, 8, 16, 32]
    block_values = [1, 2, 4, 8, 16, 32, 64]
    output_grid = np.zeros([len(warp_values), len(block_values)])
    for i, n_warps in enumerate(warp_values):
        for j, n_blocks in enumerate(block_values):
            output_buf = numpy_to_gpu(outputs)
            with measure_time() as timer:
                opt_reduce(
                    input_buf,
                    output_buf,
                    np.int64((len(inputs) // n_blocks) // (n_warps * 32)),
                    grid=(n_blocks, 1, 1),
                    block=(32, n_warps, 1024 // (32 * n_warps)),
                )
                # Always reduce 1024 values, even though some of them
                # may be zero.
                small_reduce(
                    output_buf,
                    output_buf,
                    np.int64(1),
                    grid=(1, 1, 1),
                    block=(1024, 1, 1),
                )
            sync()
            results = gpu_to_numpy(output_buf, outputs.shape, outputs.dtype)
            rate = ((int(np.prod(inputs.shape)) * 4) / timer()) / (2**30)
            print(f"{n_warps=} {n_blocks=} GiB/s={rate}")
            assert results[0] == expected, f"{results[0]=} {expected=}"
            output_grid[i, j] = rate
    rows = [["", *[f"{i} warps" for i in warp_values]]]
    for label, row in zip(block_values, output_grid.T):
        rows.append([f"{label} SMs", *[f"{x:.02f} GiB/s" for x in row]])
    print("<table>")
    for row in rows:
        print("<tr>")
        for item in row:
            print(f"<td>{item}</td>")
        print("</tr>")
    print("</table>")


if __name__ == "__main__":
    reduction_all_max_flexible_multistep()
