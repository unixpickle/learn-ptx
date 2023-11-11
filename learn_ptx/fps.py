import time

import numpy as np

from .context import compile_function, gpu_to_numpy, measure_time, numpy_to_gpu, sync


def fps_block():
    fn = compile_function("fps_block.ptx", "farthestPointSampleBlock")
    inputs = np.random.normal(size=[2**19, 3]).astype(np.float32)
    outputs = np.zeros([4096, 3]).astype(np.float32)
    tmp = np.zeros([len(inputs)]).astype(np.float32)
    input_buf = numpy_to_gpu(inputs)
    output_buf = numpy_to_gpu(outputs)
    tmp_buf = numpy_to_gpu(tmp)
    with measure_time() as timer:
        fn(
            input_buf,
            tmp_buf,
            output_buf,
            np.int32(len(inputs)),
            np.int32(len(outputs)),
            grid=(1, 1, 1),
            block=(1024, 1, 1),
        )
    sync()
    results = gpu_to_numpy(output_buf, inputs.shape, inputs.dtype)
    print(f"took {timer():.05f} seconds on GPU")
    t1 = time.time()
    expected = fps_on_cpu(input_buf, len(outputs))
    t2 = time.time()
    print(f"took {(t2 - t1):.05f} seconds on CPU")
    print(f"maximum absolute error is {np.abs(results - expected).max()}")


def fps_on_cpu(points: np.ndarray, n: int) -> np.ndarray:
    results = np.zeros([n, 3], dtype=points.dtype)
    results[0] = points[0]
    dists = ((points - points[0]) ** 2).sum(-1)
    dists[0] = -1
    for i in range(1, n):
        idx = np.argmin(dists)
        point = points[idx]
        dists = np.minimum(dists, ((points - point) ** 2).sum(-1))
        dists[idx] = -1
        results[i] = point
    return results


if __name__ == "__main__":
    fps_block()
