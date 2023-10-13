from typing import Any, Callable

import numpy as np

from .context import compile_function, gpu_to_numpy, measure_time, numpy_to_gpu, sync


def matmul_simple_block_v1():
    fn = compile_function("matmul_simple_block_v1.ptx", "blockedMatmul")
    evaluate_matmul_fn(fn)


def matmul_simple_block_v2():
    fn = compile_function("matmul_simple_block_v2.ptx", "blockedMatmulV2")
    evaluate_matmul_fn(fn)


def matmul_simple_block_v3():
    fn = compile_function("matmul_simple_block_v3.ptx", "blockedMatmulV3")
    evaluate_matmul_fn(fn)


def matmul_simple_block_v4():
    fn = compile_function("matmul_simple_block_v4.ptx", "blockedMatmulV4")
    evaluate_matmul_fn(fn)


def matmul_inner_loop():
    fn = compile_function("matmul_inner_loop.ptx", "simpleMatmul")
    evaluate_matmul_fn(fn)


def matmul_big_blocks():
    fn = compile_function("matmul_big_blocks.ptx", "bigBlocksMatmul")

    def call_fn(A: np.ndarray, B: np.ndarray, A_buf: Any, B_buf: Any, out_buf: Any):
        fn(
            A_buf,
            B_buf,
            out_buf,
            np.int32(A.shape[0] // 32),
            grid=(
                A.shape[0] // 32,
                A.shape[1] // 32,
                1,
            ),
            block=(32, 8, 1),
        )

    generic_eval_matmul(call_fn)


def evaluate_matmul_fn(fn: Callable):
    def call_fn(A: np.ndarray, B: np.ndarray, A_buf: Any, B_buf: Any, out_buf: Any):
        block_size = 32
        fn(
            A_buf,
            B_buf,
            out_buf,
            np.int32(A.shape[0] // block_size),
            grid=(
                A.shape[0] // block_size,
                A.shape[1] // block_size,
                1,
            ),
            block=(block_size, block_size, 1),
        )

    generic_eval_matmul(call_fn)


def generic_eval_matmul(fn: Callable, block_mult: int = 1):
    size = 8192
    A = np.random.normal(size=[size, size]).astype(np.float32)
    B = np.random.normal(size=[size, size]).astype(np.float32)
    A_buf = numpy_to_gpu(A)
    B_buf = numpy_to_gpu(B)
    out_buf = numpy_to_gpu(A * 0)
    with measure_time() as timer:
        fn(
            A,
            B,
            A_buf,
            B_buf,
            out_buf,
        )
    sync()
    results = gpu_to_numpy(out_buf, A.shape, A.dtype)
    expected = A @ B
    print(f"maximum absolute error of matmul is {np.abs(results - expected).max()}")
    print(f"time elapsed: {timer()}")


if __name__ == "__main__":
    matmul_big_blocks()
