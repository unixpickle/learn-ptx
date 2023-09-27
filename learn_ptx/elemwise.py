from math import ceil

import numpy as np

from .context import compile_function, gpu_to_numpy, numpy_to_gpu, sync


def sqrt_example():
    code = """
    .version 7.0
    .target sm_50 // enough for my Titan X
    .address_size 64

    .visible .entry sqrtElements(
        .param .u64 inputPtr,
        .param .u64 outputPtr,
        .param .u32 n
    ) {
        .reg .pred %p1;
        .reg .u64  %addr;
        .reg .u32  %tmp<2>;
        .reg .u64  %offset;
        .reg .f32  %val;

        // Compute the offset as ctaid.x*ntid.x + tid.x
        mov.u32 %tmp0, %ctaid.x;
        mov.u32 %tmp1, %ntid.x;
        mul.lo.u32 %tmp0, %tmp0, %tmp1;
        mov.u32 %tmp1, %tid.x;
        add.u32 %tmp1, %tmp0, %tmp1;
        cvt.u64.u32 %offset, %tmp1;
        mul.lo.u64 %offset, %offset, 4;

        // Mask out out-of-bounds accesses.
        ld.param.u32 %tmp0, [n];
        setp.lt.u32 %p1, %tmp1, %tmp0;

        // Load the value from memory.
        ld.param.u64 %addr, [inputPtr];
        add.u64 %addr, %addr, %offset;
        @%p1 ld.global.f32 %val, [%addr];

        // Element-wise operation itself.
        @%p1 sqrt.approx.f32 %val, %val;

        // Store back the output.
        ld.param.u64 %addr, [outputPtr];
        add.u64 %addr, %addr, %offset;
        @%p1 st.global.f32 [%addr], %val;
    }
    """
    fn = compile_function(code, "sqrtElements")
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
