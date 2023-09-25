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
        .reg .u64  %inAddr;
        .reg .u64  %outAddr;
        .reg .u32  %offset;
        .reg .u32  %offsetTmp1;
        .reg .u32  %offsetTmp2;
        .reg .u64  %offset64;
        .reg .u32  %size;
        .reg .f32  %val;

        ld.param.u64 %inAddr, [inputPtr];
        ld.param.u64 %outAddr, [outputPtr];
        ld.param.u32 %size, [n];

        // Compute the offset as ctaid.x*ntid.x + tid.x
        mov.u32 %offsetTmp1, %ctaid.x;
        mov.u32 %offsetTmp2, %ntid.x;
        mul.lo.u32 %offset, %offsetTmp1, %offsetTmp2;
        mov.u32 %offsetTmp1, %tid.x;
        add.u32 %offset, %offset, %offsetTmp1;
        cvt.u64.u32 %offset64, %offset;
        mul.lo.u64 %offset64, %offset64, 4;

        setp.lt.u32 %p1, %offset, %size;

        add.u64 %inAddr, %inAddr, %offset64;
        add.u64 %outAddr, %outAddr, %offset64;

        @%p1 ld.global.f32 %val, [%inAddr];
        @%p1 sqrt.approx.f32 %val, %val;
        @%p1 st.global.f32 [%outAddr], %val;
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
