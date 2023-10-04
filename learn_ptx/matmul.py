from typing import Callable

import numpy as np

from .context import compile_function, gpu_to_numpy, measure_time, numpy_to_gpu, sync


def square_matrix_simple_block():
    code = """
    .version 7.0
    .target sm_50 // enough for my Titan X
    .address_size 64

    .visible .entry blockedMatmul (
        .param .u64 ptrA,
        .param .u64 ptrB,
        .param .u64 ptrOut,
        .param .u32 numBlocks
    ) {
        .reg .pred %p0;
        .reg .u64  %tmp<2>;
        .reg .u32  %halfTmp<2>;
        .reg .u32  %localOffset;
        .reg .u64  %offsetX;
        .reg .u64  %offsetY;
        .reg .u64  %stride;
        .reg .u32  %i;
        .reg .u32  %j;
        .reg .u32  %numBlocks;
        .reg .u64  %ptrA;
        .reg .u64  %ptrB;
        .reg .u64  %ptrOut;
        .reg .f32  %acc;
        .reg .f32  %val<2>;
        .shared .align 4 .f32 loadedA[1024]; // should be ntid.x*ntid.y
        .shared .align 4 .f32 loadedB[1024]; // should be ntid.x*ntid.y

        ld.param.u32 %numBlocks, [numBlocks];
        ld.param.u64 %ptrA, [ptrA];
        ld.param.u64 %ptrB, [ptrB];
        ld.param.u64 %ptrOut, [ptrOut];

        // Local offset is (tid.y*ntid.x + tid.x) * sizeof(float32)
        mov.u32 %localOffset, %tid.y;
        mov.u32 %halfTmp0, %ntid.x;
        mul.lo.u32 %localOffset, %localOffset, %halfTmp0;
        mov.u32 %halfTmp0, %tid.x;
        add.u32 %localOffset, %localOffset, %halfTmp0;
        mul.lo.u32 %localOffset, %localOffset, 4;

        // Compute offsets in the output matrix.
        // offsetX = ctaid.x * ntid.x
        cvt.u64.u32 %offsetX, %ctaid.x;
        cvt.u64.u32 %tmp0, %ntid.x;
        mul.lo.u64 %offsetX, %offsetX, %tmp0;
        // offsetY = ctaid.y * ntid.y
        cvt.u64.u32 %offsetY, %ctaid.y;
        cvt.u64.u32 %tmp0, %ntid.y;
        mul.lo.u64 %offsetY, %offsetY, %tmp0;

        // Stride is ntid.x * numBlocks
        cvt.u64.u32 %stride, %ntid.x;
        cvt.u64.u32 %tmp0, %numBlocks;
        mul.lo.u64 %stride, %stride, %tmp0;

        // Zero out our local portion of the output.
        // mov.u32 %halfTmp0, output;
        // add.u32 %halfTmp0, %halfTmp0, %localOffset;
        // mov.f32 %val0, 0.0;
        // st.shared.f32 [%halfTmp0], %val0;
        mov.f32 %acc, 0.0;

        mov.u32 %i, 0;
    loop_start:
        // Don't write into memory until other threads are
        // caught up, to avoid races.
        bar.sync 0;

        // Our block offset in A is (i*ntid.x + tid.x, offsetY + tid.y)
        cvt.u64.u32 %tmp0, %i;
        cvt.u64.u32 %tmp1, %ntid.x;
        mul.lo.u64 %tmp0, %tmp0, %tmp1;
        cvt.u64.u32 %tmp1, %tid.x;
        add.u64 %tmp0, %tmp0, %tmp1;
        cvt.u64.u32 %tmp1, %tid.y;
        add.u64 %tmp1, %tmp1, %offsetY;
        // Compute pointer as &ptrA[y*stride+x]
        mul.lo.u64 %tmp1, %tmp1, %stride;
        add.u64 %tmp0, %tmp0, %tmp1;
        mul.lo.u64 %tmp0, %tmp0, 4;
        add.u64 %tmp0, %tmp0, %ptrA;
        // Output pointer
        mov.u32 %halfTmp0, loadedA;
        add.u32 %halfTmp0, %halfTmp0, %localOffset;
        // Copy to local memory
        ld.global.f32 %val0, [%tmp0];
        st.shared.f32 [%halfTmp0], %val0;

        // Our block offset in B is (offsetX + tid.x, i*ntid.y + tid.y)
        cvt.u64.u32 %tmp0, %i;
        cvt.u64.u32 %tmp1, %ntid.y;
        mul.lo.u64 %tmp0, %tmp0, %tmp1;
        cvt.u64.u32 %tmp1, %tid.y;
        add.u64 %tmp0, %tmp0, %tmp1;
        cvt.u64.u32 %tmp1, %tid.x;
        add.u64 %tmp1, %tmp1, %offsetX;
        // Compute global offset as &ptrB[y*stride+x]
        mul.lo.u64 %tmp0, %tmp0, %stride;
        add.u64 %tmp0, %tmp0, %tmp1;
        mul.lo.u64 %tmp0, %tmp0, 4;
        add.u64 %tmp0, %tmp0, %ptrB;
        // Output pointer
        mov.u32 %halfTmp0, loadedB;
        add.u32 %halfTmp0, %halfTmp0, %localOffset;
        // Copy to local memory
        ld.global.f32 %val0, [%tmp0];
        st.shared.f32 [%halfTmp0], %val0;

        bar.sync 0;

        mov.u32 %j, 0;
    inner_loop_start:
        // Offset in loadedA is j + tid.y*ntid.x
        mov.u32 %halfTmp0, %ntid.x;
        mov.u32 %halfTmp1, %tid.y;
        mul.lo.u32 %halfTmp1, %halfTmp1, %halfTmp0;
        add.u32 %halfTmp1, %halfTmp1, %j;
        mul.lo.u32 %halfTmp1, %halfTmp1, 4;
        mov.u32 %halfTmp0, loadedA;
        add.u32 %halfTmp0, %halfTmp0, %halfTmp1;
        ld.shared.f32 %val0, [%halfTmp0];

        // Offset in loadedB is tid.x + j*ntid.x
        mov.u32 %halfTmp1, %ntid.x;
        mul.lo.u32 %halfTmp1, %halfTmp1, %j;
        mov.u32 %halfTmp0, %tid.x;
        add.u32 %halfTmp1, %halfTmp1, %halfTmp0;
        mul.lo.u32 %halfTmp1, %halfTmp1, 4;
        mov.u32 %halfTmp0, loadedB;
        add.u32 %halfTmp0, %halfTmp0, %halfTmp1;
        ld.shared.f32 %val1, [%halfTmp0];

        // Can be optimized to fused operation.
        mul.f32 %val1, %val0, %val1;
        add.f32 %acc, %acc, %val1;

        // j += 1; loop while j < ntid.x
        mov.u32 %halfTmp0, %ntid.x;
        add.u32 %j, %j, 1;
        setp.lt.u32 %p0, %j, %halfTmp0;
        @%p0 bra inner_loop_start;

    inner_loop_end:
        // i += 1; loop while i < numBlocks
        add.u32 %i, %i, 1;
        setp.lt.u32 %p0, %i, %numBlocks;
        @%p0 bra loop_start;

    loop_end:
        // Write back to output memory.

        // Output address is offsetX+tid.x + stride*(offsetY+tid.y)
        cvt.u64.u32 %tmp0, %tid.y;
        add.u64 %tmp0, %tmp0, %offsetY;
        mul.lo.u64 %tmp0, %tmp0, %stride;
        cvt.u64.u32 %tmp1, %tid.x;
        add.u64 %tmp1, %tmp1, %offsetX;
        add.u64 %tmp0, %tmp0, %tmp1;
        mul.lo.u64 %tmp0, %tmp0, 4;
        add.u64 %tmp0, %tmp0, %ptrOut;

        st.global.f32 [%tmp0], %acc;
    }
    """
    fn = compile_function(code, "blockedMatmul")
    evaluate_matmul_fn(fn)


def square_matrix_simple_block_v2():
    code = """
    .version 7.0
    .target sm_50 // enough for my Titan X
    .address_size 64

    .visible .entry blockedMatmulV2 (
        .param .u64 ptrA,
        .param .u64 ptrB,
        .param .u64 ptrOut,
        .param .u32 numBlocks
    ) {
        .reg .pred %p0;
        .reg .u64  %dtmp<2>;
        .reg .u32  %stmp<3>;

        // Offset in loadedA / loadedB that we write to.
        .reg .u32  %loadOffset;

        // Attributes of the thread/CTA.
        .reg .u32  %blockSize;
        .reg .u32  %tidX;
        .reg .u32  %tidY;

        .reg .u64  %offsetX;
        .reg .u64  %offsetY;
        .reg .u64  %stride;
        .reg .u32  %i;
        .reg .u32  %j;
        .reg .u32  %numBlocks;
        .reg .u64  %ptrA;
        .reg .u64  %ptrB;
        .reg .u64  %ptrOut;
        .reg .f32  %acc;
        .reg .f32  %val<2>;
        .shared .align 4 .f32 loadedA[1024]; // should be at least blockSize^2
        .shared .align 4 .f32 loadedB[1024]; // should be at least blockSize^2

        ld.param.u32 %numBlocks, [numBlocks];
        ld.param.u64 %ptrA, [ptrA];
        ld.param.u64 %ptrB, [ptrB];
        ld.param.u64 %ptrOut, [ptrOut];

        mov.u32 %blockSize, %ntid.x;
        mov.u32 %tidX, %tid.x;
        mov.u32 %tidY, %tid.y;

        // Local offset is (tid.y*blockSize + tid.x) * sizeof(float32)
        mul.lo.u32 %loadOffset, %tidY, %blockSize;
        add.u32 %loadOffset, %loadOffset, %tidX;
        shl.b32 %loadOffset, %loadOffset, 2;

        // For computing offsetX, offsetY, and stride, we use
        // %dtmp0 to store a 64-bit version of %blockSize.
        cvt.u64.u32 %dtmp0, %blockSize; // %dtmp0 = %blockSize

        // Compute offsets in the output matrix.
        // offsetX = ctaid.x * ntid.x = ctaid.x * blockSize
        cvt.u64.u32 %offsetX, %ctaid.x;
        mul.lo.u64 %offsetX, %offsetX, %dtmp0;
        // offsetY = ctaid.y * ntid.y = ctaid.y * blockSize
        cvt.u64.u32 %offsetY, %ctaid.y;
        mul.lo.u64 %offsetY, %offsetY, %dtmp0;

        // Stride is blockSize * numBlocks;
        cvt.u64.u32 %stride, %numBlocks;
        mul.lo.u64 %stride, %stride, %dtmp0;

        // We will accumulate into this register.
        mov.f32 %acc, 0.0;

        // We will calculate block offset in A in %ptrA as
        // (i*ntid.x + tid.x, offsetY + tid.y)
        // = i*ntid.x + tid.x + stride*(offsetY+tid.y)
        cvt.u64.u32 %dtmp0, %tidY;
        add.u64 %dtmp0, %dtmp0, %offsetY;
        mul.lo.u64 %dtmp0, %dtmp0, %stride;
        cvt.u64.u32 %dtmp1, %tidX;
        add.u64 %dtmp0, %dtmp0, %dtmp1;
        shl.b64 %dtmp0, %dtmp0, 2;
        add.u64 %ptrA, %ptrA, %dtmp0;

        // We will calculate our block offset in B in %ptrB as
        // (offsetX + tid.x, i*ntid.y + tid.y)
        // = offsetX + tid.x + i*stride*blockSize + stride*tid.y
        cvt.u64.u32 %dtmp0, %tidY;
        mul.lo.u64 %dtmp0, %dtmp0, %stride;
        cvt.u64.u32 %dtmp1, %tidX;
        add.u64 %dtmp0, %dtmp0, %dtmp1;
        add.u64 %dtmp0, %dtmp0, %offsetX;
        shl.b64 %dtmp0, %dtmp0, 2;
        add.u64 %ptrB, %ptrB, %dtmp0;

        // Set %dtmp0 and %dtmp1 to strides in A and B, respectively.
        // Stride in ptrA is blockSize*4
        cvt.u64.u32 %dtmp0, %blockSize;
        shl.b64 %dtmp0, %dtmp0, 2;
        // Stride in ptrB is stride*blockSize*4
        mul.lo.u64 %dtmp1, %dtmp0, %stride;

        mov.u32 %i, 0;
    loop_start:
        // Don't write into memory until other threads are
        // caught up, to avoid races.
        bar.sync 0;

        // Read our entry from A into shared memory.
        mov.u32 %stmp0, loadedA;
        add.u32 %stmp0, %stmp0, %loadOffset;
        // Copy to local memory
        ld.global.f32 %val0, [%ptrA];
        st.shared.f32 [%stmp0], %val0;
        add.u64 %ptrA, %ptrA, %dtmp0;

        // Read our entry from B into shared memory.
        mov.u32 %stmp0, loadedB;
        add.u32 %stmp0, %stmp0, %loadOffset;
        // Copy to local memory
        ld.global.f32 %val0, [%ptrB];
        st.shared.f32 [%stmp0], %val0;
        add.u64 %ptrB, %ptrB, %dtmp1;

        bar.sync 0;

        // %stmp0 will be address in A.
        // It will be &loadedA[j + tid.y*ntid.x], starting at j=0
        mul.lo.u32 %stmp0, %tidY, %blockSize;
        shl.b32 %stmp0, %stmp0, 2;
        mov.u32 %stmp1, loadedA;
        add.u32 %stmp0, %stmp0, %stmp1;

        // %stmp1 will be address in B.
        // It will be &loadedB[tid.x + j*ntid.x] starting at j=0
        mov.u32 %stmp1, loadedB;
        shl.b32 %stmp2, %tidX, 2;
        add.u32 %stmp1, %stmp1, %stmp2;
        shl.b32 %stmp2, %blockSize, 2;

        mov.u32 %j, 0;
    inner_loop_start:
        // Offset in loadedA is j + tid.y*ntid.x
        ld.shared.f32 %val0, [%stmp0];
        add.u32 %stmp0, %stmp0, 4;

        // Offset in loadedB is tid.x + j*ntid.x
        ld.shared.f32 %val1, [%stmp1];
        add.u32 %stmp1, %stmp1, %stmp2;

        // Can be optimized to fused operation.
        mul.f32 %val1, %val0, %val1;
        add.f32 %acc, %acc, %val1;

        // j += 1; loop while j < ntid.x
        add.u32 %j, %j, 1;
        setp.lt.u32 %p0, %j, %blockSize;
        @%p0 bra inner_loop_start;

    inner_loop_end:
        // i += 1; loop while i < numBlocks
        add.u32 %i, %i, 1;
        setp.lt.u32 %p0, %i, %numBlocks;
        @%p0 bra loop_start;

    loop_end:
        // Write back to output memory.

        // Output address is offsetX+tid.x + stride*(offsetY+tid.y)
        cvt.u64.u32 %dtmp0, %tidY;
        add.u64 %dtmp0, %dtmp0, %offsetY;
        mul.lo.u64 %dtmp0, %dtmp0, %stride;
        cvt.u64.u32 %dtmp1, %tidX;
        add.u64 %dtmp1, %dtmp1, %offsetX;
        add.u64 %dtmp0, %dtmp0, %dtmp1;
        shl.b64 %dtmp0, %dtmp0, 2;
        add.u64 %dtmp0, %dtmp0, %ptrOut;

        st.global.f32 [%dtmp0], %acc;
    }
    """
    fn = compile_function(code, "blockedMatmulV2")
    evaluate_matmul_fn(fn)


def evaluate_matmul_fn(fn: Callable):
    size = 8192
    A = np.random.normal(size=[size, size]).astype(np.float32)
    B = np.random.normal(size=[size, size]).astype(np.float32)
    A_buf = numpy_to_gpu(A)
    B_buf = numpy_to_gpu(B)
    out_buf = numpy_to_gpu(A * 0)
    with measure_time() as timer:
        block_size = 32
        fn(
            A_buf,
            B_buf,
            out_buf,
            np.int32(A.shape[0] // block_size),
            grid=(A.shape[0] // block_size, A.shape[1] // block_size, 1),
            block=(block_size, block_size, 1),
        )
    sync()
    results = gpu_to_numpy(out_buf, A.shape, A.dtype)
    expected = A @ B
    print(f"maximum absolute error of matmul is {np.abs(results - expected).max()}")
    print(f"time elapsed: {timer()}")


if __name__ == "__main__":
    square_matrix_simple_block_v2()
