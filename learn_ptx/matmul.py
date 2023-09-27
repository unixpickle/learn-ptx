import numpy as np

from .context import compile_function, gpu_to_numpy, numpy_to_gpu, sync


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
        .reg .u64  %localOffset;
        .reg .u64  %offsetX;
        .reg .u64  %offsetY;
        .reg .u64  %stride;
        .reg .u32  %i;
        .reg .u32  %numBlocks;
        .reg .u64  %ptrA;
        .reg .u64  %ptrB;
        .reg .u64  %ptrOut;
        .reg .f32  %val;
        .shared .align 4 .f32 loadedA[1024]; // should be ntid.x*ntid.y
        .shared .align 4 .f32 loadedB[1024]; // should be ntid.x*ntid.y
        .shared .align 4 .f32 output[1024]; // should be ntid.x*ntid.y

        ld.param.u32 %numBlocks, [numBlocks];
        ld.param.u64 %ptrA, [ptrA];
        ld.param.u64 %ptrB, [ptrB];
        ld.param.u64 %ptrOut, [ptrOut];

        // Local offset is tid.y*ntid.x + tid.x
        cvt.u64.u32 %localOffset, %tid.y;
        cvt.u64.u32 %tmp0, %ntid.x;
        mul.lo.u64 %localOffset, %localOffset, %tmp0;
        cvt.u64.u32 %tmp0, %tid.x;
        add.u64 %localOffset, %localOffset, %tmp0;

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
        cvta.shared.u64 %tmp0, output;
        add.u64 %tmp0, %tmp0, %localOffset;
        mov.f32 %val, 0.0;
        st.shared.f32 [%tmp0], %val;

        mov.u32 %i, 0;
    loop_start:
        // Our block offset in A is (offsetX + i*ntid.x + tid.x, offsetY + tid.y)
        cvt.u64.u32 %tmp0, %i;
        cvt.u64.u32 %tmp1, %ntid.x;
        mul.lo.u64 %tmp0, %tmp0, %tmp1;
        cvt.u64.u32 %tmp1, %tid.x;
        add.u64 %tmp0, %tmp0, %tmp1;
        add.u64 %tmp0, %tmp0, %offsetX;
        cvt.u64.u32 %tmp1, %tid.y;
        add.u64 %tmp1, %tmp1, %offsetY;
        // Compute global offset as y*stride+x
        mul.lo.u64 %tmp1, %tmp1, %stride;
        add.u64 %tmp0, %tmp0, %tmp1;
        cvta.shared.u64 %tmp1, loadedA;
        add.u64 %tmp1, %tmp1, %localOffset;
        // Copy to local memory
        ld.global.f32 %val, [%tmp0];
        st.shared.f32 [%tmp1], %val;

        // Our block offset in B is (offsetX + tid.x, offsetY + i*ntid.y + tid.y)
        cvt.u64.u32 %tmp0, %i;
        cvt.u64.u32 %tmp1, %ntid.y;
        mul.lo.u64 %tmp0, %tmp0, %tmp1;
        cvt.u64.u32 %tmp1, %tid.y;
        add.u64 %tmp0, %tmp0, %tmp1;
        add.u64 %tmp0, %tmp0, %offsetY;
        cvt.u64.u32 %tmp1, %tid.x;
        add.u64 %tmp1, %tmp1, %offsetX;
        // Compute global offset as y*stride+x
        mul.lo.u64 %tmp0, %tmp0, %stride;
        add.u64 %tmp1, %tmp1, %tmp0;
        cvta.shared.u64 %tmp1, loadedB;
        add.u64 %tmp1, %tmp1, %localOffset;
        // Copy to local memory
        ld.global.f32 %val, [%tmp0];
        st.shared.f32 [%tmp1], %val;

        bar.sync 0;

        // TODO: core matrix multiplication inner-loop here.

        add.u32 %i, %i, 1;
        setp.lt.u32 %p0, %i, %numBlocks;
        @%p0 bra loop_start;
    
    loop_end:
        // Write back to output memory.

        // Output address is stride*(offsetX+tid.x) + offsetY+tid.y
        cvt.u64.u32 %tmp0, %tid.x;
        add.u64 %tmp0, %tmp0, %offsetX;
        mul.lo.u64 %tmp0, %tmp0, %stride;
        cvt.u64.u32 %tmp1, %tid.y;
        add.u64 %tmp1, %tmp1, %offsetY;
        add.u64 %tmp0, %tmp0, %tmp1;
        add.u64 %tmp0, %tmp0, %ptrOut;

        // Input address is given by %localOffset
        cvta.shared.u64 %tmp1, output;
        add.u64 %tmp1, %tmp1, %localOffset;

        ld.global.f32 %val, [%tmp1];
        st.global.f32 [%tmp0], %val;
    }
    """
    fn = compile_function(code, "blockedMatmul")
    A = np.random.normal(size=[768, 768]).astype(np.float32)
    B = np.random.normal(size=[768, 768]).astype(np.float32)
    A_buf = numpy_to_gpu(A)
    B_buf = numpy_to_gpu(B)
    out_buf = numpy_to_gpu(A * 0)
    block_size = 32
    fn(
        A_buf,
        B_buf,
        out_buf,
        np.int32(A.shape[0] // block_size),
        grid=(block_size, block_size, 1),
        block=(A.shape[0] // block_size, A.shape[1] // block_size, 1),
    )
    sync()
    results = gpu_to_numpy(out_buf, A.shape, A.dtype)
    expected = A @ B
    print(f"maximum absolute error of matmul is {np.abs(results - expected).max()}")


if __name__ == "__main__":
    square_matrix_simple_block()
