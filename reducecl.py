import pyopencl as cl
import numpy as np
mf = cl.mem_flags

def reduce_sum(ctx, queue, a_buf, N, o_buf):
    n_threads = ctx.get_info(cl.context_info.DEVICES)[0].max_work_group_size
    r = np.empty(n_threads).astype(np.float32)
    r_buf = cl.Buffer(ctx, mf.READ_WRITE, size=r.nbytes)
    loc_buf = cl.LocalMemory(4*n_threads)
    prg = cl.Program(ctx, """
    __kernel void reduce(__global float *a,
    __global float *r,
    __local float *b)
    {
        uint gid = get_global_id(0);
        uint wid = get_group_id(0);
        uint lid = get_local_id(0);
        uint gs = get_local_size(0);
        b[lid] = a[gid];
        barrier(CLK_LOCAL_MEM_FENCE);
        for(uint s = gs/2; s > 0; s >>= 1) {
            if(lid < s) {
                b[lid] += b[lid+s];
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        if(lid == 0) r[wid] = b[lid];
    }
    """).build()
    print("N==", N, "n_threads==", n_threads)
    evt = prg.reduce(queue, (N,), (n_threads,), a_buf, r_buf, loc_buf)
    evt.wait()
    #print(evt.profile.end - evt.profile.start)
    evt = prg.reduce(queue, (n_threads,), (n_threads,), r_buf, o_buf, loc_buf)
    evt.wait()
    #print(evt.profile.end - evt.profile.start)
