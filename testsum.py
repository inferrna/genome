import pyopencl as cl
import numpy as np
ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)
n_threads = ctx.get_info(cl.context_info.DEVICES)[0].max_work_group_size
N = n_threads*n_threads
a = np.arange(N).astype(np.float32)
r = np.empty(n_threads).astype(np.float32)
o = np.empty(1).astype(np.float32)
print("Reducing {0:d} numbers...".format(N))
mf = cl.mem_flags
a_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)
r_buf = cl.Buffer(ctx, mf.READ_WRITE, size=r.nbytes)
o_buf = cl.Buffer(ctx, mf.WRITE_ONLY, size=o.nbytes)
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
int res = 0;
uint tst[128] = {0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 7, 8, 8, 8, 9, 9, 9, 10, 10, 10, 11, 11, 11, 12, 12, 12, 13, 13, 13, 13, 14, 14, 14, 14, 15, 15, 15, 15, 16, 16, 16, 16, 17, 17, 17, 17, 18, 18, 18, 18, 19, 19, 19, 19, 19, 20, 20, 20, 20, 20, 21, 21, 21, 21, 21, 22, 22, 22, 22, 22, 23, 23, 23, 23, 23, 24, 24, 24, 24, 24, 25, 25, 25, 25, 25, 25, 26, 26, 26, 26, 26, 26, 27, 27, 27, 27, 27, 27, 28, 28, 28, 28, 28, 28, 29, 29, 29, 29, 29, 29, 30, 30, 30, 30, 30, 30, 31, 31, 31, 31, 31, 31};
uint tse[128] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127};
for(uint i=0; i<128; i++){
    res += (gid - tst[i]) + lid*tse[i];
}
*r = 1.0*res;
/*b[lid] = a[gid];
barrier(CLK_LOCAL_MEM_FENCE);
for(uint s = gs/2; s > 0; s >>= 1) {
if(lid < s) {
b[lid] += b[lid+s];
}
barrier(CLK_LOCAL_MEM_FENCE);
}
if(lid == 0) r[wid] = b[lid];*/
}
""").build()
print("N==", N, "n_threads==", n_threads)
evt = prg.reduce(queue, (N,), (n_threads,), a_buf, r_buf, loc_buf)
evt.wait()
print(evt.profile.end - evt.profile.start)
evt = prg.reduce(queue, (n_threads,), (n_threads,), r_buf, o_buf, loc_buf)
evt.wait()
print(evt.profile.end - evt.profile.start)
cl.enqueue_copy(queue, o, o_buf)
print(o, np.sum(a))
