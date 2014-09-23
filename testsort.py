import pyopencl as cl
import numpy as np
ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)
n_threads = ctx.get_info(cl.context_info.DEVICES)[0].max_work_group_size
N = n_threads
a = np.random.rand(N).astype(np.float32)
o = np.empty(N).astype(np.float32)
print("Reducing {0:d} numbers...".format(N))
mf = cl.mem_flags
a_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)
o_buf = cl.Buffer(ctx, mf.WRITE_ONLY, size=o.nbytes)
loc_buf = cl.LocalMemory(4*n_threads)
prg = cl.Program(ctx, """
#define data_t float
__kernel void ParallelBitonic_Local(__global const data_t * in,__global data_t * out,__local data_t * aux)
{
  int i = get_local_id(0); // index in workgroup
  int wg = get_local_size(0); // workgroup size = block size, power of 2
  bool smaller, swap;
  data_t iData, jData;
  // Move IN, OUT to block start
  int offset = get_group_id(0) * wg;
  in += offset; out += offset;
  // Load block in AUX[WG]
  aux[i] = in[i];
  barrier(CLK_LOCAL_MEM_FENCE); // make sure AUX is entirely up to date

  // Loop on sorted sequence length
  for (int length=1;length<wg;length<<=1)
  {
    bool direction = ((i & (length<<1)) != 0); // direction of sort: 0=asc, 1=desc
    // Loop on comparison distance (between keys)
    for (int inc=length;inc>0;inc>>=1)
    {
      int j = i ^ inc; // sibling to compare
      iData = aux[i];
      jData = aux[j];
      smaller = (jData < iData) || ( jData == iData && j < i );
      swap = smaller ^ (j < i) ^ direction;
      barrier(CLK_LOCAL_MEM_FENCE);
      aux[i] = (swap)?jData:iData;
      barrier(CLK_LOCAL_MEM_FENCE);
    }
  }

  // Write output
  out[i] = aux[i];
}
""").build()
print("N==", N, "n_threads==", n_threads)
evt = prg.ParallelBitonic_Local(queue, (n_threads,), (n_threads,), a_buf, o_buf, loc_buf)
evt.wait()
cl.enqueue_copy(queue, o, o_buf)
print(o, np.sort(a))
