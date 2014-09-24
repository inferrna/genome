import pyopencl as cl
import numpy as np
ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)
n_threads = ctx.get_info(cl.context_info.DEVICES)[0].max_work_group_size
N = n_threads*8
a = np.random.rand(N).astype(np.float32)
o = np.empty(N).astype(np.float32)
print("Reducing {0:d} numbers...".format(N))
mf = cl.mem_flags
a_buf = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=a)
o_buf = cl.Buffer(ctx, mf.WRITE_ONLY, size=o.nbytes)
loc_buf = cl.LocalMemory(8*4*n_threads)
defines = """
#define data_t float
#define data_t4 float4
#define getKey(a) (a)
#define getValue(a) (0)
#define makeData(k,v) (k)
#ifndef BLOCK_FACTOR
#define BLOCK_FACTOR 1
#endif
#define ORDER(a,b) { bool swap = reverse ^ (getKey(a)<getKey(b)); data_t auxa = a; data_t auxb = b; a = (swap)?auxb:auxa; b = (swap)?auxa:auxb; }
#define ORDERV(x,a,b) { bool swap = reverse ^ (getKey(x[a])<getKey(x[b])); data_t auxa = x[a]; data_t auxb = x[b]; x[a] = (swap)?auxb:auxa; x[b] = (swap)?auxa:auxb; }
#define B2V(x,a) { ORDERV(x,a,a+1) }
#define B4V(x,a) { for (int i4=0;i4<2;i4++) { ORDERV(x,a+i4,a+i4+2) } B2V(x,a) B2V(x,a+2) }
#define B8V(x,a) { for (int i8=0;i8<4;i8++) { ORDERV(x,a+i8,a+i8+4) } B4V(x,a) B4V(x,a+4) }
#define B16V(x,a) { for (int i16=0;i16<8;i16++) { ORDERV(x,a+i16,a+i16+8) } B8V(x,a) B8V(x,a+8) }
"""
prgg = cl.Program(ctx, defines+"""
__kernel void ParallelBitonic_C4(__global data_t * data, __global data_t * out, __global int * inc0, __local data_t * aux)
{
  int t = get_global_id(0); // thread index
  int wgBits = 4*get_local_size(0) - 1; // bit mask to get index in local memory AUX (size is 4*WG)
  int inc,low,i, dir = 1;
  bool reverse;
  data_t x[4];

  // First iteration, global input, local output
  inc0 = *inc0>>1;
  inc = inc0;
  low = t & (inc - 1); // low order bits (below INC)
  i = ((t - low) << 2) + low; // insert 00 at position INC
  reverse = ((dir & i) == 0); // asc/desc order
  for (int k=0;k<4;k++) x[k] = data[i+k*inc];
  B4V(x,0);
  for (int k=0;k<4;k++) aux[(i+k*inc) & wgBits] = x[k];
  barrier(CLK_LOCAL_MEM_FENCE);

  // Internal iterations, local input and output
  for ( ;inc>1;inc>>=2)
  {
    low = t & (inc - 1); // low order bits (below INC)
    i = ((t - low) << 2) + low; // insert 00 at position INC
    reverse = ((dir & i) == 0); // asc/desc order
    for (int k=0;k<4;k++) x[k] = aux[(i+k*inc) & wgBits];
    B4V(x,0);
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int k=0;k<4;k++) aux[(i+k*inc) & wgBits] = x[k];
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  // Final iteration, local input, global output, INC=1
  i = t << 2;
  reverse = ((dir & i) == 0); // asc/desc order
  for (int k=0;k<4;k++) x[k] = aux[(i+k) & wgBits];
  B4V(x,0);
  for (int k=0;k<4;k++) data[i+k] = x[k];
  if(get_global_id(0)==0) *inc0<<=1;
}
""").build()
inch = np.empty(1).astype(np.int)
inch[0] = n_threads>>2
n = n_threads
incg = cl.Buffer(ctx, mf.WRITE_ONLY| mf.COPY_HOST_PTR, hostbuf=inch)

print("N==", N, "n_threads==", n_threads)
while n < N:
    print("n==", n, " of ", N)
    evt = prgg.ParallelBitonic_C4(queue, (n,), (n_threads,), a_buf, o_buf, incg, loc_buf)
    evt.wait()
    n = n*2;
cl.enqueue_copy(queue, o, a_buf)
print(o[-16:], np.sort(a)[-16:])
