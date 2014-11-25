import pyopencl as cl
import numpy as np
from pyopencl.algorithm import RadixSort

mf = cl.mem_flags
class cl_reduce():
    def __init__(self, ctx, numitems):
        self.ctx = ctx
        self.n_threads = ctx.get_info(cl.context_info.DEVICES)[0].max_work_group_size
        #Init parents with indexes
        parentsh   = np.arange(start=0, stop=numitems, dtype=np.ushort)
        self.parentsg  = cl.Buffer(ctx, mf.WRITE_ONLY | mf.COPY_HOST_PTR, hostbuf=parentsh)
        self.gparentsg = cl.Buffer(ctx, mf.WRITE_ONLY | mf.COPY_HOST_PTR, hostbuf=parentsh)
        self.prgsm_fst = cl.Program(ctx, """
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
        self.prgsm_med = cl.Program(ctx, """
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
            if(lid == 0) r[wid] = b[lid]/"""+str(numitems)+""";
        }
        """).build()
#Minimal
        self.prgmna = cl.Program(ctx, """
        __kernel void reduce(__global float *a,
        __global float *r,
        __global uint *q,
        __global uint *o_lid,
        __local float *b,
        __local uint *c)
        {
            uint gid = get_global_id(0);
            uint wid = get_group_id(0);
            uint lid = get_local_id(0);
            uint gs = get_local_size(0);
            b[lid] = a[gid];
            c[lid] = gid;
            barrier(CLK_LOCAL_MEM_FENCE);
            for(uint s = gs/2; s > 0; s >>= 1) {
                if(lid < s && b[lid] > b[lid+s]) {
                    b[lid] = b[lid+s];
                    c[lid] = c[lid+s];
                }
                barrier(CLK_LOCAL_MEM_FENCE);
            }
            if(lid == 0) {
                r[wid] = b[0];
                q[wid] = c[0];    //1st iteration, save min's gid for each group
            }
        }
        """).build()
        self.prgmnb = cl.Program(ctx, """
        __kernel void reduce(__global float *a,
        __global float *r,
        __global uint *q,
        __global uint *o_lid,
        __local float *b,
        __local uint *c)
        {
            uint gid = get_global_id(0);
            uint wid = get_group_id(0);
            uint lid = get_local_id(0);
            uint gs = get_local_size(0);
            b[lid] = a[gid];
            c[lid] = gid;
            barrier(CLK_LOCAL_MEM_FENCE);
            for(uint s = gs/2; s > 0; s >>= 1) {
                if(lid < s && b[lid] > b[lid+s]) {
                    b[lid] = b[lid+s];
                    c[lid] = c[lid+s];
                }
                barrier(CLK_LOCAL_MEM_FENCE);
            }
            if(lid == 0) {
                r[wid] = b[0];
                o_lid[wid] = q[c[0]]; //2nd iteration, return value of gid
            }
        }
        """).build()
#Sort
        self.prgsrt = cl.Program(ctx, """
            #define data_t float
            __kernel void ParallelBitonic_Local(__global const data_t * in, __global uint * out, __local data_t * aux, __local uint * idxs)
            {
              int i = get_local_id(0); // index in workgroup
              int wg = get_local_size(0); // workgroup size = block size, power of 2
              bool smaller, swap;
              data_t iData, jData;
              uint jidx, iidx;
              // Move IN, OUT to block start
              int offset = get_group_id(0) * wg;
              in += offset; out += offset;
              // Load block in AUX[WG]
              aux[i] = in[i];
              idxs[i] = i;
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
                  iidx = idxs[i];
                  jidx = idxs[j];
                  smaller = (jData < iData) || ( jData == iData && j < i );
                  swap = smaller ^ (j < i) ^ direction;
                  barrier(CLK_LOCAL_MEM_FENCE);
                  aux[i] = (swap)?jData:iData;
                  idxs[i] = (swap)?jidx:iidx;
                  barrier(CLK_LOCAL_MEM_FENCE);
                }
              }
            
              // Write output
              out[i] = idxs[i];
            }
        """).build()
    def reduce_sum(self, queue, a_buf, N, o_buf):
        r = np.empty(self.n_threads).astype(np.float32)
        r_buf = cl.Buffer(self.ctx, mf.READ_WRITE, size=r.nbytes)
        loc_buf = cl.LocalMemory(4*self.n_threads)
        #print("N==", N, "n_threads==", self.n_threads)
        evt = self.prgsm_fst.reduce(queue, (N,), (self.n_threads,), a_buf, r_buf, loc_buf)
        evt.wait()
        #print(evt.profile.end - evt.profile.start)
        n_threads = N//self.n_threads
        evt = self.prgsm_med.reduce(queue, (n_threads,), (n_threads,), r_buf, o_buf, loc_buf)
        evt.wait()
        #print(evt.profile.end - evt.profile.start)
    def reduce_min(self, queue, a_buf, N, o_buf, o_lid):
        r = np.empty(self.n_threads).astype(np.float32)
        r_buf = cl.Buffer(self.ctx, mf.READ_WRITE, size=r.nbytes)
        q_buf = cl.Buffer(self.ctx, mf.READ_WRITE, size=r.nbytes)
        loc_buf = cl.LocalMemory(4*self.n_threads)
        loc_lid = cl.LocalMemory(4*self.n_threads)
        #print("N==", N, "n_threads==", self.n_threads)
        minnt = min(N, self.n_threads)
        evt = self.prgmna.reduce(queue, (N,), (minnt,), a_buf, r_buf, q_buf, o_lid, loc_buf, loc_lid)
        evt.wait()
        #print(evt.profile.end - evt.profile.start)
        n_threads = N//minnt
        evt = self.prgmnb.reduce(queue, (n_threads,), (n_threads,), r_buf, o_buf, q_buf, o_lid, loc_buf, loc_lid)
        evt.wait()
        #print(evt.profile.end - evt.profile.start)

    def sort(self, queue, N, a_buf, o_buf):
        loc_aux = cl.LocalMemory(16*self.n_threads)
        loc_idx = cl.LocalMemory(16*self.n_threads)
        #print("N==", N, "n_threads==", self.n_threads)
        minnt = min(N, self.n_threads)
        evt = self.prgsrt.ParallelBitonic_Local(queue, (minnt,), (minnt,), a_buf, o_buf, loc_aux, loc_idx)
        evt.wait()

