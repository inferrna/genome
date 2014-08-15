#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pyopencl as cl

print( cl.get_cl_header_version() )

result = 3.0
equation = "a_g[gid]+2*b_g[gid]+3*c_g[gid]+4*d_g[gid] - "+str(result)+""
nvars = 4
nsamp = 500

arr_np = np.random.rand(nvars*nsamp).astype(np.float32) - np.random.rand(nvars*nsamp).astype(np.float32)

ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)

arr4np = arr_np.view().reshape(nvars, nsamp)

mf = cl.mem_flags
run = cl.Program(ctx, """
__kernel void sum(__global const float *a_g, __global const float *b_g, __global const float *c_g, __global const float *d_g, __global float *res_g) {
  int gid = get_global_id(0);
  res_g[gid] = """+equation+""";
}
""").build()
a_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=arr4np[0])
b_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=arr4np[1])
c_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=arr4np[2])
d_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=arr4np[3])
res_g = cl.Buffer(ctx, mf.WRITE_ONLY, arr4np[0].nbytes)
res_np = np.empty_like(arr4np[0])

for cy in range(0, 3):
   # if cy>0 and cl.enqueue_fill_buffer:
   #     cl.enqueue_fill_buffer(queue, a_g, arr4np[0], 0, nsamp)
   #     cl.enqueue_fill_buffer(queue, b_g, arr4np[1], 0, nsamp)
   #     cl.enqueue_fill_buffer(queue, c_g, arr4np[2], 0, nsamp)
   #     cl.enqueue_fill_buffer(queue, d_g, arr4np[3], 0, nsamp)
    if cy>0:
        a_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=arr4np[0])
        b_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=arr4np[1])
        c_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=arr4np[2])
        d_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=arr4np[3])
    run.sum(queue, arr_np.shape, None, a_g, b_g, c_g, d_g, res_g)
    cl.enqueue_copy(queue, res_np, res_g)

#Sort by given result
    ordr = np.argsort(res_np)
    print(res_np[ordr])

    arr4np[0] = arr4np[0][ordr]
    arr4np[1] = arr4np[1][ordr]
    arr4np[2] = arr4np[2][ordr]
    arr4np[3] = arr4np[3][ordr]
    for i in range(0, nsamp//2):
        a = np.arange(nvars)
        np.random.shuffle(a)
        idxs = a[:2]
#Swap as ..
        for idx in idxs:
            frm = arr4np[0][i]
            too = arr4np[0][-(i+1)]
            arr4np[idx][i] = too
            arr4np[idx][-(i+1)] = frm
# Check on CPU with Numpy:
