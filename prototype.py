#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pyopencl as cl
import random

print( cl.get_cl_header_version() )

result = 3.0
equation = "(a_g[gid]+2*b_g[gid])*(3*c_g[gid]+4*d_g[gid]) - "+str(result)+""
nvars = 4
nsamp = 500

arr_np = np.random.rand(nvars*nsamp).astype(np.float32) - np.random.rand(nvars*nsamp).astype(np.float32)

ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)

arr4np = arr_np.reshape(nvars, nsamp)

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
_arr4np = [[0]]*nvars

for cy in range(0, 130):
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
    res_ordrd = res_np[ordr]
    abs_res = abs(res_ordrd)
    fltr = abs_res<100
    abs_resf = abs_res[fltr]
    mind = abs_resf.argmin()
    nwlen = len(abs_resf)
    rll = nwlen//2 - mind
    print("Median index is", mind)
    print("Best result  is", res_ordrd[fltr][mind])
    if abs(res_ordrd[mind]) < 0.001:
        print("Solution for", equation,"is\n", [x[ordr][fltr][mind] for x in arr4np])
        break
    #print(_arr4np)
    for j in range(0, len(arr4np)):
        _arr4np[j] = np.roll(arr4np[j][ordr][fltr], rll)
    diff = nsamp - nwlen
    print("diff==", diff)
    for i in range(0, nwlen//2):
        a = np.arange(nvars)
        np.random.shuffle(a)
        idxs = a[:2]
#Swap as ..
        for idx in idxs:
            frm = _arr4np[idx][i]
            too = _arr4np[idx][-(i+1)]
            _arr4np[idx][i] = too
            _arr4np[idx][-(i+1)] = frm
    print("Newlen == ", len(_arr4np[0]))
    d1 = np.array([[0]*diff]*4)
    for i in range(0, diff):
        r = random.randint(0, nwlen-1)
        for j in range(0, len(d1)):
            d1[j][i] = _arr4np[j][r]
        ri = random.randint(0, len(d1)-1)
        d1[ri][i] = d1[ri][i] + random.choice([-2,2])*_arr4np[ri].max()*random.random()
    for j in range(0, len(d1)):
        arr4np[j] = np.concatenate([_arr4np[j], d1[j]])
# Check on CPU with Numpy:
