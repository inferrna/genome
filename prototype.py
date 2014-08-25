#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pyopencl as cl
import random
from string import ascii_lowercase

def dec2str(num):
    k = []
    s = str(num)
    for a in s:
        k.append(ascii_lowercase[int(a)])
    return ''.join(k)

print( cl.get_cl_header_version() )

result = 3.0
#results = [3.0, 5.0, 2.0]
#ivalues = [[1.0, 2.0, 3.0, 4.0], [1.0, 3.0, 1.0, 2.0], [4.0, 1.0, 1.0, 2.0]]
coeffs = np.random.random(9)
equation = "(a_g[gid]+2*b_g[gid])*(3*c_g[gid]+4*d_g[gid]) - "+str(result)+""
#eqlst = ["a_g[gid]"*str(iv[0])
#exit()
nvars = 4
nsamp = 500
varnames = []
valnames = []
for i in range(0, nvars):
    varnames.append(dec2str(i)+"_gnm")
    valnames.append(dec2str(i)+"_val")

print(varnames)
print(valnames)


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
arrs_g = [[]]*(nvars+1)
res_g = cl.Buffer(ctx, mf.WRITE_ONLY, arr4np[0].nbytes)
res_np = np.empty_like(arr4np[0])
_arr4np = [[0]]*nvars
global_offset = None
g_times_l = False
wait_for = None

for cy in range(0, 130):
   # if cy>0 and cl.enqueue_fill_buffer:
   #     cl.enqueue_fill_buffer(queue, a_g, arr4np[0], 0, nsamp)
   #     cl.enqueue_fill_buffer(queue, b_g, arr4np[1], 0, nsamp)
   #     cl.enqueue_fill_buffer(queue, c_g, arr4np[2], 0, nsamp)
   #     cl.enqueue_fill_buffer(queue, d_g, arr4np[3], 0, nsamp)
    for i in range(0, nvars):
        arrs_g[i] = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=arr4np[i])
    arrs_g[-1] = res_g
    #run.sum.set_args(*arrs_g)
    #cl.enqueue_nd_range_kernel(queue, run.sum, arr4np[i].shape, None, global_offset, wait_for, g_times_l=g_times_l)
    run.sum(queue, arr4np[i].shape, None, *arrs_g)#arrs_g[0], arrs_g[1], arrs_g[2], arrs_g[3], res_g)
    print("enqueue ok")
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
