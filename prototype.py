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

result = 1.0
ninpt = 50   #Samples count
nvars = 40   #Count of equations members
nsamp = 50   #Genome samples count
varnames = []
valnames = []
#Names for kernel parameters
for i in range(0, nvars):
    varnames.append(dec2str(i)+"_gnm")
    valnames.append(dec2str(i)+"_val")
gstruct = """
struct genomes {
    """+'\n    '.join(['float '+dec2str(i)+';' for i in range(0, nvars)])+"""
};
"""
vstruct = """
struct vars {
    """+'\n    '.join(['float '+dec2str(i)+';' for i in range(0, nvars)])+"""
};
"""
print(struct)
varsofid = [var+'[gid]' for var in varnames]
valsof_i = [val+'[i]' for val in valnames]
varsprms = ['__global const float *'+var for var in varnames]
valsprms = ['__global const float *'+val for val in valnames]
eq = ['*'.join(c) for c in zip(varsofid, valsof_i)]
equation = '+'.join(eq)+" - "+str(result)+""
varspstr = ', '.join(varsprms+valsprms)
print(varsofid)
print(varsprms)
print(varnames)
print(valnames)
print(eq)
print(equation)
print(varspstr)
#exit()

#Random init genome
arr_np = np.random.rand(nvars*nsamp).astype(np.float32) - np.random.rand(nvars*nsamp).astype(np.float32)
arr4np = arr_np.reshape(nvars, nsamp)
#Random init equation members
inp_np = np.random.rand(nvars*ninpt).astype(np.float32) - np.random.rand(nvars*ninpt).astype(np.float32)
inp4np = inp_np.reshape(nvars, ninpt)

ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)


mf = cl.mem_flags
run = cl.Program(ctx, """
__kernel void sum("""+varspstr+""", __global float *res_g) {
  int gid = get_global_id(0);
  float _res = 0.0;
  for(int i=0; i<"""+str(ninpt)+"""; i++){
    _res += fabs("""+equation+""");
    //_res += """+equation+""";
  }
  res_g[gid] = _res;
}
""").build()
#Metabuffer for opencl datas
arrs_g = [[]]*(2*nvars+1)
#Results buffers (as genome counts)
res_g = cl.Buffer(ctx, mf.WRITE_ONLY, arr4np[0].nbytes)
res_np = np.empty_like(arr4np[0])
_arr4np = [[0]]*nvars
global_offset = None
g_times_l = False
wait_for = None

for cy in range(0, 13000):
   # if cy>0 and cl.enqueue_fill_buffer:
   #     cl.enqueue_fill_buffer(queue, a_g, arr4np[0], 0, nsamp)
   #     cl.enqueue_fill_buffer(queue, b_g, arr4np[1], 0, nsamp)
   #     cl.enqueue_fill_buffer(queue, c_g, arr4np[2], 0, nsamp)
   #     cl.enqueue_fill_buffer(queue, d_g, arr4np[3], 0, nsamp)
    for i in range(0, nvars):
        arrs_g[i] = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=arr4np[i])
        arrs_g[i+nvars] = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=inp4np[i])
    arrs_g[-1] = res_g
    #run.sum.set_args(*arrs_g)
    #cl.enqueue_nd_range_kernel(queue, run.sum, arr4np[i].shape, None, global_offset, wait_for, g_times_l=g_times_l)
    run.sum(queue, arr4np[i].shape, None, *arrs_g)#arrs_g[0], arrs_g[1], arrs_g[2], arrs_g[3], res_g)
    print("enqueue ok")
    cl.enqueue_copy(queue, res_np, res_g)

#Sort by given result
    res_np, unfltr = np.unique(res_np, return_index=True)
    ordr = np.argsort(res_np)           #Get order
    res_ordrd = res_np[ordr]            #Sort by order
    abs_res = abs(res_ordrd)            #Set sorted values to absolute
    fltr = abs_res<=abs_res.mean()      #Get filter
    abs_resf = abs_res[fltr]  #
    mind = abs_resf.argmin()
    nwlen = len(abs_resf)
    rll = nwlen//2 - mind
    print("Median index is", mind)
    print("Filtered length is", nwlen)
    print("Best result  is", res_ordrd[fltr][mind])
    if abs(res_ordrd[mind]) < 0.001:
        print("Solution for", equation,"is\n", [x[unfltr][ordr][fltr][mind] for x in arr4np])
        break
    for j in range(0, len(arr4np)):
        _arr4np[j] = np.roll(arr4np[j][unfltr][ordr][fltr], rll)
    #print(_arr4np)
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
    d1 = np.array([[0.0]*diff]*nvars)
    for i in range(0, diff):
        r = random.randint(0, nwlen-1)
        for j in range(0, len(d1)):
            d1[j][i] = _arr4np[j][r]
            #print("_arr4np[",j,"][",r,"]==", _arr4np[j][r])
        ri = random.randint(0, len(d1)-1)
        d1[ri][i] = d1[ri][i] + (2*random.random()-1)
    #print(d1)
    for j in range(0, len(d1)):
        arr4np[j] = np.concatenate([_arr4np[j], d1[j]])
# Check on CPU with Numpy:
