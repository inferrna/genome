#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pyopencl as cl
from pyopencl.reduction import ReductionKernel
import random
from string import ascii_lowercase
from reducecl import cl_reduce

def dec2str(num):
    k = []
    s = str(num)
    for a in s:
        k.append(ascii_lowercase[int(a)])
    return 'qq'+''.join(k)

print( cl.get_cl_header_version() )

result = 1.0
ninpt = 3   #Samples count
nvars = 9   #Count of equations members
nsamp = 4096   #Genome samples count
varnames = [dec2str(i) for i in range(0, nvars)]
gstruct = """
struct genomes {
    """+'\n    '.join(['float '+v+';' for v in varnames])+"""
};
"""
vstruct = """
struct vars {
    """+'\n    '.join(['float '+v+';' for v in varnames])+"""
};
"""
print(gstruct)
varsofid = ['gms[gid].'+var for var in varnames]
valsof_i = ['vs[i].'+val for val in varnames]
eq = ['*'.join(c) for c in zip(varsofid, valsof_i)]
equation = '+'.join(eq)+" - "+str(result)+""
print(varsofid)
print(varnames)
print(eq)
print(equation)
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
run = cl.Program(ctx, '\n'.join([gstruct, vstruct])+"""
__kernel void sum(__global struct vars *vs, __global struct genomes *gms, __global float *res_g) {
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
arrs_g = [[]]*3
#Results buffers (as genome counts)
res_g = cl.Buffer(ctx, mf.WRITE_ONLY, arr4np[0].nbytes)
res_np = np.empty_like(arr4np[0])
_arr4np = [[0]]*nvars
global_offset = None
g_times_l = False
wait_for = None
currmin = (9999999999, 0,)
obuf = np.empty(1).astype(np.float32)
olid = np.empty(1).astype(np.uint32)
o_med = cl.Buffer(ctx, mf.WRITE_ONLY, size=obuf.nbytes)
o_min = cl.Buffer(ctx, mf.WRITE_ONLY, size=obuf.nbytes)
o_lid = cl.Buffer(ctx, mf.WRITE_ONLY, size=olid.nbytes)
clreducer = cl_reduce(ctx, nsamp)
parentsh   = np.arange(start=0, stop=numitems, dtype=np.ushort)
parentsg  = cl.Buffer(ctx, mf.WRITE_ONLY | mf.COPY_HOST_PTR, hostbuf=parentsh)
gparentsg = cl.Buffer(ctx, mf.WRITE_ONLY | mf.COPY_HOST_PTR, hostbuf=parentsh)
sort = RadixSort(ctx, "float *res", key_expr="res[i]", sort_arg_names=["res"])



for cy in range(1, 1300):
    arrs_g[0] = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=inp_np)
    arrs_g[1] = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=arr_np)
    arrs_g[2] = res_g
    #run.sum.set_args(*arrs_g)
    #cl.enqueue_nd_range_kernel(queue, run.sum, arr4np[i].shape, None, global_offset, wait_for, g_times_l=g_times_l)
    run.sum(queue, arr4np[0].shape, None, *arrs_g)
    print("enqueue ok")
    cl.enqueue_copy(queue, res_np, res_g)
    #Reduce sum
    clreducer.reduce_sum(queue, res_g, nsamp, o_med)
    cl.enqueue_copy(queue, obuf, o_med)
    print("total sum is", obuf, np.sum(res_np))
    #Reduce minimal
    clreducer.reduce_min(queue, res_g, nsamp, o_min, o_lid)
    cl.enqueue_copy(queue, obuf, o_min)
    cl.enqueue_copy(queue, olid, o_lid)
    print("min value is", obuf, np.min(res_np))
    print("min index is", olid, res_np.argmin(axis=0))
#Sort by given result
    res_np, unfltr = np.unique(res_np, return_index=True)
    ordr = np.argsort(res_np)                                   #Get order
    res_ordrd = res_np[ordr]                                    #Sort by order
    abs_res = abs(res_ordrd)                                    #Set sorted values to absolute
    fltrm = abs_res<=np.median(abs_res)                         #Get median filter
    fltrr = (abs_res<=2*np.median(abs_res))*(np.random.randint(0, 2, len(abs_res)) > 0)           #Get random filter
    fltr = fltrm #+ fltrr
    abs_resf = abs_res[fltr]  #
    mind = abs_resf.argmin()
    nwlen = len(abs_resf)
    rll = nwlen//2 - mind
    best_res = res_ordrd[fltr][mind]
    print("Median index is", mind)
    print("Filtered length is", nwlen)
    print("Best result  is", best_res)
    if best_res<currmin[0]:
        currmin = (best_res, cy,)
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
        d1[ri][i] = d1[ri][i] + (2*random.random()-1)*(best_res/nvars)
    #print(d1)
    for j in range(0, len(d1)):
        arr4np[j] = np.concatenate([_arr4np[j], d1[j]])
# Check on CPU with Numpy:
print(currmin)
