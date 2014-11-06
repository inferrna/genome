#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pyopencl as cl
from pyopencl.reduction import ReductionKernel
from pyopencl.algorithm import RadixSort
import pyopencl.array as cl_array
import random
from string import ascii_lowercase
from reducecl import cl_reduce
from randfloat import randfloat
import genn
import pprint
pp = pprint.PrettyPrinter(depth=5)

def dec2str(num):
    k = []
    s = str(num)
    for a in s:
        k.append(ascii_lowercase[int(a)])
    return 'qq'+''.join(k)

print( cl.get_cl_header_version() )

ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)

result = 1.0
ninpt = 2   #Samples count
nvarsd = 5   #Count of equations members
topology = [nvarsd, 4, 3, 2, 1]
nvarsg = genn.countcns(topology)     #Count of equations members
print("Total connections is", nvarsg)
nsamp = ctx.get_info(cl.context_info.DEVICES)[0].max_work_group_size #Genome samples count (current sort limitation to local_size)
print("Population count is", nsamp)
clreducer = cl_reduce(ctx, nsamp)
#exit()

#Random init genome
arr_np = np.random.rand(nvarsg*nsamp).astype(np.float32) - np.random.rand(nvarsg*nsamp).astype(np.float32)
arr4np = arr_np.reshape(nsamp, nvarsg)
#Random init equation members
inp_np = np.random.rand(nvarsd*ninpt).astype(np.float32) - np.random.rand(nvarsd*ninpt).astype(np.float32)
inp4np = inp_np.reshape(ninpt, nvarsd)
#Results
vsr = np.array([1.0]*ninpt, dtype=np.float32)

mf = cl.mem_flags
#Generate indices for cloning
s = np.concatenate((np.array([0], dtype=np.uint), np.linspace(2, 7, num=nsamp//4).astype(np.uint).cumsum(),))
hs = np.empty(nsamp, dtype=np.uint)     #Distribution of sotred indexes to new genome
hs.fill(0)
for x in range(0, len(s)-1):
    sx = np.arange(s[x], s[x+1]).astype(np.uint)
    for sxi in sx:
        if sxi<len(s): hs[sxi] = x

defines = \
"#define nvarsd "+str(nvarsd)+"\n"+\
"#define nvarsg "+str(nvarsg)+"\n"+\
"#define ninpt "+str(ninpt)+"\n\n"
kernels = genn.genkern2(ninpt, topology, lambda x: cl.Program(ctx, x).build())
print(kernels)
#exit()
run = cl.Program(ctx, defines+genn.genkern(ninpt, topology)+"\n"+\
"""
__kernel void copy_inp(__global float *inpt, __global float *dnr){
    uint gid = get_global_id(0);
    dnr[gid] = inpt[gid];
}

__kernel void replicate_mutate(__global float *_gms, __global float *_tmpgms,\
                               __global uint *srt_idxs, __global float *res_g,\
                               __global float *_rnd, __global uint *_nvarsg, __global uint *_shiftsg) {
  uint gid = get_global_id(0);
  const uint hs[] = {"""+", ".join([str(hh) for hh in hs])+"""}; //Indexes for allocate cutted population to full
  uint h = hs[gid];                           
  uint i, idx = srt_idxs[h];                  //Sorted indexes of population
  __global float *gms = _gms + idx*nvarsg+_shiftsg[0];
  __global float *rnd = _rnd + gid*nvarsg+_shiftsg[0];
  __global float *tmpgms = _tmpgms + gid*nvarsg;//+_shiftsg[0];
  //float gml[nvarsg];
  float res = res_g[idx]/100.0;//<0.01?res_g[idx]:0.01;
  for(i=0; i<_nvarsg[0]; i++)
      tmpgms[i] = gms[i]+rnd[i]*res;

}

__kernel void savebest(__global float *_gms, __global float *_gm, __global float *res_g, __global float *bestres, __global uint *srt_idxs,\
                       __global uint *_nvarsg, __global uint *_shiftsg){
    uint idx = srt_idxs[0];
    if(res_g[idx] < bestres[0]){
        bestres[0] = res_g[idx];
        __global float *gms = _gms + idx*nvarsg+_shiftsg[0];
        for(uint i=0; i<_nvarsg[0]; i++)
            _gm[i] = gms[i];
    }
}
__kernel void loadbest(__global float *_gms, __global float *_gm, __global float *res_g, __global float *bestres, __global uint *srt_idxs,\
                       __global uint *_nvarsg, __global uint *_shiftsg){
    uint idx = srt_idxs[0];
    float stillbest = res_g[idx];
    __global float *gms = _gms + idx*nvarsg+_shiftsg[0];
    if(stillbest>bestres[0]){
        for(uint i=0; i<_nvarsg[0]; i++){
            gms[i] = _gm[i];
        }
    } else {
        bestres[0] = stillbest;
        for(uint i=0; i<_nvarsg[0]; i++)
            _gm[i] = gms[i];
    }
}

__kernel void fillgms(__global float *_gms, __global float *_tmpgms, __global uint *_nvarsg, __global uint *_shiftsg) {
  uint gid = get_global_id(0);
  __global float *gms = _gms + gid*nvarsg+_shiftsg[0];
  __global float *tmpgms = _tmpgms + gid*nvarsg;//+_shiftsg[0];
  gms[gid] = tmpgms[gid];
  for(uint i=0; i<_nvarsg[0]; i++)
      gms[i] = tmpgms[i];
}
""").build()
#Metabuffer for opencl datas
global_offset = None
g_times_l = False
wait_for = None
k = 0
_k = 0
lt = len(topology)-1
currmin = (9999999999, 0,)
topconns = genn.topconns(topology)   #Layer-to-layer conns
tcshifts = np.concatenate((np.array([0], dtype=np.uint32), topconns.cumsum().astype(np.uint32)[:-1],))         #Shifts for a conns
ptcshifts = [int(k) for k in tcshifts]
print("Layer-to-layer conns is", topconns)
print("Shifts for a conns is", tcshifts)
topconnsg = [cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.array([ma], dtype=np.uint32)) for ma in topconns]
tcshiftsg = [cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.array([ma], dtype=np.uint32)) for ma in tcshifts]
obuf = np.array([99999.99999]).astype(np.float32) #Array for an best result
olid = np.empty(1).astype(np.uint32)
o_med = cl.Buffer(ctx, mf.WRITE_ONLY, size=obuf.nbytes)
o_min = cl.Buffer(ctx, mf.WRITE_ONLY, size=obuf.nbytes)
o_lid = cl.Buffer(ctx, mf.WRITE_ONLY, size=olid.nbytes)
gmbg = cl.Buffer(ctx, mf.READ_WRITE, size=nvarsg*obuf.nbytes)
brsg = cl.Buffer(ctx, mf.READ_WRITE, size=obuf.nbytes)
cl.enqueue_copy(queue, brsg, np.array([np.inf], dtype=np.float32))
vsg = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=inp_np)   #Device array of input data
dnrg = cl.Buffer(ctx, mf.READ_WRITE, size=inp_np.nbytes)                #Device array of input data
run.copy_inp(queue, (nvarsd*ninpt,), None, vsg, dnrg)                   #Copy data to fst layer
vsrg = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=vsr)     #Device array of outpus data
gms = cl.Buffer(ctx, mf.READ_WRITE| mf.COPY_HOST_PTR, hostbuf=arr_np)   #Device array of genome
tmpgms = cl.Buffer(ctx, mf.READ_WRITE, arr_np.nbytes)
#Results buffers (as genome counts)
res_np = np.empty(nsamp).astype(np.float32)             #Host array of equation results
res_g = cl.Buffer(ctx, mf.READ_WRITE, res_np.nbytes)    #Device array of equation results
ressh = np.empty(nsamp).astype(np.uint32)               #Host array of sorted indexes
ressg = cl.Buffer(ctx, mf.READ_WRITE, ressh.nbytes)     #Device array of sorted indexes
randsg = cl.Buffer(ctx, mf.READ_WRITE, arr_np.nbytes)   #Array of randoms
randg = randfloat(ctx, nvarsg*nsamp)
randg.reseed()
    
dbg = True

def printdbg(*args):
    if dbg: print(args)

for cy in range(1, 17):    
    if cy%8==0:
        clreducer.reduce_min(queue, res_g, nsamp, o_min, o_lid)
        cl.enqueue_copy(queue, obuf, o_min)
        if dbg: queue.finish()
        print("k=={0} of {1}. min value is {2}. Load best".format(k, lt, obuf[0]))
        run.loadbest(queue, (1,), None, gms, gmbg, res_g, brsg, ressg, topconnsg[k], tcshiftsg[k])
        cl.enqueue_copy(queue, brsg, np.array([np.inf], dtype=np.float32))
        if dbg: queue.finish()
        if dbg: print("Finish")
        if k < (lt-1): kernels["finish"][k].runnet(queue, (ninpt,), None, gms, dnrg, vsrg, res_g)
        #if dbg: 
        #    queue.finish()
        #    exit()
        _k+=1
        k = _k%lt; 
    if cy%512==0:
        randg.reseed()
    if obuf[0]<0.000001: break
    if k==0:
        printdbg(cy, k, "run.copy_inp Starts")
        run.copy_inp(queue, (nvarsd*ninpt,), None, vsg, dnrg)
    printdbg(cy, k, "ordinal .runnet Starts")
    if dbg:
        cl.enqueue_copy(queue, inp_np, dnrg)
        print("Device input is", inp_np, " Contains {0} NaNs.".format(np.count_nonzero(np.isnan(inp_np))))
        cl.enqueue_copy(queue, arr_np, gms)
        print("Device coeffs is", arr_np, " Contains {0} NaNs.".format(np.count_nonzero(np.isnan(arr_np))))
    kernels["ordinal"][k].runnet(queue, (nsamp,), None, gms, dnrg, vsrg, res_g)
    if dbg:
        cl.enqueue_copy(queue, res_np, res_g)
        res_np.sort()
        print("Sorted results is", res_np)

    printdbg(cy, k, "clreducer.sort Starts")
    clreducer.sort(queue, nsamp, res_g, ressg)
    if cy%8!=0: run.savebest(queue, (1,), None, gms, gmbg, res_g, brsg, ressg, topconnsg[k], tcshiftsg[k])
    #Generate randoms
    printdbg(cy, k, "randsg Starts")
    randg.randgen(randsg, int(topconns[k]), ptcshifts[k])
    printdbg(cy, k, "run.replicate_mutate Starts")
    run.replicate_mutate(queue, (nsamp,), None, gms, tmpgms, ressg, res_g, randsg, topconnsg[k], tcshiftsg[k])
    printdbg(cy, k, "run.fillgms Starts")
    run.fillgms(queue, (nsamp,), None, gms, tmpgms, topconnsg[k], tcshiftsg[k])
# Check on CPU with Numpy:
print(currmin)
cl.enqueue_copy(queue, ressh, ressg)
cl.enqueue_copy(queue, arr_np, gms)
solve = arr4np[ressh[0]]
print("\nSolve coeffs\n", solve)
print("\nInput\n", inp4np)
print("\nEquation solved\n", [s.sum() for s in inp4np*solve])
