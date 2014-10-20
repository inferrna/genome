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
ninpt = 5   #Samples count
nvarsd = 9   #Count of equations members
topology = [nvarsd, 7, 5, 1]
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
print(genn.genkern2(ninpt, topology))
print(genn.oldkernel)
exit()
run = cl.Program(ctx, defines+genn.genkern(ninpt, topology)+"\n"+\
"""
__kernel void replicate_mutate(__global float *_gms, __global float *_tmpgms,\
                               __global uint *srt_idxs, __global float *res_g,\
                               __global float *_rnd, __global uint *_nvarsg) {
  uint gid = get_global_id(0);
  const uint hs[] = {"""+", ".join([str(hh) for hh in hs])+"""}; //Indexes for allocate cutted population to full
  uint h = hs[gid];                           
  uint i, idx = srt_idxs[h];                  //Sorted indexes of population
  __global float *gms = _gms + idx*nvarsg;
  __global float *rnd = _rnd + gid*nvarsg;
  __global float *tmpgms = _tmpgms + gid*nvarsg;
  //float gml[nvarsg];
  float res = res_g[idx]/100.0;//<0.01?res_g[idx]:0.01;
  for(i=0; i<_nvarsg[0]; i++)
      tmpgms[i] = gms[i]+rnd[i]*res;

}

__kernel void savebest(__global float *_gms, __global float *_gm, __global float *res_g, __global float *bestres, __global uint *srt_idxs){
    uint idx = srt_idxs[0];
    bestres[0] = res_g[idx];
    __global float *gms = _gms + idx*nvarsg;
    for(uint i=0; i<nvarsg; i++)
        _gm[i] = gms[i];
}
__kernel void loadbest(__global float *_gms, __global float *_gm, __global float *res_g, __global float *bestres, __global uint *srt_idxs){
    uint idx = srt_idxs[0];
    float stillbest = res_g[idx];
    __global float *gms = _gms + idx*nvarsg;
    if(stillbest>bestres[0]){
        for(uint i=0; i<nvarsg; i++){
            gms[i] = _gm[i];
        }
    } else {
        bestres[0] = stillbest;
        for(uint i=0; i<nvarsg; i++)
            _gm[i] = gms[i];
    }
}

__kernel void fillgms(__global float *_gms, __global float *_tmpgms, __global uint *_nvarsg) {
  uint gid = get_global_id(0);
  __global float *gms = _gms + gid*nvarsg;
  __global float *tmpgms = _tmpgms + gid*nvarsg;
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
tcshiftsg = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=tcshifts)
obuf = np.array([99999.99999]).astype(np.float32) #Array for an best result
olid = np.empty(1).astype(np.uint32)
o_med = cl.Buffer(ctx, mf.WRITE_ONLY, size=obuf.nbytes)
o_min = cl.Buffer(ctx, mf.WRITE_ONLY, size=obuf.nbytes)
o_lid = cl.Buffer(ctx, mf.WRITE_ONLY, size=olid.nbytes)
gmbg = cl.Buffer(ctx, mf.READ_WRITE, size=nvarsg*obuf.nbytes)
brsg = cl.Buffer(ctx, mf.READ_WRITE, size=obuf.nbytes)
vsg = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=inp_np) #Device array of input data
vsrg = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=vsr)   #Device array of outpus data
gms = cl.Buffer(ctx, mf.READ_WRITE| mf.COPY_HOST_PTR, hostbuf=arr_np) #Device array of genome
tmpgms = cl.Buffer(ctx, mf.READ_WRITE, arr_np.nbytes)
#Results buffers (as genome counts)
res_np = np.empty(nsamp).astype(np.float32)             #Host array of equation results
res_g = cl.Buffer(ctx, mf.READ_WRITE, res_np.nbytes)    #Device array of equation results
ressh = np.empty(nsamp).astype(np.uint32)               #Host array of sorted indexes
ressg = cl.Buffer(ctx, mf.READ_WRITE, ressh.nbytes)     #Device array of sorted indexes
randsg = cl.Buffer(ctx, mf.READ_WRITE, arr_np.nbytes)   #Array of randoms
randg = randfloat(ctx, nvarsg*nsamp)
randg.reseed()
    


for cy in range(1, 640000):
    run.runnet(queue, (nsamp,), None, vsg, gms, vsrg, res_g)
    #print("enqueue ok")
    #cl.enqueue_copy(queue, res_np, res_g)
    #print("Result is", res_np)
    ##Reduce sum
    #clreducer.reduce_sum(queue, res_g, nsamp, o_med)
    #cl.enqueue_copy(queue, obuf, o_med)
    #print("total sum is", obuf)
    #Reduce minimal
    clreducer.sort(queue, nsamp, res_g, ressg)
    if cy%8==0:
        clreducer.reduce_min(queue, res_g, nsamp, o_min, o_lid)
        cl.enqueue_copy(queue, obuf, o_min)
        _k+=1
        k = k%lt; 
        print("min value is", obuf)
        run.loadbest(queue, (1,), None, gms, gmbg, res_g, brsg, ressg)
    else:
        run.savebest(queue, (1,), None, gms, gmbg, res_g, brsg, ressg)
    if cy%512==0:
        randg.reseed()
    if obuf[0]<0.000001: break
    #cl.enqueue_copy(queue, olid, o_lid)
    #print("min index is", olid)
    #Reduce sort
#__global float *_gms, __global float *_gm, __global float *res_g, __global float *bestres, __global uint *srt_idxs)
    #cl.enqueue_copy(queue, ressh, ressg)
    #Generate randoms
    randg.randgen(randsg, int(topconns[k]), ptcshifts[k])
    #print(len(res_np), " vs ", ressh.max() )
    #print(res_np[ressh])
    #Mutate
    run.replicate_mutate(queue, (nsamp,), None, gms, tmpgms, ressg, res_g, randsg, topconnsg[k], global_offset=(ptcshifts[k],))
    run.fillgms(queue, (nsamp,), None, gms, tmpgms, topconnsg[k], global_offset=(ptcshifts[k],))
# Check on CPU with Numpy:
print(currmin)
cl.enqueue_copy(queue, ressh, ressg)
cl.enqueue_copy(queue, arr_np, gms)
solve = arr4np[ressh[0]]
print("\nSolve coeffs\n", solve)
print("\nInput\n", inp4np)
print("\nEquation solved\n", [s.sum() for s in inp4np*solve])
