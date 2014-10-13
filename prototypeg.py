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
ninpt = 3   #Samples count
nvarsd = 9   #Count of equations members
nvarsg = 9   #Count of equations members
nsamp = ctx.get_info(cl.context_info.DEVICES)[0].max_work_group_size #Genome samples count (current sort limitation to local_size)
clreducer = cl_reduce(ctx, nsamp)
#exit()

#Random init genome
arr_np = np.random.rand(nvarsg*nsamp).astype(np.float32) - np.random.rand(nvarsg*nsamp).astype(np.float32)
arr4np = arr_np.reshape(nsamp, nvarsg)
#Random init equation members
inp_np = np.random.rand(nvarsd*ninpt).astype(np.float32) - np.random.rand(nvarsd*ninpt).astype(np.float32)
inp4np = inp_np.reshape(ninpt, nvarsd)


mf = cl.mem_flags
#Generate indices for cloning
s = np.concatenate((np.array([0], dtype=np.uint), np.linspace(2, 7, num=nsamp//4).astype(np.uint).cumsum(),))
hs = np.empty(nsamp, dtype=np.uint)     #Distribution of sotred indexes to new genome
hs.fill(0)
for x in range(0, len(s)-1):
    sx = np.arange(s[x], s[x+1]).astype(np.uint)
    for sxi in sx:
        if sxi<len(s): hs[sxi] = x
run = cl.Program(ctx, 
"#define nvarsd "+str(nvarsd)+"\n"+
"#define nvarsg "+str(nvarsg)+"\n"+
"#define ninpt "+str(ninpt)+"\n"+
"""__kernel void sum(__global float *_vs, __global float *_gms, __global float *res_g) {
  uint i, j, gid = get_global_id(0);
  __global float *gms = _gms + gid*nvarsg;
  __global float *vs = _vs;
  float rsi = 0.0, _res = 0.0;
  for(i=0; i<ninpt; i++){
    rsi = 0.0;
    for(j=0; j<nvarsd; j++){
        rsi += vs[j]*gms[j];
    }
    vs += nvarsd;
    _res += fabs(rsi-1.0);
    //_res = gms[j-1];
  }
  res_g[gid] = _res;
}

__kernel void replicate_mutate(__global float *_gms, __global float *_tmpgms,\
                               __global uint *srt_idxs, __global float *res_g,\
                               __global float *_rnd) {
  uint gid = get_global_id(0);
  const uint hs[] = {"""+", ".join([str(hh) for hh in hs])+"""}; //Indexes for allocate cutted population to full
  uint h = hs[gid];                           
  uint i, idx = srt_idxs[h];                  //Sorted indexes of population
  __global float *gms = _gms + idx*nvarsg;
  __global float *rnd = _rnd + gid*nvarsg;
  __global float *tmpgms = _tmpgms + gid*nvarsg;
  //float gml[nvarsg];
  float res = res_g[idx]<1.0?res_g[idx]:1.0;
  for(i=0; i<nvarsg; i++)
      tmpgms[i] = gms[i]+rnd[i]*res;

}

__kernel void fillgms(__global float *_gms, __global float *_tmpgms) {
  uint gid = get_global_id(0);
  __global float *gms = _gms + gid*nvarsg;
  __global float *tmpgms = _tmpgms + gid*nvarsg;
  gms[gid] = tmpgms[gid];
  for(uint i=0; i<nvarsg; i++)
      gms[i] = tmpgms[i];
}
""").build()
#Metabuffer for opencl datas
global_offset = None
g_times_l = False
wait_for = None
currmin = (9999999999, 0,)
obuf = np.empty(1).astype(np.float32)
olid = np.empty(1).astype(np.uint32)
o_med = cl.Buffer(ctx, mf.WRITE_ONLY, size=obuf.nbytes)
o_min = cl.Buffer(ctx, mf.WRITE_ONLY, size=obuf.nbytes)
o_lid = cl.Buffer(ctx, mf.WRITE_ONLY, size=olid.nbytes)
vsg = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=inp_np) #Device array of input data
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
    

for cy in range(0, 64):
    run.sum(queue, (nsamp,), None, vsg, gms, res_g)
    print("enqueue ok")
    #cl.enqueue_copy(queue, res_np, res_g)
    #print("Result is", res_np)
    ##Reduce sum
    #clreducer.reduce_sum(queue, res_g, nsamp, o_med)
    #cl.enqueue_copy(queue, obuf, o_med)
    #print("total sum is", obuf)
    #Reduce minimal
    clreducer.reduce_min(queue, res_g, nsamp, o_min, o_lid)
    cl.enqueue_copy(queue, obuf, o_min)
    #cl.enqueue_copy(queue, olid, o_lid)
    print("min value is", obuf)
    #print("min index is", olid)
    #Reduce sort
    clreducer.sort(queue, nsamp, res_g, ressg)
    #cl.enqueue_copy(queue, ressh, ressg)
    if obuf[0]<0.000001: break
    #Generate randoms
    randg.randgen(randsg)
    #print(len(res_np), " vs ", ressh.max() )
    #print(res_np[ressh])
    #Mutate
    run.replicate_mutate(queue, (nsamp,), None, gms, tmpgms, ressg, res_g, randsg)
    run.fillgms(queue, (nsamp,), None, gms, tmpgms)
# Check on CPU with Numpy:
print(currmin)
cl.enqueue_copy(queue, ressh, ressg)
cl.enqueue_copy(queue, arr_np, gms)
solve = arr4np[ressh[0]]
print("\nSolve coeffs\n", solve)
print("\nInput\n", inp4np)
print("\nEquation solved\n", [s.sum() for s in inp4np*solve])
