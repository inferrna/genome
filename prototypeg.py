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
nvars = 9   #Count of equations members
nsamp = ctx.get_info(cl.context_info.DEVICES)[0].max_work_group_size #Genome samples count (current sort limitation to local_size)
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


mf = cl.mem_flags
#Generate indices for cloning
s = np.concatenate((np.array([0], dtype=np.uint), np.linspace(2, 7, num=nsamp//4).astype(np.uint).cumsum(),))
hs = np.empty(nsamp, dtype=np.uint)
hs.fill(0)
for x in range(0, len(s)-1):
    sx = np.arange(s[x], s[x+1]).astype(np.uint)
    for sxi in sx:
        if sxi<len(s): hs[sxi] = x
run = cl.Program(ctx, 
"#define nvars "+str(nvars)+"\n"+
'\n'.join([gstruct, vstruct])+"""
__kernel void sum(__global struct vars *vs, __global struct genomes *gms, __global float *res_g) {
  int gid = get_global_id(0);
  float _res = 0.0;
  for(int i=0; i<"""+str(ninpt)+"""; i++){
    _res += fabs("""+equation+""");
    //_res += """+equation+""";
  }
  res_g[gid] = _res;
}

__kernel void replicate_mutate(__global struct genomes *gms, __global struct genomes *tmpgms,\
                               __global uint *srt_idxs, __global float *res_g,\
                               __global struct genomes *rands) {
  int gid = get_global_id(0);
  const uint hs[] = {"""+", ".join([str(hh) for hh in hs])+"""}; //Indexes for allocate cutted population to full
  uint h = hs[gid];                           
  int i, idx = srt_idxs[h];                  //Sorted indexes of population
  struct genomes gnml = gms[idx];
  struct genomes rand = rands[gid];
  float *gnma = &gnml;
  float *randa = &rand;
  float res = res_g[idx]<1.0?res_g[idx]:1.0;
  for(i=0; i<nvars; i++)
      gnma[i] = gnma[i]+randa[i]*res;
  tmpgms[gid] = gnml;

}

__kernel void fillgms(__global struct genomes *gms, __global struct genomes *tmpgms) {
  int gid = get_global_id(0);
  gms[gid] = tmpgms[gid];
}
""").build()
#Metabuffer for opencl datas
arrs_g = [[]]*3
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
vsg = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=inp_np)
gms = cl.Buffer(ctx, mf.WRITE_ONLY| mf.COPY_HOST_PTR, hostbuf=arr_np)
tmpgms = cl.Buffer(ctx, mf.WRITE_ONLY, arr_np.nbytes)
clreducer = cl_reduce(ctx, nsamp)
#Parents and grandparents
numsg = cl_array.arange(queue, 0, nsamp, 1, dtype=np.ushort) 
numsh = np.empty(nsamp).astype(np.ushort)
#Results buffers (as genome counts)
res_np = np.empty(nsamp).astype(np.float32)
res_g = cl.Buffer(ctx, mf.WRITE_ONLY, res_np.nbytes)
res_np = np.empty(nsamp).astype(np.float32)
ressh = np.empty(nsamp).astype(np.uint32)
ressg = cl.Buffer(ctx, mf.WRITE_ONLY, ressh.nbytes)     #Sorted indexes
randsg = cl.Buffer(ctx, mf.WRITE_ONLY, arr_np.nbytes)   #Array of randoms
randg = randfloat(ctx, nvars*nsamp)
randg.reseed()
    

for cy in range(1, 3):
    run.sum(queue, (nsamp,), None, vsg, gms, res_g)
    print("enqueue ok")
    #cl.enqueue_copy(queue, res_np, res_g)
    #Reduce sum
    clreducer.reduce_sum(queue, res_g, nsamp, o_med)
    cl.enqueue_copy(queue, obuf, o_med)
    print("total sum is", obuf)
    #Reduce minimal
    clreducer.reduce_min(queue, res_g, nsamp, o_min, o_lid)
    cl.enqueue_copy(queue, obuf, o_min)
    cl.enqueue_copy(queue, olid, o_lid)
    print("min value is", obuf)
    print("min index is", olid)
    #Reduce sort
    clreducer.sort(queue, nsamp, res_g, ressg)
    cl.enqueue_copy(queue, ressh, ressg)
    #Generate randoms
    randg.randgen(randsg)
    print(len(res_np), " vs ", ressh.max() )
    #print(res_np[ressh])
    #Mutate
    run.replicate_mutate(queue, (nsamp,), None, gms, tmpgms, ressg, res_g, randsg)
    run.fillgms(queue, (nsamp,), None, gms, tmpgms)
# Check on CPU with Numpy:
print(currmin)
