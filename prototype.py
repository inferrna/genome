#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pyopencl as cl

equation = "a_g[gid]+2*b_g[gid]+3*c_g[gid]+4*d_g[gid]"
result = 3

a_np = np.random.rand(500).astype(np.float32) - np.random.rand(500).astype(np.float32)
b_np = np.random.rand(500).astype(np.float32) - np.random.rand(500).astype(np.float32)
c_np = np.random.rand(500).astype(np.float32) - np.random.rand(500).astype(np.float32)
d_np = np.random.rand(500).astype(np.float32) - np.random.rand(500).astype(np.float32)

ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)

mf = cl.mem_flags
a_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a_np)
b_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b_np)
c_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=c_np)
d_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=d_np)

run = cl.Program(ctx, """
__kernel void sum(__global const float *a_g, __global const float *b_g, __global const float *c_g, __global const float *d_g, __global float *res_g) {
  int gid = get_global_id(0);
  res_g[gid] = """+equation+""";
}
""").build()

res_g = cl.Buffer(ctx, mf.WRITE_ONLY, a_np.nbytes)
run.sum(queue, a_np.shape, None, a_g, b_g, c_g, d_g, res_g)

res_np = np.empty_like(a_np)
cl.enqueue_copy(queue, res_np, res_g)

# Check on CPU with Numpy:
print(res_np)
