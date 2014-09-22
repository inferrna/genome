#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pyopencl as cl
mf = cl.mem_flags
sz = 64

a_np = np.random.randint(low=1, high=65535, size=sz).astype(np.ushort)

ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)
o_np = np.empty(sz).astype(np.float32)

a_g = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=a_np)
o_g = cl.Buffer(ctx, mf.WRITE_ONLY, size=o_np.nbytes)
res_g = cl.Buffer(ctx, mf.WRITE_ONLY, a_np.nbytes)

prg = cl.Program(ctx, """
int rand(ushort* seed) // 1 <= *seed < m
{
    int const a = 16807; //ie 7**5
    int const m = 2147483647; //ie 2**31-1

    *seed = convert_ushort((*seed * a)%m);
    return(*seed);
}
__kernel void rands(__global ushort* seed_memory, __global float* out) {
{
    int gid = get_global_id(0); // Get the global id in 1D.

    // Since the Park-Miller PRNG generates a SEQUENCE of random numbers
    // we have to keep track of the previous random number, because the next
    // random number will be generated using the previous one.
    ushort seed = seed_memory[gid];
    rand(&seed); // Generate the next random number in the sequence.
    seed_memory[gid] = seed;
    out[gid] = seed/65536.0; // Save the seed for the next time this kernel gets enqueued.
}

}
""").build()
iterations = 3072
ress = np.empty(iterations).astype(np.float32)
for i in range(0, iterations):
    prg.rands(queue, a_np.shape, None, a_g, o_g)
    cl.enqueue_copy(queue, o_np, o_g)
# Check on CPU with Numpy:
    ress[i] = o_np[9]
uniqw, inverse = np.unique(ress, return_inverse=True)
counts = np.bincount(inverse)
idxes = np.where(counts > 1)[0]
print(idxes)
print(ress[idxes])

