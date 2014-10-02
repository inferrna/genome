#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pyopencl as cl
mf = cl.mem_flags

class randfloat():
    def __init__(self, ctx, numitems):
        self.sz = numitems
        self.queue = cl.CommandQueue(ctx)
        self.ctx = ctx
        self.a_g =  cl.Buffer(self.ctx, mf.WRITE_ONLY, numitems*4)
        self.prg =  cl.Program(self.ctx, """
        int rand(uint* seed) // 1 <= *seed < m
        {
            uint const a = 16807; //ie 7**5
            uint const m = 2147483647; //ie 2**31-1

            *seed = convert_uint((*seed * a)%m);
            return(*seed);
        }
        __kernel void rands(__global uint* seed_memory, __global float* out) {
        {
            int gid = get_global_id(0); // Get the global id in 1D.

            // Since the Park-Miller PRNG generates a SEQUENCE of random numbers
            // we have to keep track of the previous random number, because the next
            // random number will be generated using the previous one.
            uint seed = seed_memory[gid];
            rand(&seed); // Generate the next random number in the sequence.
            seed_memory[gid] = seed;
            out[gid] = 2*((seed/2147483647.0)-0.5); // Save the seed for the next time this kernel gets enqueued.
        }

        }
        """).build()
    def randgen(self, o_g):
        self.prg.rands(self.queue, (self.sz,), None, self.a_g, o_g)
    def reseed(self):
        self.a_np = np.random.randint(low=1, high=2147483647, size=self.sz).astype(np.uint32)
        cl.enqueue_copy(self.queue, self.a_g, self.a_np)
