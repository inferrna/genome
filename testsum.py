import numpy as np
import pyopencl as cl
import pyopencl.array as cl_array
from reducecl import cl_reduce
from math import log2, ceil
from time import time
mf = cl.mem_flags


ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)
N = 5
tosum = 13000000
tosumr = pow(2, ceil(log2(tosum)))
nsamp = tosumr*N
print("tosumr==", tosumr)
arr_np = np.empty(nsamp).astype(np.float32)
arrsnp = arr_np.reshape(N, tosumr)
obuf = np.empty(5).astype(np.float32)
for arr in arrsnp:
    np.copyto(arr[:tosum], np.random.rand(tosum).astype(np.float32))

reducer = cl_reduce(ctx, nsamp)
brsg = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=arr_np)
obufg = cl.Buffer(ctx, mf.READ_WRITE, obuf.nbytes)
t1 = time()
cpubuf = [a.sum() for a in arrsnp]
t2 = time()
reducer.reduce_sum(queue, brsg, nsamp, obufg, N)
t3 = time()
print("cpu totsum == ", arr_np.sum())
cl.enqueue_copy(queue, obuf, obufg)
print(obuf, t3-t2)
print(cpubuf, t2-t1)
