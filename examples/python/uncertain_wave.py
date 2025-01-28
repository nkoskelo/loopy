import numpy as np
import pyopencl as cl
import pyopencl.array
import pyopencl.clrandom

import loopy as lp
lp.set_caching_enabled(False)
from loopy.version import LOOPY_USE_LANGUAGE_VERSION_2018_2

from warnings import filterwarnings, catch_warnings
filterwarnings("error", category=lp.LoopyWarning)
from loopy.diagnostic import DirectCallUncachedWarning
filterwarnings("ignore", category=DirectCallUncachedWarning)

ctx = cl.create_some_context(interactive=False)
queue = cl.CommandQueue(ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)

n = 16*16
M = int(5e6)
x_vec_dev = cl.clrandom.rand(queue, n, dtype=np.float32)
y_vec_dev = cl.clrandom.rand(queue, n, dtype=np.float32)
z_vec_dev = cl.clrandom.rand(queue, n, dtype=np.float32)
a_mat_dev = cl.clrandom.rand(queue, (16,16), dtype=np.float32)
b_mat_dev = cl.clrandom.rand(queue, (16,M), dtype=np.float32)

"""
    The equation:

    u_t = -c u_x where c is uncertain.

    I am going to discritize with a finite difference scheme.
    This will be constant throughout the trials for modifying c.

    The discritization will be first order forward in time.
    2nd order central difference in space.
"""

n = 5000
ht = 1e-4
x_grid = np.linspace(-5, 5, n).astype(np.float32)
hx = x_grid[1] - x_grid[0]
M = 10
c = np.linspace(1,15, M).astype(np.float32).reshape(M,1)
K = np.arange(n)
Kp1 = np.roll(K, -1) # Roll left
Km1 = np.roll(K, 1) # Roll right

knl = lp.make_kernel(
        "{ [space_ind, inst]: 0<=space_ind<N and 0<=inst<M}",
        "state[inst, space_ind] = in_state[space_ind] -c[inst] * ht / (2 * hx) * (in_state[Kp1[space_ind]] - in_state[Km1[space_ind]])")

knl = lp.set_options(knl, write_code=True)
print(knl)
print("==================================")
knl = lp.add_and_infer_dtypes(knl, {"ht,hx,c,in_state": np.float32, 
                                    "Km1": Km1.dtype,
                                    "Kp1": Kp1.dtype})
print(lp.generate_code_v2(knl).device_code())
print("==================================")

nt = 100
times_basic = np.zeros(nt)
for i in range(nt):
    for j in range(M):
        evt, _ = knl(queue, c = c[j], in_state = x_grid, ht = ht, hx = hx, Kp1=Kp1, Km1 = Km1)
        if isinstance(evt, list):
            [evt[j].wait() for j in range(len(evt))]
            t = sum([evt[j].profile.end - evt[j].profile.start for j in range(len(evt))])
            times_basic[i] += t
        else:
            evt.wait()
            times_basic[i] += evt.profile.end - evt.profile.start


breakpoint()

n = 5000
ht = 1e-4
x_grid = np.linspace(-5, 5, n).astype(np.float32)
hx = x_grid[1] - x_grid[0]
M = 10
c = np.linspace(1,15, M).astype(np.float32).reshape(M,1)
K = np.arange(n)
Kp1 = np.roll(K, -1) # Roll left
Km1 = np.roll(K, 1) # Roll right

knl = lp.make_kernel(
        "{ [space_ind, inst]: 0<=space_ind<N and 0<=inst<1}",
        "state[inst, space_ind] = in_state[space_ind] -c[inst] * ht / (2 * hx) * (in_state[Kp1[space_ind]] - in_state[Km1[space_ind]])")

print("THE BATCHED VERSION!!!!!!!!!!!!")

knl = lp.set_options(knl, write_code=True)
knl = lp.to_batched(knl, M, ["c", "state"])
knl = lp.tag_inames(knl, dict(ibatch="ilp"))
print(knl)
print("==================================")
knl = lp.add_and_infer_dtypes(knl, {"ht,hx,c,in_state": np.float32, 
                                    "Km1": Km1.dtype,
                                    "Kp1": Kp1.dtype})
print(lp.generate_code_v2(knl).device_code())
print("==================================")


nt = 100
times = np.zeros(nt)
for i in range(nt):
    evt, _ = knl(queue, c = c, in_state = x_grid, ht = ht, hx = hx, Kp1=Kp1, Km1 = Km1)
    if isinstance(evt, list):
        [evt[j].wait() for j in range(len(evt))]
        t = sum([evt[j].profile.end - evt[j].profile.start for j in range(len(evt))])
        times[i] = t
    else:
        evt.wait()
        times[i] = evt.profile.end - evt.profile.start
print("Batched: ", times)
print("Basic Times: ", times_basic)
