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
queue = cl.CommandQueue(ctx)

n = 16*16
M = int(5e6)
x_vec_dev = cl.clrandom.rand(queue, n, dtype=np.float32)
y_vec_dev = cl.clrandom.rand(queue, n, dtype=np.float32)
z_vec_dev = cl.clrandom.rand(queue, n, dtype=np.float32)
a_mat_dev = cl.clrandom.rand(queue, (16,16), dtype=np.float32)
b_mat_dev = cl.clrandom.rand(queue, (16,M), dtype=np.float32)


x_vect_host = np.random.randn(n).astype(np.float32)
y_vect_host = np.random.randn(n).astype(np.float32)

"""
    My kernel is to apply matrix multiplication to A and B.
"""

knl = lp.make_kernel(
        "{ [i,j,k]: 0<=i<16 and 0<=j<M and 0<=k<16}",
        "c[i,j] = sum(k, a[i,k] * b[k,j])",
        assumptions="M>=1 and M mod 50 = 0") #,
        #assumptions="n>=1 and n mod 16 = 0")

#knl = lp.split_iname(knl, "k", 16)
#knl = lp.prioritize_loops(knl, "k_outer,k_inner")
#knl = lp.tag_inames(knl, dict(k_inner="unr"))
#knl = lp.add_and_infer_dtypes(knl, dict(a=np.float32, b=np.float32))
knl = lp.split_iname(knl, "i", 16, outer_tag="g.1", inner_tag="l.1")
knl = lp.split_iname(knl, "j", 50, outer_tag="g.0", inner_tag="l.0")
knl = lp.tag_inames(knl, dict(k="unr"))
knl = lp.set_options(knl, write_code=True)

import time

tmp_times = []
for i in range(1000):
    start_time = time.monotonic_ns()
    evt, (out,) = knl(queue, a=a_mat_dev, b=b_mat_dev)
    end_time = time.monotonic_ns()
    tmp_times.append(end_time - start_time)

tmp_times = np.array(tmp_times)
np.savetxt("out_times.csv",tmp_times, delimiter=",")

ein = lp.make_einsum("ij,jk->ik", ["a","b"])
ein = lp.prioritize_loops(ein, "k,i,j")
print("Ein: ",ein)
tmp_times = []
for i in range(1000):
    start_time = time.monotonic_ns()
    evt, (out,) = ein(queue, a=a_mat_dev, b=b_mat_dev)
    end_time = time.monotonic_ns()
    tmp_times.append(end_time - start_time)

tmp_times = np.array(tmp_times)
np.savetxt("base_einsum_times.csv",tmp_times, delimiter=",")

#print(ein)
print(knl)

knl_2 = lp.make_kernel(
        "{ [i,k]: 0<=i,j<n and 0<=k<8}",
        "c[i] = sum(k, a[i,k] * b[k])",
        assumptions="n>=1 and n mod 16 = 0")

knl_2 = lp.split_iname(knl_2, "i", 16, outer_tag="g.0", inner_tag="l.0")
knl_2 = lp.tag_inames(knl_2, dict(k="unr"))
knl_2 = lp.to_batched(knl_2, 1000, "b")
knl_2 = lp.set_options(knl_2, write_code=True)

#code = lp.generate_code_v2(knl_2).device_code()
#print("K2 code: ", code)

#print(knl_2)
"""
knl_example = lp.make_kernel(
        "{ [i]: 0<=i<n }",
        "a[i]=0", assumptions="n>=0")
knl_example = lp.split_iname(knl_example, "i", 128,
                             outer_tag="g.0", inner_tag="l.0")
knl_example = lp.set_options(knl_example, write_code=True)
evt, (out,) = knl_example(queue, a=x_vec_dev)
glob, loc = knl_example["loopy_kernel"].get_grid_size_upper_bounds(knl_example.callables_table)

print(glob)
print(loc)
"""

#op_map = lp.get_op_map(knl, subgroup_size="guess")

#print(op_map)
#print("Tgt: ", knl.target)
#print("Device: ", knl.target.device)

knl = lp.make_kernel(
        "{ [i,j,k]: 0<=i<16 and 0<=j<M and 0<=k<16}",
        "c[i,j] = sum(k, a[i,k] * b[k,j])") #,
        #assumptions="n>=1 and n mod 16 = 0")

#knl = lp.split_iname(knl, "k", 16)
#knl = lp.prioritize_loops(knl, "k_outer,k_inner")
#knl = lp.tag_inames(knl, dict(k_inner="unr"))
#knl = lp.add_and_infer_dtypes(knl, dict(a=np.float32, b=np.float32))
#knl = lp.tag_inames(knl, dict(k="unr"))
knl = lp.prioritize_loops(knl, "i,k,j")
knl = lp.set_options(knl, write_code=True)

print("New knl: ", knl)

tmp_times = []
for i in range(1000):
    start_time = time.monotonic_ns()
    evt, (out,) = knl(queue, a=a_mat_dev, b=b_mat_dev)
    end_time = time.monotonic_ns()
    tmp_times.append(end_time - start_time)

tmp_times = np.array(tmp_times)
np.savetxt("just_inner_times.csv",tmp_times, delimiter=",")
