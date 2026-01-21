import sys
from time import time
import numpy as np
from my_naive_allreduce import *
from my_ring_allreduce import *
from mpi4py import MPI


la_comm = MPI.COMM_WORLD
ma_rank = la_comm.Get_rank()
la_size = la_comm.Get_size()


def _op(x, y):
    # TODO: add your code
    return x+y


for size in [2**12, 2**15, 2**18, 2**21]:
    print("Testing array size:", size)

    data = np.random.rand(size)
    res1 = np.zeros_like(data)
    res2 = np.zeros_like(data)

    start1 = time()
    allreduce(data, res1, la_comm, _op)
    end1 = time()

    print("Naive all-reduce time:", end1-start1)

    start1 = time()
    ringallreduce(data, res2, la_comm, _op)
    end1 = time()

    print("Ring all-reduce time:", end1-start1)

    ground_truth = np.zeros_like(data)
    la_comm.Allreduce(data, ground_truth, op=MPI.SUM) # change the operator according to _op!

    print("Comparing results...")
    print("Naive all-reduce correct:", np.allclose(res1, ground_truth))
    print("Ring all-reduce correct:", np.allclose(res2, ground_truth))
    print()
    print()
    
import sys
from time import time
import numpy as np
from my_naive_allreduce import *
from my_ring_allreduce import *
from mpi4py import MPI


la_comm = MPI.COMM_WORLD
ma_rank = la_comm.Get_rank()
la_size = la_comm.Get_size()


def _op(x, y):
    # TODO: add your code
    return x+y


for size in [2**12, 2**15, 2**18, 2**21]:
    print("Testing array size:", size)

    data = np.random.rand(size)
    res1 = np.zeros_like(data)
    res2 = np.zeros_like(data)

    start1 = time()
    allreduce(data, res1, la_comm, _op)
    end1 = time()

    print("Naive all-reduce time:", end1-start1)

    start1 = time()
    ringallreduce(data, res2, la_comm, _op)
    end1 = time()

    print("Ring all-reduce time:", end1-start1)

    ground_truth = np.zeros_like(data)
    la_comm.Allreduce(data, ground_truth, op=MPI.SUM) # change the operator according to _op!

    print("Comparing results...")
    print("Naive all-reduce correct:", np.allclose(res1, ground_truth))
    print("Ring all-reduce correct:", np.allclose(res2, ground_truth))
    print()
    print()