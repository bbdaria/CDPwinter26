import numpy as np
import math
from numba import cuda, njit, prange, float32
import timeit

def max_cpu(A, B):
    """
     Returns
     -------
     np.array
         element-wise maximum between A and B
     """
    return [[max(A[i][j], B[i][j]) for j in range(len(A[0]))] for i in range(len(A))]
    


@njit(parallel=True)
def max_numba(A, B):
    """
     Returns
     -------
     np.array
         element-wise maximum between A and B
     """
    C = np.zeros(A.shape, dtype=A.dtype)
    for i in prange(A.shape[0]):       
        for j in range(A.shape[1]):      
            a = A[i, j]
            b = B[i, j]
            C[i, j] = a if a > b else b 
    return C
    


def max_gpu(A, B):
    """
     Returns
     -------
     np.array
         element-wise maximum between A and B
     """
    dev_c = cuda.device_array(A.shape, A.dtype)
    threads_per_block = 128
    blocks_per_grid = math.ceil(A.size / threads_per_block)
    max_kernel[blocks_per_grid, threads_per_block](A, B, dev_c)
    C = dev_c.copy_to_host()
    return C
    


@cuda.jit
def max_kernel(A, B, C):
    tx = cuda.threadIdx.x
    bx = cuda.blockIdx.x
    bw = cuda.blockDim.x
    idx = tx + bx * bw

    n = A.shape[0] * A.shape[1]

    if idx < n:
        cols = A.shape[1]
        i = idx // cols
        j = idx % cols
        a = A[i, j]
        b = B[i, j]
        C[i, j] = a if a > b else b


def verify_solution():
    A = np.random.randint(0, 256, (1000, 1000))
    B = np.random.randint(0, 256, (1000, 1000))

    if not np.all(max_cpu(A, B) == np.maximum(A, B)):
        print('[-] max_cpu failed')
        exit(0)
    else:
        print('[+] max_cpu passed')

    if not np.all(max_numba(A, B) == np.maximum(A, B)):
        print('[-] max_numba failed')
        exit(0)
    else:
        print('[+] max_numba passed')

    if not np.all(max_gpu(A, B) == np.maximum(A, B)):
        print('[-] max_gpu failed')
        exit(0)
    else:
        print('[+] max_gpu passed')

    print('[+] All tests passed\n')


# this is the comparison function - keep it as it is.
def max_comparison():
    A = np.random.randint(0, 256, (1000, 1000))
    B = np.random.randint(0, 256, (1000, 1000))

    def timer(f):
        return min(timeit.Timer(lambda: f(A, B)).repeat(3, 20))

    print('[*] CPU:', timer(max_cpu))
    print('[*] Numba:', timer(max_numba))
    print('[*] CUDA:', timer(max_gpu))


if __name__ == '__main__':
    verify_solution()
    max_comparison()
