import numpy as np
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
    C = np.empty_like(A)
    for i in prange(A.shape[0]):       
        for j in prange(A.shape[1]):      
            if A[i,j] >= B[i,j]:
                C[i,j] = A[i,j]
            else:
                C[i,j] = B[i,j] 
    return C
    


def max_gpu(A, B):
    """
     Returns
     -------
     np.array
         element-wise maximum between A and B
     """
    dev_A = cuda.to_device(A)
    dev_B = cuda.to_device(B)
    dev_C = cuda.device_array(np.shape(A))
    threadsperblock = 1000 
    blockspergrid = 1000
    max_kernel[blockspergrid, threadsperblock](dev_A, dev_B, dev_C)
    C = dev_C.copy_to_host()
    return C
    


@cuda.jit
def max_kernel(A, B, C):
    tx = cuda.threadIdx.x
    bx = cuda.blockIdx.x
    C[bx,tx] = max(A[bx,tx], B[bx,tx])


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
