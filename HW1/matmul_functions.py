import numpy as np
from numba import njit, cuda, prange
import timeit


def matmul_transpose_trivial(X):
    C = np.empty((len(X),len(X)))
    for row in range(len(X)):
        for col in range(len(X)):
            sum = 0
            for i in range(len(X[row])):
                sum = sum + (X[row][i] * X[col][i])
            C[row][col] = sum
    return C

@njit(parallel=True)
def matmul_transpose_numba(X):
    C = np.empty((len(X),len(X)))
    for row in prange(len(X)):
        for col in prange(len(X)):
            sum = 0
            for i in range(len(X[row])):
                sum = sum + (X[row][i] * X[col][i])
            C[row][col] = sum
    return C


def matmul_transpose_gpu(X):
    dev_X = cuda.to_device(X)
    dev_C = cuda.device_array(np.shape(X))
    threadsperblock = 1024
    blockspergrid = 1
    matmul_kernel[blockspergrid,threadsperblock](dev_X,dev_C)
    C = dev_C.copy_to_host()
    return C


@cuda.jit
def matmul_kernel(A, C):
    rowsA, colsA = A.shape
    tx = cuda.threadIdx.x
    while(tx < (rowsA)**2):
        i = tx/rowsA
        j = tx % rowsA
        C[i,j] = 0
        for k in range(colsA):
            C[i, j] += A[i, k] * A[j, k]
        tx += 1024
    #bx = cuda.blockIdx.x
    #for i in range(len(A[0])):
    #    sum = sum + (A[bx][i] * A[tx][i])
    #C[bx][tx] = sum
    #for col in prange(len(A)):
    #    sum = 0
    #    for i in range(len(A[0])):
    #        sum = sum + (A[tx][i] * A[col][i])
    #C[tx][col] = sum


def verify_solution():
    X = np.random.randn(784, 128)
    Xt = X.copy().transpose()

    if not np.allclose(matmul_transpose_trivial(X), np.matmul(X, Xt)):
        print('[-] matmul_transpose_trivial failed')
        exit(0)
    else:
        print('[+] matmul_transpose_trivial passed')

    if not np.allclose(matmul_transpose_numba(X), np.matmul(X, Xt)):
        print('[-] matmul_transpose_numba failed')
        exit(0)
    else:
        print('[+] matmul_transpose_numba passed')

    if not np.allclose(matmul_transpose_gpu(X), np.matmul(X, Xt)):
        print('[-] matmul_transpose_gpu failed')
        exit(0)
    else:
        print('[+] matmul_transpose_gpu passed')

    print('[+] All tests passed\n')


# this is the comparison function - keep it as it is, don't change X or Y.
def matmul_comparison():
    X = np.random.randn(784, 128)
    Xt = X.copy().transpose()

    def timer(f, functionParameters):
        return min(timeit.Timer(lambda: f(X) if functionParameters == 1 else f(X, Xt)).repeat(3, 100))

    # print('Python:', timer(matmul_transpose_trivial, 1)) we will not consider this since it takes infinite time :)
    print('Numpy:', timer(np.matmul, 2))
    print('Numba:', timer(matmul_transpose_numba, 1))
    print('CUDA:', timer(matmul_transpose_gpu, 1))


if __name__ == '__main__':
    verify_solution()
    matmul_comparison()
