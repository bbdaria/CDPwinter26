import numpy as np
from numba import njit, cuda, prange
import timeit


def matmul_transpose_trivial(X):
    raise NotImplementedError("To be implemented")


@njit(parallel=True)
def matmul_transpose_numba(X):
    raise NotImplementedError("To be implemented")


def matmul_transpose_gpu(X):
    raise NotImplementedError("To be implemented")


@cuda.jit
def matmul_kernel(A, C):
    raise NotImplementedError("To be implemented")


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
