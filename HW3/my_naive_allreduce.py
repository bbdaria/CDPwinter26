import numpy as np


def allreduce(send, recv, comm, op):
    """ Naive all reduce implementation

    Parameters
    ----------
    send : numpy array
        the array of the current process
    recv : numpy array
        an array to store the result of the reduction. Of same shape as send
    comm : MPI.Comm
    op : associative commutative binary operator
    """
    rank = comm.Get_rank()
    size = comm.Get_size()

    np.copyto(recv, send)

    for src in range(size):
        if src == rank:
            continue

        tmp = np.empty_like(send)
        comm.Sendrecv(
            sendbuf=send,
            dest=src,
            recvbuf=tmp,
            source=src
        )
        
        recv[:] = op(recv, tmp)

    return recv
