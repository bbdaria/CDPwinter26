import numpy as np


def ringallreduce(send, recv, comm, op):
    """ ring all reduce implementation
    You need to use the algorithm shown in the lecture.

    Parameters
    ----------
    send : numpy array
        the array of the current process
    recv : numpy array
        an array to store the result of the reduction. Of same shape as send
    comm : MPI.Comm
    op : associative commutative binary operator
    """
    p = comm.Get_size()
    rank = comm.Get_rank()

    chunk_size = send.size // p
    
    chunks = [(chunk_size * r, chunk_size * (r + 1)) for r in range(p - 1)]
    chunks.append((chunk_size * (p - 1), send.size))
    right = (rank + 1) % p
    left = (rank - 1 + p) % p
    np.copyto(recv, send)
    current = rank

    for i in range(p - 1):
        # scatter
        send_start, send_end = chunks[current]
        req = comm.Isend(recv[send_start:send_end], dest=right, tag=0)

        current = (current - 1) % p
        recv_start, recv_end = chunks[current]

        tmp = np.empty(recv_end - recv_start, dtype=recv.dtype)
        comm.Recv(tmp, source=left, tag=0)

        # reduce received chunk into the accumulator chunk
        recv[recv_start:recv_end] = op(recv[recv_start:recv_end], tmp)

        req.Wait()
    
    current = (rank - 1) % p
    for i in range(p - 1):
        # gather
        send_start, send_end = chunks[current]
        req = comm.Isend(recv[send_start:send_end], dest=right, tag=1)

        current = (current - 1) % p
        recv_start, recv_end = chunks[current]

        tmp = np.empty(recv_end - recv_start, dtype=recv.dtype)
        comm.Recv(tmp, source=left, tag=1)

        # just place the chunk (no reduction in all-gather)
        recv[recv_start:recv_end] = tmp

        req.Wait()
        
    return recv





    
