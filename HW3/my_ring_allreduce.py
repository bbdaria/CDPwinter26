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

    # 1. Copy data to recv buffer
    np.copyto(recv, send)

    # 2. Create a flat view of the array to handle 2D/3D inputs correctly
    #    This allows us to treat matrices as a long stream of numbers
    recv_flat = recv.reshape(-1)

    # 3. Calculate chunks based on the FLATTENED size
    total_elements = recv_flat.size
    chunk_size = total_elements // p
    
    chunks = [(chunk_size * r, chunk_size * (r + 1)) for r in range(p - 1)]
    chunks.append((chunk_size * (p - 1), total_elements))

    right = (rank + 1) % p
    left = (rank - 1 + p) % p
    current = rank

    # --- Phase 1: Scatter-Reduce ---
    for i in range(p - 1):
        # scatter
        send_start, send_end = chunks[current]
        # Use recv_flat for slicing!
        req = comm.Isend(recv_flat[send_start:send_end], dest=right, tag=0)

        current = (current - 1) % p
        recv_start, recv_end = chunks[current]

        tmp = np.empty(recv_end - recv_start, dtype=recv.dtype)
        comm.Recv(tmp, source=left, tag=0)

        # reduce using the flat view
        recv_flat[recv_start:recv_end] = op(recv_flat[recv_start:recv_end], tmp)

        req.Wait()
    
    # --- Phase 2: All-Gather ---
    current = (rank - 1) % p
    for i in range(p - 1):
        # gather
        send_start, send_end = chunks[current]
        req = comm.Isend(recv_flat[send_start:send_end], dest=right, tag=1)

        current = (current - 1) % p
        recv_start, recv_end = chunks[current]

        tmp = np.empty(recv_end - recv_start, dtype=recv.dtype)
        comm.Recv(tmp, source=left, tag=1)

        # store result in flat view
        recv_flat[recv_start:recv_end] = tmp

        req.Wait()
        
    # No need to reshape back, modifying recv_flat modifies recv in-place (if contiguous)
    return recv





    
