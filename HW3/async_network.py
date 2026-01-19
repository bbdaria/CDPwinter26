from network import *
import itertools
import sys
import numpy as np
import math
import mpi4py
from time import time

mpi4py.rc(initialize=False, finalize=False)
from mpi4py import MPI


class AsynchronicNeuralNetwork(NeuralNetwork):

    def __init__(self, sizes=list(), learning_rate=1.0, mini_batch_size=16, number_of_batches=16,
                 epochs=10, number_of_masters=1, matmul=np.matmul):
        # calling super constructor
        super().__init__(sizes, learning_rate, mini_batch_size, number_of_batches, epochs, matmul)
        # setting number of workers and masters
        self.num_masters = number_of_masters

    def fit(self, training_data, validation_data=None):
        # MPI setup
        MPI.Init()
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        self.num_workers = self.size - self.num_masters

        self.layers_per_master = self.num_layers // self.num_masters

        # split up work
        if self.rank < self.num_masters:
            self.do_master(validation_data)
        else:
            self.do_worker(training_data)

        # when all is done
        self.comm.Barrier()
        MPI.Finalize()

    def do_worker(self, training_data):
        """
        worker functionality
        :param training_data: a tuple of data and labels to train the NN with
        """
        # setting up the number of batches the worker should do every epoch
        #Divide the batches between all workers
        batches_per_worker = self.number_of_batches // self.num_workers
        if self.rank == self.num_workers - 1:
            # Last worker takes the remainder
            self.number_of_batches -= (self.num_workers - 1) * batches_per_worker
        else:
            self.number_of_batches = batches_per_worker

        for epoch in range(self.epochs):
            # creating batches for epoch
            data = training_data[0]
            labels = training_data[1]
            mini_batches = self.create_batches(data, labels, self.mini_batch_size)
            
            for x, y in mini_batches:
                self.forward_prop(x)
                nabla_b, nabla_w = self.back_prop(y)

                # 2. FIX: Use Uppercase Isend (sending arrays individually)
                # Send nabla_w and nabla_b to the correct masters
                requests = []
                for l in range(self.num_layers):
                    # Determine destination master for this layer
                    dest_master = l % self.num_masters
                    
                    # Tag strategy: Layer index for Weights, Layer+NumLayers for Biases
                    requests.append(self.comm.Isend(nabla_w[l], dest=dest_master, tag=l))
                    requests.append(self.comm.Isend(nabla_b[l], dest=dest_master, tag=l + self.num_layers))

                # Must wait for sends to clear the buffer before modifying/proceeding
                #for r in requests:
                #    r.Wait()

                # 3. FIX: Use Uppercase Irecv to get updated parameters
                requests = []
                for l in range(self.num_layers):
                    src_master = l % self.num_masters
                    requests.append(self.comm.Irecv(self.weights[l], source=src_master, tag=l))
                    requests.append(self.comm.Irecv(self.biases[l], source=src_master, tag=l + self.num_layers))
                
                # Wait for new weights to arrive before next batch
                for r in requests:
                    r.Wait()

    def do_master(self, validation_data):
        """
        master functionality
        :param validation_data: a tuple of data and labels to train the NN with
        """
        # setting up the layers this master does
        nabla_w = []
        nabla_b = []
        for i in range(self.rank, self.num_layers, self.num_masters):
            nabla_w.append(np.zeros_like(self.weights[i]))
            nabla_b.append(np.zeros_like(self.biases[i]))

        for epoch in range(self.epochs):
            for batch in range(self.number_of_batches):
                
                # 2. Receive gradients from ANY worker
                # We start by listening for the first layer's weight gradient from MPI.ANY_SOURCE.
                # This allows us to identify which worker is ready.
                
                # Identify the first layer index this master manages
                first_layer_idx = self.rank
                
                # Buffer for the first message
                status = MPI.Status()
                
                # Wait for the first weight gradient from any worker
                req = self.comm.Irecv(nabla_w[0], source=MPI.ANY_SOURCE, tag=first_layer_idx)
                req.Wait(status)
                
                # Identify the worker who sent the message
                worker_rank = status.Get_source()
                
                # 3. Receive the rest of the gradients from that SAME worker
                requests = []
                
                # We already got nabla_w[0], now get the corresponding bias
                requests.append(self.comm.Irecv(nabla_b[0], source=worker_rank, tag=first_layer_idx + self.num_layers))
                
                # Receive the rest of the layers this master manages
                # We enumerate starting from 1 because index 0 (nabla_w[0]) is already received
                for i, l in enumerate(range(self.rank + self.num_masters, self.num_layers, self.num_masters), 1):
                    # Tag 'l' for weights, 'l + num_layers' for biases
                    requests.append(self.comm.Irecv(nabla_w[i], source=worker_rank, tag=l))
                    requests.append(self.comm.Irecv(nabla_b[i], source=worker_rank, tag=l + self.num_layers))
                
                # Wait for all gradient parts to arrive
                for r in requests:
                    r.Wait()

                # 4. Perform Gradient Descent Update
                # Note: 'i' is the local index in nabla lists, 'l' is the global layer index
                for i, l in enumerate(range(self.rank, self.num_layers, self.num_masters)):
                    self.weights[l] = self.weights[l] - self.eta * nabla_w[i]
                    self.biases[l] = self.biases[l] - self.eta * nabla_b[i]

                # 5. Send updated parameters back to the SAME worker
                requests = []
                for l in range(self.rank, self.num_layers, self.num_masters):
                    # Send updated weights and biases using non-blocking Isend
                    requests.append(self.comm.Isend(self.weights[l], dest=worker_rank, tag=l))
                    requests.append(self.comm.Isend(self.biases[l], dest=worker_rank, tag=l + self.num_layers))
                
                # Wait for sends to complete
                for r in requests:
                    r.Wait()

            # End of epoch progress print
            self.print_progress(validation_data, epoch)

        