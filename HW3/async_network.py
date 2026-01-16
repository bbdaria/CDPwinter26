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
        for epoch in range(self.epochs):
            # creating batches for epoch
            data = training_data[0]
            labels = training_data[1]
            mini_batches = self.create_batches(data, labels, self.mini_batch_size)
            for x, y in mini_batches:
                # do work - don't change this
                self.forward_prop(x)
                nabla_b, nabla_w = self.back_prop(y)

                # send nabla_b, nabla_w to masters
                requests = []

                # iterate over each master to send them their specific layers
                for m_rank in range(self.num_masters):
                    # Identify which layers belong to master 'm_rank'
                    layers_indices = range(m_rank, self.num_layers, self.num_masters)

                    # Pack the gradients for these specific layers
                    m_nabla_w = [nabla_w[i] for i in layers_indices]
                    m_nabla_b = [nabla_b[i] for i in layers_indices]

                    # Asynchronously send to the master (Tag 1 for gradients)
                    req = self.comm.isend((m_nabla_w, m_nabla_b), dest=m_rank, tag=1)
                    requests.append(req)

                # recieve new self.weight and self.biases values from masters
                for m_rank in range(self.num_masters):
                    # Asynchronously receive updated weights (Tag 2 for weights)
                    req = self.comm.irecv(source=m_rank, tag=2)
                    new_w, new_b = req.wait()  # Wait to ensure we have data before next batch

                    # Update local parameters
                    layers_indices = range(m_rank, self.num_layers, self.num_masters)
                    for i, w, b in zip(layers_indices, new_w, new_b):
                        self.weights[i] = w
                        self.biases[i] = b

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

                # wait for any worker to finish batch and
                # get the nabla_w, nabla_b for the master's layers
                # TODO: add your code
                status = MPI.Status()
                #  wait for any worker to send us gradients (Tag 1)
                req = self.comm.irecv(source=MPI.ANY_SOURCE, tag=1)
                data = req.wait(status=status)  # Wait for the data

                received_nabla_w, received_nabla_b = data
                worker_rank = status.Get_source()  # Identify who sent this

                # Update the local temporary variables
                nabla_w = received_nabla_w
                nabla_b = received_nabla_b

                # calculate new weights and biases (of layers in charge)
                for i, dw, db in zip(range(self.rank, self.num_layers, self.num_masters), nabla_w, nabla_b):
                    self.weights[i] = self.weights[i] - self.eta * dw
                    self.biases[i] = self.biases[i] - self.eta * db

                # send new values (of layers in charge)
                # prepare the list of current weights/biases to send back
                layers_indices = range(self.rank, self.num_layers, self.num_masters)
                current_w = [self.weights[i] for i in layers_indices]
                current_b = [self.biases[i] for i in layers_indices]

                # Send back to the worker who asked (Tag 2)
                self.comm.isend((current_w, current_b), dest=worker_rank, tag=2)

            self.print_progress(validation_data, epoch)

        # gather relevant weight and biases to process 0
        # TODO: add your code
        my_layers_data = {
            'indices': list(range(self.rank, self.num_layers, self.num_masters)),
            'weights': [self.weights[i] for i in range(self.rank, self.num_layers, self.num_masters)],
            'biases': [self.biases[i] for i in range(self.rank, self.num_layers, self.num_masters)]
        }

        gathered_data = self.comm.gather(my_layers_data, root=0)

        if self.rank == 0:
            # Reconstruct the full model from the gathered pieces
            # Only Ranks 0 to num_masters-1 have real data. Workers (if any returned) are ignored.

            for m_data in gathered_data:
                if m_data is None: continue  # Skip workers if they participated in gather

                indices = m_data['indices']
                w_list = m_data['weights']
                b_list = m_data['biases']

                for idx, w, b in zip(indices, w_list, b_list):
                    self.weights[idx] = w
                    self.biases[idx] = b