from network import *
import itertools
import sys
import numpy as np
import math
import mpi4py
from time import time

mpi4py.rc(initialize=False, finalize=False)
from mpi4py import MPI

TAG_GRADS  = 10001
TAG_PARAMS = 10002
TAG_GATHER = 10003
TAG_INIT   = 10004


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
        # Divide the batches between all workers
        worker_id = self.rank - self.num_masters

        base = self.number_of_batches // self.num_workers
        rem  = self.number_of_batches % self.num_workers

        my_batches = base + (1 if worker_id < rem else 0)
        start_idx  = worker_id * base + min(worker_id, rem)
        end_idx    = start_idx + my_batches

        for epoch in range(self.epochs):
            # creating batches for epoch
            data = training_data[0]
            labels = training_data[1]
            mini_batches = self.create_batches(data, labels, self.mini_batch_size)
            
            mini_batches = mini_batches[:self.number_of_batches]
            mini_batches = mini_batches[start_idx:end_idx]

            for x, y in mini_batches:
                # do work - don't change this
                self.forward_prop(x)
                nabla_b, nabla_w = self.back_prop(y)

                # send nabla_b, nabla_w to masters 
                for m in range(self.num_masters):
                    layer_ids = list(range(m, self.num_layers, self.num_masters))
                    grads_w = [nabla_w[i] for i in layer_ids]
                    grads_b = [nabla_b[i] for i in layer_ids]
                    payload = (self.rank, layer_ids, grads_w, grads_b)
                    self.comm.send(payload, dest=m, tag=TAG_GRADS)


                # recieve new self.weight and self.biases values from masters
                for m in range(self.num_masters):
                    layer_ids, new_w, new_b = self.comm.recv(source=m, tag=TAG_PARAMS)
                    for i, w_i, b_i in zip(layer_ids, new_w, new_b):
                        self.weights[i] = w_i
                        self.biases[i] = b_i

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

        master_layer_ids = list(range(self.rank, self.num_layers, self.num_masters))

        for epoch in range(self.epochs):
            for batch in range(self.number_of_batches):

                # wait for any worker to finish batch and
                # get the nabla_w, nabla_b for the master's layers
                worker_rank, layer_ids, grads_w, grads_b = self.comm.recv(source=MPI.ANY_SOURCE, tag=TAG_GRADS)

                nabla_w = grads_w
                nabla_b = grads_b

                # calculate new weights and biases (of layers in charge)
                for i, dw, db in zip(range(self.rank, self.num_layers, self.num_masters), nabla_w, nabla_b):
                    self.weights[i] = self.weights[i] - self.eta * dw
                    self.biases[i] = self.biases[i] - self.eta * db

                # send new values (of layers in charge)
                updated_w = [self.weights[i] for i in master_layer_ids]
                updated_b = [self.biases[i] for i in master_layer_ids]
                self.comm.send((master_layer_ids, updated_w, updated_b), dest=worker_rank, tag=TAG_PARAMS)

            self.print_progress(validation_data, epoch)

        # gather relevant weight and biases to process 0
        payload = (master_layer_ids,
                   [self.weights[i] for i in master_layer_ids],
                   [self.biases[i] for i in master_layer_ids])

        if self.rank == 0:
            for m in range(1, self.num_masters):
                layer_ids, w_list, b_list = self.comm.recv(source=m, tag=TAG_GATHER)
                for i, w_i, b_i in zip(layer_ids, w_list, b_list):
                    self.weights[i] = w_i
                    self.biases[i] = b_i
        else:
            self.comm.send(payload, dest=0, tag=TAG_GATHER) 