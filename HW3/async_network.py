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
        # Divide the batches between all workers
        worker_num_of_batches = self.number_of_batches // self.num_workers
        if self.rank == self.num_workers - 1:
            self.number_of_batches -= (self.num_workers - 1) * worker_num_of_batches
        else:
            self.number_of_batches = worker_num_of_batches

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
                for l in range(self.num_layers):
                    requests.append(self.comm.Isend(nabla_w[l], dest=(l%self.num_masters), tag=l))
                    requests.append(self.comm.Isend(nabla_b[l], dest=(l%self.num_masters), tag=l+self.num_layers)) 

                for r in requests:
                    r.Wait()

                # recieve new self.weight and self.biases values from masters
                # TODO: add your code
                requests = []
                for l in range(self.num_layers):
                    requests.append(self.comm.Irecv(self.weights[l], src=(l%self.num_masters), tag=l))
                    requests.append(self.comm.Irecv(self.biases[l], src=(l%self.num_masters), tag=l+self.num_layers)) 
                    
                # waiting on all the send/recv calls before continueing to next epoch
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
                # wait for any worker to finish batch and
                # get the nabla_w, nabla_b for the master's layers
                
                # --- START OF FIXES ---
                status = MPI.Status() # Create a Status object to hold message info
                
                # Use MPI.ANY_SOURCE instead of ANY_SRC
                first_req = self.comm.Irecv(nabla_w[0], src=MPI.ANY_SOURCE, tag=self.rank)
                
                # Pass status to Wait to populate it
                first_req.Wait(status)
                
                # Get the source rank from the status object
                source = status.Get_source()
                
                # Logic fix: Tag for bias must match worker's send (rank + num_layers)
                requests = [self.comm.Irecv(nabla_b[0], src=source, tag=self.rank + self.num_layers)]
                # --- END OF FIXES ---

                i = 1
                for l in range(self.rank + self.num_masters, self.num_layers, self.num_masters):
                    requests.append(self.comm.Irecv(nabla_w[i], src=source, tag=l))
                    requests.append(self.comm.Irecv(nabla_b[i], src=source, tag=l+self.num_layers))
                    i += 1
                
                for r in requests:
                    r.Wait()

                # calculate new weights and biases (of layers in charge)
                for i, dw, db in zip(range(self.rank, self.num_layers, self.num_masters), nabla_w, nabla_b):
                    self.weights[i] = self.weights[i] - self.eta * dw
                    self.biases[i] = self.biases[i] - self.eta * db

                # send new values (of layers in charge)
                requests = []
                for l in range(self.rank, self.num_layers, self.num_masters):
                    requests.append(self.comm.Isend(self.weights[l], dest=source, tag=l))
                    requests.append(self.comm.Isend(self.biases[l], dest=source, tag=l+self.num_layers))
                
                for r in requests:
                    r.Wait()

            self.print_progress(validation_data, epoch)

        # gather relevant weight and biases to process 0
        requests = []
        if self.rank == 0:
            for l in range(self.num_layers):
                if l%self.num_masters != 0:
                    requests.append(self.comm.Irecv(self.weights[l], src=(l%self.num_masters), tag=l))
                    requests.append(self.comm.Irecv(self.biases[l], src=(l%self.num_masters), tag=l+self.num_layers)) 
        else:
            for l in range(self.rank, self.num_layers, self.num_masters):
                requests.append(self.comm.Isend(self.weights[l], dest=0, tag=l))
                requests.append(self.comm.Isend(self.biases[l], dest=0, tag=l+self.num_layers))
        
        # waiting on all the send/recv calls before continueing to next epoch
        for r in requests:
            r.Wait()