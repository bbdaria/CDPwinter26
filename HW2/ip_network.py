#
#   @date:  [TODO: Today's date]
#   @author: [TODO: Student Names]
#
# This file is for the solutions of the wet part of HW2 in
# "Concurrent and Distributed Programming for Data processing
# and Machine Learning" course (02360370), Winter 2025
#
import multiprocessing
import os

from my_queue import MyQueue
from network import *
from preprocessor import Worker


class IPNeuralNetwork(NeuralNetwork):
    
    def __init__(self, sizes=None, learning_rate=1.0, mini_batch_size=16, number_of_batches=16,
                 epochs=10, matmul=np.matmul):
        super().__init__(sizes, learning_rate, mini_batch_size, number_of_batches, epochs, matmul)
        self.workers = None
        self.results = None
        self.jobs = None

    def fit(self, training_data, validation_data=None):
        '''
        Override this function to create and destroy workers
        '''
        self.jobs = multiprocessing.JoinableQueue()
        self.results = multiprocessing.SimpleQueue()
        # 1. Create Workers
        try:
            num_workers = int(os.environ['SLURM_CPUS_PER_TASK'])
        except KeyError:
            #Fallback for local launch (if we are not on a server)
            num_workers = multiprocessing.cpu_count()
        # (Call Worker() with self.mini_batch_size as the batch_size)
        if num_workers < 2: #TODO: review instructions
            num_workers = 2
        print(f"Starting {num_workers} workers...")
        self.workers = []
        for _ in range(num_workers):
            w = Worker(
                jobs=self.jobs,
                result=self.results,
                training_data=training_data,
                batch_size=self.mini_batch_size
            )
            w.start()
            self.workers.append(w)
        # 2. Set jobs
        # Call the parent's fit. Notice how create_batches is called inside super.fit().
        super().fit(training_data, validation_data)
        # 3. Stop Workers
        for i in range(num_workers):
            self.jobs.put(None)
        
        self.jobs.join()
        for i in range(num_workers):
            self.workers[i].join()

        
        
    
    def create_batches(self, data, labels, batch_size):
        '''
        Override this function to return self.number_of_batches batches created by workers
        Hint: you can either generate (i.e sample randomly from the training data) the image batches here OR in Worker.run()
        '''
        # 1. Send signal to worker
        for _ in range(self.number_of_batches):
            self.jobs.put(1)

        # 2. Get results
        new_batches = []
        for _ in range(self.number_of_batches):
            new_batches.append(self.results.get())

        return new_batches
    