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
    
    def fit(self, training_data, validation_data=None):
        '''
        Override this function to create and destroy workers
        '''
        self.jobs = multiprocessing.JoinableQueue()
        self.results = MyQueue()
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
        for w in self.workers:
            w.terminate()
            w.join()  # wait to finish

        
        
    
    def create_batches(self, data, labels, batch_size):
        '''
        Override this function to return self.number_of_batches batches created by workers
        Hint: you can either generate (i.e sample randomly from the training data) the image batches here OR in Worker.run()
        '''
        mini_batches = []

        # Loop for the number of batches required per epoch
        for i in range(self.number_of_batches):

            # 1. Sample random indices from the data
            # We select 'batch_size' random indices from the available data
            #TODO: use option 2 - Worker.run()
            keys = np.random.choice(len(data), batch_size,
                                    replace=False)  # ensures unique images in a batch
            # Extract the corresponding labels
            batch_labels = labels[keys]

            # 2. Send jobs (images) to the Workers
            for key in keys:
                self.jobs.put(data[key])

            # 3. Collect results from the Workers
            batch_data = []
            for _ in range(batch_size):
                # Wait for a result from the results queue
                augmented_image = self.results.get()
                batch_data.append(augmented_image)

            batch_data = np.array(batch_data)
            mini_batches.append((batch_data, batch_labels))

        return mini_batches