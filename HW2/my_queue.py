#
#   @date:  [TODO: 5.1.2026]
#   @author: [TODO: Daria Bebin, Kseniia Filonenko]
#
# This file is for the solutions of the wet part of HW2 in
# "Concurrent and Distributed Programming for Data processing
# and Machine Learning" course (02360370), Winter 2025
#
from multiprocessing import Pipe 
from multiprocessing import Lock

    
class MyQueue(object):

    def __init__(self):
        self.readers, self.writers = Pipe(False)
        self.lock = Lock()
        self.count = 0
        

    def put(self, msg):
        '''Put the given message in queue.

        Parameters
        ----------
        msg : object
            the message to put.
        '''
        with self.lock:
            self.writers.send(msg)
            self.count += 1
            


    def get(self):
        '''Get the next message from queue (FIFO)
            
        Return
        ------
        An object
        '''
        self.count -= 1
        return self.readers.recv()
            
    
    def empty(self):
        '''Get whether the queue is currently empty
            
        Return
        ------
        A boolean value
        '''
    
        return self.count == 0
