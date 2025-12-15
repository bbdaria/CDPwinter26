#
#   @date:  [TODO: Today's date]
#   @author: [TODO: Student Names]
#
# This file is for the solutions of the wet part of HW2 in
# "Concurrent and Distributed Programming for Data processing
# and Machine Learning" course (02360370), Winter 2025
#
class MyQueue(object):

    def __init__(self):
        ''' Initialize MyQueue and it's members.
        '''
        raise NotImplementedError("To be implemented")

    def put(self, msg):
        '''Put the given message in queue.

        Parameters
        ----------
        msg : object
            the message to put.
        '''
        raise NotImplementedError("To be implemented")

    def get(self):
        '''Get the next message from queue (FIFO)
            
        Return
        ------
        An object
        '''
        raise NotImplementedError("To be implemented")
    
    def empty(self):
        '''Get whether the queue is currently empty
            
        Return
        ------
        A boolean value
        '''
        raise NotImplementedError("To be implemented")
