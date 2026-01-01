#
#   @date:  [TODO: Today's date]
#   @author: [TODO: Student Names]
#
# This file is for the solutions of the wet part of HW2 in
# "Concurrent and Distributed Programming for Data processing
# and Machine Learning" course (02360370), Winter 2025
#
import multiprocessing
import my_queue
from scipy import ndimage
from scipy.stats import skew
import numpy as np


class Worker(multiprocessing.Process):
    
    def __init__(self, jobs, result, training_data, batch_size):
        super().__init__()

        ''' Initialize Worker and it's members.

        Parameters
        ----------
        jobs: JoinableQueue
            A jobs Queue for the worker.
        result: Queue
            A results Queue for the worker to put it's results in.
		training_data: 
			A tuple of (training images array, image lables array)
		batch_size:
			workers batch size of images (mini batch size)
        
        You should add parameters if you think you need to.
        '''

        self.jobs = jobs
        self.result = result
        self.training_data = training_data
        self.batch_size = batch_size

    @staticmethod
    def _to_2d(image):
        # Accept (784,) or (28,28)
        if image.ndim == 1:
            return image.reshape(28, 28)
        return image

    @staticmethod
    def _to_1d(image, original_ndim):
        if original_ndim == 1:
            return image.reshape(-1)
        return image

    @staticmethod
    def rotate(image, angle):
        '''Rotate given image to the given angle

        Parameters
        ----------
        image : numpy array
            An array of size 784 of pixels
        angle : int
            The angle to rotate the image
            
        Return
        ------
        An numpy array of same shape
        '''
        img2 = Worker._to_2d(image)
        rotated_image = ndimage.rotate(img2, angle, reshape=False, mode="constant", cval=0) 
        return Worker._to_1d(rotated_image, image.ndim)

    @staticmethod
    def shift(image, dx, dy):
        '''Shift given image

        Parameters
        ----------
        image : numpy array
            An array of shape 784 of pixels
        dx : int
            The number of pixels to move in the x-axis
        dy : int
            The number of pixels to move in the y-axis
            
        Return
        ------
        An numpy array of same shape
        '''
        img2 = Worker._to_2d(image)
        shift_amount = (-dx, -dy)
        shifted_image = ndimage.shift(img2, shift_amount, mode='constant', cval=0)
        return shifted_image
    
    @staticmethod
    def add_noise(image, noise):
        '''Add noise to the image
        for each pixel a value is selected uniformly from the 
        range [-noise, noise] and added to it. 

        Parameters
        ----------
        image : numpy array
            An array of shape 784 of pixels
        noise : float
            The maximum amount of noise that can be added to a pixel

        Return
        ------
        An numpy array of same shape
        '''
        new_image = np.shape(image)
        rows,cols = image.shape
        for i in range(0, rows - 1):
            for j in range(0, cols - 1):
                new_image[i][j] = image[i][j] + np.random.uniform(low = -1 * noise, high =noise)
        return new_image


    @staticmethod
    def skew(image, tilt):
        '''Skew the image

        Parameters
        ----------
        image : numpy array
            An array of size 784 of pixels
        tilt : float
            The skew paramater

        Return
        ------
        An numpy array of same shape
        '''
        img2 = Worker._to_2d(image)
        new_image = np.zeros(img2.shape)
        rows, cols = img2.shape
        for i in range(0, rows - 1):
            for j in range(0, cols - 1):
                x = j + i*tilt
                if x >= 0 or x <= cols:
                    new_image[i][j] = img2[i][x]
        return Worker._to_1d(new_image, image.ndim)
        

    def process_image(self, image):
        '''Apply the image process functions
		Experiment with the random bounds for the functions to see which produces good accuracies.

        Parameters
        ----------
        image: numpy array
            An array of size 784 of pixels

        Return
        ------
        An numpy array of same shape
        '''
        angle = random.randint(-15, 15)
        dx = random.randint(-2, 2)
        dy = random.randint(-2, 2)
        noise = random.uniform(0, 0.15)
        skew_val = random.uniform(-0.15, 0.15)

        new_image = image.copy()
        new_image = rotate(new_image, angle)
        new_image = shift(new_image, dx, dy)
        new_image = add_noise(new_image, noise)
        new_image = skew(new_image, skew_val)

        return new_image



    def run(self):
        '''Process images from the jobs queue and add the result to the result queue.
		Hint: you can either generate (i.e sample randomly from the training data)
		the image batches here OR in ip_network.create_batches
        '''
        while True:
            job = self.jobs.get()
            if job is None:
                self.jobs.task_done()
                return
            
            training_images, training_labels = self.training_data[0], self.training_data[1]

            indexes = random.sample(range(0, training_images.shape[0]), self.batch_size)
            images, labels = (training_images[indexes], training_labels[indexes])

            batch_images = np.empty(shape=(self.batch_size * 2, images.shape[1]))
            batch_labels = np.empty(shape=(self.batch_size * 2, labels.shape[1]))

            for i in range(0, self.batch_size):
                batch_images[i*2]  = self.process_image(images[i])
                batch_images[i*2 + 1] = images[i]
                batch_labels[i*2] = labels[i]
                batch_labels[i*2 + 1] = labels[i]
                
            indexes = random.sample(range(0, self.batch_size * 2), self.batch_size)
            self.result.put((batch_images[indexes], batch_labels[indexes]))
            self.jobs.task_done()
