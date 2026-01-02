#
#   @date:  [TODO: Today's date]
#   @author: [TODO: Student Names]
#
# This file is for the solutions of the wet part of HW2 in
# "Concurrent and Distributed Programming for Data processing
# and Machine Learning" course (02360370), Winter 2025
#
from numba import cuda
from numba import njit
import imageio
import matplotlib.pyplot as plt
import numpy as np

def correlation_gpu(kernel, image):
    '''Correlate using gpu
    Parameters
    ----------
    kernel : numpy array
        A small matrix
    image : numpy array
        A larger matrix of the image pixels
            
    Return
    ------
    An numpy array of same shape as image
    '''
    raise NotImplementedError("To be implemented")


@njit
def correlation_numba(kernel, image):
    '''Correlate using numba
    Parameters
    ----------
    kernel : numpy array
        A small matrix
    image : numpy array
        A larger matrix of the image pixels
            
    Return
    ------
    An numpy array of same shape as image
    '''
    # Image and kernel dimensions
    image_height, image_width = image.shape
    kernel_height, kernel_width = kernel.shape

    # Find center of kernel (correlation)
    pad_h = kernel_height // 2
    pad_w = kernel_width // 2

    result = np.zeros_like(image)

    #iterate through pixels of an image
    for i in range(image_height):
        for j in range(image_width):

            pixel_sum = 0.0

            # Go over all elements of kernel
            for k in range(kernel_height):
                for l in range(kernel_width):

                    # Coordinates of a neighbour
                    # adjust them, so they on the center of a kernel
                    ni = i - pad_h + k
                    nj = j - pad_w + l

                    # Check borders
                    if 0 <= ni < image_height and 0 <= nj < image_width:
                        pixel_sum += image[ni, nj] * kernel[k, l]

            result[i, j] = pixel_sum
    return result

def sobel_operator():
    '''Load the image and perform the operator
        ----------
        Return
        ------
        An numpy array of the image
        '''
    pic = load_image()
    # your calculations

    sobel_matrix = [[1,0,-1],[2,0,-2],[1,0,-1]]
    sobel_filter = np.array(sobel_matrix)

    Gx = correlation_numba(sobel_filter, pic)
    Gy = correlation_numba(np.transpose(sobel_filter), pic)

    rows, cols = Gx.shape
    result = [[np.sqrt(np.pow(Gx[i][j],2)+np.pow(Gy[i][j],2)) for i in range(rows)] for j in range(cols)]

    return result


def load_image(): 
    fname = 'data/image.jpg'
    pic = imageio.imread(fname)
    to_gray = lambda rgb : np.dot(rgb[... , :3] , [0.299 , 0.587, 0.114])
    gray_pic = to_gray(pic)
    return gray_pic


def show_image(image):
    """ Plot an image with matplotlib

    Parameters
    ----------
    image: list
        2d list of pixels
    """
    plt.imshow(image, cmap='gray')
    plt.show()
