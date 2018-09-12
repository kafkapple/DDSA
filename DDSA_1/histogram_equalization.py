# -*- coding: utf-8 -*-
"""
Created on Thu May 31 20:35:36 2018

@author: 2014_Joon_IBS
"""

import numpy as np

def image_histogram_equalization(image, number_bins=256):
    # from http://www.janeriksolem.net/2009/06/histogram-equalization-with-python-and.html

    # get image histogram
    image_histogram, bins = np.histogram(image.flatten(), number_bins, normed=True)
    cdf = image_histogram.cumsum() # cumulative distribution function
    cdf = 255 * cdf / cdf[-1] # normalize

    # use linear interpolation of cdf to find new pixel values
    image_equalized = np.interp(image.flatten(), bins[:-1], cdf)

    return image_equalized.reshape(image.shape), cdf

if __name__ == '__main__':

    # generate some test data with shape 1000, 1, 96, 96
    data = np.random.rand(1000, 1, 96, 96)

    # loop over them
    data_equalized = np.zeros(data.shape)
    for i in range(data.shape[0]):
        image = data[i, 0, :, :]
        data_equalized[i, 0, :, :] = image_histogram_equalization(image)[0]
        
    print('hi')
    