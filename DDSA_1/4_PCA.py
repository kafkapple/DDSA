# -*- coding: utf-8 -*-
"""
Created on Wed May 30 13:39:46 2018

@author: 2014_Joon_IBS
"""

import sklearn.decomposition
import matplotlib.pyplot as plt
import numpy as np
import elice_utils

import csv

def main():
    X, attributes = input_data()
    pca_array = normalize(X)
    pca, pca_array = run_PCA(X, 2)
    visualize_2d_wine(pca_array)

def input_data():
    X = []
    attributes = []
    
    with open("data/attributes.txt") as fp:
        attributes = fp.readlines()
    
    attributes = [x.strip() for x in attributes]
    
    csvreader = csv.reader(open("data/wine.csv"))
    
    for line in csvreader:
        float_numbers = [float(i) for i in line]
        X.append(float_numbers)
    
    
    return np.array(X), attributes

def run_PCA(X, num_components):
    pca = sklearn.decomposition.PCA(n_components = num_components)
    pca.fit(X)
    pca_array = pca.transform(X)
    

    return pca, pca_array
    
def normalize(X):
    '''
    각각의 feature에 대해,
    178개의 데이터에 나타나는 해당하는 feature의 값이 최소 0, 최대 1이 되도록
    선형적으로 데이터를 이동시킵니다.
    '''
    for i in range(X.shape[1]):
        X[:,i] = (X[:,i] - np.min(X[:,i])) / np.max(X[:,i])
        
    return X

def visualize_2d_wine(X):
    '''X를 시각화하는 코드를 구현합니다.'''
    plt.scatter(X[:,0],X[:,1])
    plt.savefig("image.png")
    elice_utils.send_image("image.png")

if __name__ == '__main__':
    main()