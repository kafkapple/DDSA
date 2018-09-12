# -*- coding: utf-8 -*-
"""
Created on Thu Jun  7 12:04:31 2018

@author: 2014_Joon_IBS
"""

from scipy.stats import linregress
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import math
import operator

def main():    
    # 여기에 내용을 채우세요.
    words = read_data()
    words = sorted(words, key = lambda w:w[1], reverse = True )
    
    
    X = list(range(1,len(words)+1))
    Y = [i[1] for i in words]
    
    #HW5 zipf law 적용시, log 
    #log_x = [math.log(x_i) for x_i in X]
    #log_y = [math.log(y_i) for y_i in Y]
    
    slope, intercept = do_linear_regression( X, Y  )
    return slope, intercept

def read_data():
    # 여기에 내용을 채우세요.
    words = []
    with open('words.txt') as w:
        for i in w:
            line = i.strip().split(',')
            line_x = [line[0], int(line[1])]
            words.append(line_x)
        
    return words

def draw_chart(X, Y, slope, intercept):
    fig = plt.figure()
    
    # 여기에 내용을 채우세요.
    
    plt.savefig('chart.png')
    

def do_linear_regression(X, Y):
    # 여기에 내용을 채우세요.
    slope, intercept, r_value, p_value, std_err = linregress(X,Y)
    return slope, intercept

if __name__ == "__main__":
    main()
