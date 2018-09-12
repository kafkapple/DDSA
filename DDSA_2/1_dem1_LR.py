# -*- coding: utf-8 -*-
"""
Created on Fri Jun  8 21:18:24 2018

@author: 2014_Joon_IBS
"""

from scipy.stats import linregress
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import elice_utils

def main():
    '''
        read_data의 파라미터는 다음 중 하나에서 선택할 수 있습니다.
        * sleep_time
        * study_time
        * stress_level
        * exercise_activity
        * social_activity
    '''
    x_col = 'social_activity' # 여기에 있는 파라미터를 바꾸면서 실행해보세요
    
    (X, Y) = read_data(x_col)
    (slope, intercept) = do_linear_regression(X, Y)  # linear regression을 하는 함수
    draw_chart(X, Y, slope, intercept, x_col) # 그래프를 그리는 함수

def read_data(col_name):
    student_data = []
    
    
    with open('students.txt') as fp:
        columns = fp.readline().strip().split(' ')
        for line in fp:
            line_test = line.split()
            print(line_test)
            line_data = line.strip().split(' ')
            line_data_numeric = [float(x) for x in line_data]
            student_data.append(line_data_numeric)
    
    student_data = np.array(student_data)
    col_index = columns.index(col_name)
    print(col_index)
    # X must be numpy.array in (30 * 5) shape.
    # Y must be 1-dimensional numpy.array.
    X = student_data[:,col_index]
    Y = student_data[:,-1]
    return (X, Y)

def do_linear_regression(X, Y):
    slope, intercept, r_value, p_value, std_err = linregress(X, Y)
    return (slope, intercept)
    
def draw_chart(X, Y, slope, intercept, x_col):
    fig = plt.figure()
    fig.suptitle('Linear Regression for Class Data')
    ax = fig.add_subplot(111)
    ax.set_xlabel(x_col)
    ax.set_ylabel('GPA')
    
    plt.scatter(X, Y)
    
    min_X = min(X)
    max_X = max(X)
    min_Y = min_X * slope + intercept
    max_Y = max_X * slope + intercept
    plt.plot([min_X, max_X], [min_Y, max_Y], 
             color='red',
             linestyle='--',
             linewidth=3.0)
    
    ax.text(min_X, min_Y + 0.1, r'$y = %.2lfx + %.2lf$' % (slope, intercept), fontsize=15)
    
    plt.savefig('chart.svg')
    elice_utils.send_image('chart.svg')

if __name__ == "__main__":
    main()
