# -*- coding: utf-8 -*-
"""
Created on Mon Jun 18 21:51:51 2018

@author: 2014_Joon_IBS
"""

from sklearn import datasets
from sklearn.linear_model import LogisticRegression
import numpy as np


def load_data():
	return datasets.load_iris(return_X_y=True)


def binary_classification(X, binary_label):
	# input : X (features), shape:(number of samples, number of features)
	# input : binary_label (1 or 0), shape:(number of samples, )
	# output : probability to belong to that class, shape:(number of samples, )
    regr = LogisticRegression()
    regr.fit(X, binary_label)
    predicted_p = regr.predict_proba(X)
    predicted_p = predicted_p[:,1]
    return 


def multiclass_classification(X, y):
	# input : X (features), shape:(number of samples, number of features)
	# input : y (labels), shape:(number of samples,)
	# output : multiclass classification accuracy, shape:(1, )
   class_list = list(set(y))  #unique 값 만 남김. 이를 통해 class 종류 확인
     #class 마다 predction 값 저장할 공간 생성
   
   y_temp = np.zeros(len(y))    #class 마다 binary label 담는 y vector 생성
   pred_class = np.array([])
   for i in class_list:
       idx_i = np.where(y==i)
       idx_i_not = np.where(y!=i)
       y_temp[idx_i]=1
       y_temp[idx_i_not]=0       
       np.concatenate(pred_class,binary_classification(X,y_temp))

   final_class = np.where(np.max(pred_class))
   regr = LogisticRegression()
   regr.fit(X,y)
   
   accuracy = regr.score(X,final_class)
   
   return accuracy


def main():
	data = load_data()
	multiclass_classification(data[0], data[1])
	return



if __name__ == '__main__':
	main()