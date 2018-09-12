# -*- coding: utf-8 -*-
"""
Created on Tue Jun 12 22:24:15 2018

@author: 2014_Joon_IBS
"""

import re
import math
import naivebayes_utils

def main():
    training1_sentence = input()
    training2_sentence = input()
    testing_sentence = input()

    alpha = float(input())
    prob1 = float(input())
    prob2 = float(input())

    print(naive_bayes(training1_sentence, training2_sentence, testing_sentence, alpha, prob1, prob2))

def naive_bayes(training1_sentence, training2_sentence, testing_sentence, alpha, prob1, prob2):
    # Implement Naive Bayes Algorithm here...
    # Return normalized log probability 
    # of p(training1_sentence|testing_sentence) and p(training2_sentence|testing_sentence)
    
    bow_train1 = create_BOW(training1_sentence)
    bow_train2 = create_BOW(training2_sentence)
    bow_test = create_BOW(testing_sentence)
    
    d_words1 = len(bow_train1)  #단어의 종류 수
    d_words2 = len(bow_train2)
    
    
    n_words1 = sum(bow_train1.values()) #총 단어의 count 수
    n_words2 = sum(bow_train2.values())
    
    for word in bow_test:
        theta1 = bow_train1.get(word)  # 1에 해당 단어 있을 경우, 빈도수
        theta2 = bow_train2.get(word)
        if theta1 is None:
            theta1 = 0
        if theta2 is None:
            theta2 = 0
        prob1 += math.log(((theta1 + alpha) / (n_words1 + d_words1 * alpha))** bow_test.get(word))
        prob2 += math.log(((theta2 + alpha) / (n_words2 + d_words2 * alpha))** bow_test.get(word))
        
    classify1 = prob1
    classify2 = prob2

    return normalize_log_prob(classify1, classify2)

def normalize_log_prob(prob1, prob2):
    return naivebayes_utils.normalize_log_prob(prob1, prob2)

def log_likelihood(training_model, testing_model, alpha):
    return naivebayes_utils.calculate_doc_prob(training_model, testing_model, alpha)

def create_BOW(sentence):
    # bag-of-words
    # 문장에 들어있는 각 단어를 key로, 해당 단어가 문장에 나타나는 빈도수를 value로 하는 dictionary를 반환합니다.
    # 예: {'elice':3, 'data':1, ...}
    return naivebayes_utils.create_BOW(sentence)

if __name__ == "__main__":
    main()
