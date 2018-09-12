# -*- coding: utf-8 -*-
"""
Created on Mon May 28 15:36:17 2018

@author: 2014_Joon_IBS
"""

def main():
    sensitivity = float(input())
    prior_prob = float(input()) #전체 인구 중 유방함 확률
    false_alarm = float(input())    #병 없는데 진단될 경우

    print("%.2lf%%" % (100 * mammogram_test(sensitivity, prior_prob, false_alarm)))

def mammogram_test(sensitivity, prior_prob, false_alarm):
    p_a1_b1 = sensitivity # p(A = 1 | B = 1)

    p_b1 = prior_prob    # p(B = 1)

    p_b0 = 1 - p_b1    # p(B = 0)

    p_a1_b0 = false_alarm # p(A = 1|B = 0)

    p_a1 = p_b0*p_a1_b0 + p_b1*p_a1_b1    # p(A = 1)

    p_b1_a1 = p_a1_b1*p_b1/p_a1 # p(B = 1|A = 1)

    return p_b1_a1

if __name__ == "__main__":
    main()