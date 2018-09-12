# -*- coding: utf-8 -*-
"""
Created on Tue Jun 12 22:24:31 2018

@author: 2014_Joon_IBS
"""

import re
import math

def calculate_doc_prob(training_model, testing_model, alpha):
    # Implement likelihood function here...
    bow_train = training_model
    bow_test = testing_model

    total_tokens_train = sum([bow_train[x] for x in bow_train])

    logprob = 0
    for word in bow_test:
        for i in range(0, bow_test[word]): # word count
            if word in bow_train:
            	logprob += math.log(bow_train[word] + alpha)
            else:
                logprob += math.log(alpha)
            logprob -= math.log(total_tokens_train + len(bow_train) * alpha)

    return logprob

def create_BOW(sentence):
    # Exercise
    bow = {}

    lowered = sentence.lower()
    alphabet_only = replace_non_alphabetic_chars_to_space(lowered)
    split_lowered = alphabet_only.lower().split()

    for token in split_lowered:
        if len(token) < 1:
            continue
        if token in bow:
            bow[token] += 1
        else:
            bow[token] = 1

    return bow

def replace_non_alphabetic_chars_to_space(sentence):
    return re.sub(r'[^a-z]+', ' ', sentence)

def normalize_log_prob(prob1, prob2):
    maxprob = max(prob1, prob2)

    prob1 -= maxprob
    prob2 -= maxprob
    prob1 = math.exp(prob1)
    prob2 = math.exp(prob2)

    normalize_constant = 1.0 / float(prob1 + prob2)
    prob1 *= normalize_constant
    prob2 *= normalize_constant

    return (prob1, prob2)

def visualize_boxplot(title, values, labels):
    width = .35

    fig, ax = plt.subplots()
    ind = numpy.arange(len(values))
    rects = ax.bar(ind, values, width)
    ax.set_title(title)
    ax.bar(ind, values, width=width)
    ax.set_xticks(ind + width/2)
    ax.set_xticklabels(labels)

    def autolabel(rects):
        # attach some text labels
        for rect in rects:
            height = rect.get_height()
            ax.text(rect.get_x()+rect.get_width()/2., height + 0.01, '%.2lf%%' % (height * 100), ha='center', va='bottom')

    autolabel(rects)

    plt.savefig("image.svg", format="svg")
    elice_utils.send_image("image.svg")
