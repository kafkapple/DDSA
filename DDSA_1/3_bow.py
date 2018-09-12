# -*- coding: utf-8 -*-
"""
Created on Mon May 28 15:54:30 2018

@author: 2014_Joon_IBS
"""

import re

special_chars_remover = re.compile("[^\w'|_]")  #정규식 표현 re 사용. 특수문자 제거 (regular expression)

def main():
    sentence = "Bag-of-Words 모델을 Python으로 직접 구현하겠습니다."
    bow = create_BOW(sentence)

    print(bow)


def create_BOW(sentence):
    bow = {}
    
    sentence_lowered = sentence.lower()
    sentence_without_special_characters = remove_special_characters(sentence_lowered)
    splitted_sentence = sentence_without_special_characters.split()
    
    splitted_sentence_filtered = [  #token 중 한 글자 이상인 것만 
        token
        for token in splitted_sentence
        if len(token) >= 1
        
    ]
    
    
    
    for token in splitted_sentence_filtered:
        bow.setdefault(token, 0)    #token 없으면 0으로 리셋
        bow[token] += 1
        
    return bow


def remove_special_characters(sentence):
    return special_chars_remover.sub(' ', sentence)


if __name__ == "__main__":
    main()
