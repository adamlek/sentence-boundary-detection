#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  5 13:03:26 2017

@author: adam

Features to consider:
    next/prev_type: int, str, char
    OnTheFly: POS-tagging
    
"""

import math

end = ['.','?','!']
start_s = ['(', '[', '{', ')', ']', '}']
#1 - upper
#2 - lower
#3 - number
#0 - other

def construct_featureset(left, right, word, next_p, prev_p, in_p, config=False):
    if config:
        pass
        
    ### right context
    if right:
        if right[0].isalpha():
            if right[0].isnumeric():
                next_u = 1 # number
            else:
                if right[0].isupper():
                    next_u = 1 # Uppercase
                else:
                    next_u = 0 # lowercase
        else:
            next_u = 0 # .! ? ! ) ) ) is going on!
        next_w = right.lower()
        next_l = len(right)
    else:
        next_u = 0
        next_w = 'NULL'
        next_l = 0

    ### left context
    if left:
        if left[0].isalpha():
            if left[0].isnumeric():
                prev_u = 0
            else:
                if left[0].isupper():
                    prev_u = 0
                else:
                    prev_u = 1
        else:
            prev_u = 0
        prev_w = left.lower()
        prev_l = len(left)
    else:
        prev_u = 0
        prev_w = 'NULL'
        prev_l = 0
    
    next_test = 0
    prev_test = 0
    
    ###### log(punctuation_dist)
    if next_p != 0:
        next_p = next_p
    
    if prev_p != 0:
        prev_p = prev_p
        
    next_test = math.fabs(next_p-next_l)
    
    if in_p:
        in_p = 1
    else:
        in_p = 0

    return {
            'next_u': next_u, 
            'prev_u': prev_u, 
            'next_l': next_l, 
#            'prev_l': prev_l,
            'next_w': next_w, 
            'prev_w': prev_w,
            'prev_p': next_p,
            'next_p': prev_p,
#            'next_test': next_test,
#            'prev_test': prev_test,
#            'in_p': in_p,
            'punct': end.index(word)
            }
