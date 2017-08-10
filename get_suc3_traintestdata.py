#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 15:42:01 2017

@author: adam

Fixa träningsdata från suc-train.conll
"""
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import chi2

from scipy import stats

from os import listdir as listdir
from collections import defaultdict
import numpy as np
import re
import math

from create_fs import construct_featureset


quot = ['"',"'"]
start_s = ['(', '[', '{', ')', ']', '}']
end = ['.','!','?']

'}'

def get_train(fts = False):
    end = ['.','!','?']
    o_alltext = [[]]
    feat_list = []
    source = '/home/adam/data/SUC3.0/corpus/conll/'
#    source = '/home/adam/data/SUC3.0/corpus/conll/'
#    output = '/home/adam/data/SUC3.0/corpus/rawtext/'

    for dirr in listdir(source):
        if not dirr == 'suc-train.conll':
            continue
    
        with open(source + dirr) as f1:
            text = f1.read().split('\n')
            
        for i, ln in enumerate(text):
            ln = ln.rstrip().split()
            if ln != []:
                if any(x for x in end if x in ln[1] and ln[1] != x):
                    for sy in end:
                        if sy in ln[1]:
                            wordn = ln[1].replace('{0}'.format(sy), ' {0} '.format(sy))                            
                    
                    for w in wordn.split():
                        o_alltext[-1].append(w)
                else:
                    o_alltext[-1].append(ln[1])
                    # <--- END
            else:
                o_alltext.append([])
                # <--- END
             
    alltext = [x for x in o_alltext if x]
    index_se_tokens = [i for (i, x) in enumerate([y for x in o_alltext for y in x]) if x in end]
#    print(index_se_tokens)
    
    in_parenthesis = False
    es_track = 0
    for i, item in enumerate(alltext):
        # find all SE-tokens
        for n, word in enumerate(item):
            if '(' in word:
                in_parenthesis = True
                
            if word in end:
                if es_track == 0:
                    r_nextp = int(math.fabs(index_se_tokens[es_track] - index_se_tokens[es_track+1]))
                    l_nextp = 0
                elif es_track+1 == len(index_se_tokens):
                    l_nextp = int(math.fabs(index_se_tokens[es_track] - index_se_tokens[es_track-1]))
                    r_nextp = 0
                else:
                    r_nextp = int(math.fabs(index_se_tokens[es_track] - index_se_tokens[es_track+1]))
                    l_nextp = int(math.fabs(index_se_tokens[es_track] - index_se_tokens[es_track-1]))
                    # <--- END
                    
                sb = 1
                if n+1 == len(item):
                    if i+1 == len(alltext):
                        right = ''
                    else:
                        right = alltext[i+1][0]
                        # <--- END
                    left = item[n-1]
                    punct = word
                    sb = True
                elif n == 0:
                    # current_i - nextp_i = distance from current_i to nextp_i
                    right = item[n+1]
                    left = ''
                    punct = word
                    
                    sb = False
                    
                else:
                    if n+2 == len(item) and item[-1] in quot+end: 
                        right = alltext[i+1][0]
                        sb = True
                    else:
                        right = item[n+1]
                        sb = False
                        # <--- END
                        
                    left = item[n-1]
                    punct = word
                    # <--- END

                es_track += 1
                feat_list.append((construct_featureset(left, right, punct, 
                                                       r_nextp, l_nextp, in_parenthesis), sb))
                
                if ')' in word:
                    in_parenthesis = False
                # <--- END
    return feat_list

def get_test(fts = False):
    ### BLOG or DEV?
    files = ['suc-dev.conll']
#    files = ['blogs.conll']
    end = ['.','!','?']
    o_alltext = [[]]
    feat_list = []
    source = '/home/adam/data/SUC3.0/corpus/conll/'
#    output = '/home/adam/data/SUC3.0/corpus/rawtext/'

    for dirr in listdir(source):
        if not dirr in files:
            continue
    
        with open(source + dirr) as f1:
            text = f1.read().split('\n')
            
        for i, ln in enumerate(text):
            ln = ln.rstrip().split()
            if ln != []:
                if any(x for x in end if x in ln[1] and ln[1] != x):
                    for sy in end:
                        if sy in ln[1]:
                            wordn = ln[1].replace('{0}'.format(sy), ' {0} '.format(sy))                            
                    
                    for w in wordn.split():
                        o_alltext[-1].append(w)
                else:
                    o_alltext[-1].append(ln[1])
                    # <--- END
            else:
                o_alltext.append([])
                # <--- END
    
    index_se_tokens = [i for (i, x) in enumerate([y for x in o_alltext for y in x]) if x in end]

    corr = []
    es_track = 0
    in_parenthesis = False

    for i, item in enumerate(o_alltext):
        if not item:
            continue

        for n, word in enumerate(item):
            if '(' in word:
                in_parenthesis = True

            sb = 1 #positive or negative feature

            if word in end:
                if es_track == 0:
                    r_nextp = int(math.fabs(index_se_tokens[es_track] - index_se_tokens[es_track+1]))
                    l_nextp = 0
                elif es_track+1 == len(index_se_tokens):
                    l_nextp = int(math.fabs(index_se_tokens[es_track] - index_se_tokens[es_track-1]))
                    r_nextp = 0
                else:
                    r_nextp = int(math.fabs(index_se_tokens[es_track] - index_se_tokens[es_track+1]))
                    l_nextp = int(math.fabs(index_se_tokens[es_track] - index_se_tokens[es_track-1]))
                    # <--- END
                    
                if n+1 == len(item):
                    if i+1 == len(o_alltext):
                        right = ''
                    else:
                        if o_alltext[i+1] != []:
                            right = o_alltext[i+1][0]
                        else:
                            right = ''
                        # <--- END
                    left = item[n-1]
                    punct = word
                    sb = True
                elif n == 0:
                    right = item[n+1]
                    left = ''
                    punct = word
                    sb = False
                else:
                    if n+2 == len(item) and item[-1] in quot + end: 
                        right = o_alltext[i+1][0]
                        sb = True
                    else:
                        right = item[n+1]
                        sb = False
                        # <--- END
                    left = item[n-1]
                    punct = word
                    # <--- END

#                if len([x for x in item if x in end]) > 1:
#                    print(item, r_nextp, l_nextp)
#                    print(construct_featureset(left, right, punct, r_nextp, l_nextp), sb)
#                    print(n, '\n')

                es_track += 1
                corr.append(sb)
                feat_list.append((construct_featureset(left, right, punct, 
                                                       r_nextp, l_nextp, in_parenthesis), sb))
                if ')' in word:
                    in_parenthesis = False
                # <--- END
                    
    return feat_list, corr

def extract_features(text):
    pass


def get_dev():
    pass

#trainf = get_train()
testf, c = get_test()
#
#T = defaultdict(int)
#F = defaultdict(int)
#
#X, y = [], []
#print(len(testf) + len(trainf))
#for feat, cl in testf + trainf:
##    if feat['next_w'] in end + quot:
##        print(feat, cl)
#    if cl:
#        if len(T) <= len(F):
#            T[feat['next_w']] += 1
#        else:
#            pass
#    else:
#        F[feat['next_w']] += 1
#        
#print(sum(T.values())/len(T))
#print(sum(F.values())/len(F))
#
#mtr = [0,0,0,0]
#
#for k, v in T.items():
#    mtr[0] += v
#    if k in F.keys():
#        mtr[1] += F[k]
#
#for k, v in F.items():
#    mtr[3] += v
#    if k in T.keys():
#        mtr[2] += T[k]
#        
#print(mtr)
#r1 = mtr[0] + mtr[1]
#r2 = mtr[2] + mtr[3]
#c1 = mtr[0] + mtr[2]
#c2 = mtr[1] + mtr[3]    
#
#N = mtr[0] + mtr[1] + mtr[2] + mtr[3]
#
#A = (r1 * c1) / N
#B = (r1 * c2) / N
#C = (r2 * c1) / N
#D = (r2 * c2) / N
#
#eA = math.pow(mtr[0] - A, 2)/A
#eB = math.pow(mtr[1] - B, 2)/B
#eC = math.pow(mtr[2] - C, 2)/C
#eD = math.pow(mtr[3] - D, 2)/D
#
#print(eA+eB+eC+eD)

#mean_T = sum(T)/len(T)
#mean_F = sum(F)/len(F)
#
#std_T = np.std(T)
#std_F = np.std(F)
#
#cv_T = np.std(T)/mean_T
#cv_F = np.std(T)/mean_F
#
#print('DATA:')
#print(mean_T, std_T, cv_T)
#print(mean_F, std_F, cv_F)      

















  

