# -*- coding: utf-8 -*-
"""
Created on Fri Jun 16 10:07:04 2017

@author: adam

Additional features:
    next word form/lemma
    
Procedure changes:
    Read tokens with .!? within, e.g. ['Hi!'] not ['Hi','!']

"""
from os import listdir as listdir
import re
import pickle
from collections import defaultdict
import nltk.stem.snowball as stem

class MaxEnt:
    def __init__(self):
        self.lemmalist = defaultdict(list)
        self.lemmaf = stem.SwedishStemmer()
    
    # read file and extract information
    def read_text(self, filename): #SUC documents
        ann_data = []
        sbd = []
        end = ['.','!','?', '...']
        with open(filename) as file:
            for i, line in enumerate(file):
                line = line.rstrip()
                if line:
                    # add new list for sentence
                    if re.match('<s id=".*">', line):
                        ann_data.append([])
                        
                    # add symbol index of sentence boundary
                    elif re.match('</s>', line):
                        if ann_data[-1][-1][0] in end:
                            sbd.append(len(ann_data[-1])-1) # add entry to sbd list
                        else:
#                            print(ann_data[-1])
                            if len(ann_data[-1]) > 1:
                                if ann_data[-1][-2] in end:
                                    sbd.append(len(ann_data[-1])-2)
#                                    print('>',ann_data[-1], len(ann_data[-1])-2)
                                else:
                                    sbd.append('NULL')
                            else:
                                sbd.append('NULL') # no clear sentence boundary
                    else:
                        # select word from xml structure
                        word = re.search('<(d|w) n="\d{1,4}">(.*)<ana><ps>(.*)</ps>(<m>(.*)</m>)?<b>(.*)</b>',line)
                        # add word to sentence
                        if word:  
                            entry = word.group(2)
                            
                            if len(entry) > 1 and entry[0] in ['.','!','?']:
                                if entry[0]*len(entry) == entry:
                                    for ind, sy in enumerate(entry):
                                        if ind > 0:
                                            ann_data[-1].append(sy)
                                    entry = entry[0]
                                
                            ann_data[-1].append(entry)
                            
        # ann_data = [{features}, ...], sbd = [int, int, int, ...]
        return ann_data, sbd

    # Features: next_word_isupper, next_word_postag, previous_word_islower, previous_word_length, puntuation symbol                            
    # extract feature set from data, label as True or False. parag contains lists of sentences, 
    # the last symbol of the sent is the sbd.
    def get_features(self, parag, sb):
        extracted_features = []
                    
        for i, sent in enumerate(parag):
#            print(sent, len(parag))
            for n, w in enumerate(sent):
                if w in ['.','!','?']:
                    features = False
                    if i+1 != len(parag):
                        if parag[i+1][0][0].isalpha():
                            next_u = parag[i+1][0][0].isupper()
                        else:
                            next_u = True
                            
                        features = {'next_upper': next_u,
                                    'next_w' : parag[i+1][0],
                                    'next_len' : len(parag[i+1][0]),
                                    'prev_lower': parag[i][n-1].islower(), 
                                    'prev_len': len(parag[i][n-1]), 
                                    'punct': parag[i][n]}   
                    else:
                        features = {'next_upper': True,
                                    'next_w': 'NULL',
                                    'next_len' : 0,
                                    'prev_lower': parag[i][n-1].islower(), 
                                    'prev_len': len(parag[i][n-1]), 
                                    'punct': parag[i][n],
                                    }                          
                    if features:
                        if n+1 == len(sent):
                            extracted_features.append((features, True))
                        else:
                            ### handle -> "asdasd."
                            if parag[i][n+1] in ['"',"'"]:
                                if n+2 >= len(sent):
                                    extracted_features.append((features, True))
                                else:
                                    extracted_features.append((features, False))
                            else:
                                extracted_features.append((features, False))                        

        return extracted_features
        
    def get_keys(self, target, dictionary):
        for item in dictionary:
            if target in item:
                return item
                
if __name__ == '__main__':    

    load = False

    if load:        
        f = open('maxent.pickle', 'rb')
        classifier = pickle.load(f)
        f.close()
    else:
        features = []        
        
        source = '/home/adam/data/SUC3.0/corpus/xml/'
        
        ME = MaxEnt()
        for file in listdir(source):
#            if file.endswith('.xml'):
            if 'aa' in file:
                data, sbd = ME.read_text(source + file)
                features += ME.get_features(data, sbd)
#                print(features)
                
        nullers = 0
        for f in features:
            if f[1] == False:
                nullers += 1
#        print(nullers)
        
        size = int(len(features) * 0.1)
        train_set, test_set = features[size:], features[:size]
        
#        classifier = nltk.classify.MaxentClassifier.train(features, 'IIS', trace=0, max_iter=1000)
        
#        f = open('maxent_suc3.pickle', 'wb')
#        pickle.dump(classifier, f)
#        f.close()
        
##################################### DO STUFF
#    ME = MaxEnt()
#    
#    test = {'next_upper': False,
#            'next_ps': 'JJ',
#            'prev_lower': True, 
#            'prev_len': 1, 
#            'punct': '.'}  

#    print(classifier.classify(test))   
#    print(nltk.classify.accuracy(classifier, test_set))
#    print(classifier.show_most_informative_features(4))
    
#    print(me.sbd)
#    if data:
#        for i, sent in enumerate(me.ann_data):
#            print(sent)
#            print('')