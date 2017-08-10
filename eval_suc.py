# -*- coding: utf-8 -*-
"""
sentence boundary detection evaluation

Created on Thu Jun 15 12:41:09 2017

@author: adam
"""

import nltk
import pickle
import subprocess
from os import listdir

import numpy as np
import matplotlib.pyplot as plt

from get_suc3_traintestdata import get_test, get_train

from sklearn import tree
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import f_classif
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

feature_map = {
    'next_u': 0, 
    'prev_u': 1, 
    'next_l': 2, 
    'prev_l': 3,
    'prev_p': 4,
    'next_p': 5,
    'next_test': 6,
    'prev_test': 7,
    'in_p': 8,
    'punct': 9
    }
end = ['.','?','!']
cap = False
name = False
s = ''

ev = [0,0,0,0]
sbd_base = []
ev_base = [0,0,0,0]


#test maxent
maxent_test = True
save = False

rule_test = False
baseline = False

def model_test():
    if save:
        name = 'maxent_'
        suff = '.pickle'
        i = 0        
        while name + suff in listdir('/home/adam/august/code/pickle/'):
            i += 1
            if i <= 9:
                name = name[:-1] + str(i)
            else:
                name = name[:-2] + str(i)
            
        print(name)       
        
#        train = train[:-1]
#        features = ME.get_features(train, train_s)

        features = get_train()
#        print(features)
        print('N(trainingdata):', len(features))
        
#        nltk.config_megam('/home/adam/Downloads/MEGAM/megam-64')
        
        classifier = nltk.classify.MaxentClassifier.train(features, 'GIS', 
                                                          trace=5, max_iter=1000) 
        
        f = open('/home/adam/august/code/pickle/{0}.pickle'.format(name), 'wb')
        pickle.dump(classifier, f)
        f.close()

    elif not save:
        # load from file
        f = open('/home/adam/august/code/pickle/maxen34.pickle', 'rb')
        classifier = pickle.load(f)
        f.close()
    else:
        return None
        
    testdata, key = get_test()
    print('N(testdata):', len(testdata))
    for i, feat in enumerate(testdata):
#            print(feat[0])
        bd = classifier.classify(feat[0])
        if key[i]:
            if bd:
                ev[0] += 1
            else:
                ev[2] += 1
        else:
            if bd:
                ev[1] += 1
            else:
                ev[3] += 1
        
        if baseline:
            pass
    return ev
            
            
def feature_tests():
    traindata = get_train()
    testdata, kys = get_test()
    X = [] # X
    lbls = [] # y
    
    for dp in traindata + testdata:
        lb, ft = dp[1], dp[0]
        
        if lb:
            lbls.append(1)
        else:
            lbls.append(0)
        
        thing = [0,0,0,0,0,0,0,0,0,0]
        for w in ft:
            if w == 'next_u':
                thing[0] = ft[w]
            elif w == 'next_l':
                thing[1] = ft[w]
            elif w == 'next_p':
                thing[2] = ft[w]
            elif w == 'prev_u':
                thing[3] = ft[w]
            elif w == 'prev_l':
                thing[4] = ft[w]
            elif w == 'prev_p':
                thing[5] = ft[w]
            elif w == 'next_test':
                thing[6] = ft[w]
            elif w == 'prev_test':
                thing[7] = ft[w]
            elif w == 'in_p':
                thing[8] = ft[w]
            elif w == 'punct':
                thing[9] = ft[w]
                
        X.append(thing)
        
#        print(len(X), len(lbls))
    X1 = np.asarray(X)
    
#    pca = PCA(n_components=4)
#    pca.fit(X1, lbls)
#    print(pca.explained_variance_ratio_)

    print('LogisticRegression(RFE):')
    model = LogisticRegression()
    rfe = RFE(model, 5)
    fit = rfe.fit(X1, lbls)
    print('T:', [get_k(feature_map, i) for (i, x) in enumerate(fit.support_) if x])
    print('F:', [get_k(feature_map, i) for (i, x) in enumerate(fit.support_) if not x])
    print('')
#
#    feats = SelectKBest(f_classif, k=4)
#    feats.fit(X1, lbls)
#    print(feats.scores_)
#    print(feats.score_func)
#    ply = PolynomialFeatures(3)
#    X1 = ply.fit_transform(X)
#    
    dtr = tree.DecisionTreeClassifier(random_state=0)
    dtr.fit(X1, lbls)
    
    print('DecisionTreeClassifier:')
    for i, ft in enumerate([dtr.feature_importances_[x] for x in np.argsort(dtr.feature_importances_)[::-1]]):
        print(i+1, get_k(feature_map, i), ft)#get_k(feature_map, i), ft)
    print(cross_val_score(dtr, X1, lbls, cv=5, scoring='f1'))
    print('')
    
        
    
    show_f_importance_graph(X1, lbls)
        
def get_k(tdict, i):
    for x in tdict:
        if tdict[x] == i:
            return x
    

def show_f_importance_graph(X1, lbls):
    frst = ExtraTreesClassifier(n_estimators=250, random_state=0)
    frst.fit(X1, lbls)
    
    imp = frst.feature_importances_
    std = np.std([tr.feature_importances_ for tr in frst.estimators_],
         axis=0)
    
    ind = np.argsort(imp)[::-1]
    std_rnk = []
    names = []
    print('ExtraTreesClassifier:')
    for f in range(X1.shape[1]):
        if ind[f] == 0:
            names.append('next_u')
            std_rnk.append(std[f])
        elif ind[f] == 1:
            names.append('next_l')
            std_rnk.append(std[f])
        elif ind[f] == 2:
            names.append('next_p')
            std_rnk.append(std[f])
        elif ind[f] == 3:
            names.append('prev_u')
            std_rnk.append(std[f])
        elif ind[f] == 4:
            names.append('prev_l')
            std_rnk.append(std[f])
        elif ind[f] == 5:
            names.append('prev_p')
            std_rnk.append(std[f])
        elif ind[f] == 6:
            names.append('next_test')
            std_rnk.append(std[f])
        elif ind[f] == 7:
            names.append('prev_test')
            std_rnk.append(std[f])
        elif ind[f] == 8:
            names.append('in_p')
            std_rnk.append(std[f])
        elif ind[f] == 9:
            names.append('punct')
            std_rnk.append(std[f])            
        else:
            names.append('{0}'.format(f))
            std_rnk.append(0)
            
        print("{0}. {1} \t {2} (std = {3})".format(f + 1, names[-1], imp[ind[f]], std[f]))

    plt.figure()
    plt.title("Feature importances")
    plt.bar(range(X1.shape[1]), imp[ind],
           color="r", yerr=std[ind], align="center")
    plt.xticks(range(X1.shape[1]), names)
    plt.xlim([-1, X1.shape[1]])
    plt.show()

def visualize_tree(tr, feature_names):
    with open("dt.dot", 'w') as f:
        tree.export_graphviz(tr, out_file=f,
                        feature_names=feature_names)

    command = ["dot", "-Tpng", "dt.dot", "-o", "dt.png"]
    try:
        subprocess.check_call(command)
    except:
        exit("Could not run dot, ie graphviz, to "
             "produce visualization")

#visualize_tree(dtr, thing)

def show_res(ev):
    if name:
        print(name)
        
    y_true = [1]*ev[0] + [1]*ev[1]+ [0]*ev[2]
    y_score = [1]*ev[0]+ [0]*ev[1]+ [1]*ev[2]
    print('ROC', roc_auc_score(y_true, y_score))
    
    try:
        print(ev, sum(ev))
        pre = ev[0]/(ev[0]+ev[1])
        rec = ev[0]/(ev[0]+ev[2])
        print('pr', pre)
        print('re', rec)
        print('f-score', (2*pre*rec)/(pre+rec))    
    except:
        pass

if __name__ == '__main__':
    model_cmatrix = model_test()
    show_res(model_cmatrix)
    
#    feature_tests()

