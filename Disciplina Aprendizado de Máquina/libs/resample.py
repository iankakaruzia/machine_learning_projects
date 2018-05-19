import numpy as np
import math

def split_train_test(n_elem, perc_train, seed):
    a = [i for i in range(n_elem)]
    np.random.seed(seed)
    np.random.shuffle(a)
    idx_train = a[:int(len(a) * perc_train)]
    idx_test = a[int(len(a) * perc_train):]
    return idx_train, idx_test

def split_k_fold(n_elem, n_splits=3, shuffle=True, seed=0):
    if n_splits <= 2:
        return "Valor de n_splits deve ser maior que 2"
    
    a = [i for i in range(n_elem)]
    
    if shuffle:
        np.random.seed(seed)
        np.random.shuffle(a)
    
    splits_size = int(n_elem/n_splits)
    idx_train = list()
    idx_test = list()
    x=0
    
    for i in range(n_splits):
        a_copy = list(a)
        fold = list()
        while len(fold) < splits_size:
            b = a_copy[x]
            fold.append(b)
            x = x+1
        idx_test.append(fold)
        for k in range(len(fold)):
            a_copy.remove(fold[k])
        idx_train.append(a_copy)
    
    return idx_train, idx_test

def split_stratified_train_test(y, perc_train, seed):
    n_elem = len(y)
    idx_train = list()
    idx_test = list()
    
    i0 = list()
    i1 = list()
    
    for x in range(n_elem):
        if(y[x] == 0):
            i0.append(x)
        else:
            i1.append(x)
    
    np.random.seed(seed)
    np.random.shuffle(i0)
    np.random.shuffle(i1)
    
    idx_train.extend(i0[:math.floor(len(i0) * perc_train)])
    idx_train.extend(i1[:math.floor(len(i1) * perc_train)])
    
    idx_test.extend(i0[math.floor(len(i0) * perc_train):])
    idx_test.extend(i1[math.floor(len(i1) * perc_train):])
        
    return idx_train, idx_test
    





















