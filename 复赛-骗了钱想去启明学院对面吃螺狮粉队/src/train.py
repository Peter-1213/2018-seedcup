# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals # For Python2
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import time
import gc
import sys

sys.path.append('./liblinear-multicore-2.20/python') # import liblinear
from liblinearutil import *

from multiprocessing import cpu_count # For adaptive multicore processing


def compute_class_weight(class_weight, classes, y):
    """
    Estimate class weights for unbalanced datasets.

    Parameters
    ----------
    class_weight : dict, 'balanced' or None
        If 'balanced', class weights will be given by
        ``n_samples / (n_classes * np.bincount(y))``.
        If a dictionary is given, keys are classes and values
        are corresponding class weights.
        If None is given, the class weights will be uniform.

    classes : ndarray
        Array of the classes occurring in the data, as given by
        ``np.unique(y_org)`` with ``y_org`` the original class labels.

    y : array-like, shape (n_samples,)
        Array of original class labels per sample;

    Returns
    -------
    class_weight_vect : ndarray, shape (n_classes,)
        Array with class_weight_vect[i] the weight for i-th class

    References
    ----------
    The "balanced" heuristic is inspired by
    Logistic Regression in Rare Events Data, King, Zen, 2001.
    """
    # Import error caused by circular imports.
    from sklearn.preprocessing import LabelEncoder

    if set(y) - set(classes):
        raise ValueError("classes should include all valid labels that can "
                         "be in y")
    if class_weight is None or len(class_weight) == 0:
        # uniform class weights
        weight = np.ones(classes.shape[0], dtype=np.float64, order='C')
    elif class_weight == 'balanced':
        # Find the weight of each class as present in y.
        le = LabelEncoder()
        y_ind = le.fit_transform(y)
        if not all(np.in1d(classes, le.classes_)):
            raise ValueError("classes should have valid labels that are in y")

        recip_freq = len(y) / (len(le.classes_) *
                               np.bincount(y_ind).astype(np.float64))
        weight = recip_freq[le.transform(classes)]
    else:
        # user-defined dictionary
        weight = np.ones(classes.shape[0], dtype=np.float64, order='C')
        if not isinstance(class_weight, dict):
            raise ValueError("class_weight must be dict, 'balanced', or None,"
                             " got: %r" % class_weight)
        for c in class_weight:
            i = np.searchsorted(classes, c)
            if i >= len(classes) or classes[i] != c:
                raise ValueError("Class label {} not present.".format(c))
            else:
                weight[i] = class_weight[c]

    return weight

def weight_array_to_param_string(weight_array,classes):
    """
    Transform balanced weight vector into the format of liblinear parameters

    Parameters
    -----------
    weight_array : ndarray
        Array of weights for each class ** in the order of parameter classes **.

    classes : ndarray
        Array of the classes occurring in the data, as given by
        ``np.unique(y_org)`` with ``y_org`` the original class labels.

    Returns
    -------
    A string which suits format of liblinear parameter.
    like '-wi weight_of_i -wj weight_of_j ...'
    """
    param_string = ''
    if len(weight_array)!=len(classes):
        raise ValueError("Length of weight_array does not euqal number of classes given")
    for weight_item, single_class in zip(weight_array, classes):
        local_string = ' '
        local_string = '-w'+str(single_class)+' '+str(weight_item)+' '
        param_string += local_string
    print('weight calculated')
    return param_string

def Data_Loader():
    """
    Loading data for training/validating

    Returns
    -------
    train_df, test_df : pd.Dataframe
    Dataframe of given dataset
    """
    if len(sys.argv)>1:  
        if sys.argv[1] == 'valid':
            # valid_b_cut is randomly selected from all datasets(size 100,000)
            test_df = pd.read_csv('valid_b_cut.txt', delimiter='\t') 
            # train_b_cut is the remaining.
            train_df = pd.read_csv('train_b_cut.txt', delimiter='\t') 
        elif sys.argv[1] == 'train':
            # train_b.txt is a mix of all datasets provided.
            train_df = pd.read_csv('train_b.txt', delimiter='\t') 
            test_df = pd.read_csv('test_b.txt', delimiter='\t')
        else:
            raise ValueError("Input Argument is invalid.")
    else:
        raise ValueError("You must specify a training mode in the argument")
    print("Dataset Loaded")
    return train_df ,test_df

def Tfidf(train_df, test_df):
    """
    Transform dataset into Tfidf sparse matrices

    Parameters
    -----------
    train_df, test_df: pd.Dataframe
    Dataframe of datasets.
    
    Returns
    -------
    Tfidf_train,Tfidf_test: scipy.sparse.spmatrix
    sparse matrices of Tfidfvectorized dataset.
    """
    vec = TfidfVectorizer(min_df=1, max_df=0.6, use_idf=1, smooth_idf=1, sublinear_tf=1) 
    Tfidf_train = vec.fit_transform(train_df[feature_column]) # fit and standardizarion the train dataset.
    Tfidf_test = vec.transform(test_df[feature_column]) # standardization the test set.
    print("Tfidf Matrix Trained")
    return Tfidf_train,Tfidf_test

def Train(training_class,train_df,test_df,Tfidf_train,Tfidf_test):
    print(training_class+' Started')
    unique_list_of_cate = np.unique(train_df[training_class])
    # Mapping categories to numbers
    mapping_dict = {unique_list_of_cate[i]:i for i in range(len(unique_list_of_cate))} 
    # Mapping back from numbers to categories
    return_dict = dict([(v, k) for (k, v) in mapping_dict.items()]) 
    y = list((train_df[training_class].map(mapping_dict)).astype(int))

    Your_Param = '-s 1 -c 0.6 -e 0.00002 -q ' # Modify according to ./liblinear-multicore-2.20/python/liblinearutil.py
    # Adaptive Multithreads:
    Threads = cpu_count() - 1 # -1 to avoid eating up all resources. Delete it to enjoy best performance.
    Thread_param = str('-n ' + str(Threads)+ ' ')
    
    # For weight params:
    classes_mapped = np.unique((train_df[training_class].map(mapping_dict)).astype(int))
    # Calculate weights for each classes
    weight_array = compute_class_weight('balanced',classes_mapped, y)
    # Transform weights array to liblinear parameters
    param_weight_string = weight_array_to_param_string(weight_array, classes_mapped)

    param = str(Your_Param + Thread_param + param_weight_string)
    Linear_clf = train(y, Tfidf_train, param)
    print('The following accuracy is incorrect, use valid.py instead.')
    preds , _ , _ = predict([], Tfidf_test, Linear_clf)

    #Output to file:
    fid0 = open(training_class+'.csv','w')
    fid0.write("id,"+training_class+"\n")
    i = 0
    for item in preds: 
        fid0.write(str(test_df['item_id'][i]) + "," + str(return_dict[item]) + "\n")
        i = i + 1
    fid0.close()
    print(training_class+' finished!')

    #Collecting RAMs
    del Linear_clf
    del preds
    gc.collect()


if __name__ == '__main__':
    list_of_class=['cate1_id','cate2_id','cate3_id']
    feature_column='title_words' # Specify the feature column

    train_df, test_df = Data_Loader()
    Tfidf_train, Tfidf_test = Tfidf(train_df=train_df, test_df=test_df)

    for training_class in list_of_class:
        Train(training_class=training_class, train_df=train_df, test_df=test_df, Tfidf_train=Tfidf_train, Tfidf_test=Tfidf_test)