import pandas as pd, numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn import svm
import sklearn.metrics as skm

li = ['cate1_id','cate2_id','cate3_id']
column = 'title_words'
train = pd.read_csv('train_a.txt', delimiter='\t')
#test = pd.read_csv('test_a.txt', delimiter='\t')
test = pd.read_csv('valid_a.txt', delimiter='\t')
test_id = test["item_id"].copy()
vec = TfidfVectorizer(ngram_range=(1,2),min_df=3, max_df=0.5, use_idf=1, smooth_idf=1, sublinear_tf=1)
trn_term_doc = vec.fit_transform(train[column])
test_term_doc = vec.transform(test[column])
for classifier in li:
    fid0=open(classifier+'.csv','w')

    y=(train[classifier]).astype(int)-1
    Gaussian_clf = svm.LinearSVC(class_weight='balanced', tol=0.000001, C=0.6,max_iter = 50000)
    #Gaussian_clf = svm.SVC()
    Gaussian_clf.fit(trn_term_doc,y)
    preds = Gaussian_clf.predict(test_term_doc)
    i=0
    fid0.write("id,"+classifier+"\n")
    for item in preds:
        fid0.write(str(test['item_id'][i])+","+str(item+1)+"\n")
        i=i+1
    fid0.close()
    print(classifier+' finished!')

