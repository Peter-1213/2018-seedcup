import lightgbm as lgb
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import pickle
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

print("Loading Data ... ")

# 导入数据
li = ['cate2_id','cate3_id']
column = 'title_words'
train = pd.read_csv('train_a.txt', delimiter='\t')
valid = pd.read_csv('valid_a.txt', delimiter='\t')
test = pd.read_csv('test_a.txt', delimiter='\t')
test_id = test["item_id"].copy()
vec = TfidfVectorizer(ngram_range=(1,2),min_df=3, max_df=0.5, use_idf=1, smooth_idf=1, sublinear_tf=1)
trn_term_doc = vec.fit_transform(train[column])
test_term_doc = vec.transform(test[column])
valid_term_doc = vec.transform(valid[column])
train_x, test_x, valid_x = trn_term_doc, test_term_doc, valid_term_doc
leavesmapping={'cate1_id':500, 'cate2_id':200, 'cate3_id':100}

for classes in li:
    list_of_cate = list(set(train[classes]))
    mapping_dict = {list_of_cate[i]:i for i in range(len(list_of_cate))}
    return_dict = dict([(v, k) for (k, v) in mapping_dict.items()])

    train_y = (train[classes].map(mapping_dict)).astype(int)
    valid_y = (valid[classes].map(mapping_dict)).astype(int)

    X_train = train_x
    y_train = train_y
    X_test = valid_x
    y_test = valid_y


    # create dataset for lightgbm
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)
    # specify your configurations as a dict
    params = {
        'boosting_type': 'dart',
        # 'device': 'gpu',
        'max_depth': -1,
        'max_bin': 300,
        'objective': 'multiclassova',
        'num_class': len(list_of_cate),
        'metric': 'multi_error',
        'num_leaves': leavesmapping[classes],
        'min_data_in_leaf': 20,
        'num_iterations': 1000,
        'learning_rate': 0.15,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'lambda_l1': 0.4,
        'lambda_l2': 0.5,
        'min_gain_to_split': 0.2,
        'verbose': 5,
        'is_unbalance': True
    }

    # train
    print('Start training...')
    gbm = lgb.train(params,
                    lgb_train,
                    num_boost_round=10000,
                    valid_sets=lgb_eval,
                    early_stopping_rounds=8)

    print('Start predicting...')

    preds = gbm.predict(test_x, num_iteration=gbm.best_iteration)  # 输出的是概率结果

    # 导出结果
    fid0 = open(classes + '.csv', 'w')
    fid0.write("id," + classes + "\n")
    i = 0
    for pred in preds:
        result = int(np.argmax(pred))
        fid0.write(str(test['item_id'][i]) + "," + str(return_dict[result]) + "\n")
        i = i + 1
    fid0.close()
    print(classes + ' finished!')   

