import pandas as pd
import sklearn.metrics as skm

valid_data = pd.read_csv('valid_b_cut.txt', delimiter='\t')
cate3_ids = pd.read_csv('cate3_id.csv')
cate2_ids = pd.read_csv('cate2_id.csv')
cate1_ids = pd.read_csv('cate1_id.csv')

f1score3 = skm.f1_score(valid_data['cate3_id'], cate3_ids['cate3_id'],average='macro')
f1score2 = skm.f1_score(valid_data['cate2_id'], cate2_ids['cate2_id'],average='macro')
f1score1 = skm.f1_score(valid_data['cate1_id'], cate1_ids['cate1_id'],average='macro')
f1score = (0.1*f1score1+0.3*f1score2+0.6*f1score3)

print("f1score = ", f1score)
print("f1score1 = ", f1score1)
print("f1score2 = ", f1score2)
print("f1score3 = ", f1score3)



