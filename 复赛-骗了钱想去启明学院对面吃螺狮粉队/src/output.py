import pandas as pd

data1 = pd.read_csv('cate1_id.csv')
data2 = pd.read_csv('cate2_id.csv')
data3 = pd.read_csv('cate3_id.csv')
data1['cate2_id'] = data2['cate2_id']
data1['cate3_id'] = data3['cate3_id']
data1.to_csv('answer.txt',sep = '\t',index=False)