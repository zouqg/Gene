import pandas as pd
import itertools
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

from sklearn.metrics import roc_auc_score
import os
import joblib

dataset=pd.read_csv('./datasets/Train.csv')
df_predict = pd.read_csv('./datasets/test_dataset.csv')
mirna_seqdf=pd.read_csv('./datasets/mirna_seq.csv')#(['mirna', 'seq']
gene_seqdf=pd.read_csv('./datasets/gene_seq.csv')#'label', 'sequence'

dataset_mirna=dataset['miRNA']
dataset_gene=dataset['gene']
dataset_label=dataset['label']
gene_index=gene_seqdf['label'].values.tolist()
gene_seq=gene_seqdf['sequence']
mirna_index=mirna_seqdf['mirna'].values.tolist()
mirna_seq=mirna_seqdf['seq']

key_set = {}
key_set_T = {}
for i in itertools.product('UCGA', repeat=3):  # itertools.product('BCDEF', repeat = 2):
    # print(i)
    obj = ''.join(i)
    # print(obj)
    ky = {'{}'.format(obj): 0}
    key_set.update(ky)
for i in itertools.product('TCGA', repeat=3):  # itertools.product('BCDEF', repeat = 2):
    # print(i)
    obj = ''.join(i)
    # print(obj)
    ky = {'{}'.format(obj): 0}
    key_set_T.update(ky)

def clean_key_set(key_set):
    for i, key in enumerate(key_set):
        # print(i,key,key_set[key])
        key_set[key] = 0
    return key_set



def return_features(n,seq):
    clean_key_set(key_set)
    key=key_set
    if '\n' in seq:
        seq=seq[0:-1]
    for i in range(n,len(seq)+1-n):
        win=seq[i:i+n]
        #print(win)
        ori=key_set['{}'.format(win)]
        key_set['{}'.format(win)]=ori+1
    return key_set
def return_gene_features(n,seq):
    clean_key_set(key_set_T)
    key=key_set_T
    if '\n' in seq:
        seq=seq[0:-1]
    for i in range(n,len(seq)+1-n):
        win=seq[i:i+n]
        #print(win)
        ori=key_set_T['{}'.format(win)]
        key_set_T['{}'.format(win)]=ori+1
    return key_set_T

def construct_dataset(dataset_mirna,dataset_gene):
    list_mirna_feature=[]
    list_gene_feature=[]
    for i in range(0,len(dataset_mirna)):
        try:
            mirna=dataset_mirna[i]  #  mirna的名字
            m_index=mirna_index.index(mirna)
            mirna_f=return_features(3,mirna_seq[m_index])

            gene=dataset_gene[i]
            g_index=gene_index.index(gene)
            gene_f=return_gene_features(3, gene_seq[g_index])
            # print(gene_seq[g_index])
            # print(gene_f)

            mirna_feature=mirna_f.copy()
            gene_feature=gene_f.copy()
            list_mirna_feature.append(mirna_feature)
            list_gene_feature.append(gene_feature)
        except:
            mirna=dataset_mirna[i]
            gene=dataset_gene[i]
            print('error detected',i,mirna,gene)
    lmpd=pd.DataFrame(list_mirna_feature)
    lgpd=pd.DataFrame(list_gene_feature)
    X=pd.concat([lmpd,lgpd],axis=1)
    return X


#标签换为数字
Y=[]
for i,label in enumerate(dataset_label):
    if label =='Functional MTI':
        Y.append(1)
    else:
        Y.append(0)

X=construct_dataset(dataset_mirna,dataset_gene)
#lmpd.to_csv('gene_features.csv',index=None)

X_train, X_test, y_train, y_test = train_test_split(X, Y,test_size=0.3, random_state=2)

#切分训练集进行调参
def train():
  clf = RandomForestClassifier(n_estimators=40)
  clf.fit(X_train,y_train)
  y_p=clf.predict(X_test)

  #acc = metrics.accuracy_score(y_test,y_p)
  #print('RF_ACC',acc)
  y_pb=clf.predict_proba(X_test)
  #print(y_p)
  f1score=metrics.f1_score(y_test, y_p)
  print('RF_F1',f1score)
  MCC=metrics.matthews_corrcoef(y_test, y_p)

if __name__ == '__main__':
    train()
    # 最终模型
    clf_final = RandomForestClassifier(n_estimators=40)
    clf_final.fit(X, Y)

    predict_mirna = df_predict['miRNA']
    predict_gene = df_predict['gene']

    # 测试数据生成器
    X_predict = construct_dataset(predict_mirna, predict_gene)

    # 预测结果
    final_result = clf_final.predict(X_predict)

    # 预测结果
    final_result = clf_final.predict(X_predict)


    #  生成结果
    df_predict['results'] = final_result
    df_predict.to_csv('./datasets/submission.csv', index=None)