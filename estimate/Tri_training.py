#coding:utf-8
import string
import numpy as np
import pandas as pd
import time
import sklearn
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import random
import math
import os
from sklearn.ensemble import BaggingClassifier

train_name=['train/Monitor_1839_1009_pm25','train/Monitor_1841_1009_pm25','train/Monitor_1842_1009_pm25','train/Monitor_1843_1009_pm25','train/Monitor_1844_1009_pm25','train/Monitor_1847_1009_pm25']
test_name=['test/Monitor_1839_1009_pm25','test/Monitor_1841_1009_pm25','test/Monitor_1842_1009_pm25','test/Monitor_1843_1009_pm25','test/Monitor_1844_1009_pm25','test/Monitor_1847_1009_pm25']
#train_crf=['train\\Monitor_1839_1009_pm25.data','train\\Monitor_1841_1009_pm25.data','train\\Monitor_1842_1009_pm25.data','train\\Monitor_1843_1009_pm25.data','train\\Monitor_1844_1009_pm25.data','train\\Monitor_1847_1009_pm25.data']
test_crf=['test\\Monitor_1839_1009_pm25.data','test\\Monitor_1841_1009_pm25.data','test\\Monitor_1842_1009_pm25.data','test\\Monitor_1843_1009_pm25.data','test\\Monitor_1844_1009_pm25.data','test\\Monitor_1847_1009_pm25.data']

class CRF_classifier():
    CRF_count=0
    def __init__(self):
        self.count=CRF_classifier.CRF_count
        CRF_classifier.CRF_count+=1
        self.model_name='model'+str(self.count)
        self.train_info_name='train_info'+str(self.count)
    def train(self,train):
        np.savetxt('train.data',train,delimiter='\t')
        command1 ='CRF++-0.58\\crf_learn -c 4.0 CRF++-0.58\\template train.data '+self.model_name+'>>'+self.train_info_name
        os.system(command1)
    def predict_lab(self,Test):
        t0=time.clock()
        t0=self.model_name+str(t0)
        np.savetxt('test.data',Test,delimiter='\t')
        command1 ='CRF++-0.58\\crf_test -v2 -m '+self.model_name+' test.data>>'+t0
        os.system(command1)
        crf_result=[]
        buf=list()
        f=open(t0,'r')
        buf=f.readline();
        for i in range(0,len(Test)):
            buf=f.readline();
            buf=buf.strip('\n')
            buf=buf.split('\t')
            buf=buf[len(buf)-8].split('/')
            buf=float(buf[0])
            crf_result.append(buf)
        f.close()
        np.array(crf_result)
        command2='del '+t0
        os.system(command2)
        return crf_result
    def predict_prob(self,Test):
        t0=time.clock()
        t0=self.model_name+str(t0)
        np.savetxt('test.data',Test,delimiter='\t')
        command1 ='CRF++-0.58\\crf_test -v2 -m '+self.model_name+' test.data>>'+t0
        os.system(command1)
        #read file
        crf_result=[]
        buf=list()
        f=open(t0,'r')
        buf=f.readline();
        for i in range(0,len(Test)):
            buf=f.readline()
            buf=buf.strip('\n')
            buf=buf.split('\t')
            for i in range(len(buf)-7,len(buf)):
                buf[i]=buf[i].split('/')
                buf[i]=float(buf[i][1])
            crf_result.append(buf[-7:])
        f.close()
        np.array(crf_result)
        command2='del '+t0
        os.system(command2)
        return crf_result

class Other_claasifier():
    def __init__(self,classifier):
        self.myclassfier=classifier
        self.myclassfier.classes_=[1,2,3,4,5,6,7]
    def train(self,test):
        self.myclassfier=self.myclassfier.fit(test[:,40:-1],test[:,-1])
    def predict_lab(self,Test):
        prediction=self.myclassfier.predict(Test[:,40:-1])
        return prediction
    def predict_prob(self,Test):
        Pre_pro=self.myclassfier.predict_proba(Test[:,40:-1])
        return Pre_pro

def MeasureError(classifier_base,Labeled_F,i):
   "对错误率进行测量"
   count_err=0.0
   err=0.0
   prediction1=classifier_base[(i+1)%3].predict_lab(Labeled_F)
   prediction2=classifier_base[(i+2)%3].predict_lab(Labeled_F)
   for j in range(0,len(Labeled_F)):
       if prediction1[j]==prediction2[j]:
           count_err+=1.0
           if prediction1[j]!=Labeled_F[j,-1]:
               err+=1.0
   if err!=0.0:
       err=err/count_err
   return err

#domian return 1 if domian and 0 if week domian else return -1

def accu_rate(classifier_base,Test):
    Pre_pro=[[[0.0 for k in range(0,6)] for i in range(0,len(Test))]for j in range(0,3)]
    Pre_result=[0 for i in range(0,len(Test))]
    Pro_sum=[[0.0 for i in range(0,6)]for j in range(0,len(Test))]
    for i in range(0,3):
        Pre_pro[i]=classifier_base[i].predict_prob(Test)
    for i in range(0,len(Test)):
        for j in range(0,6):
            for k in range(0,3):
                Pro_sum[i][j]+=Pre_pro[k][i][j]
        max_pro=max(Pro_sum[i])
        ind=Pro_sum[i].index(max_pro)
        Pre_result[i]=ind+1
    error_t=0.0
    for i in range(0,len(Test)):
        if Pre_result[i]!=Test[i,-1]:
            error_t+=1.0
    return (float((len(Test)-error_t))/len(Test))
  
def bootstrap_resample(Labeled_F,n=None):
    if n==None:
        n=len(Labeled_F)
    resample_i=np.floor(np.random.rand(n)*len(Labeled_F)).astype(int)
    X_resample=Labeled_F[resample_i]
    return X_resample

print 'CRF \NB \LR'
print "iteration,train_accuracy rate,test_accuracy rate: "
test_accuracy=[0.0,0.0,0.0,0.0,0.0,0.0,0.0]
train_accuracy=[0.0,0.0,0.0,0.0,0.0,0.0,0.0]
acc_detail=[[0.0 for i in range(0,8)]for j in range(0,8)]
for r1 in range(0,6):
    print train_name[r1]
    
    #three classification
    classifier_base=[CRF_classifier(),Other_claasifier(GaussianNB()),Other_claasifier(LogisticRegression())]
    #function MeasureError()
    #input
    #Labeled 
    t0=time.clock();
    Labeled_F=np.loadtxt(train_name[r1])
    #Test
    Test=np.loadtxt(test_name[r1])
    #U:unlabeled
    Unlabeled_F=np.loadtxt('unmonitor_1009')
    #Learn
    #for i in range(0,3):
    Sub_L=[[[0.0 for i in range(0,len(Labeled_F[0]))] for i in range(0,len(Labeled_F))] for i in range(0,3)]#Si
    Sub_L=np.array(Sub_L)
    Err_pri=[0.0 for i in range(0,3)]#e'_i 
    L_pri=[0 for i in range(0,3)]#l'_i 
    Err=[0.0 for i in range(0,3)]#e_i 
    for i in range(0,3): 
        Sub_L[i]=bootstrap_resample(Labeled_F) #L_i <-- Bootstrap(L)
        Sub_L[i]=np.array(Sub_L[i])
        classifier_base[i].train(Sub_L[i])  #h_i <-- Learn(L_i)
        Err_pri[i]=0.5#e'_i <-- .5
        L_pri[i]=0#l'_i <-- 0
    #endfor    
    iteration=0
    train_accuracy[r1]=accu_rate(classifier_base,Labeled_F)
    test_accuracy[r1]=accu_rate(classifier_base,Test)
    acc_detail[r1][iteration]=accu_rate(classifier_base,Test)
    print iteration,train_accuracy[r1],test_accuracy[r1] 
    #repeat until none of h_i ( i \in {1...3} ) changes 
    Bchanged=True
    while Bchanged :
        Bchanged=False
        L_add=[[[0.0 for i in range(0,53)]for j in range(0,200000)]for k in range(0,3)]                           #L_i <-- \phi
        #L_add=np.array(L_add)
        S_add=[]
        for i in range(0,3):
            S_add.append([])
        count_L=[0,0,0]
        Bupdated=[0 for i in range(0,3)]
        for i in range(0,3):                     #for i \in {1...3} do 
            Bupdated[i]=False
            Err[i]=MeasureError(classifier_base,Labeled_F,i)     #e_i <-- MeasureError(h_j & h_k) (j, k \ne i)
            if Err[i]< Err_pri[i]:               #if (e_i < e'_i)
                #Unlabeled_F
                prediction1=classifier_base[(i+1)%3].predict_lab(Unlabeled_F)
                prediction2=classifier_base[(i+2)%3].predict_lab(Unlabeled_F)
                for j in range(0,len(Unlabeled_F)):
                    if prediction1[j]==prediction2[j]:
                        L_add[i][count_L[i]][0:-1]=Unlabeled_F[j,0:-1]
                        L_add[i][count_L[i]][-1:]=prediction1[j]
                        count_L[i]+=1
                if  L_pri[i]==0:
                    L_pri[i]=int(math.floor((float(Err[i])/(Err_pri[i]-Err[i])+1)))
                if L_pri[i]<count_L[i]:
                    if (Err[i]*count_L[i])<(Err_pri[i]*L_pri[i]):
                        Bupdated[i]=True
                    elif L_pri[i]>(float(Err[i])/(Err_pri[i]-Err[i])):
                        S_add[i]=random.sample(L_add[i][0:count_L[i]],int(math.ceil(Err_pri[i]*L_pri[i]/float(Err[i])-1)))
                        Bupdated[i]=True
        for i in range(0,3):
            if Bupdated[i]==True:
                L_tmp=list(Labeled_F)+list(S_add[i])
                L_tmp=np.array(L_tmp)
                classifier_base[i].train(L_tmp)
                Err_pri[i]=Err[i]
                L_pri[i]=len(S_add[i])
                Bchanged=True
        iteration+=1
        train_accuracy[r1]=accu_rate(classifier_base,Labeled_F)
        test_accuracy[r1]=accu_rate(classifier_base,Test)
        acc_detail[r1][iteration]=accu_rate(classifier_base,Test)
        print iteration,train_accuracy[r1],test_accuracy[r1] 
#test
    t6=time.clock()
    print "total time is: "
    print (t6-t0)

test_accuracy[6]=(test_accuracy[0]*2924.0+test_accuracy[1]*1241.0+test_accuracy[2]*3002.0+test_accuracy[3]*2987.0+test_accuracy[4]*2997.0+test_accuracy[5]*718.0)/13869.0
np.savetxt('test_accuracy.txt',test_accuracy,fmt="%f")
train_accuracy[6]=(train_accuracy[0]*13152.0+train_accuracy[1]*10873.0+train_accuracy[2]*10883.0+train_accuracy[3]*10868.0+train_accuracy[4]*12629.0+train_accuracy[5]*10946.0)/69351.0
np.savetxt('train_accuracy.txt',train_accuracy,fmt="%f")
for i in range(0,8):
    acc_detail[6][i]=(acc_detail[0][i]*2924.0+acc_detail[1][i]*1241.0+acc_detail[2][i]*3002.0+acc_detail[3][i]*2987.0+acc_detail[4][i]*2997.0+acc_detail[5][i]*718.0)/13869.0
    acc_detail[7][i]=(acc_detail[0][i]*2924.0+acc_detail[2][i]*3002.0+acc_detail[3][i]*2987.0+acc_detail[4][i]*2997.0)/11910.0
np.savetxt('acc_detail.txt',acc_detail,fmt="%f")