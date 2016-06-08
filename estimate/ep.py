#coding:utf-8
import string
import numpy as np
import pandas as pd
import time
import sklearn
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn import naive_bayes
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import random
import math
import os
from sklearn.ensemble import BaggingClassifier

train_name=['train/Monitor_1839_1009_pm25','train/Monitor_1841_1009_pm25','train/Monitor_1842_1009_pm25','train/Monitor_1843_1009_pm25','train/Monitor_1844_1009_pm25','train/Monitor_1847_1009_pm25']
test_name=['test/Monitor_1839_1009_pm25','test/Monitor_1841_1009_pm25','test/Monitor_1842_1009_pm25','test/Monitor_1843_1009_pm25','test/Monitor_1844_1009_pm25','test/Monitor_1847_1009_pm25']

#CRF
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
   count=0.0
   err=0.0
   prediction=[[0 for j in range(0,len(Labeled_F))]for k in range(0,len(classifier_base))]
   for j in range(0,len(classifier_base)):
       prediction[j]=classifier_base[j].predict_lab(Labeled_F)
   for j in range(0,len(Labeled_F)):
       for k in range(0,len(classifier_base)):
           flag_count=0
           tmp=prediction[(i+1)%len(classifier_base)][j]
           for g in range(2,len(classifier_base)):
               if prediction[(i+g)%len(classifier_base)][j]==tmp:
                  tmp=prediction[(i+g)%len(classifier_base)][j]
                  flag_count+=1
               else:
                   break;
           if flag_count==len(classifier_base)-2 :
               count+=1.0
               if tmp!=prediction[i][j]:
                   err+=1.0
   err=err/count
   return err

def div_ij(i,j,classifier_base,Labeled_F):
    div_count=0.0;
    prediction1=classifier_base[i].predict_lab(Labeled_F)
    prediction2=classifier_base[j].predict_lab(Labeled_F)
    for k in range(0,len(Labeled_F)):
        if prediction1[k]!=prediction2[k]:
            div_count+=1.0
    div_count=div_count/len(Labeled_F)
    return div_count

#ensemble pruning sub_function
def diversity(S,classifier_base,Labeled_F):
    if size(S)<=1:
        div=0
    else:
        sum_div=0.0
        for i in range(0,size(S)-1):
            if S[i]==1:
                for j in range(i+1,size(S)):
                    if S[j]==1:
                        sum_div+=div_ij(i,j,classifier_base,Labeled_F)
        div=sum_div/(size(S)*(size(S)-1))
    return div



def size(S):
    count=0
    for i in range(0,len(S)):
        if S[i]==1:
            count+=1
    return count

#domian return 1 if domian and 0 if week domian else return -1
def domain(S_z,S_new,classifier_base,Labeled_F):
    div1=diversity(S_z,classifier_base,Labeled_F)
    div2=diversity(S_new,classifier_base,Labeled_F)
    acc1=accracy(S_z,classifier_base,Labeled_F)
    acc2=accracy(S_new,classifier_base,Labeled_F)
    size1=size(S_z)
    size2=size(S_new)
    flag1=0
    flag2=0
    if div1>=div2 and acc1>=acc2 and size1>=size2 :
        flag1=1
        if div1>div2 or acc1>acc2 or size1>size2 :
            flag2=1
    if flag2==1:
        return 1
    elif flag1==1:
        return 0
    else:
        return -1
def domain1(S_z,S_new,classifier_base,Labeled_F):
    div1=diversity(S_z,classifier_base,Labeled_F)
    div2=diversity(S_new,classifier_base,Labeled_F)
    acc1=accracy(S_z,classifier_base,Labeled_F)
    acc2=accracy(S_new,classifier_base,Labeled_F)
    flag1=0
    flag2=0
    if div1>=div2 and acc1>=acc2:
        flag1=1
        if div1>div2 or acc1>acc2:
            flag2=1
    if flag2==1:
        return 1
    elif flag1==1:
        return 0
    else:
        return -1

def domain2(S_z,S_new,classifier_base,Labeled_F):
    acc1=accracy(S_z,classifier_base,Labeled_F)
    acc2=accracy(S_new,classifier_base,Labeled_F)
    flag1=0
    flag2=0
    if acc1>=acc2:
        flag1=1
        if or acc1>acc2:
            flag2=1
    if flag2==1:
        return 1
    elif flag1==1:
        return 0
    else:
        return -1
def domain3(S_z,S_new,classifier_base,Labeled_F):
    div1=diversity(S_z,classifier_base,Labeled_F)
    div2=diversity(S_new,classifier_base,Labeled_F)
    flag1=0
    flag2=0
    if div1>=div2:
        flag1=1
        if or div1>div2:
            flag2=1
    if flag2==1:
        return 1
    elif flag1==1:
        return 0
    else:
        return -1
		
def accracy(S_final,classifier_base,Test):
    pro=[[[0.0 for i in range(0,7)]for j in range(0,len(Test))]for k in range(0,size(S_final))]
    pro=np.array(pro)
    count=0
    for i in range(0,len(S_final)):
        if S_final[i]==1:
            pro[count]=classifier_base[i].predict_prob(Test)
            count+=1
    sum_pro=[[0.0 for i in range(0,7)]for j in range(0,len(Test))]
    for i in range(0,len(Test)):
        for j in range(0,7):
            for k in range(0,count):
                sum_pro[i][j]+=pro[k][i][j]
    sum_lab=[0 for i in range(0,len(Test))]
    count_acc=0.0
    for i in range(0,len(Test)):
        maxs=max(sum_pro[i])
        sum_lab[i]=sum_pro[i].index(maxs)+1
        if sum_lab[i]==Test[i,-1]:
            count_acc+=1.0
    count_acc=count_acc/len(Test)
    return count_acc
  
def bootstrap_resample(Labeled_F,n=None):
    if n==None:
        n=len(Labeled_F)
    resample_i=np.floor(np.random.rand(n)*len(Labeled_F)).astype(int)
    X_resample=Labeled_F[resample_i]
    return X_resample
  
acc_detail=[[0.0 for i in range(0,11)]for j in range(0,8)]
S_detail=[]
Str_name='EP-RF-0.8-len-3-acc-20'
print Str_name
#U:unlabeled
t0=time.clock()
Unlabeled_F=np.loadtxt('unmonitor_1009')
for ins_count in range(0,6):
    #disa_based semi
    #classifier_base
    classifier_base=[Other_claasifier(RandomForestClassifier()),Other_claasifier(RandomForestClassifier()),Other_claasifier(RandomForestClassifier()),Other_claasifier(RandomForestClassifier()),Other_claasifier(RandomForestClassifier()),Other_claasifier(RandomForestClassifier()),Other_claasifier(RandomForestClassifier()),Other_claasifier(RandomForestClassifier()),Other_claasifier(RandomForestClassifier()),Other_claasifier(RandomForestClassifier()),Other_claasifier(RandomForestClassifier()),Other_claasifier(RandomForestClassifier()),Other_claasifier(RandomForestClassifier()),Other_claasifier(RandomForestClassifier()),Other_claasifier(RandomForestClassifier()),Other_claasifier(RandomForestClassifier()),Other_claasifier(RandomForestClassifier()),Other_claasifier(RandomForestClassifier()),Other_claasifier(RandomForestClassifier()),Other_claasifier(RandomForestClassifier())]
    print len(classifier_base)
    #input
    #Labeled 
    Labeled_F=np.loadtxt(train_name[ins_count])
    #Test
    Test=np.loadtxt(test_name[ins_count])
    #Learn
    #for i in range(0,3):
    Sample=[[[0.0 for i in range(0,53)]for j in range(0,len(Labeled_F))]for k in range(0,len(classifier_base))]
    Sample=np.array(Sample)
    Err_pri=[0.0 for i in range(0,len(classifier_base))]#e'_i 
    L_pri=[0 for i in range(0,len(classifier_base))]#l'_i 
    Err=[0.0 for i in range(0,len(classifier_base))]#e_i 
    for i in range(0,len(classifier_base)): 
        Sample[i]=bootstrap_resample(Labeled_F) #L_i <-- Bootstrap(L)
        Sample[i]=np.array(Sample[i])
        classifier_base[i].train(Sample[i])
        Err_pri[i]=0.5#e'_i <-- .5
        L_pri[i]=0#l'_i <-- 0
    iteration=0
    stotal=[1 for i in range(0,len(classifier_base))]
    acc_detail[ins_count][iteration]=accracy(stotal,classifier_base,Test)
    print iteration,acc_detail[ins_count][iteration]
    #endf
    #repeat until none of h_i ( i \in {1...3} ) changes 
    changed_total=True
    while changed_total :
        changed_total=False
        L_add=[]
        for i in range(0,len(classifier_base)):
            L_add.append([])
        tmp_add=[0.0 for i in range(0,53)]
        #L_add=np.array(L_add)
        S_add=[]
        for i in range(0,len(classifier_base)):
            S_add.append([])
        count_L=[0 for i in range(0,len(classifier_base))]
        updated_base=[0 for i in range(0,len(classifier_base))]
        for i in range(0,len(classifier_base)):                   
            updated_base[i]=False
            Err[i]=MeasureError(classifier_base,Labeled_F,i)     
            if Err[i]< Err_pri[i]:           
                #Unlabeled_F1 
                prediction=[[0 for j in range(0,len(Unlabeled_F))]for k in range(0,len(classifier_base))]
                for j in range(0,len(classifier_base)):
                    prediction[j]=classifier_base[j].predict_lab(Unlabeled_F)
                for j in range(0,len(Unlabeled_F)):
                    tmp=[0 for m in range(0,7)]
                    tmp2=[0 for m in range(0,7)]
                    for k in range(1,len(classifier_base)):
                        tmp[int(prediction[(i+k)%len(classifier_base)][j])-1]+=1
                        tmp2=tmp[:]
                        tmp2.sort(reverse=True)
                    for k in range(0,7):
                        if tmp2[k]>len(classifier_base)*0.8:
                            if (tmp.index(tmp2[k])+1)!=prediction[i][j]:
                                tmp_add[:]=Unlabeled_F[j][:]
                                tmp_add[-1]=tmp.index(tmp2[k])+1
                                L_add[i].append(tmp_add)
                                count_L[i]+=1
                print count_L[i]
                if  L_pri[i]==0:
                    L_pri[i]=int(math.floor((float(Err[i])/(Err_pri[i]-Err[i])+1)))
                if L_pri[i]<count_L[i]:
                    if (Err[i]*count_L[i])<(Err_pri[i]*L_pri[i]):
                        updated_base[i]=True
                    elif L_pri[i]>(float(Err[i])/(Err_pri[i]-Err[i])):
                        S_add[i]=random.sample(L_add[i][0:count_L[i]],int(math.ceil(Err_pri[i]*L_pri[i]/float(Err[i])-1)))
                        updated_base[i]=True
        for i in range(0,len(classifier_base)):
            if updated_base[i]==True:
                L_tmp=list(Labeled_F)+list(S_add[i])
                L_tmp=np.array(L_tmp)
                classifier_base[i].train(L_tmp)
                Err_pri[i]=Err[i]
                L_pri[i]=len(S_add[i])
                changed_total=True
        iteration+=1
        acc_detail[ins_count][iteration]=accracy(stotal,classifier_base,Test)
        print iteration,acc_detail[ins_count][iteration] 
		
    #ensemble pruning 
	
	#0.8-3-acc-div
    Pruning=[]
    Sub_base=[0 for i in range(0,len(classifier_base))]
    for i in range(0,len(classifier_base)):
        Sub_base[i]=random.randint(0,1)
    while size(Sub_base)<=1:
        for i in range(0,len(classifier_base)):
            Sub_base[i]=random.randint(0,1)
    Pruning.append(Sub_base)
    Pupdated=True
    count_p=0
    count1=0
    while count1<len(Pruning):
        count_p+=1
        Pupdated=False
        select_num=random.randint(0,len(Pruning)-1)
        S_select=Pruning[select_num]
        S_new=S_select[:]
        change_bit=random.randint(0,len(classifier_base)-1)
        S_new[change_bit]=(S_select[change_bit]+1)%2
        while size(S_new)<=0:
            change_bit=random.randint(0,len(classifier_base)-1)
            S_new[change_bit]=(S_select[change_bit]+1)%2
        flag=0
        for i in range(0,len(Pruning)):
            if domain(Pruning[i],S_new,classifier_base,Labeled_F)==1:
                flag=1
                break
        count1+=1
        if flag==0:
            count1=0
            P_new=[]
            P_new.append(S_new)
            for i in range(0,len(Pruning)):
                if domain(S_new,Pruning[i],classifier_base,Labeled_F)==-1:
                    P_new.append(Pruning[i])
            Pruning=P_new
            Pupdated=True
    #select the best and compute the accuracy
    print count_p,len(Pruning)
    S_final=Pruning[0]
    for j in range(1,len(Pruning)):
        if accracy(Pruning[j],classifier_base,Labeled_F)>accracy(S_final,classifier_base,Labeled_F):
            S_final=Pruning[j]
    S_detail.append(S_final)
    #compute the accracy
    acc_detail[ins_count][iteration+1]=accracy(S_final,classifier_base,Test)
    print acc_detail[ins_count][iteration+1],S_final
	#0.8-3-div
	S_final=Pruning[0]
    for j in range(1,len(Pruning)):
        if diversity(Pruning[j],classifier_base,Labeled_F)>diversity(S_final,classifier_base,Labeled_F):
            S_final=Pruning[j]
    S_detail.append(S_final)
    #compute the accracy
    acc_detail[ins_count][iteration+2]=accracy(S_final,classifier_base,Test)
    print acc_detail[ins_count][iteration+2],S_final
	#0.8-2-acc
	Pruning=[]
    Sub_base=[0 for i in range(0,len(classifier_base))]
    for i in range(0,len(classifier_base)):
        Sub_base[i]=random.randint(0,1)
    while size(Sub_base)<=1:
        for i in range(0,len(classifier_base)):
            Sub_base[i]=random.randint(0,1)
    Pruning.append(Sub_base)
    Pupdated=True
    count_p=0
    count1=0
    while count1<len(Pruning):
        count_p+=1
        Pupdated=False
        select_num=random.randint(0,len(Pruning)-1)
        S_select=Pruning[select_num]
        S_new=S_select[:]
        change_bit=random.randint(0,len(classifier_base)-1)
        S_new[change_bit]=(S_select[change_bit]+1)%2
        while size(S_new)<=0:
            change_bit=random.randint(0,len(classifier_base)-1)
            S_new[change_bit]=(S_select[change_bit]+1)%2
        flag=0
        for i in range(0,len(Pruning)):
            if domain1(Pruning[i],S_new,classifier_base,Labeled_F)==1:
                flag=1
                break
        count1+=1
        if flag==0:
            count1=0
            P_new=[]
            P_new.append(S_new)
            for i in range(0,len(Pruning)):
                if domain1(S_new,Pruning[i],classifier_base,Labeled_F)==-1:
                    P_new.append(Pruning[i])
            Pruning=P_new
            Pupdated=True
    #select the best and compute the accuracy
    print count_p,len(Pruning)
    S_final=Pruning[0]
    for j in range(1,len(Pruning)):
        if accracy(Pruning[j],classifier_base,Labeled_F)>accracy(S_final,classifier_base,Labeled_F):
            S_final=Pruning[j]
    S_detail.append(S_final)
    #compute the accracy
    acc_detail[ins_count][iteration+3]=accracy(S_final,classifier_base,Test)
    print acc_detail[ins_count][iteration+3],S_final
	#0.8-acc
	Pruning=[]
    Sub_base=[0 for i in range(0,len(classifier_base))]
    for i in range(0,len(classifier_base)):
        Sub_base[i]=random.randint(0,1)
    while size(Sub_base)<=1:
        for i in range(0,len(classifier_base)):
            Sub_base[i]=random.randint(0,1)
    Pruning.append(Sub_base)
    Pupdated=True
    count_p=0
    count1=0
    while count1<len(Pruning):
        count_p+=1
        Pupdated=False
        select_num=random.randint(0,len(Pruning)-1)
        S_select=Pruning[select_num]
        S_new=S_select[:]
        change_bit=random.randint(0,len(classifier_base)-1)
        S_new[change_bit]=(S_select[change_bit]+1)%2
        while size(S_new)<=0:
            change_bit=random.randint(0,len(classifier_base)-1)
            S_new[change_bit]=(S_select[change_bit]+1)%2
        flag=0
        for i in range(0,len(Pruning)):
            if domain2(Pruning[i],S_new,classifier_base,Labeled_F)==1:
                flag=1
                break
        count1+=1
        if flag==0:
            count1=0
            P_new=[]
            P_new.append(S_new)
            for i in range(0,len(Pruning)):
                if domain2(S_new,Pruning[i],classifier_base,Labeled_F)==-1:
                    P_new.append(Pruning[i])
            Pruning=P_new
            Pupdated=True
    #select the best and compute the accuracy
    print count_p,len(Pruning)
    S_final=Pruning[0]
    S_detail.append(S_final)
    #compute the accracy
    acc_detail[ins_count][iteration+4]=accracy(S_final,classifier_base,Test)
    print acc_detail[ins_count][iteration+4],S_final
	#0.8-div
	Pruning=[]
    Sub_base=[0 for i in range(0,len(classifier_base))]
    for i in range(0,len(classifier_base)):
        Sub_base[i]=random.randint(0,1)
    while size(Sub_base)<=1:
        for i in range(0,len(classifier_base)):
            Sub_base[i]=random.randint(0,1)
    Pruning.append(Sub_base)
    Pupdated=True
    count_p=0
    count1=0
    while count1<len(Pruning):
        count_p+=1
        Pupdated=False
        select_num=random.randint(0,len(Pruning)-1)
        S_select=Pruning[select_num]
        S_new=S_select[:]
        change_bit=random.randint(0,len(classifier_base)-1)
        S_new[change_bit]=(S_select[change_bit]+1)%2
        while size(S_new)<=0:
            change_bit=random.randint(0,len(classifier_base)-1)
            S_new[change_bit]=(S_select[change_bit]+1)%2
        flag=0
        for i in range(0,len(Pruning)):
            if domain3(Pruning[i],S_new,classifier_base,Labeled_F)==1:
                flag=1
                break
        count1+=1
        if flag==0:
            count1=0
            P_new=[]
            P_new.append(S_new)
            for i in range(0,len(Pruning)):
                if domain3(S_new,Pruning[i],classifier_base,Labeled_F)==-1:
                    P_new.append(Pruning[i])
            Pruning=P_new
            Pupdated=True
    #select the best and compute the accuracy
    print count_p,len(Pruning)
    S_final=Pruning[0]
    S_detail.append(S_final)
    #compute the accracy
    acc_detail[ins_count][iteration+5]=accracy(S_final,classifier_base,Test)
    print acc_detail[ins_count][iteration+5],S_final
	
t1=time.clock()
t2=t1-t0
print t2
for i in range(0,11):
    acc_detail[6][i]=(acc_detail[0][i]*2924.0+acc_detail[1][i]*1241.0+acc_detail[2][i]*3002.0+acc_detail[3][i]*2987.0+acc_detail[4][i]*2997.0+acc_detail[5][i]*718.0)/13869.0
    acc_detail[7][i]=(acc_detail[0][i]*2924.0+acc_detail[2][i]*3002.0+acc_detail[3][i]*2987.0+acc_detail[4][i]*2997.0)/11910.0
tmp1=Str_name+'.txt'
np.savetxt(tmp1,acc_detail,fmt="%f")
tmp2=Str_name+'S_detail.txt'
np.savetxt(tmp2,S_detail,fmt="%f")