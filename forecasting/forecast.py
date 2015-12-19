#coding:utf-8
import string
import numpy as np
import pandas as pd
import time
import random
import math
import os
import sklearn
from sklearn import tree
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import linear_model, datasets, metrics
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD

#定义时序分类器
class TemClassifier():
	def _init_(self,classifier):
		self.myclassfier=classifier
	def train(self,traindata):
		#towrite
	def predict(self,testdata):
		#towrite

#定义空间分类器
class SpaClassifier():
	def _init_(self,classifier):
		self.myclassfier=classifier
		self.myclassfier=self.myclassfier.add(Dense(64, input_dim=20, init='uniform'))
		self.myclassfier=self.myclassfier.add(Activation('tanh'))
		self.myclassfier=self.myclassfier.add(Dropout(0.5))
		self.myclassfier=self.myclassfier.add(Dense(64, init='uniform'))
		self.myclassfier=self.myclassfier.add(Activation('tanh'))
		self.myclassfier=self.myclassfier.add(Dropout(0.5))
		self.myclassfier=self.myclassfier.add(Dense(2, init='uniform'))
		self.myclassfier=self.myclassfier.add(Activation('softmax'))
	def train(self,traindata):
		sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
		self.myclassfier=self.myclassfier.compile(loss='mean_squared_error', optimizer=sgd)
		#self.myclassfier=self.myclassfier.fit(traindata_X, traindata_Y, nb_epoch=20, batch_size=16)
	def predict(self,testdata):
		#towrite

#定义聚合分类器
class AGClassifier():
	def _init_(self,classifier):
		self.myclassfier=classifier
	def train(self,traindata):
		#towrite
	def predict(self,testdata):
		#towrite

#读取文件数据
def ReadFile(i):
	aqi=np.loadtxt(aqi_name[i])
	spatial=np.loadtxt(spatial_name[i])
	weather=np.loadtxt(weather_name)
	forecast=np.loadtxt(forecast_name)
	time=np.loadtxt(time_name)


acc=[0.0 for i in range(0,11)]
size=[0 for i in range(0,11)]

#array or list 
aqi=[]
weather=[]
forecast=[]
spatial=[]
time=[]

#file name and path
aqi_name=['data/a37.txt','data/a38.txt','data/a39.txt','data/a40.txt','data/a41.txt','data/a42.txt','data/a43.txt','data/a44.txt','data/a45.txt','data/a46.txt','data/a47.txt']
spatial_name=['data/sp1837.txt','data/sp1838.txt','data/sp1839.txt','data/sp1840.txt','data/sp1841.txt','data/sp1842.txt','data/sp1843.txt','data/sp1844.txt','data/sp1845.txt','data/sp1846.txt','data/sp1847.txt']
weather_name='data/weather.txt'
forecast_name='data/forecast.txt'
time_name='data/time.txt'
#对于每一个监测站点
for i in range(0,11):
	#读取数据
	readfile(i)
	#读取时序分类器数据:过去3小时的aqi和weather以及当前时间
	T_train=np.hstack((aqi[0:3000,:],aqi[1:3001,:],aqi[2:3002,:],aqi[3:3003,:],weather[3:3003,:],time[3:3003,:]))#aqi,weather,time
	T_test=np.hstack((aqi[3000:4400,:],aqi[3001:4401,:],aqi[3002:4402,:],aqi[3003:4403,:],weather[3003:4403,:],time[3003:4403,:]))
	#读取空间分类器数据:过去3小时周围站点的aqi和weather以及当前时间和当前时间站点的aqi
	S_train_f=hstack((spatial[0:3000,:],spatial[1:3001,:],spatial[2:3002,:],spatial[3:3003,:],aqi[3:3003,-1:]))
	S_test_f=hstack((spatial[3000:4400,:],spatial[3001:4401,:],spatial[3002:4402,:],spatial[3003:4403,:],aqi[3003:4403,-1:]))
	#定义时序分类器
	tc=[TemClassifier(linear_model.LinearRegression()),TemClassifier(linear_model.LinearRegression()),TemClassifier(linear_model.LinearRegression()),TemClassifier(linear_model.LinearRegression()),TemClassifier(linear_model.LinearRegression()),
		TemClassifier(linear_model.LinearRegression()),TemClassifier(linear_model.LinearRegression()),TemClassifier(linear_model.LinearRegression()),TemClassifier(linear_model.LinearRegression()),TemClassifier(linear_model.LinearRegression()),
		TemClassifier(linear_model.LinearRegression()),TemClassifier(linear_model.LinearRegression()),TemClassifier(linear_model.LinearRegression()),TemClassifier(linear_model.LinearRegression()),TemClassifier(linear_model.LinearRegression()),
		TemClassifier(linear_model.LinearRegression()),TemClassifier(linear_model.LinearRegression()),TemClassifier(linear_model.LinearRegression()),TemClassifier(linear_model.LinearRegression()),TemClassifier(linear_model.LinearRegression()),
		TemClassifier(linear_model.LinearRegression()),TemClassifier(linear_model.LinearRegression()),TemClassifier(linear_model.LinearRegression()),TemClassifier(linear_model.LinearRegression())]
	#定义空间分类器
	sc=[SpaClassifier(Sequential()),SpaClassifier(Sequential()),SpaClassifier(Sequential()),SpaClassifier(Sequential()),SpaClassifier(Sequential()),
		SpaClassifier(Sequential()),SpaClassifier(Sequential()),SpaClassifier(Sequential()),SpaClassifier(Sequential()),SpaClassifier(Sequential()),
		SpaClassifier(Sequential()),SpaClassifier(Sequential()),SpaClassifier(Sequential()),SpaClassifier(Sequential()),SpaClassifier(Sequential()),
		SpaClassifier(Sequential()),SpaClassifier(Sequential()),SpaClassifier(Sequential()),SpaClassifier(Sequential()),SpaClassifier(Sequential()),
		SpaClassifier(Sequential()),SpaClassifier(Sequential()),SpaClassifier(Sequential()),SpaClassifier(Sequential())]
	#定义聚合分类器-回归树
	pa=[AGClassifier(tree.DecisionTreeRegressor()),AGClassifier(tree.DecisionTreeRegressor()),AGClassifier(tree.DecisionTreeRegressor()),AGClassifier(tree.DecisionTreeRegressor()),AGClassifier(tree.DecisionTreeRegressor()),
		AGClassifier(tree.DecisionTreeRegressor()),AGClassifier(tree.DecisionTreeRegressor()),AGClassifier(tree.DecisionTreeRegressor()),AGClassifier(tree.DecisionTreeRegressor()),AGClassifier(tree.DecisionTreeRegressor()),
		AGClassifier(tree.DecisionTreeRegressor()),AGClassifier(tree.DecisionTreeRegressor()),AGClassifier(tree.DecisionTreeRegressor()),AGClassifier(tree.DecisionTreeRegressor()),AGClassifier(tree.DecisionTreeRegressor()),
		AGClassifier(tree.DecisionTreeRegressor()),AGClassifier(tree.DecisionTreeRegressor()),AGClassifier(tree.DecisionTreeRegressor()),AGClassifier(tree.DecisionTreeRegressor()),AGClassifier(tree.DecisionTreeRegressor()),
		AGClassifier(tree.DecisionTreeRegressor()),AGClassifier(tree.DecisionTreeRegressor()),AGClassifier(tree.DecisionTreeRegressor()),AGClassifier(tree.DecisionTreeRegressor())]
	#训练时序分类器
	for j in range(0,len(tc)):
		tc[j].train(np.hstack((T_train,forcast[3:3003,(3*(j/6)):(3+3*(j/6))],aqi[(4+j):(3004+j),-1:])))
	#训练空间分类器
	for j in range(0,len(sc)):
		sc[i].train(np.hstack((S_train,aqi[(4+j):(3004+j),-1:])))
	#训练聚合分类器

	for j in range(0,len(pa)):
		pa[j].train(np.hstack(((aqi[3:3003,-1:]-aqi[(4+j):(3004+j),-1:]),(aqi[3:3003,-1:]-aqi[(4+j):(3004+j),-1:]),weather[3:3003,:],(aqi[3:3003,-1:]-aqi[(4+j):(3004+j),-1:]))))
	#训练特殊情况分类器

	#时序分类器对测试数据进行分类
	T_result=[(0.0 for k in range(0,1400)) for l in range(0,len(tc))]
	DT_result=[(0.0 for k in range(0,1400)) for l in range(0,len(tc))]
	for j in range(0,len(tc)):
		T_result[j]=tc[j].predict(np.hstack((T_train,forcast[3003:4403,(3*(j/6)):(3+3*(j/6))])))
		DT_result[j]=aqi[3003:4403,-1:]-T_result[j]
	#计算时序结果的delt值
	#空间分类器对测试数据进行分类
	S_result=[(0.0 for k in range(0,1400)) for l in range(0,len(sc))]
	DS_result=[(0.0 for k in range(0,1400)) for l in range(0,len(sc))]
	for j in range(0,len(sc)):
		S_result[j]=sc[j].predict(S_test)
		DS_result[j]=aqi[3003:4403,-1:]-S_result[j]
	#分类聚合器对测试结果聚合
	AR_result=[0.0 for k in range(0,len(T_R))]
	for j in range(0,len(pa)):
		AR_result[j]=pa[j].predict(np.hstack((DT_result[j],DS_result[j],weather[3003:4403,:])))
	#特殊情况结果整合
		
	#计算错误率和精确度
