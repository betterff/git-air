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


aqi_name=['data/a37.txt','data/a38.txt','data/a39.txt','data/a40.txt','data/a41.txt','data/a42.txt','data/a43.txt','data/a44.txt','data/a45.txt','data/a46.txt','data/a47.txt']
weather_name='weathernew.txt'
time_name='data/time.txt'

aqi=[[] for i in range(0,len(aqi_name))]
for i in range(0,len(aqi_name)):
	aqi[i]=np.loadtxt(aqi_name[i])
weather=np.loadtxt(weather_name)

weatherd=[]
weathero=[]
count=0.0
for i in range(0,3000):
	flag=0
	for j in range(0,len(aqi_name)):
		if aqi[j][i][-1]>=200 and aqi[j][i+2][-1]<100:
			flag=1
	if flag==1:
		weatherd.append(weather[i])
		count+=1.0
	else:
		weathero.append(weather[i])
#print count
np.savetxt('distridata.txt',weatherd,fmt="%f")
print weatherd 
print len(weatherd)
print count

#计算每个特征的范围
rg=[[0.0 for i in range(0,2)] for j in range(0,5)]
for i in range(0,3000):
	for j in range(0,5):
		if rg[j][0]>weather[i][j]:
			rg[j][0]=weather[i][j]
		if rg[j][1]<weather[i][j]:
			rg[j][1]=weather[i][j]
np.savetxt('range.txt',rg,fmt="%f")
#计算分布
sum1=int(math.ceil(rg[0][1])-math.floor(rg[0][0])+1)
sum2=int(math.ceil(rg[1][1])-math.floor(rg[1][0])+1)
sum3=int(math.ceil(rg[2][1])-math.floor(rg[2][0])+1)
sum4=int(math.ceil(rg[3][1])-math.floor(rg[3][0])+1)
sum5=int(math.ceil(rg[4][1])-math.floor(rg[4][0])+1)
print sum1,sum2,sum3,sum4,sum5
tmp=[[0.0 for j in range(0,6)] for i in range(0,sum1)]
hum=[[0.0 for j in range(0,6)] for i in range(0,sum2)]
wnf=[[0.0 for j in range(0,6)] for i in range(0,sum3)]
wnd=[[0.0 for j in range(0,6)] for i in range(0,sum4)]
rain=[[0.0 for j in range(0,6)] for i in range(0,sum5)]
for i in range(0,sum1):
	tmp[i][0]=math.floor(rg[0][0])+i
for i in range(0,sum2):
	hum[i][0]=math.floor(rg[1][0])+i
for i in range(0,sum3):
	wnf[i][0]=math.floor(rg[2][0])+i
for i in range(0,sum4):
	wnd[i][0]=math.floor(rg[3][0])+i
for i in range(0,sum5):
	rain[i][0]=math.floor(rg[4][0])+i

for i in range(0,len(weatherd)):
	for j in range(0,sum1):
		if tmp[j][0]==math.floor(weatherd[i][0]):
			tmp[j][1]+=(1.0/count)
			tmp[j][4]+=1
	for j in range(0,sum2):
		if hum[j][0]==math.floor(weatherd[i][1]):
			hum[j][1]+=(1.0/count)
			hum[j][4]+=1
	for j in range(0,sum3):
		if wnf[j][0]==math.floor(weatherd[i][2]):
			wnf[j][1]+=(1.0/count)
			wnf[j][4]+=1
	for j in range(0,sum4):
		if wnd[j][0]==math.floor(weatherd[i][3]):
			wnd[j][1]+=(1.0/count)
			wnd[j][4]+=1
	for j in range(0,sum5):
		if rain[j][0]==math.floor(weatherd[i][4]):
			rain[j][1]+=(1.0/count)
			rain[j][4]+=1

for i in range(0,len(weathero)):
	for j in range(0,sum1):
		if tmp[j][0]==math.floor(weathero[i][0]):
			tmp[j][2]+=(1.0/2985)
			tmp[j][5]+=1
	for j in range(0,sum2):
		if hum[j][0]==math.floor(weathero[i][1]):
			hum[j][2]+=(1.0/2985)
			hum[j][5]+=1
	for j in range(0,sum3):
		if wnf[j][0]==math.floor(weathero[i][2]):
			wnf[j][2]+=(1.0/2985)
			wnf[j][5]+=1
	for j in range(0,sum4):
		if wnd[j][0]==math.floor(weathero[i][3]):
			wnd[j][2]+=(1.0/2985)
			wnd[j][5]+=1
	for j in range(0,sum5):
		if rain[j][0]==math.floor(weathero[i][4]):
			rain[j][2]+=(1.0/2985)
			rain[j][5]+=1

print 'tmp:'
for i in range(0,sum1):
	if tmp[i][1]>tmp[i][2]:
		tmp[i][3]=1
		print tmp[i][0],tmp[i][1],tmp[i][2],tmp[i][4],tmp[i][5]
print 'hum:'
for i in range(0,sum2):
	if hum[i][1]>hum[i][2]:
		hum[i][3]=1,
		print hum[i][0],hum[i][1],hum[i][2],hum[i][4],hum[i][5]
print 'wnf:'
for i in range(0,sum3):
	if wnf[i][1]>wnf[i][2]:
		wnf[i][3]=1
		print wnf[i][0],wnf[i][1],wnf[i][2],wnf[i][4],wnf[i][5]
print 'wnd:'
for i in range(0,sum4):
	if wnd[i][1]>wnd[i][2]:
		wnd[i][3]=1
		print wnd[i][0],wnd[i][1],wnd[i][2],wnd[i][4],wnd[i][5]
print 'rain:'
for i in range(0,sum5):
	if rain[i][1]>rain[i][2]:
		rain[i][3]=1
		print rain[i][0],rain[i][1],rain[i][2],rain[i][4],rain[i][5]
	




