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
weather_name='data/weather.txt'
time_name='data/time.txt'

aqi=[[] for i in range(0,len(aqi_name))]
for i in range(0,len(aqi_name)):
	aqi[i]=np.loadtxt(aqi_name[i])
weather=np.loadtxt(weather_name)

for i in range(0,len(weather)):
	if weather[i][3]>365:
		weather[i][3]=365
np.savetxt('weathernew.txt',weather,fmt="%f")