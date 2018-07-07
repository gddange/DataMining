#coding:utf-8

import pandas as pd 
import numpy as np 
from numpy import *
import matplotlib.pyplot as plt 

'''读取数据'''
path = '/data/pokemon.csv'
data = pd.read_csv(path,engine='python')
data = data.set_index('Name')
newdata = data.drop(columns=['Legendary','#','Generation'])[:400]
water = newdata[newdata['Type 1'] == 'Water']
water.drop(columns=['Type 1','Type 2'],inplace = True)
normal = newdata[newdata['Type 1'] == 'Normal']
normal.drop(columns=['Type 1','Type 2'],inplace = True)

#元素个数
water_len = len(water)
normal_len = len(normal)
print('################### water_len is %d normal_len is %d'%(water_len,normal_len))

D = len(water.columns)
Pi = np.pi 

#均值向量
water_means = mat(water.mean())
normal_means = mat(normal.mean())

#协方差矩阵
water_cov = mat(water.cov())
normal_cov = mat(normal.cov())
common_cov = water_len/(water_len+normal_len)*water_cov+normal_len/(normal_len+water_len)*normal_cov

#正态分布函数
def func(means,covs,X):
	return 1/power(2*Pi,D/2)*power(linalg.det(covs),1/2)*exp(-1/2*(X-means)*covs.I*(X-means).T)

def Bayes(func1,func2):
	p_water = water_len/(water_len + normal_len)
	p_normal = normal_len/(normal_len+water_len)
	p = (func1*p_water)/(func1*p_water+func2*p_normal)
	#print('func1 is %f, func2 is %f,p_water is %f, p_normal is %f'%(func1,func2,p_water,p_normal))
	return p


'''用从训练集上学到的参数信息来到测试集上检测结果'''
test_data = data.drop(columns=['Legendary','#','Generation'])[400:]
test_water = test_data[test_data['Type 1'] == 'Water']
test_water.drop(columns=['Type 1','Type 2'],inplace = True)
test_normal = test_data[test_data['Type 1'] == 'Normal']
test_normal.drop(columns=['Type 1','Type 2'],inplace = True)

loss = 0
waterp_Y = []     #预测的分类
normalp_Y = []
water_Y = []    #真实的分类
normal_Y = []

def lossfunc(Y,predict_Y):
	loss = 0
	for i in range(len(Y)):
		if predict_Y[i] != Y[i]:
			loss +=1
	return loss

for i in range(len(test_water)):
	water_Y.append(0)
	X = mat(np.array(test_water.iloc[i]))
	test_p = Bayes(func(water_means,common_cov,X),func(normal_means,common_cov,X))
	if test_p > 0.5:
		waterp_Y.append(0)
	else:
		waterp_Y.append(1)

for i in range(len(test_normal)):
	normal_Y.append(1)
	X = mat(np.array(test_normal.iloc[i]))
	test_p = Bayes(func(water_means,common_cov,X),func(normal_means,common_cov,X))    #当两个分布使用同样的协方差矩阵之后预测准确率确实提高了
	if test_p < 0.5:
		normalp_Y.append(1)
	else:
		normalp_Y.append(0)

loss +=lossfunc(water_Y,waterp_Y)
loss +=lossfunc(normal_Y,normalp_Y)
print(len(test_water))
print(len(test_normal))
loss = float(loss)
print('loss is %f, acuracy is: %f'%(loss,1-loss/(len(test_water)+len(test_normal))))
#print('test数据的个数为: %d'%sumtest)
#print('预测正确率为: %f'%rate)









