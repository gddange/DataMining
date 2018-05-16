#coding:utf-8
'''
Apriori算法的两个输入参数分别是最小支持度和数据集。
Ck是Lk的超集，其中Ck中的项可以是频繁的也可以是非频繁的，而Lk中的项则必然是频繁的
所以需要有两个数据类，一个用于存储Ck和Lk,另一个则用于存储Ck以及其支持度
'''

#msupport = input("please input you minmal support: ")  #最小支持度，用于筛选项集

def loadData():
	'''
	具体实现的时候可以从文本中读取数据
	'''
	return [[1,2,5],[2,4],[2,3],[1,2,4],[1,3],[2,3],[1,3],[1,2,3,5],[1,2,3]]

def createC1(dataSet):
	C1 = []    #为了统一，因为在后来的C2,C3...中，每一项都是一个list，所以现在这里list的项也应该是list
	for transaction in dataSet:
		for item in transaction:
			if [item] not in C1:
				C1.append([item])
	C1.sort()
	return list(map(frozenset,C1))  #这样返回的是list，但是list中每个元素都是set,保证元素不重复

#连接L(k-1)产生Ck,这里需要return Ck并压缩Ck的长度
def createCn(L_origin,k):
	Ck = []
	for i in range(len(L_origin)):
		for li in L_origin[i+1:]:
			list1 = list(L_origin[i])
			list2 = list(li)
			list1.sort()
			list2.sort()
			if list1[:k-2] == list2[:k-2]:
				newCk = frozenset(list1+list2)
				if(len(L_origin[0])>1): 
					for tid in L_origin[L_origin.index(li):]:
						myset = (list1[-1],list2[-1])       #压缩Ck的大小，若是Ck 的k-1项子集有不存在于L中，则也不会是频繁集，剔除
						if frozenset(myset).issubset(tid):
							Ck.append(newCk)
				else:
					Ck.append(newCk)
	return Ck

#获取Ck的支持度
def create_supportData(Ck,dataSet):
	supportData = {}
	for item in Ck:
		for tid in dataSet:
			if item.issubset(tid):
				if supportData.get(item):
					supportData[item] +=1
				else:
					supportData[item] = 1
	return supportData

#对Ck筛选得到Lk
def filterCk(supportData,lenD,support=0.5):
	lenD = float(lenD)
	Lk = []
	supportLkData = {}
	for key,value in supportData.items():
		if value/lenD >= support:
			Lk.append(key)
			supportLkData[key] = value
	return Lk,supportLkData

#获取所有的频繁项集合
def apriori(dataSet,support = 0.5):
	lenD = len(dataSet)
	C1 = createC1(dataSet)
	supportData = create_supportData(C1,dataSet)
	Lf,supportLkData = filterCk(supportData,lenD,support = support)
	L1 = len(Lf)
	L = []   #总的频繁集
	k = 2
	supportData = {}
	L.append(Lf)
	while(k<=L1):
		Ck = createCn(Lf,k)
		supportLk = create_supportData(Ck,dataSet)
		Lk,tempSpData= filterCk(supportLk,lenD,support = support)
		L.append(Lk)
		supportData.update(supportLk)
		supportLkData.update(tempSpData)
		supportLk = create_supportData(Ck,dataSet)
		Lf = Lk
		k +=1
	return L,supportData,supportLkData

#计算置信度，函数需要输入A和B，输出A和B的置信度
def get_confidence(A,B,supportData):
	if isinstance(A,int):
		A = (A,)
	if isinstance(B,int):
		B = (B,)
	setAB=(list(A)+list(B))  #刚好set转为list的时候会忽略最后一个逗号
	valueA = None
	valueB = None
	for key,value in supportData.items():
		if key == frozenset(A):
			valueA = float(value)
			break
	for key,value in supportData.items():
		if key == frozenset(setAB):
			valueB = value
			break
	if(valueA != None and valueB != None):
		return valueB/valueA
	else:return 0

if __name__ == '__main__':
	data = loadData()
	L,supportData,supportLkData = apriori(data,support = 0.22)
	print('The L is: \n',L)
	print('The support is: \n',supportData)
	print('larger than support is: \n',supportLkData)
	print('The confidence of (1,2)=>5 is :',get_confidence((1,2),(5),supportLkData))



