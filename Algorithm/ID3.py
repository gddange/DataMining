#coding:utf-8
import pandas as pd 
import cmath
import testplottree

data = [['youth','high','no','fair','no'],
	        ['youth','high','no','excellent','no'],
	        ['middle_age','high','no','fair','yes'],
	        ['senior','medium','no','fair','yes'],
	        ['senior','low','yes','fair','yes'],
	        ['senior','low','yes','excellent','no'],
	        ['middle_age','low','yes','excellent','yes'],
	        ['youth','medium','no','fair','no'],
	        ['youth','low','yes','fair','yes'],
	        ['senior','medium','yes','fair','yes'],
	        ['youth','medium','yes','excellent','yes'],
	        ['middle_age','medium','no','excellent','yes'],
	        ['middle_age','high','yes','fair','yes'],
	        ['senior','medium','no','excellent','no']
	]

attribute_list = ['age','income','student','credit_rating','buy']

dataDF = pd.DataFrame(data,columns = attribute_list)

def cacaulategaininfo(x,y,z):
	return z*(-1*x*cmath.log(x,2)-y*cmath.log(y,2))

def gain_info(attr_count,lenDj):
	infos = {}
	for item in attr_count:
		for key,attr in item.items():
			attr_matrix = attr.unstack()
			attr_matrix = attr_matrix.fillna(value = 0)
			attrsum = attr_matrix.sum(1).tolist()
			attr_matrix = attr_matrix.apply(lambda x:x/attr_matrix.sum(1))   #得到百分数函数
			attr_matrix['sum'] = attrsum
			attr_matrix['sum'] = attr_matrix['sum']/lenDj    #接下来就只用对dataframe进行数学计算
			result = attr_matrix.apply(lambda row: cacaulategaininfo(row['yes'],row['no'],row['sum']),axis = 1)
			result = result.fillna(value = 0)
			result = result.sum()
			infos[key] = result   #计算得每个属性的gain_info存入dict中
	return infos

def attribute_selection(Dj,attribute_list):
	'''根据具体的数据集以及其属性列表来对其进行分支,扫描属性，分别计算增益率，选择信息增益最大的那个'''
	yes = Dj['buy'].value_counts()['yes']
	no = Dj['buy'].value_counts()['no']
	lenDj = float(len(Dj))
	yes = yes/lenDj
	no = no/lenDj
	attr_count = []
	infoD = -(yes*cmath.log(yes,2)+no*cmath.log(no,2))
	for attr in attribute_list[:-1]:
		attrDict = {}
		attrDf = Dj.groupby([attr,'buy']).size()
		attrDict[attr] = attrDf
		attr_count.append(attrDict)
	infos = gain_info(attr_count,lenDj)
	for key,value in infos.items():
		infos[key] = infoD - value
	infos = sorted(infos.items(),key = lambda item:item[1],reverse = True)
	return list(infos)[0][0]        #获取gaininfo最大的属性，用其来进行split

def split_attribtue(Dj,attr,Sa = None):
	'''根据attr的具体值进行split'''
	DJs = {}
	DJ = {}
	if len(Dj[attr].unique()) == len(Dj):
		attrlist = Dj[attr].tolist()
		attrlist.sort()
		splitPoint = (attrlist[len(attrlist/2)] + attrlist[(len(attrlist/2))+1])/2
		DJ['lt'+splitPoint] = Dj[Dj[attr]<=splitPoint]
		DJ['gt'+splitPoint] = Dj[Dj[attr]>splitPoint]
		DJs[attr] = DJ
	else:
		if Sa == None:
			attrvalues = list(Dj[attr].unique())
			for value in attrvalues:
				DJ[value] = Dj[Dj[attr] == value]
			DJs[attr] = DJ
		else:
			DJ['isin'+Sa] = Dj[Dj[attr].isin(Sa)]
			DJ['notin'+Sa] = Dj[~Dj[attr].isin(Sa)]
			DJs[attr] = DJ
	return DJs

def createTree(rootDj,attrlist):
	'''递归生成决策树'''
	tree = {}     #表示Tree的字典
	firstAttr = list(rootDj.keys())[0]   #Tree的根节点
	valueDict = rootDj[firstAttr]
	for key in valueDict.keys():
		childtree = {}
		if len(attrlist) == 1:
			continue
		if len(valueDict[key]['buy'].unique()) == 1:
			tree[key] = valueDict[key]['buy'].unique()[0]
		else:
			sortAttr = attribute_selection(valueDict[key],attrlist)
			rootDj = split_attribtue(valueDict[key],sortAttr)
			sort_attrlist = attrlist[:attrlist.index(sortAttr)] + attrlist[attrlist.index(sortAttr)+1:]
			childtree[sortAttr] = createTree(rootDj,sort_attrlist)
			tree[key] = childtree
	return tree

def ID3():
	'''这个函数需要一边split一边select一边再生成决策树'''
	rootTree = {}
	sortAttr = attribute_selection(dataDF,attribute_list)
	rootDj = split_attribtue(dataDF,sortAttr)
	sort_attrlist = attribute_list[:attribute_list.index(sortAttr)] + attribute_list[attribute_list.index(sortAttr)+1:]
	rootTree[sortAttr] = createTree(rootDj,sort_attrlist)
	testplottree.creatPlot(rootTree)

if __name__ == '__main__':
	ID3()









