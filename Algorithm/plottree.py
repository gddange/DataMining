#coding:utf-8
import matplotlib.pyplot as plt 

decisionNode = dict(boxstyle='sawtooth',fc='0.8')
leafNode = dict(boxstyle='round4',fc='0.8')
arrow_args = dict(arrowstyle='<-')

def plotNode(Nodename, centerPt, parentPt, nodeType):  #  centerPt箭头坐标  parentPt箭尾坐标
    creatPlot.ax1.annotate(Nodename, xy=parentPt, xycoords='axes fraction', xytext=centerPt, textcoords='axes fraction', va="center", ha="center", bbox=nodeType, arrowprops=arrow_args)

def getNumLeafs(myTree):
	'''知道有多少个叶子节点，便于可以知道x轴的宽度'''
	numLeafs = 0
	firstStr = list(myTree.keys())[0]
	secondDict = myTree[firstStr]
	for key in secondDict.keys():
		#以下三行测试节点的数据类型是否为字典
		if type(secondDict[key]).__name__=='dict':
			numLeafs +=getNumLeafs(secondDict[key])
		else:
			numLeafs +=1
	return numLeafs

def getTreeDepth(myTree):
	'''知道tree有多少层，以便知道y轴有多高'''
	maxDepth = 0
	firstStr = list(myTree.keys())[0]
	secondDict = myTree[firstStr]
	for key in secondDict.keys():
		if type(secondDict[key]).__name__ == 'dict':
			thisDepth = 1+ getTreeDepth(secondDict[key])
		else:
			thisDepth = 1
		if thisDepth > maxDepth:
			maxDepth = thisDepth
	return maxDepth

def plotMidText(cntrPt,parentPt,txtString):
	xMid = (parentPt[0] - cntrPt[0]) / 2.0 + cntrPt[0]
	yMid = (parentPt[1] - cntrPt[1])/2.0 + cntrPt[1]
	creatPlot.ax1.text(xMid,yMid,txtString,va='center',ha='center',rotation=30)

def plotTree(myTree,parentPt,nodeTxt):
	numLeafs = getNumLeafs(myTree)
	depth = getTreeDepth(myTree)
	firstStr = list(myTree.keys())[0]
	cntrPt = (plotTree.xOff + (1.0 + float(numLeafs))/2.0/plotTree.totalW,plotTree.yOff)
	plotMidText(cntrPt,parentPt,nodeTxt)
	plotNode(firstStr,cntrPt,parentPt,decisionNode)
	secondDict = myTree[firstStr]
	plotTree.yOff = plotTree.yOff - 1.0/plotTree.totalD
	for key in secondDict.keys():
		if type(secondDict[key]).__name__ == 'dict':
			plotTree(secondDict[key],cntrPt,str(key))
		else:
			plotTree.xOff = plotTree.xOff + 1.0/plotTree.totalW
			plotNode(secondDict[key],(plotTree.xOff,plotTree.yOff),cntrPt,leafNode)
			plotMidText((plotTree.xOff,plotTree.yOff),cntrPt,str(key))
	plotTree.yOff = plotTree.yOff + 1.0/plotTree.totalD

def creatPlot(inTree):
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    asprops = dict(xticks = [],yticks = [])
    creatPlot.ax1 = plt.subplot(111,frameon = False,**asprops)
    plotTree.totalW = float(getNumLeafs(inTree))
    plotTree.totalD = float(getTreeDepth(inTree))
    plotTree.xOff = -0.5 / plotTree.totalW;plotTree.yOff = 1.0;
    plotTree(inTree,(0.5,1.0),'')
    plt.show()

def retrieveTree(i):
	listOfTrees =[{'no surfacing': {0: 'no', 1: {'flippers':{0: 'no', 1: 'yes'}}}},{'no surfacing': {0: 'no', 1: {'flippers':{0: {'head': {0: 'no', 1: 'yes'}}, 1: 'no'}}}}]
	return listOfTrees[i]


if __name__ == '__main__':
	myTree = retrieveTree(0)
	myTree['no surfacing'][3] = 'maybe'
	creatPlot(myTree)