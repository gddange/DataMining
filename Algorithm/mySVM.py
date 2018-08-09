#coding:utf-8

from numpy import *

#读取数据
def loadData(filename):
	dataMat = [];labelMat = []
	with open(filename) as tf:
		for line in tf.readlines():
			dataAttr = line.strip().split('\t')
			dataMat.append([float(dataAttr[0]),float(dataAttr[1])])
			labelMat.append(float(dataAttr[-1]))
	return dataMat,labelMat

#随机选择一个a和已经选定的a配对更新
def selectRand(i,m):
	j = i
	if (j == i):
		j = int(random.uniform(0,m))
	return j

#这个函数用于在smo里得到aj的时候要更新aj的范围，因为每个alpha都有一个约束，在0，c之间
def clipAlpha(aj,L,H):
	if aj < L:
		aj = L
	elif aj > H:
		aj = H
	return aj

#smo算法具体实现过程，需要的参数，C，数据，迭代次数
def smoSimple(dataMat,classLabels,C,toler,maxIter):
	iter = 0     #用于count的迭代次数
	dataMat = mat(dataMat);labelMat = mat(classLabels).T      #把数组元素转换为numpy中的matrix便于计算
	b = 0     #初始化b
	m,n = shape(dataMat)
	alphas= mat(zeros((m,1)))    #初始化alpha向量,给每个元素都填充值为0，具体的a个数则是与数据的个数相同
	while(iter < maxIter):
		alphaPairsChanged = 0         #用于计算alpha对是否进行了更新，如果一直没有根棍则迭代一定次数就循环
		for i in range(m):
			'''内层循环，每次循环就更新一对alpha'''
			fxi = float(multiply(alphas,labelMat).T * (dataMat*dataMat[i,:].T)) + b #这里multiply是对应元素相乘，乘了以后要转置是为了使得最后的乘积为一个数字
			Ei = fxi - float(labelMat[i])        #这是误差项，用来计算alpha
			if((labelMat[i]*Ei < -toler) and (alphas[i] < C)) or ((labelMat[i]*Ei > toler) and (alphas[i] > 0)):
				#得到了这个alpha要判断这个alpha是否需要优化，判断的原则在于这个alpha对应的数据是否满足svm的约束条件，不满足的时候才需要进行优化,满足的时候就alpha为0就可以满足KTT条件了
				j = selectRand(i,m)
				fxj = float(multiply(alphas,labelMat).T * (dataMat*dataMat[j,:].T)) + b
				Ej = fxj - float(labelMat[j])
				alphaIold = alphas[i].copy()
				alphaJold = alphas[j].copy()        #这里用copy()函数是因为，如果直接赋值的话，两个引用会指向同一个存储地址，于是就会导致更改一个另一个也会跟着变化
				#先更新alphaj ，要注意alphaj的新范围，是否超过了0，C的区间，然后对其进行调整
				if(labelMat[i] != labelMat[j]):
					L = max(0,alphas[j] - alphas[i])
					H = min(C,(C+alphas[j] - alphas[i]))
				elif(labelMat[j] == labelMat[i]):
					L = max(0,(alphas[j] + alphas[i] - C))
					H = min(C,(alphas[j] + alphas[i]))
				if (L == H):print("L==H");continue              #如果L==H则代表这个alpha已经不能进行优化了，重新选择一个alpha进行优化
				eta = 2*dataMat[i,:]*dataMat[j,:].T - dataMat[i,:]*dataMat[i,:].T - dataMat[j,:]*dataMat[j,:].T
				alphas[j] -= labelMat[j]*(Ei-Ej)/eta        #SMo算法里alpha的更新公式
				alphas[j] = clipAlpha(alphas[j],L,H)        #对alpha更新以后要注意新的是否超出了它的范围，如果超出了就进行调整
				if(abs(alphas[j] - alphaJold) < 0.00001):print("J is not moving enough!");continue      #如果这个alpha没有足够的更新，代表这两个alpha之间的差距不是很大，故此重新选择一个alpha进行更新
				alphas[i] +=labelMat[j]*labelMat[i]*(alphaJold - alphas[j])     #更新ai
				#更新b，更新的原理在于更新以后的ai或者aj要满足KKT条件，则ai或者aj要在界内，且满足fx*yi = 1，只有这样，a才不需要为0
				b1 = b - Ei- labelMat[i]*(alphas[i]-alphaIold)*dataMat[i,:]*dataMat[i,:].T - labelMat[j]*(alphas[j]-alphaJold)*dataMat[i,:]*dataMat[j,:].T
				b2 = b - Ej- labelMat[i]*(alphas[i]-alphaIold)*dataMat[i,:]*dataMat[j,:].T - labelMat[j]*(alphas[j]-alphaJold)*dataMat[j,:]*dataMat[j,:].T
				if((alphas[i] > 0) and (alphas[i] < C)): b = b1
				elif((alphas[j] > 0) and (alphas[j] < C)):b = b2
				else:
					b = (b1+b2)/2.0
				alphaPairsChanged +=1     #这一次有更新alpha，因此要加一
				print("iteration %d, for i: %d alphas changed: %d"%(iter,i,alphaPairsChanged))
		if(alphaPairsChanged == 0):
			iter +=1
		else:
			iter = 0
		print("iter is %d"%iter)
	return b,alphas

class optStruct:
	def __init__(self,dataMatIn,classLabels,C,toler):
		self.X = dataMatIn
		self.labelMat = classLabels
		self.C = C
		self.toler = toler
		self.m = shape(dataMatIn)[0]
		self.alphas = mat(zeros((self.m,1)))
		self.b = 0
		self.eCache = mat(zeros((self.m,2)))      #这个缓存E的数组，第一列是E是否有效的标志，第二列是E的实际值

def calcEk(oS,k):
	'''计算os这个数据集的第k个数据的函数值以及预测误差'''
	fxk = float(multiply(oS.alphas,oS.labelMat).T*(oS.X*oS.X[k,:].T)) + oS.b
	Ek = fxk - float(oS.labelMat[k])
	return Ek

def selectJ(i,oS,Ei):
	'''比起随机的选择第二个alpha来进行更新，这里选择的是步长最大的alpha，其实就是两者之间差距最大的alpha，这代表两个点之间差距很大'''
	maxK = -1;maxDeltaE = 0;Ej = 0
	oS.eCache[i] = [1,Ei]        #将第i个数据的E设置为有效的，有效即代表它已经计算好了
	validEcacheList = nonzero(oS.eCache[:,0].A)[0]     #nonzero是numpy中切片操作，返回输入参数中非零元素的索引,这个A是返回矩阵的ndarray形式
	if(len(validEcacheList)) >1:
		'''从已经计算过的所有的E中选择与当前i差距最大的alpha来进行更新'''
		for k in validEcacheList:
			if k ==i:continue
			Ek = calcEk(oS,k)
			deltaE = abs(Ei - Ek)
			if (deltaE > maxDeltaE):
				maxK = k; maxDeltaE = deltaE;Ej = Ek
		return maxK,Ej
	else:
		j = selectRand(i,oS.m)
		Ej = calcEk(oS,j)
	return j,Ej

def updateEk(oS,k):
	Ek = calcEk(oS,k)
	oS.eCache[k] = [1,Ek]

def innerL(oS,i):
	'''这个函数的返回值是对alpha对进行更改的次数'''
	Ei = calcEk(oS,i)
	#判断这个i是否在间隔以内，即它的值是否为0，如果在间隔以外则其值为0不需要进行更新,否则更新
	if((oS.labelMat[i]*Ei < -oS.toler) and (oS.alphas[i] < oS.C)) or ((oS.labelMat[i]*Ei > oS.toler) and (oS.alphas[i] > 0)):
		#选择最大步长的配对alpha j
		j,Ej = selectJ(i,oS,Ei)
		alphaIold = oS.alphas[i].copy()
		alphaJold = oS.alphas[j].copy()
		#计算ahpha j的范围
		if (oS.labelMat[i] != oS.labelMat[j]):
			L = max(0,oS.alphas[j] - oS.alphas[i])
			H = min(oS.C,oS.C + oS.alphas[j] - oS.alphas[i])
		if(oS.labelMat[i] == oS.labelMat[j]):
			L = max(0,oS.alphas[j] + oS.alphas[i] - oS.C)
			H = min(oS.C,oS.alphas[j] + oS.alphas[i])
		if(L == H):print("L == H"); return 0
		#更新alpha j
		eta = 2*oS.X[i,:]*oS.X[j,:].T - oS.X[i,:]*oS.X[i,:].T - oS.X[j,:]*oS.X[j,:].T
		if eta>=0:print("eta >= 0");return 0
		oS.alphas[j] -= oS.labelMat[j]*(Ei - Ej)/eta
		oS.alphas[j] = clipAlpha(oS.alphas[j],L,H)
		updateEk(oS,j)      #将第k个数据的E标志为有效
		if (abs(oS.alphas[j] - alphaJold) < 0.00001):
			print("alpha j is not movint enough")
			return 0
		#更新alpha i
		oS.alphas[i] +=oS.labelMat[i]*oS.labelMat[j]*(alphaJold - oS.alphas[j])
		updateEk(oS,i)
		#更新b
		b1 = oS.b - Ei- oS.labelMat[i]*(oS.alphas[i]-alphaIold)*oS.X[i,:]*oS.X[i,:].T - oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.X[i,:]*oS.X[j,:].T
		b2 = oS.b - Ej- oS.labelMat[i]*(oS.alphas[i]-alphaIold)*oS.X[i,:]*oS.X[j,:].T - oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.X[j,:]*oS.X[j,:].T
		if(oS.alphas[i] < oS.C) and (oS.alphas[i] > 0):
			oS.b = b1
		elif(oS.alphas[j] < oS.C) and (oS.alphas[j] > 0):
			oS.b = b2
		else:
			oS.b = (b1+b2)/2.0
		return 1
	else:return 0

def smoP(dataMatIn,classLabels,C,toler,maxIter,kTup = ('lin',0)):
	oS = optStruct(mat(dataMatIn),mat(classLabels).T,C,toler)
	iter = 0
	entereSet = True; alphaPairsChanged = 0
	#iter小于最大的迭代次数和对alpha进行了更改都是进入循环的条件
	while(iter < maxIter) and ((alphaPairsChanged >0) or (entereSet)):
		'''选择第一个alpha时采用两种方式交替进行，第一种遍历全部的数据，第二个则是遍历边界上的数据'''
		alphaPairsChanged = 0
		if entereSet:
			for i in range(oS.m):
				alphaPairsChanged +=innerL(oS,i)
				print("full set ,iterate %d, i is %d, pairs changed %d times"%(iter,i,alphaPairsChanged))
			iter +=1    #没有对alpha对进行更改的话就给iter +1 
		else:
			nonBoundIs = nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]
			for i in nonBoundIs:
				alphaPairsChanged +=innerL(oS,i)
				print("non-bound, iterate: %d, i is :%d , pairs changed:%d"%(iter,i,alphaPairsChanged))
			iter +=1
		if entereSet:entereSet = False
		elif(alphaPairsChanged == 0):entereSet = True
		print("iteration number is %d"%iter)
	return oS.b,oS.alphas

#计算超平面，即得到分类器
def calcW(alphas,dataArr,classLabels):
	X = mat(dataArr);labelMat = mat(classLabels).T
	m,n = shape(X)
	w = zeros((n,1))
	for i in range(m):
		w += multiply(alphas[i]*labelMat[i],X[i,:].T)
	return w

if __name__ == "__main__":
	dataMat,labelMat = loadData('testSet.txt')
	b,alphas = smoP(dataMat,labelMat,0.6,0.001,40)
	ws = calcW(alphas,dataMat,labelMat)
	fx0 = dataMat[0]*mat(ws) + b
	print(fx0,'     ',labelMat[0])














