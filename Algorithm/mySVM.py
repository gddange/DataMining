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

if __name__ == '__main__':
	dataMat,labelMat = loadData('testSet.txt')
	b,alphas = smoSimple(dataMat,labelMat,0.6,0.001,40)
	print(b,'\n',alphas)











