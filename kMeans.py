# coding:utf-8
from numpy import *
import numpy.linalg as la

def distEclud(vecA, vecB):     # 传进来的参数：1维的ndarray
    return sqrt(sum(power(vecA - vecB, 2))) #la.norm(vecA-vecB)

# 余弦距离
def distCos(vecA, vecB):
	vecA,vecB = vecA.flatten(), vecB.flatten()
	lenA,lenB = la.norm( vecA ), la.norm( vecB )
	if lenA == 0 or lenB == 0: return 0
	return inner(vecA,vecB) / (lenA * lenB)

def randCent(dataSet, k, distMeas=None):
    n = shape(dataSet)[1]
    centroids = mat(zeros((k,n)))#create centroid mat
    for j in range(n):#create random cluster centers, within bounds of each dimension
        minJ = min(dataSet[:,j]) 
        rangeJ = float(max(dataSet[:,j]) - minJ)
        centroids[:,j] = mat(minJ + rangeJ * random.rand(k,1))
    return centroids
    
# 最小最大法选择k个初始点
def minMaxCent(dataSet, k, distMeas=distEclud):
	print 'Initializing centroids with minMaxCent...'
	# 初始质点在dataSet中的行号
	centroidNos = []
	# load distMat
	distMat_file = 'distMat_cos.txt' if distMeas == distCos else 'distMat.txt'
	distMat = loadtxt(distMat_file)
	# done load
	while True:
		if len(centroidNos) >= k: break
		if len(centroidNos) == 0:
			c1,c2 = unravel_index(argmax( distMat ), distMat.shape)
			centroidNos += [c1.flat[0],c2.flat[0]]; continue
		minDist2c = amin(distMat[centroidNos,:], axis=0).flatten()
		centroidNos.append( argmax( minDist2c ) )
	centroids = dataSet[ centroidNos, : ]
	return centroids

# 密度法选择k个初始点	
# 使用欧几里德距离
def densCent(dataSet, k, distMeas=distEclud, radius=475., ratio=1.1):
	print 'Initializing centroids with densCent...'
	c2c = radius * ratio
	print 'radius = %.2f, c2c / radius = %.2f' % (radius, c2c / radius)
	# load distMat
	distMat_file = 'distMat_cos.txt' if distMeas == distCos else 'distMat.txt'
	distMat = loadtxt(distMat_file)
	# done load
	densArr = add.reduce((distMat <= radius), axis=0)
	dtype = [('No', int), ('density', float)]
	values = list(enumerate( densArr.tolist() ))
	sorted_dens = sort( array( values, dtype=dtype ), order='density' )
	sorted_dens = fliplr( sorted_dens.reshape(1,sorted_dens.shape[0]) ).flatten()
	centroidNos = []
	for No,dens in sorted_dens.tolist():
		if len(centroidNos) >= k: break
		if len(centroidNos) == 0:
			centroidNos.append(No); continue
		if not all(distMat[No, centroidNos] >= c2c): continue
		centroidNos.append(No)
	return dataSet[ centroidNos, : ]

# 输入：dataSet:二维数组
def kMeans(dataSet, k, distMeas=distEclud, createCent=randCent, n_traverse=1000):
	m = shape(dataSet)[0]
	clusterAssment = mat(zeros((m,2)))#create mat to assign data points 
									  #to a centroid, also holds SE of each point
	centroids = createCent(dataSet, k, distMeas)
	clusterChanged = True
	n_run=0
	while clusterChanged and n_run <= n_traverse:
		if n_run % 10 == 0: print 'Run',n_run; n_run+=1
		clusterChanged = False
		for i in range(m):#for each data point assign it to the closest centroid
			minDist = inf; minIndex = -1
			for j in range(k):
				distJI = distMeas(centroids[j,:],dataSet[i,:])  # 广播
				if distJI < minDist:
					minDist = distJI; minIndex = j
			if clusterAssment[i,0] != minIndex: clusterChanged = True
			clusterAssment[i,:] = minIndex,minDist**2
		for cent in range(k):#recalculate centroids
			ptsInClust = dataSet[ nonzero( clusterAssment[:,0].A == cent )[0] ]#get all the point in this cluster
			centroids[cent,:] = mean(ptsInClust, axis=0) #assign centroid to mean 
	return centroids, clusterAssment
    
def biKmeans(dataSet, k, distMeas=distEclud, createCent=randCent, n_traverse=1000):
    m = shape(dataSet)[0]
    clusterAssment = mat(zeros((m,2)))
    centroid0 = mean(dataSet, axis=0).tolist()[0]
    centList =[centroid0] #create a list with one centroid
    for j in range(m):#calc initial Error
        clusterAssment[j,1] = distMeas(mat(centroid0), dataSet[j,:])**2
    while (len(centList) < k):
        lowestSSE = inf
        for i in range(len(centList)):
            ptsInCurrCluster = dataSet[nonzero(clusterAssment[:,0].A==i)[0],:]#get the data points currently in cluster i
            centroidMat, splitClustAss = kMeans(ptsInCurrCluster, 2, distMeas, createCent, n_traverse)
            sseSplit = sum(splitClustAss[:,1])#compare the SSE to the currrent minimum
            sseNotSplit = sum(clusterAssment[nonzero(clusterAssment[:,0].A!=i)[0],1])
            if (sseSplit + sseNotSplit) < lowestSSE:
                bestCentToSplit = i
                bestNewCents = centroidMat
                bestClustAss = splitClustAss.copy()
                lowestSSE = sseSplit + sseNotSplit
        bestClustAss[nonzero(bestClustAss[:,0].A == 1)[0],0] = len(centList) #change 1 to 3,4, or whatever
        bestClustAss[nonzero(bestClustAss[:,0].A == 0)[0],0] = bestCentToSplit
        centList[bestCentToSplit] = bestNewCents[0,:].tolist()[0]#replace a centroid with two best centroids 
        centList.append(bestNewCents[1,:].tolist()[0])
        clusterAssment[nonzero(clusterAssment[:,0].A == bestCentToSplit)[0],:]= bestClustAss#reassign new clusters, and SSE
    return mat(centList), clusterAssment