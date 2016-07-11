# coding:utf-8
from numpy import *
from kMeans import distEclud,distCos
from load_pic_data import loadDataSet

def calc_distMat(dataSet, distMeas=distEclud):
	n_obj = dataSet.shape[0]
	distMat_ = zeros( (n_obj,n_obj) )
	for i in range(n_obj):
		for j in range(i+1,n_obj):
			distMat_[j,i] = distMat_[i,j] = distMeas(dataSet[i,:],dataSet[j,:])
	return distMat_

dataset, labels = loadDataSet()
savetxt('distMat.txt', calc_distMat(dataset))