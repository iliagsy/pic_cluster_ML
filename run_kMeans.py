# coding:utf-8
import os
import sys
import timeit

import numpy as np

from load_pic_data import loadDataSet
from kMeans import *
from test import *

dataset, labels = loadDataSet()
labels = labels.reshape(labels.shape[0],1)
nCluster = 10
times = 2
mean_algrs = [kMeans, biKmeans]
mean_algr = mean_algrs[0]
createCents = [randCent, minMaxCent, densCent]
createCent = createCents[0]

if not createCent == randCent: times = 1 # 只有randCent需要运行多次

entropy_lst = []
F_lst = []

for i in range( times ):
	start_time = timeit.default_timer()

	centroids, clusterAssment = mean_algr( 
		dataset, 
		nCluster,
		distMeas=distCos,
		createCent=createCent,
		n_traverse=50
	)

	end_time = timeit.default_timer()
	print 'Clustering process complete.'
	print 'KMeans run for %.2fm' % ((end_time - start_time) / 60.)

	cluster_labels = clusterAssment[:, 0]

	entropy = calc_entropy( cluster_labels, labels, nCluster )
	F = calc_F_value( cluster_labels, labels, nCluster )

	entropy_lst.append( entropy )
	F_lst.append( F )

	print 'Run %d completed.' % i
	print "Entropy =", entropy, "F value =", F

print 'Mean entropy = %.2f, mean F value = %.2f' % (np.mean(entropy_lst), np.mean(F_lst))
print 'SD of each run: entropy = %.3f, F value = %.3f' % (np.std(entropy_lst),np.std(F_lst))

