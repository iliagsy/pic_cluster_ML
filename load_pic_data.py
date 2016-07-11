# coding:utf-8
from preprocessing import *
import numpy as np

# 打乱顺序
sub_sets = []

for i in range(10):
	sub_set = sum( [ sub_data_set[ len(sub_data_set)*i/10 : len(sub_data_set)*(i+1)/10 ] 
		for sub_data_set in data_set ], [] )
	sub_sets.append( sub_set )

data_set = sum( sub_sets, [] )

data_set_x = np.array( [x for x,y in data_set] )
data_set_y = np.array( [y for x,y in data_set] )

# 返回值：labels为0维数组
def loadDataSet():
	return (data_set_x, data_set_y)
	