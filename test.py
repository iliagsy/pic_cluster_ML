# coding:utf-8
import numpy as np
from load_pic_data import nClass

# 计算n_ij矩阵
# 返回值：float
def calc_n_ij_mat( cluster_labels, labels, nCluster ):
# n_ij: 聚类j，类i
	n_obj = cluster_labels.shape[0]
	labels = np.hstack( (labels.reshape(n_obj,1), cluster_labels.reshape(n_obj,1) ) )
	n_ij_mat = np.zeros( (nClass, nCluster) )
	for label,cluster in labels.tolist():
		n_ij_mat[label, cluster] += 1
	n_ij_mat.dtype = np.float
	return n_ij_mat

# 返回1维（不是0维！）数组
def calc_n_cluster_j( n_ij_mat ):
	n_cluster_j = np.add.reduce( n_ij_mat, axis=0, keepdims=True )
	n_cluster_j[ n_cluster_j == 0 ] = np.inf
	return n_cluster_j

# 返回1维（不是0维！）数组
def calc_n_label_i( n_ij_mat ):
	n_label_i = np.add.reduce( n_ij_mat, axis=1, keepdims=True )
	n_label_i[ n_label_i == 0 ] = np.inf
	return n_label_i

# p_i_cond_j：聚类j中对象属于类i的概率，= n_ij / n_c_j
# 条件概率
def calc_p_i_cond_j_mat( n_ij_mat ):
	# n_cluster_j: 聚类j中的对象数目
	n_cluster_j = calc_n_cluster_j( n_ij_mat )
	p_i_cond_j_mat = n_ij_mat / np.repeat( n_cluster_j, n_ij_mat.shape[0], axis=0 )
	return p_i_cond_j_mat

# 输入：cluster_labels, labels 均为(n_obj,1)维数组
def calc_entropy( cluster_labels, labels, nCluster ):
	n_obj = cluster_labels.shape[0]
	n_ij_mat = calc_n_ij_mat( cluster_labels, labels, nCluster )
	p_ij_mat = n_ij_mat / n_obj;
	
	p_i_cond_j_mat = calc_p_i_cond_j_mat( n_ij_mat )
	info_i_cond_j_mat = -np.log( p_i_cond_j_mat )
	info_i_cond_j_mat[ info_i_cond_j_mat == np.inf ] = 0
	
	entropy = info_i_cond_j_mat * p_ij_mat
	entropy = np.add.reduce( entropy, axis=tuple(range(entropy.ndim)) )
	
	return entropy
	

def calc_F_value( cluster_labels, labels, nCluster ):
	n_obj = cluster_labels.shape[0]
	n_ij_mat = calc_n_ij_mat( cluster_labels, labels, nCluster )
	n_label_i = calc_n_label_i( n_ij_mat )
	
	p_ij_mat = calc_p_i_cond_j_mat( n_ij_mat )
	r_ij_mat = n_ij_mat / np.repeat( n_label_i, n_ij_mat.shape[1], axis=1 )
	
	F_ij_mat = 2 * r_ij_mat * p_ij_mat / (r_ij_mat + p_ij_mat)
	F_ij_mat[ np.isnan( F_ij_mat ) ] = 0
	# F_label_i: 类i中最大的F值
	F_label_i = np.amax( F_ij_mat, axis=1, keepdims=True )
	F = np.sum( (n_label_i * F_label_i).flatten() ) / n_obj
	
	return F
	