# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 23:10:39 2019

@author: TF
"""

import tensorflow as tf
from sklearn.model_selection import KFold
import numpy as np
import math
import sklearn as sk

def conv2d(x,w):
    return tf.nn.conv2d(x,w,strides=[1,1,1,1],padding='SAME')

def max_pool(x):
    return tf.nn.max_pool(x,ksize=[1,5,1,1],strides=[1,5,1,1],padding='SAME')

def weight_variable(shape):
    initial = tf.truncated_normal(shape,stddev=0.1)#生成一个截断的正态分布
    return tf.Variable(initial) # 返回初始化的结果

#初始化偏置
def bias_variable(shape):
    initial = tf.constant(0.1,shape=shape)
    return tf.Variable(initial)

class DealData:
    # 数据加载函数
    def load(self, posfilenameonehot, nagfilenameonehot):       
        _dataonehot = []
        _label = []
        
        file = open(posfilenameonehot, 'r')
        for line in file.readlines():
            line = line.strip('\n')         # 除去换行
            line = line.split(' ')          # 文件以“ ”分隔
            if "" in line:                  # 解决每行结尾有空格的问题
                line.remove("")
            _dataonehot.append(line)
            _label.append([1,0])           
        file.close()
        
        file = open(nagfilenameonehot, 'r')
        for line in file.readlines():
            line = line.strip('\n')         # 除去换行
            line = line.split(' ')          # 文件以“ ”分隔
            if "" in line:                  # 解决每行结尾有空格的问题
                line.remove("")
            _dataonehot.append(line)
            _label.append([0,1])
        file.close()
        
        label = np.array(_label,dtype=int)
        dataonehot = np.array(_dataonehot,dtype=float)
        shuffle = np.random.permutation(dataonehot.shape[0])
        dataonehot = dataonehot[shuffle,:]
        label = label[shuffle,:]
        return dataonehot,label

x_input=tf.placeholder(tf.float32,[None,549*7])
x_reshape=tf.reshape(x_input,[-1,549,1,7])
y=tf.placeholder(tf.float32,[None,2])

W_conv1=weight_variable([3,1,7,7])
b_conv1=bias_variable([7])
x_conv1=conv2d(x_reshape,W_conv1)+b_conv1
relu1=tf.nn.relu(x_conv1)
x_pool1=max_pool(relu1)

W_conv2=weight_variable([5,1,7,7])
b_conv2=bias_variable([7])
x_conv2=conv2d(x_pool1,W_conv2)+b_conv2
relu2=tf.nn.relu(x_conv2)
x_pool2=max_pool(relu2)

reshape=tf.reshape(x_pool2,[-1,22*7])
W_fc1=weight_variable([22*7,1024])
b_fc1=bias_variable([1024])
wx_plus_b1=tf.matmul(reshape,W_fc1)+b_fc1
h_fc1=tf.nn.relu(wx_plus_b1)

keep_prob=tf.placeholder(tf.float32)
h_fc1_drop=tf.nn.dropout(h_fc1,keep_prob)
W_fc2=weight_variable([1024,2])
b_fc2=bias_variable([2])
wx_plus_b2=tf.matmul(h_fc1_drop,W_fc2)+b_fc2
prediction=tf.nn.softmax(wx_plus_b2)

cross_entropy=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y,logits=prediction))
train_step=tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cross_entropy)
correct_prediction=tf.equal(tf.argmax(prediction,1),tf.argmax(y,1))
accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    deal=DealData()
    Xonehot,Y = deal.load('D:\\项目\\data\\final_data\\25\\2875-bind-protein.7onehot',
                          'D:\\项目\\data\\final_data\\25\\6782-non-bind-protein.7onehot')
    kf = KFold(n_splits=10,shuffle=True)
    batch_size=966
    for i in range(2001):        
        result = 0
		TP=0
        TN=0
        FP=0
        FN=0
        AUC=0
        for train_index, test_index in kf.split(Xonehot):    
            for j in range(0,len(train_index),batch_size):
                sess.run(train_step,feed_dict={x_input:Xonehot[train_index[j:j+batch_size]],
                                               y:Y[train_index[j:j+batch_size]],
                                               keep_prob:0.25})
            #sess.run(train_step,feed_dict={x_conv1:Xonehot[train_index],x_conv2:X343[train_index],y:Y[train_index],keep_prob:0.25})
            acc,pre = sess.run([accuracy,prediction],feed_dict={x_input:Xonehot[test_index],
                                                               y:Y[test_index],
                                                               keep_prob:0.25})
            y_true=np.argmax(Y[test_index],1)
            y_predict=np.argmax(pre,1)
            AUC+=sk.metrics.roc_auc_score(Y[test_index],pre)
            TP+=np.count_nonzero((y_true-1)*(y_predict-1))
            TN+=np.count_nonzero(y_true*y_predict)
            FP+=np.count_nonzero((y_true-1)*y_predict)
            FN+=np.count_nonzero(y_true*(y_predict-1))
            #print(x_input[0,:,:,:])
            result += acc
        PRE=TP/(TP+FP)
        if(TP+FN==0 or TP+FP==0 or TN+FP==0 or TN+FN==0):
            MCC=0
        else:
            MCC=(TP*TN-FP*FN)/math.sqrt((TP+FN)*(TP+FP)*(TN+FP)*(TN+FN))
        if(TP+FN==0):
            SN=0
        else:
            SN=TP/(TP+FN)
        if(TN+FP==0):
            SP=0
        else:
            SP=TN/(TN+FP)
        with open('./test.txt', 'w') as f:
            f.write("Iter "+str(i)+",TP="+str(TP/10)+",TN="+str(TN/10)+",FP="+str(FP/10)+",FN="+str(FN/10))
            f.write("\n")
            f.write("Accuracy=%.4f,PRE=%.4f,MCC=%.4f,SN=%.4f,SP=%.4f,AUC=%.4f" %(result/10,PRE,MCC,SN,SP,AUC/10))
            f.write("\n")