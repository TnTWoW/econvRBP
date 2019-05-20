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

def conv3d(x,w):
    return tf.nn.conv3d(x,w,strides=[1,1,1,1,1],padding='SAME')

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
    def load(self, posfilename343, nagfilename343,posfilenameonehot, nagfilenameonehot):
        _data343 = []        
        _dataonehot = []
        _label = []
        file = open(posfilename343, 'r')
        for line in file.readlines():
            line = line.strip('\n')         # 除去换行
            line = line.split(' ')          # 文件以“ ”分隔
            if "" in line:                  # 解决每行结尾有空格的问题
                line.remove("")
            _data343.append(line)
        file.close()
        
        file = open(nagfilename343, 'r')
        for line in file.readlines():
            line = line.strip('\n')         # 除去换行
            line = line.split(' ')          # 文件以“ ”分隔
            if "" in line:                  # 解决每行结尾有空格的问题
                line.remove("")
            _data343.append(line)            
        file.close()
        
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
        
        data343=np.array(_data343,dtype=float)
        label=np.array(_label,dtype=float)
        dataonehot=np.array(_dataonehot,dtype=float)
        return data343,dataonehot,label

x_input1=tf.placeholder(tf.float32,[None,549*7])
x_reshape1=tf.reshape(x_input1,[-1,549,1,7])
x_input2=tf.placeholder(tf.float32,[None,7*7*7])
x_reshape2=tf.reshape(x_input2,[-1,7,7,7,1])
y=tf.placeholder(tf.float32,[None,2])

W_conv11=weight_variable([3,1,7,7])
b_conv11=bias_variable([7])
x_conv11=conv2d(x_reshape1,W_conv11)+b_conv11
relu11=tf.nn.relu(x_conv11)
x_pool11=max_pool(relu11)

W_conv12=weight_variable([5,1,7,7])
b_conv12=bias_variable([7])
x_conv12=conv2d(x_pool11,W_conv12)+b_conv12
relu12=tf.nn.relu(x_conv12)
x_pool12=max_pool(relu12)

W_conv21=weight_variable([1,1,1,1,1])
b_conv21=bias_variable([1])
x_conv21=conv3d(x_reshape2,W_conv21)+b_conv21
relu21=tf.nn.relu(x_conv21)

W_conv22=weight_variable([3,3,3,1,1])
b_conv22=bias_variable([1])
x_conv22=conv3d(relu21,W_conv22)+b_conv22
relu22=tf.nn.relu(x_conv22)

W_conv23=weight_variable([3,3,3,1,1])
b_conv23=bias_variable([1])
x_conv23=conv3d(relu22,W_conv23)+b_conv23
relu23=tf.nn.relu(x_conv23)

reshape1=tf.reshape(x_pool12,[-1,22*7])
reshape2=tf.reshape(x_conv23,[-1,7*7*7])
concat=tf.concat([reshape1,reshape2],1)
W_fc1=weight_variable([22*7+7*7*7,1024])
b_fc1=bias_variable([1024])
wx_plus_b1=tf.matmul(concat,W_fc1)+b_fc1
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
    X343,Xonehot,Y = deal.load('D:\\项目\\data\\final_data\\25\\2875-bind-protein.trid',
                               'D:\\项目\\data\\final_data\\25\\6782-non-bind-protein.trid',
                               'D:\\项目\\data\\final_data\\25\\2875-bind-protein.7onehot',
                               'D:\\项目\\data\\final_data\\25\\6782-non-bind-protein.7onehot')
    kf = KFold(n_splits=10,shuffle=True)
    batch_size=966
    for i in range(2001):        
        result = 0
        for train_index, test_index in kf.split(Xonehot):
            TP=0
            TN=0
            FP=0
            FN=0
            AUC=0
            for j in range(0,len(train_index),batch_size):
                sess.run(train_step,feed_dict={x_input1:Xonehot[train_index[j:j+batch_size]],
                                               x_input2:X343[train_index[j:j+batch_size]],
                                               y:Y[train_index[j:j+batch_size]],
                                               keep_prob:0.25})
            #sess.run(train_step,feed_dict={x_conv1:Xonehot[train_index],x_conv2:X343[train_index],y:Y[train_index],keep_prob:0.25})
            acc,pre = sess.run([accuracy,prediction],feed_dict={x_input1:Xonehot[test_index],
                                                               x_input2:X343[test_index],
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
        with open('./result_withoutRes', 'w') as f:
            f.write("Iter "+str(i)+",TP="+str(TP/10)+",TN="+str(TN/10)+",FP="+str(FP/10)+",FN="+str(FN/10))
            f.write("\n")
            f.write("Accuracy=%.4f,PRE=%.4f,MCC=%.4f,SN=%.4f,SP=%.4f,AUC=%.4f" %(result/10,PRE,MCC,SN,SP,AUC/10))
            f.write("\n")