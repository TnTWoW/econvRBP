# -*- coding: utf-8 -*-
"""
Created on Wed Nov 21 15:21:23 2018

@author: TF
"""
import numpy as np
import tensorflow as tf
from sklearn.model_selection import KFold
import sklearn as sk
import math
K=10
tf.reset_default_graph()
batch_size = 966

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
        
        data343 = np.array(_data343,dtype=float)
        label = np.array(_label,dtype=int)
        dataonehot = np.array(_dataonehot,dtype=int)
        shuffle = np.random.permutation(data343.shape[0])
        data343 = data343[shuffle,:]
        label = label[shuffle,:]
        dataonehot = dataonehot[shuffle,:]
        return data343,dataonehot,label

#初始化权值
def weight_variable(shape,name):
    initial = tf.truncated_normal(shape,stddev=0.1)#生成一个截断的正态分布
    return tf.Variable(initial,name=name) # 返回初始化的结果

#初始化偏置
def bias_variable(shape,name):
    initial = tf.constant(0.1,shape=shape)
    return tf.Variable(initial,name=name)

#卷积层

def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')    

def conv3d(x,W):
    return tf.nn.conv3d(x,W,strides=[1,1,1,1,1],padding='SAME')

def max_pool_2(x):
    return tf.nn.max_pool(x,ksize=[1,5,1,1],strides=[1,5,1,1],padding='SAME')

#命名空间
with tf.name_scope('x_input1'):    
    x_conv1 = tf.placeholder(tf.float32,[None,549*7],name='x_conv1')
    with tf.name_scope('reshape1'):
        x_input1 = tf.reshape(x_conv1,[-1,549,1,7],name='x_input1')

with tf.name_scope('x_input2'):
    x_conv2 = tf.placeholder(tf.float32,[None,7*7*7],name='x_conv2')   
    with tf.name_scope('reshape2'):
        x_input2 = tf.reshape(x_conv2,[-1,7,7,7,1],name='x_input2')
 
with tf.name_scope('y_input'):
    y = tf.placeholder(tf.float32,[None,2],name='y-input')

with tf.name_scope('Conv11'):
    W_conv11 = weight_variable([3,1,7,7],name='W_conv11')
    b_conv11 = bias_variable([7],name='b_conv11')
    conv_11 = conv2d(x_input1,W_conv11)+b_conv11
    relu11 = tf.nn.relu(conv_11)
    h_pool11 = max_pool_2(relu11)
    
with tf.name_scope('Conv12'):
    W_conv12 = weight_variable([5,1,7,7],name='W_conv12')
    b_conv12 = bias_variable([7],name='b_conv12')
    conv_12 = conv2d(h_pool11,W_conv12)+b_conv12
    relu12 = tf.nn.relu(conv_12)
    h_pool12 = max_pool_2(relu12)
    reshape1 = tf.reshape(h_pool12,[-1,22*7])
    
with tf.name_scope('Conv21'):
    W_conv21 = weight_variable([3,3,3,1,1],name='W_conv21')
    b_conv21 = bias_variable([1],name='b_conv21')
    conv_21 = conv3d(x_input2,W_conv21)+b_conv21
    relu21 = tf.nn.relu(conv_21)
    
with tf.name_scope('Conv22'):
    W_conv22 = weight_variable([3,3,3,1,1],name='W_conv22')
    b_conv22 = bias_variable([1],name='b_conv22')
    conv_22 = conv3d(relu21,W_conv22)+b_conv22
    relu22 = tf.nn.relu(conv_22)
    
with tf.name_scope('Conv23'):
    W_conv23 = weight_variable([3,3,3,1,1],name='W_conv23')
    b_conv23 = bias_variable([1],name='b_conv23')
    conv_23 = conv3d(relu22,W_conv23)+b_conv23
    relu23 = tf.nn.relu(conv_23)
    add = tf.add(relu23,x_input2)
    reshape2 = tf.reshape(add,[-1,7*7*7])
    
with tf.name_scope('concat'):
    flat = tf.concat([reshape1,reshape2],1,name='flat')

with tf.name_scope('fc1'):
    with tf.name_scope('W_fc1'):
        W_fc1 = weight_variable([22*7+7*7*7,1024],name='W_fc1')#上一场有64个神经元，全连接层有1024个神经元
    with tf.name_scope('b_fc1'):
        b_fc1 = bias_variable([1024],name='b_fc1')#1024个节点
    with tf.name_scope('wx_plus_b1'):
        wx_plus_b1 = tf.matmul(flat,W_fc1) + b_fc1
    with tf.name_scope('relu1'):
        h_fc1 = tf.nn.relu(wx_plus_b1)   # 第一个全连接层的输出
            
with tf.name_scope('keep_prob'):
    keep_prob = tf.placeholder(tf.float32,name='keep_prob')  
with tf.name_scope('h_fc1_drop'):
    h_fc1_drop = tf.nn.dropout(h_fc1,keep_prob,name='h_fc1_drop')
    
with tf.name_scope('fc2'):
    with tf.name_scope('W_fc2'):
        W_fc2 = weight_variable([1024,2],name='W_fc2')
    with tf.name_scope('b_fc2'):    
        b_fc2 = bias_variable([2],name='b_fc2')
    with tf.name_scope('wx_plus_b2'):
        wx_plus_b2 = tf.matmul(h_fc1_drop,W_fc2) + b_fc2
    with tf.name_scope('softmax'):
        #计算输出
        prediction = tf.nn.softmax(wx_plus_b2)

#交叉熵代价函数
with tf.name_scope('cross_entropy'):
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y,logits=prediction),name='cross_entropy')
    tf.summary.scalar('cross_entropy',cross_entropy)
    
with tf.name_scope('train'):
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

#求准确率
with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
        correct_prediction = tf.equal(tf.argmax(prediction,1),tf.argmax(y,1))#argmax返回一维张量中最大的值所在的位置
    with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
        tf.summary.scalar('accuracy',accuracy)

merged = tf.summary.merge_all()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    #debug_sess = tf_debug.LocalCLIDebugWrapperSession(sess)
    train_writer = tf.summary.FileWriter('logs/train',sess.graph)
    test_writer = tf.summary.FileWriter('logs/test',sess.graph)
    deal=DealData()
    X343,Xonehot,Y = deal.load('D:\\项目\\data\\final_data\\25\\2875-bind-protein.trid',
                               'D:\\项目\\data\\final_data\\25\\6782-non-bind-protein.trid',
                               'D:\\项目\\data\\final_data\\25\\2875-bind-protein.7onehot',
                               'D:\\项目\\data\\final_data\\25\\6782-non-bind-protein.7onehot')
    kf = KFold(n_splits=K,shuffle=True)
    for i in range(2001):        
        result = 0
        TP=0
        TN=0
        FP=0
        FN=0
        AUC=0
        for train_index, test_index in kf.split(X343):
            for j in range(0,len(train_index),batch_size):
                sess.run(train_step,feed_dict={x_conv1:Xonehot[train_index[j:j+batch_size]],
                                               x_conv2:X343[train_index[j:j+batch_size]],
                                               y:Y[train_index[j:j+batch_size]],
                                               keep_prob:0.25})
            acc,pre = sess.run([accuracy,prediction],feed_dict={x_conv1:Xonehot[test_index],
                                               x_conv2:X343[test_index],
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
        print("Iter "+str(i)+",TP="+str(TP/10)+",TN="+str(TN/10)+",FP="+str(FP/10)+",FN="+str(FN/10))
        print("Accuracy=%.4f,PRE=%.4f,MCC=%.4f,SN=%.4f,SP=%.4f,AUC=%.4f" %(result/10,PRE,MCC,SN,SP,AUC/10))