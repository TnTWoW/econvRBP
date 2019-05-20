# -*- coding: utf-8 -*-
from django.shortcuts import render,render_to_response
from django import forms #重点要导入,使用 Django 的 表单 
from django.http import HttpResponse
import math
import numpy as np
import tensorflow as tf

def submission(request):
    return render(request, 'submission.html')

def help(request):
    return render(request, 'help.html')
	
def developers(request):
    return render(request, 'developers.html')
	
def home(request):
    return render(request, 'home.html')

def get_weight(shape):
    return tf.Variable(tf.random_normal(shape),dtype=tf.float32)

#初始化权值
def weight_variable(shape,name):
    initial = tf.truncated_normal(shape,stddev=0.1)#生成一个截断的正态分布
    return tf.Variable(initial,name=name) # 返回初始化的结果

#初始化偏置
def bias_variable(shape,name):
    initial = tf.constant(0.1,shape=shape)
    return tf.Variable(initial,name=name)
	
def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')
	
def conv3d(x,W):
    return tf.nn.conv3d(x,W,strides=[1,1,1,1,1],padding='SAME')
	
def max_pool_2(x):
    return tf.nn.max_pool(x,ksize=[1,5,1,1],strides=[1,5,1,1],padding='SAME')

def max_pool_3(x):
    return tf.nn.max_pool3d(x,ksize=[1,2,2,2,1],strides=[1,2,2,2,1],padding='SAME')

def model(xonehot,x343):
    with tf.name_scope('x_input1'):    
        x_conv1 = tf.placeholder(tf.float32,[None,549*7],name='x_conv1')
        x_input1 = tf.reshape(x_conv1,[-1,549,1,7],name='x_input1')

    with tf.name_scope('x_input2'):
        x_conv2 = tf.placeholder(tf.float32,[None,7,7,7],name='x_conv2')
        x_input2 = tf.reshape(x_conv2,[-1,7,7,7,1],name='x_input2')
        
    with tf.name_scope('Conv11'):
        W_conv11 = weight_variable([3,1,7,7],name='W_conv11')
        b_conv11 = bias_variable([7],name='b_conv11')
        conv_11 = conv2d(x_input1,W_conv11)+b_conv11
        bn11 = tf.layers.batch_normalization(conv_11)
        relu11 = tf.nn.relu(bn11)
        h_pool11 = max_pool_2(relu11)
    
    with tf.name_scope('Conv12'):
        W_conv12 = weight_variable([5,1,7,7],name='W_conv12')
        b_conv12 = bias_variable([7],name='b_conv12')
        conv_12 = conv2d(h_pool11,W_conv12)+b_conv12
        bn12 = tf.layers.batch_normalization(conv_12)
        relu12 = tf.nn.relu(bn12)
        h_pool12 = max_pool_2(relu12)
        reshape1 = tf.reshape(h_pool12,[-1,22*7])
		
    with tf.name_scope('Conv21'):
        W_conv21 = weight_variable([1,1,1,1,1],name='W_conv21')
        b_conv21 = bias_variable([1],name='b_conv21')
        conv_21 = conv3d(x_input2,W_conv21)+b_conv21
        bn21 = tf.layers.batch_normalization(conv_21)
        relu21 = tf.nn.relu(bn21)
		
    with tf.name_scope('Conv22'):
        W_conv22 = weight_variable([3,3,3,1,1],name='W_conv22')
        b_conv22 = bias_variable([1],name='b_conv22')
        conv_22 = conv3d(relu21,W_conv22)+b_conv22
        bn22 = tf.layers.batch_normalization(conv_22)
        relu22 = tf.nn.relu(bn22)
		
    with tf.name_scope('Conv23'):
        W_conv23 = weight_variable([3,3,3,1,1],name='W_conv23')
        b_conv23 = bias_variable([1],name='b_conv23')
        conv_23 = conv3d(relu22,W_conv23)+b_conv23
        bn23 = tf.layers.batch_normalization(conv_23)
        relu23 = tf.nn.relu(bn23)
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
            prediction = tf.nn.softmax(wx_plus_b2)

			
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        model_file = tf.train.latest_checkpoint('D:/项目/python sourse code self/my_net/')
        tf.train.Saver().restore(sess,model_file)
        y_ = sess.run(prediction,feed_dict={x_conv1:xonehot,
											x_conv2:x343,
											keep_prob:0.5})
    return y_
	
def remove(x):
    x = x.replace(',','')
    x = x.replace("'",'')
    x = x.replace("\n",'')
    x = x.replace(' ','')
    x = x.replace('[','')
    x = x.replace(']','')
    return x
    
def process_onehot(x):
    data = []
    for line in x:
        line = str(line)
        line = remove(line)
        line = line.replace('0','0000000')
        line = line.replace('1','1000000')
        line = line.replace('2','0100000')
        line = line.replace('3','0010000')
        line = line.replace('4','0001000')
        line = line.replace('5','0000100')
        line = line.replace('6','0000010')
        line = line.replace('7','0000001')
        line = list(line)
        line = np.array(line,dtype=int)
        data.append(line)
    data = np.array(data)
    return data

def process_343(x):
    data = []
    for line in x:
        line = str(line)
        line = remove(line)
        length = len(line)
        time =np.zeros((7,7,7))
        for i in range(length - 2):
            time[int(line[i]) - int('1')][int(line[i + 1]) - int('1')][int(line[i + 2]) - int('1')] += 1
        mi = 10
        mx = 0
        for i in range(7):
            for j in range(7):
                for k in range(7):
                    if(time[i][j][k] < mi):
                        mi = time[i][j][k]
                    if(time[i][j][k] > mx):
                        mx = time[i][j][k]
        for i in range(7):
            for j in range(7):
                for k in range(7):
                    time[i][j][k] = (time[i][j][k] - mi) / mx
        data.append(time)
    data = np.array(data)
    return data

def process_data(file_name):
    data = []
    head = []
    with open(file_name, 'r') as f:
        for line in f:
            if line[0]!='>':
                line = line.strip('\n')
                line = line.replace('A','1')
                line = line.replace('G','1')
                line = line.replace('V','1')
                line = line.replace('I','2')
                line = line.replace('L','2')
                line = line.replace('F','2')
                line = line.replace('P','2')
                line = line.replace('Y','3')
                line = line.replace('M','3')
                line = line.replace('T','3')
                line = line.replace('S','3')
                line = line.replace('H','4')
                line = line.replace('N','4')
                line = line.replace('Q','4')
                line = line.replace('W','4')
                line = line.replace('R','5')
                line = line.replace('K','5')
                line = line.replace('D','6')
                line = line.replace('E','6')
                line = line.replace('C','7')
                data.append(list(line))
            else:
                head.append(line)
    f.close()
    x343 = process_343(data)
    avelength = 549
    for line in data:
        if len(line) >= avelength:
            data[data.index(line)] = line[0:avelength]
        else:
            new_line = line
            for _ in range(avelength-len(line)):
                new_line.append('0')
            data[data.index(line)] = new_line  
    xonehot = process_onehot(data)    
    return xonehot,x343,head

class UserForm(forms.Form):
    #title = forms.CharField(max_length=50)
    file = forms.FileField()

def handle_uploaded_file(f):
    file_name = 'some/file/temp.fasta'
    with open(file_name, 'wb+') as destination:
        for chunk in f.chunks():
            destination.write(chunk)
    xonehot,x343,head = process_data(file_name)
    result = model(xonehot,x343)
    y_predict = np.argmax(result,1)
    data = []
    for i in range(len(y_predict)):
        if y_predict[i]==0:
            data.append(str(head[i])+'<br>'+'binding protein'+'<br>')
        else:
            data.append(str(head[i])+'<br>'+'non binding protein'+'<br>')
    return data
	
def register(request):
	if request.method == 'POST':
		uf = UserForm(request.POST, request.FILES)#uf为表单
		if uf.is_valid():
			result = handle_uploaded_file(request.FILES['file'])
			return HttpResponse(result)
	else:
		uf = UserForm()
	return render(request,'register.html', {'uf':uf})