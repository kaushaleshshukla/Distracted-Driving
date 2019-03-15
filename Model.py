# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 20:53:20 2018

@author: Kaushalesh Shukla
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import time

train_path = "./imgs/train/"
test_path = "./imgs/test/"
driver_imgs_list = pd.read_csv('driver_imgs_list.csv')

#Preparing data
def prepare_data(data):
    data = data.drop(['subject'] , axis=1)
    data = pd.get_dummies(data , columns=['classname'])
    
#    Suffling data
    data = data.iloc[ np.random.permutation( len(data) ) ]
    
    img_list = data['img'].values
    img_labels = data.drop( ['img'] , axis=1).values
    
    return img_list , img_labels

img_list , img_labels = prepare_data(driver_imgs_list)
img_list.reshape(1,22424)

#function for generating batches
def next_batch(ite , batch=10):
    img , labels = [] , []
    for i in range(ite*batch , ite*batch+batch):
#        Reading training data
        img.append( plt.imread( os.path.join( train_path , img_list[i] ) ) )
        labels.append( img_labels[i] )
        
    return np.asarray(img) , np.asarray(labels)


def load_data(ite, batch=40):
    img = []
    for i in range(ite*batch, ite*batch+batch):
        img.append( plt.imread( os.path.join(test_path, test_img_list[i]) ) )
        
    return np.asarray(img)


#function to convolve the 2-D matrix
def convolve(x, in_size, out_size , filter_size=3):
    w_init = tf.contrib.layers.variance_scaling_initializer()
    b_init = tf.constant_initializer(0.0)
    w = tf.get_variable("weight", [3, 3 , in_size, out_size], dtype=tf.float32, initializer=w_init)
    b = tf.get_variable("bias", [out_size], dtype=tf.float32, initializer=b_init)
    conv = tf.nn.conv2d(x, w, [1, 1, 1, 1], padding="SAME") + b
    bn = tf.layers.batch_normalization(conv, training=True)
    relu = tf.nn.relu(bn, name="relu")    
    
    return relu


#function for fully connected neural network
def fully_connected(x, in_size, out_size, prob=1, activation=tf.nn.relu):
    w_init = tf.contrib.layers.variance_scaling_initializer()
    b_init = tf.constant_initializer(0.0)
    w = tf.get_variable("weight", [in_size, out_size], dtype=tf.float32, initializer=w_init)
    b = tf.get_variable("bias", [out_size], dtype=tf.float32, initializer=b_init)
    w = tf.nn.dropout(w, keep_prob=prob)
    mul = tf.matmul(x, w) + b
    bn = tf.layers.batch_normalization(mul, training=True)
    
    if(activation==None):
        return mul
    return activation(bn)


tf.reset_default_graph()
X = tf.placeholder(dtype=tf.float32 , shape=[None , 480 , 640 , 3] , name='Image' )
y = tf.placeholder(dtype=tf.float32 , shape=[None,10])

#Resizing input images
X_resized = tf.image.resize_images(X, [240,320], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

#Normalization of Input images
X_normalized = X_resized/255 

#Layer 1
with tf.variable_scope('Layer1'):
    conv_l1 = convolve(X_normalized , 3 , 16)
    
with tf.variable_scope('Layer2'):
    conv_l2 = convolve(conv_l1, 16, 16)
conv_l2 = tf.nn.max_pool(conv_l2 , ksize=[1,2,2,1] , strides=[1,2,2,1] , padding='SAME' , name='Maxpool1')

#Layer 2
with tf.variable_scope('Layer3'):
    conv_l3 = convolve(conv_l2, 16, 32)

with tf.variable_scope('Layer4'):
    conv_l4 = convolve(conv_l3, 32, 32)
conv_l4 = tf.nn.max_pool(conv_l4 , ksize=[1,2,2,1] , strides=[1,2,2,1] , padding='SAME' , name='Maxpool2')

#Layer 5
with tf.variable_scope('Layer5'):
    conv_l5 = convolve(conv_l4 , 32 ,64)

with tf.variable_scope('Layer6'):
    conv_l6 = convolve(conv_l5, 64, 64)
conv_l6 = tf.nn.max_pool(conv_l6 , ksize=[1,2,2,1] , strides=[1,2,2,1] , padding='SAME' , name='Maxpool3')


#Layer 4
with tf.variable_scope('Layer7'):
    conv_l7 = convolve(conv_l6 , 64 ,128)
    
with tf.variable_scope('Layer8'):
    conv_l8 = convolve(conv_l7, 128, 128)
    
conv_l8 = tf.nn.max_pool(conv_l8 , ksize=[1,2,2,1] , strides=[1,2,2,1] , padding='SAME' , name='Maxpool4')

#Layer 5
with tf.variable_scope('Layer9'):
    conv_l9 = convolve(conv_l8 , 128 , 256)

with tf.variable_scope('Layer10'):
    conv_l10 = convolve(conv_l9, 256, 256)
    

with tf.variable_scope('Layer11'):
    conv_l11 = convolve(conv_l10 , 256 , 512)
conv_l11 = tf.nn.max_pool(conv_l11 , ksize=[1,3,4,1] , strides=[1,3,4,1] , padding='SAME' , name='Maxpool6')


#Fully connected layer 1
flatten = tf.contrib.layers.flatten(conv_l11)

with tf.variable_scope('fclayer1'):
    fc1= fully_connected(flatten, 5*5*512, 1024, prob=0.6)  
with tf.variable_scope('fclayer2'):
    fc2 = fully_connected(fc1, 1024, 512, prob=0.7)

with tf.variable_scope('fclayer3'):
    fc3 = fully_connected(fc2, 512, 10, activation=None)
#Fully connected layer 2

#Softmax layer
y_ = tf.nn.softmax(fc3, name='softmax_output')
y_onehot = tf.one_hot(tf.argmax(y_, axis=1), depth=10, name='output_labels')

#Cross entropy
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=fc3 , labels=y))

training_op = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_, axis=1), tf.argmax(y, axis=1)), tf.float32))

#Defining saver for parameters
saver = tf.train.Saver()

init = tf.global_variables_initializer()

run_options = tf.RunOptions(report_tensor_allocations_upon_oom = True)


#===============================================================Training Part==========================================


epoch = 30
batch = 16
with tf.Session() as sess:
    sess.run(init)
    
#    Restoring parameters
    print('Looking for parameters to restore')
    try:
        saver.restore(sess , './2nd_Model_parameters/parameters.ckpt')
        print('Parameters Restored')
        
    except:
        print('Parameter not available')
        
    print('Training Started.....')
    for ep in range(epoch):  
        print("#################################################################")
        print("training started for epoch {}".format(ep))
        for ite in range(img_labels.shape[0]//batch):
            a = time.time()
            batch_x , batch_y = next_batch(ite,batch)
            b = time.time()
            load_time = b-a
            a = time.time()
            sess.run(training_op , feed_dict={X:batch_x , y:batch_y} , options=run_options)
            b = time.time()
            training_time = b-a
#        Evaluating Performance after each 25 iterations
            if(ite%25==0):
                
                score , loss= sess.run([accuracy , cross_entropy] , feed_dict={X:batch_x , y:batch_y})
                print("Accuracy on iter {} is {}".format(ite,score*100))
                print('Loss is {}'.format(loss))
                print('Loading Time is {}\nTraining Time is {}'.format(load_time,training_time))
                
#        Saving graph and parameters
                saver.save(sess , './2nd_Model_parameters/parameters.ckpt')

    print('Training Completed.')
#=========================================================Prediction Part=============================================
                
              
def load_residual(batch=40):
    res_img = []
    res = len(test_img_list)%batch
    
    for i in range( len(test_img_list)-res, len(test_img_list) ):
        res_img.append( plt.imread( os.path.join(test_path, test_img_list[i]) ) )
        
    return np.asarray(res_img)


batch = 64
test_img_list = os.listdir(test_path)

with tf.Session() as sess:
    
    sess.run(init)

#Loading parameters
    print('Loading Parameters')
    try:
        saver.restore(sess, './2nd_Model_parameters/parameters.ckpt')
        print('Parameters loaded')
    except:
        print('Parameters nor found')
        exit(0)

    y =  np.zeros([0,10])
    
    print('Predicting Test set')
    for ite in range(len(test_img_list)//batch):
        img = load_data(ite, batch)
        label = sess.run(y_onehot, feed_dict={X:img})
        y = np.concatenate([y,label], axis=0)
        if(ite%100==0):
            print('{} batches predicted'.format(ite+1))
            print(np.asarray(y).shape)
    
    print('Predicting residual images')      
    img = load_residual(batch)
    label = sess.run(y_onehot, feed_dict={X:img})
    y = np.concatenate([y,label], axis=0)
    print(y.shape)
    
    y = np.asarray(y)
    test_img_list = np.asarray(test_img_list)
    test_img_list.shape = [test_img_list.shape[0],1]
    
    result = np.concatenate([test_img_list,y], axis=1)
    
    df = pd.DataFrame(result, columns=['img','c0','c1','c2','c3','c4','c5','c6','c7','c8','c9'])
    
    df.to_csv('result.csv',index=False)
        
