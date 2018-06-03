
# coding: utf-8

# In[1]:


import sys
caffe_root = '/home/ll/caffe'
sys.path.insert(0, caffe_root + '/python')

import matplotlib.mlab as mlab  
import scipy.integrate as integrate
from PIL import Image
from scipy import fft
from scipy import misc
from skimage import transform
import shutil
import requests
import tempfile
import os
import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')
import time

import caffe

plt.rcParams['figure.figsize'] = (4, 4)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'


# In[2]:


def load_model():
    BATCH_SIZE = 1
    model_def = './bvlc_googlenet/deploy.prototxt'
    net_weights = './bvlc_googlenet/bvlc_googlenet.caffemodel'
    net = caffe.Net(model_def, net_weights, caffe.TEST)
    shape = list(net.blobs['data'].data.shape)
    shape[0] = BATCH_SIZE
    net.blobs['data'].reshape(*shape)
    net.blobs['prob'].reshape(BATCH_SIZE, )
    net.reshape()
    return net

def compute_gradient(image, intended_outcome):
    predict(image, display_output=False)
    probs = np.zeros_like(net.blobs['prob'].data)
    probs[0][intended_outcome] = -1 
    gradient = net.backward(prob=probs)
    return gradient['data'].copy()

def display(data):
    plt.imshow(transformer.deprocess('data', data))
    
def get_label_name(num):
    options = labels[num].split(',')
    # remove the tag
    options[0] = ' '.join(options[0].split(' ')[1:])
    return ','.join(options[:2])

def predict(data, n_preds=6, display_output=False):
    net.blobs['data'].data[...] = data
    if display_output:
        display(data)
    prob = net.forward()['prob']
    probs = prob[0]
    prediction = probs.argmax()
    top_k = probs.argsort()[::-1]
    for pred in top_k[:n_preds]:
        percent = round(probs[pred] * 100, 2)
        # display it compactly if we're displaying more than the top prediction
        pred_formatted = "%03d" % pred
        if n_preds == 1:
            format_string = "label: {cls} ({label})\ncertainty: {certainty}%"
        else:
            format_string = "label: {cls} ({label}), certainty: {certainty}%"
        if display_output:
            print format_string.format(
                cls=pred_formatted, label=get_label_name(pred), certainty=percent)
    return prob

def add_matrix(image_data,offset,ratio):
    result_data=np.zeros_like(image_data)
    for x in range(1):
        for y in range(3):
            for z in range(224):
                for v in range(224):
                    result_data[y][z][v]=image_data[y][z][v]+ratio*offset[x][y][z][v]
                    if(result_data[y][z][v]<0):
                        result_data[y][z][v]=0
                    if(result_data[y][z][v]>255):
                        result_data[y][z][v]=255
    return result_data


# In[3]:


def normalization(image_data):
    image_data[image_data<0] = 0
    image_data[image_data>255] = 255.0
    
def boxMeanFilterOperations(inputDigit, start, end, coefficient):
    retDigit = np.array(inputDigit, dtype=np.float32)
    for row in xrange(start, end):
        for col in xrange(start, end):
            retDigit[0][row][col] = sum(sum(inputDigit[0,row-start:row+start+1,col-start:col+start+1]))/coefficient
            retDigit[1][row][col] = sum(sum(inputDigit[1,row-start:row+start+1,col-start:col+start+1]))/coefficient
            retDigit[2][row][col] = sum(sum(inputDigit[2,row-start:row+start+1,col-start:col+start+1]))/coefficient
    return retDigit

def diamondMeanFilterOperations(inputDigit, kernel, start, end, coefficient):
    retDigit = np.array(inputDigit, dtype=np.float32)
    for row in xrange(start, end):
        for col in xrange(start, end):
            retDigit[0][row][col] = sum(sum(inputDigit[0,row-start:row+start+1, col-start:col+start+1]*kernel))/coefficient
            retDigit[1][row][col] = sum(sum(inputDigit[1,row-start:row+start+1, col-start:col+start+1]*kernel))/coefficient
            retDigit[2][row][col] = sum(sum(inputDigit[2,row-start:row+start+1, col-start:col+start+1]*kernel))/coefficient
    return retDigit

def crossMeanFilterOperations(inputDigit, start, end, coefficient):
    retDigit = np.array(inputDigit, dtype=np.float32)
    for row in xrange(start, end):
        for col in xrange(start, end):
            temp0 = inputDigit[0][row][col]
            temp1 = inputDigit[1][row][col]
            temp2 = inputDigit[2][row][col]
            for i in range(1,start+1):
                temp0+=inputDigit[0][row-i][col]
                temp0+=inputDigit[0][row+i][col]
                temp0+=inputDigit[0][row][col-i]
                temp0+=inputDigit[0][row][col+i]
                temp1+=inputDigit[1][row-i][col]
                temp1+=inputDigit[1][row+i][col]
                temp1+=inputDigit[1][row][col-i]
                temp1+=inputDigit[1][row][col+i]
                temp2+=inputDigit[2][row-i][col]
                temp2+=inputDigit[2][row+i][col]
                temp2+=inputDigit[2][row][col-i]
                temp2+=inputDigit[2][row][col+i]
            retDigit[0][row][col] = temp0/coefficient
            retDigit[1][row][col] = temp1/coefficient
            retDigit[2][row][col] = temp2/coefficient
    return retDigit

def scalarQuantization(inputDigit, interval, left=True):
    retDigit = np.array(inputDigit,dtype=np.float32)
    retDigit//=interval
    retDigit*=interval
    if not left:
        halfInterval = interval//2
        retDigit+=(halfInterval)
    return retDigit


# In[4]:


def trainBoxMeanFilters(rootDir,true_class):
    print(rootDir)
    for kernelSize in xrange(3,10,2):
        startTime = time.time()
        print('box filter, size = ', kernelSize)
        original_classified_wrong_number = 0
        disturbed_failure_number = 0
        Total=TTP=TP=FP=FN=0
        start = (kernelSize-1)//2
        end = 224-start
        coefficient = kernelSize**2
        list_dirs = os.walk(rootDir)
        for root, dirs, files in list_dirs:
            for f in files: 
                original_data=transformer.preprocess('data', caffe.io.load_image(os.path.join(root, f)))            
                net.blobs['data'].data[...] = original_data
                ori_out = net.forward()
                pred_class = ori_out['prob'][0].argmax()
                if pred_class != true_class:
                    original_classified_wrong_number+=1
                    continue
                grad = compute_gradient(original_data, pred_class)
                delta = np.sign(grad)
                adversarial_data=add_matrix(original_data,delta,1.0)
                normalization(adversarial_data)
                net.blobs['data'].data[...] = adversarial_data
                adv_out = net.forward()
                adv_class = adv_out['prob'][0].argmax()
                if adv_class == true_class:
                    disturbed_failure_number+=1
                    continue
                Total+=1
                ori_processed = boxMeanFilterOperations(original_data, start, end, coefficient)
                net.blobs['data'].data[...] = ori_processed
                ori_filtered_out = net.forward()
                ori_filtered_class = ori_filtered_out['prob'][0].argmax()
                adv_processed = boxMeanFilterOperations(adversarial_data, start, end, coefficient)
                net.blobs['data'].data[...] = adv_processed
                adv_filtered_out = net.forward()
                adv_filtered_class = adv_filtered_out['prob'][0].argmax()
                if(ori_filtered_class!=true_class):
                    FP+=1
                if (adv_filtered_class!=adv_class):
                    TP+=1
                    if (adv_filtered_class == true_class):
                        TTP+=1
                else:
                    FN+=1
        print("Overall results: ")
        str1 = '%d-%d-%d: TP = %d; FN = %d; FP = %d; TTP = %d' % (Total,original_classified_wrong_number,disturbed_failure_number,TP,FN,FP,TTP)
        print(str1)
        endTime  = time.time()
        print('lasting ', endTime-startTime, 'seconds')
        Recall=TP*1.0/(TP+FN)
        Precision=TP*1.0/(TP+FP)
        print('********************************')
        print ("Recall: ",Recall)
        print ("Precision: ",Precision)
        print('********************************')


# In[5]:


def trainDiamondMeanFilters(rootDir,true_class,diamonds):
    print(rootDir)
    coefficient = [5,13, 25, 41]
    kernelIndex = -1
    for kernelSize in xrange(3,10,2):
        startTime = time.time()
        print('diamond filter, size = ', kernelSize)
        original_classified_wrong_number = 0
        disturbed_failure_number = 0
        Total=TTP=TP=FP=FN=0
        start = (kernelSize-1)//2
        end = 224-start
        kernelIndex+=1
        list_dirs = os.walk(rootDir)
        for root, dirs, files in list_dirs:
            for f in files: 
                original_data=transformer.preprocess('data', caffe.io.load_image(os.path.join(root, f)))            
                net.blobs['data'].data[...] = original_data
                ori_out = net.forward()
                pred_class = ori_out['prob'][0].argmax()
                if pred_class != true_class:
                    original_classified_wrong_number+=1
                    continue
                grad = compute_gradient(original_data, pred_class)
                delta = np.sign(grad)
                adversarial_data=add_matrix(original_data,delta,1.0)
                normalization(adversarial_data)
                net.blobs['data'].data[...] = adversarial_data
                adv_out = net.forward()
                adv_class = adv_out['prob'][0].argmax()
                if adv_class == true_class:
                    disturbed_failure_number+=1
                    continue
                Total+=1
                ori_processed = diamondMeanFilterOperations(original_data, diamonds[kernelIndex], start, end, coefficient[kernelIndex])
                net.blobs['data'].data[...] = ori_processed
                ori_filtered_out = net.forward()
                ori_filtered_class = ori_filtered_out['prob'][0].argmax()
                adv_processed = diamondMeanFilterOperations(adversarial_data, diamonds[kernelIndex], start, end, coefficient[kernelIndex])
                net.blobs['data'].data[...] = adv_processed
                adv_filtered_out = net.forward()
                adv_filtered_class = adv_filtered_out['prob'][0].argmax()
                if(ori_filtered_class!=true_class):#FP
                    FP+=1
                if (adv_filtered_class!=adv_class):
                    TP+=1
                    if (adv_filtered_class == true_class):
                        TTP+=1
                else:
                    FN+=1                    
        print("Overall results: ")
        str1 = '%d-%d-%d: TP = %d; FN = %d; FP = %d; TTP = %d' % (Total,original_classified_wrong_number,disturbed_failure_number,TP,FN,FP,TTP)
        print(str1)
        endTime  = time.time()
        print('lasting ', endTime-startTime, 'seconds')
        Recall=TP*1.0/(TP+FN)
        Precision=TP*1.0/(TP+FP)
        print('********************************')
        print ("Recall: ",Recall)
        print ("Precision: ",Precision)
        print('********************************')


# In[6]:


def trainCrossMeanFilters(rootDir,true_class):
    print(rootDir)
    coefficient = [5,9, 13, 17]
    kernelIndex = -1
    for kernelSize in xrange(3,10,2):
        startTime = time.time()
        print('cross filter, size = ', kernelSize)
        list_dirs = os.walk(rootDir)
        original_classified_wrong_number = 0
        disturbed_failure_number = 0
        Total=TTP=TP=FP=FN=0
        start = (kernelSize-1)//2
        end = 224-start
        kernelIndex+=1
        for root, dirs, files in list_dirs:
            for f in files: 
                original_data=transformer.preprocess('data', caffe.io.load_image(os.path.join(root, f)))            
                net.blobs['data'].data[...] = original_data
                ori_out = net.forward()
                pred_class = ori_out['prob'][0].argmax()
                if pred_class != true_class:
                    original_classified_wrong_number+=1
                    continue
                grad = compute_gradient(original_data, pred_class)
                delta = np.sign(grad)
                adversarial_data=add_matrix(original_data,delta,1.0)
                normalization(adversarial_data)
                net.blobs['data'].data[...] = adversarial_data
                adv_out = net.forward()
                adv_class = adv_out['prob'][0].argmax()
                if adv_class == true_class:
                    disturbed_failure_number+=1
                    continue
                Total+=1
                ori_processed = crossMeanFilterOperations(original_data, start, end, coefficient[kernelIndex])
                net.blobs['data'].data[...] = ori_processed
                ori_filtered_out = net.forward()
                ori_filtered_class = ori_filtered_out['prob'][0].argmax()
                adv_processed = crossMeanFilterOperations(adversarial_data, start, end, coefficient[kernelIndex])
                net.blobs['data'].data[...] = adv_processed
                adv_filtered_out = net.forward()
                adv_filtered_class = adv_filtered_out['prob'][0].argmax()
                if(ori_filtered_class!=true_class):#FP
                    FP+=1
                if (adv_filtered_class!=adv_class):
                    TP+=1
                    if (adv_filtered_class == true_class):
                        TTP+=1
                else:
                    FN+=1
        print("Overall results: ")
        str1 = '%d-%d-%d: TP = %d; FN = %d; FP = %d; TTP = %d' % (Total,original_classified_wrong_number,disturbed_failure_number,TP,FN,FP,TTP)
        print(str1)
        endTime  = time.time()
        print('lasting ', endTime-startTime, 'seconds')
        Recall=TP*1.0/(TP+FN)
        Precision=TP*1.0/(TP+FP)
        print('********************************')
        print ("Recall: ",Recall)
        print ("Precision: ",Precision)
        print('********************************')


# In[10]:


def trainScalarQuantization(rootDir,true_class,left=True):
    print(rootDir)
    intervals = [128,85,64,51,43,37,32,28,26]
    for intervalIndex in range(9):
        startTime = time.time()
        print('NBinterval =  ', intervalIndex+2, '; interval size = ', intervals[intervalIndex])
        list_dirs = os.walk(rootDir)
        original_classified_wrong_number = 0
        disturbed_failure_number = 0
        Total=TTP=TP=FP=FN=0
        for root, dirs, files in list_dirs:
            for f in files: 
                original_data=transformer.preprocess('data', caffe.io.load_image(os.path.join(root, f)))            
                net.blobs['data'].data[...] = original_data
                ori_out = net.forward()
                pred_class = ori_out['prob'][0].argmax()
                if pred_class != true_class:
                    original_classified_wrong_number+=1
                    continue
                grad = compute_gradient(original_data, pred_class)
                delta = np.sign(grad)
                adversarial_data=add_matrix(original_data,delta,1.0)
                normalization(adversarial_data)
                net.blobs['data'].data[...] = adversarial_data
                adv_out = net.forward()
                adv_class = adv_out['prob'][0].argmax()
                if adv_class == true_class:
                    disturbed_failure_number+=1
                    continue
                Total+=1
                ori_processed = scalarQuantization(original_data, intervals[intervalIndex], left=left)
                net.blobs['data'].data[...] = ori_processed
                ori_filtered_out = net.forward()
                ori_filtered_class = ori_filtered_out['prob'][0].argmax()
                adv_processed = scalarQuantization(adversarial_data, intervals[intervalIndex], left=left)
                net.blobs['data'].data[...] = adv_processed
                adv_filtered_out = net.forward()
                adv_filtered_class = adv_filtered_out['prob'][0].argmax()
                if(ori_filtered_class!=true_class):
                    FP+=1
                if (adv_filtered_class!=adv_class):
                    TP+=1
                    if (adv_filtered_class == true_class):
                        TTP+=1
                else:
                    FN+=1
        print("Overall results: ")
        str1 = '%d-%d-%d: TP = %d; FN = %d; FP = %d; TTP = %d' % (Total,original_classified_wrong_number,disturbed_failure_number,TP,FN,FP,TTP)
        print(str1)
        endTime  = time.time()
        print('lasting ', endTime-startTime, 'seconds')
        Recall=TP*1.0/(TP+FN)
        Precision=TP*1.0/(TP+FP)
        print('********************************')
        print ("Recall: ",Recall)
        print ("Precision: ",Precision)
        print('********************************')


# In[11]:


caffe.set_mode_gpu()
net = load_model()
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2,0,1))
transformer.set_raw_scale('data', 255)  #images in [0,255] range instead of [0,1]
transformer.set_channel_swap('data', (2,1,0))  #channels in BGR order instead of RGB    
diamonds = [np.array([[0,1,0],[1,1,1],[0,1,0]]), np.array([[0,0,1,0,0],[0,1,1,1,0],[1,1,1,1,1],[0,1,1,1,0],[0,0,1,0,0]]),
                    np.array([[0,0,0,1,0,0,0],[0,0,1,1,1,0,0],[0,1,1,1,1,1,0],[1,1,1,1,1,1,1],[0,1,1,1,1,1,0],[0,0,1,1,1,0,0],[0,0,0,1,0,0,0]]),
                    np.array([[0,0,0,0,1,0,0,0,0],[0,0,0,1,1,1,0,0,0],[0,0,1,1,1,1,1,0,0],[0,1,1,1,1,1,1,1,0],[1,1,1,1,1,1,1,1,1],[0,1,1,1,1,1,1,1,0],[0,0,1,1,1,1,1,0,0],[0,0,0,1,1,1,0,0,0],[0,0,0,0,1,0,0,0,0],])]
imageDirs = ['/home/ll/DeepDetector/TestImagenet/Goldfish','/home/ll/DeepDetector/TestImagenet/Clock',
            '/home/ll/DeepDetector/TestImagenet/Pineapple']
imageLabels = [1,530,953]


# In[ ]:


for i in range(3):
    trainBoxMeanFilters(imageDirs[i],imageLabels[i])
    trainDiamondMeanFilters(imageDirs[i],imageLabels[i],diamonds)
    trainCrossMeanFilters(imageDirs[i],imageLabels[i])


# In[ ]:


for i in range(3):
    trainScalarQuantization(imageDirs[i],imageLabels[i])
    trainScalarQuantization(imageDirs[i],imageLabels[i],left=False)

