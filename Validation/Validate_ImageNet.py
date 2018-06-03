
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

import caffe

plt.rcParams['figure.figsize'] = (4, 4)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'


# In[2]:


def load_model():
    BATCH_SIZE = 1
    model_def = caffe_root + '/models/bvlc_googlenet/deploy.prototxt'
    net_weights =caffe_root + '/models/bvlc_googlenet/bvlc_googlenet.caffemodel'
    net = caffe.Net(model_def, net_weights, caffe.TEST)
    # change batch size to 1 for faster processing
    # this just means that we're only processing one image at a time instead of like 50
    shape = list(net.blobs['data'].data.shape)
    shape[0] = BATCH_SIZE
    net.blobs['data'].reshape(*shape)
    net.blobs['prob'].reshape(BATCH_SIZE, )
    net.reshape()
    return net

def compute_gradient(image, intended_outcome):
    predict(image, display_output=False)
    # Get an empty set of probabilities
    probs = np.zeros_like(net.blobs['prob'].data)
    # Set the probability for the outcome to -1
    probs[0][intended_outcome] = -1 
    # Do backpropagation to calculate the gradient for that outcome
    gradient = net.backward(prob=probs)
    return gradient['data'].copy()

def display(data):
    plt.imshow(transformer.deprocess('data', data))
    
def get_label_name(num):
    options = labels[num].split(',')
    # remove the tag
    options[0] = ' '.join(options[0].split(' ')[1:])
    return ','.join(options[:2])

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

def predict(data, n_preds=6, display_output=False):
    net.blobs['data'].data[...] = data
    #if display_output:
        #display(data)
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


# In[3]:


def normalization(image_data):
    image_data[image_data<0] = 0
    image_data[image_data>255] = 255.0

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

def scalarQuantization(inputDigit, interval):
    retDigit = np.array(inputDigit,dtype=np.float32)
    retDigit//=interval
    retDigit*=interval
    return retDigit

def oneDEntropy(inputDigit):
    expandDigit = np.array(inputDigit,dtype=np.int16)
    f1 = np.zeros(256)
    f2 = np.zeros(256)
    f3 = np.zeros(256)
    for i in range(224):
        for j in range(224):
            f1[expandDigit[0][i][j]]+=1
            f2[expandDigit[1][i][j]]+=1
            f3[expandDigit[2][i][j]]+=1
    f1/=50176.0
    f2/=50176.0
    f3/=50176.0
    H1 = 0
    H2 = 0
    H3 = 0
    for i in range(256):
        if f1[i] > 0:
            H1+=f1[i]*math.log(f1[i],2)
        if f2[i] > 0:
            H2+=f2[i]*math.log(f2[i],2)
        if f3[i] > 0:
            H3+=f3[i]*math.log(f3[i],2)
    return -(H1+H2+H3)/3.0

def chooseCloserFilter(original_data,filter_data1,filter_data2):
    result_data=np.zeros_like(original_data)
    for i in range(3):
        for j in range(224):
            for k in range(224):
                a=abs(filter_data1[i][j][k]-original_data[i][j][k])
                b=abs(filter_data2[i][j][k]-original_data[i][j][k])
                if(a<b):
                    result_data[i][j][k]=filter_data1[i][j][k]
                else:
                    result_data[i][j][k]=filter_data2[i][j][k]
    return result_data


# In[4]:


#step is the size of quantization interval,epsilon defines the perturbation(set as 1.0 in [0,255] scale)
def DDforFGSMImageNet(rootDir,true_class):
    list_dirs = os.walk(rootDir) 
    NbLowEntropy = 0
    NbMidEntropy = 0
    NbHighEntropy = 0
    lowTP = 0
    lowFN = 0
    lowFP = 0
    midTP = 0
    midFN = 0
    midFP = 0
    highTP = 0
    highFN = 0
    highFP = 0    
    disturbedFailure = 0
    for root, dirs, files in list_dirs: 
        for f in files: 
            original_data=transformer.preprocess('data', caffe.io.load_image(os.path.join(root, f)))
            net.blobs['data'].data[...] = original_data
            ori_out = net.forward()
            pred_class = ori_out['prob'][0].argmax()
            grad = compute_gradient(original_data, pred_class)
            delta = np.sign(grad)
            adversarial_data=add_matrix(original_data,delta,1.0)
            net.blobs['data'].data[...] = adversarial_data
            adv_out = net.forward()
            adv_class = adv_out['prob'][0].argmax()
            if pred_class != true_class:
                continue
            if adv_class == true_class:
                disturbedFailure+=1
                continue
            ori_entropy=oneDEntropy(original_data)
            print('ori_entropy = ', ori_entropy)
            if ori_entropy < 4:
                NbLowEntropy+=1
                oriF = scalarQuantization(original_data,128)
                net.blobs['data'].data[...] = oriF
                ori_filtered_out = net.forward()
                ori_filtered_class = ori_filtered_out['prob'][0].argmax()
                if ori_filtered_class != true_class:
                    lowFP+=1
            elif ori_entropy < 5:
                NbMidEntropy+=1
                oriF = scalarQuantization(original_data,64)
                net.blobs['data'].data[...] = oriF
                ori_filtered_out = net.forward()
                ori_filtered_class = ori_filtered_out['prob'][0].argmax()
                if ori_filtered_class != true_class:
                    midFP+=1
            else:
                NbHighEntropy+=1
                oriF = scalarQuantization(original_data,43)
                net.blobs['data'].data[...] = oriF
                ori_filtered_out = net.forward()
                ori_filtered_class = ori_filtered_out['prob'][0].argmax()
                if ori_filtered_class != true_class:
                    highFP+=1
            
            adv_entropy=oneDEntropy(adversarial_data)
            print('adv_entropy = ',adv_entropy)
            if adv_entropy < 4:
                NbLowEntropy+=1
                advF = scalarQuantization(adversarial_data,128)
                net.blobs['data'].data[...] = advF
                adv_filtered_out = net.forward()
                adv_filtered_class = adv_filtered_out['prob'][0].argmax()
                if adv_filtered_class != adv_class:
                    lowTP+=1
                else:
                    lowFN+=1
            elif adv_entropy < 5:
                NbMidEntropy+=1
                advF = scalarQuantization(adversarial_data,64)
                net.blobs['data'].data[...] = advF
                adv_filtered_out = net.forward()
                adv_filtered_class = adv_filtered_out['prob'][0].argmax()
                if adv_filtered_class != adv_class:
                    midTP+=1
                else:
                    midFN+=1
            else:
                NbHighEntropy+=1
                advF = scalarQuantization(adversarial_data,43)
                net.blobs['data'].data[...] = advF
                adv_filtered_out = net.forward()
                adv_filtered_class = adv_filtered_out['prob'][0].argmax()
                if adv_filtered_class != adv_class:
                    highTP+=1
                else:
                    highFN+=1
            
            print("disturbedFailure:", disturbedFailure)
            print("Low: ",NbLowEntropy, " - ",lowTP," - ",lowFN," - ",lowFP)
            print("Mid: ",NbMidEntropy, " - ",midTP," - ",midFN," - ",midFP)
            print("High: ",NbHighEntropy, " - ",highTP," - ",highFN," - ",highFP)
    print("Overall results: ")
    print("disturbedFailure:", disturbedFailure)
    print("Low: ",NbLowEntropy, " - ",lowTP," - ",lowFN," - ",lowFP)
    print("Mid: ",NbMidEntropy, " - ",midTP," - ",midFN," - ",midFP)
    print("High: ",NbHighEntropy, " - ",highTP," - ",highFN," - ",highFP)

    lowRecall=lowTP*1.0/(lowTP+lowFN)
    lowPrecision=lowTP*1.0/(lowTP+lowFP)
    midRecall=midTP*1.0/(midTP+midFN)
    midPrecision=midTP*1.0/(midTP+midFP)
    highRecall=highTP*1.0/(highTP+highFN)
    highPrecision=highTP*1.0/(highTP+highFP)

    print ("lowRecall: ",lowRecall)
    print ("lowPrecision: ",lowPrecision)
    print ("midRecall: ",midRecall)
    print ("midPrecision: ",midPrecision)   
    print ("highRecall: ",highRecall)
    print ("highPrecision: ",highPrecision)


# In[5]:


# Load the labels (so we know whether 242 means 'adorable puppy' or 'garbage can')
imagenet_labels_filename = caffe_root + '/data/ilsvrc12/synset_words.txt'
try:
    labels = np.loadtxt(imagenet_labels_filename, str, delimiter='\t')
except:
    get_ipython().system('/opt/caffe/data/ilsvrc12/get_ilsvrc_aux.sh')
    labels = np.loadtxt(imagenet_labels_filename, str, delimiter='\t')
    
# Set Caffe to GPU mode 
caffe.set_mode_gpu()
#Load CaffeNet for FGSM detection
# Load our model! trained by the GOOGLES! <3
net = load_model()
#transformer 
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2,0,1))
transformer.set_raw_scale('data', 255)  #images in [0,255] range instead of [0,1]
transformer.set_channel_swap('data', (2,1,0))  #channels in BGR order instead of RGB    


# In[8]:


DDforFGSMImageNet('/home/ll/DeepDetector/TotalImages/Jellyfish',107)

