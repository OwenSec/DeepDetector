
# coding: utf-8

# In[1]:

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
pd.set_option('display.mpl_style', 'default')
get_ipython().magic(u'matplotlib inline')

caffe_root = '~/caffe'
import sys
sys.path.insert(0, caffe_root + 'python')
import caffe

plt.rcParams['figure.figsize'] = (4, 4)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'


# In[2]:

def load_model():
    BATCH_SIZE = 1
    model_def = caffe_root + '/models/bvlc_googlenet/deploy.prototxt.test'
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

def predict(data, n_preds=6, display_output=True):
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

def meanFilter(image_data,kernel_size):#mean filter without padding 1/9 or 1/25
    image_data2=np.zeros_like(image_data)
    if (kernel_size==3):
        for z in range(224):
            for v in range(224):
                if (z==0) or (z==223) or (v==0) or (v==223):
                    avg_r=image_data[0][z][v]
                    avg_g=image_data[1][z][v]
                    avg_b=image_data[2][z][v]
                else:
                    avg_r=(image_data[0][z-1][v-1]+image_data[0][z-1][v]+image_data[0][z-1][v+1]+image_data[0][z][v-1]+image_data[0][z][v]+image_data[0][z][v+1]+image_data[0][z+1][v-1]+image_data[0][z+1][v]+image_data[0][z+1][v+1])/9.0
                    avg_g=(image_data[1][z-1][v-1]+image_data[1][z-1][v]+image_data[1][z-1][v+1]+image_data[1][z][v-1]+image_data[1][z][v]+image_data[1][z][v+1]+image_data[1][z+1][v-1]+image_data[1][z+1][v]+image_data[1][z+1][v+1])/9.0
                    avg_b=(image_data[2][z-1][v-1]+image_data[2][z-1][v]+image_data[2][z-1][v+1]+image_data[2][z][v-1]+image_data[2][z][v]+image_data[2][z][v+1]+image_data[2][z+1][v-1]+image_data[2][z+1][v]+image_data[2][z+1][v+1])/9.0
                image_data2[0][z][v]=avg_r
                image_data2[1][z][v]=avg_g
                image_data2[2][z][v]=avg_b
    elif (kernel_size==5):
        for z in range(224):
            for v in range(224):
                if (z<2) or (z>221) or (v<2) or (v>221):
                    avg_r=image_data[0][z][v]
                    avg_g=image_data[1][z][v]
                    avg_b=image_data[2][z][v]
                else:
                    avg_r=(image_data[0][z-2][v-2]+image_data[0][z-2][v-1]+image_data[0][z-2][v]+image_data[0][z-2][v+1]+image_data[0][z-2][v+2]+image_data[0][z-1][v-2]+image_data[0][z-1][v-1]+image_data[0][z-1][v]+image_data[0][z-1][v+1]+image_data[0][z-1][v+2]+image_data[0][z][v-2]+image_data[0][z][v-1]+image_data[0][z][v]+image_data[0][z][v+1]+image_data[0][z][v+2]+image_data[0][z+1][v-2]+image_data[0][z+1][v-1]+image_data[0][z+1][v]+image_data[0][z+1][v+1]+image_data[0][z+1][v+2]+image_data[0][z+2][v-2]+image_data[0][z+2][v-1]+image_data[0][z+2][v]+image_data[0][z+2][v+1]+image_data[0][z+2][v+2])/25.0
                    avg_g=(image_data[1][z-2][v-2]+image_data[1][z-2][v-1]+image_data[1][z-2][v]+image_data[1][z-2][v+1]+image_data[1][z-2][v+2]+image_data[1][z-1][v-2]+image_data[1][z-1][v-1]+image_data[1][z-1][v]+image_data[1][z-1][v+1]+image_data[1][z-1][v+2]+image_data[1][z][v-2]+image_data[1][z][v-1]+image_data[1][z][v]+image_data[1][z][v+1]+image_data[1][z][v+2]+image_data[1][z+1][v-2]+image_data[1][z+1][v-1]+image_data[1][z+1][v]+image_data[1][z+1][v+1]+image_data[1][z+1][v+2]+image_data[1][z+2][v-2]+image_data[1][z+2][v-1]+image_data[1][z+2][v]+image_data[1][z+2][v+1]+image_data[1][z+2][v+2])/25.0
                    avg_b=(image_data[2][z-2][v-2]+image_data[2][z-2][v-1]+image_data[2][z-2][v]+image_data[2][z-2][v+1]+image_data[2][z-2][v+2]+image_data[2][z-1][v-2]+image_data[2][z-1][v-1]+image_data[2][z-1][v]+image_data[2][z-1][v+1]+image_data[2][z-1][v+2]+image_data[2][z][v-2]+image_data[2][z][v-1]+image_data[2][z][v]+image_data[2][z][v+1]+image_data[2][z][v+2]+image_data[2][z+1][v-2]+image_data[2][z+1][v-1]+image_data[2][z+1][v]+image_data[2][z+1][v+1]+image_data[2][z+1][v+2]+image_data[2][z+2][v-2]+image_data[2][z+2][v-1]+image_data[2][z+2][v]+image_data[2][z+2][v+1]+image_data[2][z+2][v+2])/25.0
                image_data2[0][z][v]=avg_r
                image_data2[1][z][v]=avg_g
                image_data2[2][z][v]=avg_b
    return image_data2

def meanFilter1(image_data,kernel_size):#cross mean filter 1/5 or 1/9
    image_data2=np.zeros_like(image_data)
    if (kernel_size==3):
        for z in range(224):
            for v in range(224):
                if (z==0) or (z==223) or (v==0) or (v==223):
                    avg_r=image_data[0][z][v]
                    avg_g=image_data[1][z][v]
                    avg_b=image_data[2][z][v]
                else:
                    avg_r=(image_data[0][z-1][v]+image_data[0][z][v-1]+image_data[0][z][v]+image_data[0][z][v+1]+image_data[0][z+1][v])/5.0
                    avg_g=(image_data[1][z-1][v]+image_data[1][z][v-1]+image_data[1][z][v]+image_data[1][z][v+1]+image_data[1][z+1][v])/5.0
                    avg_b=(image_data[2][z-1][v]+image_data[2][z][v-1]+image_data[2][z][v]+image_data[2][z][v+1]+image_data[2][z+1][v])/5.0
                image_data2[0][z][v]=avg_r
                image_data2[1][z][v]=avg_g
                image_data2[2][z][v]=avg_b
    elif (kernel_size==5):
        for z in range(224):
            for v in range(224):
                if (z<2) or (z>221) or (v<2) or (v>221):
                    avg_r=image_data[0][z][v]
                    avg_g=image_data[1][z][v]
                    avg_b=image_data[2][z][v]
                else:
                    avg_r=(image_data[0][z-2][v]+image_data[0][z-1][v]+image_data[0][z][v-2]+image_data[0][z][v-1]+image_data[0][z][v]+image_data[0][z][v+1]+image_data[0][z][v+2]+image_data[0][z+1][v]+image_data[0][z+2][v])/9.0
                    avg_g=(image_data[1][z-2][v]+image_data[1][z-1][v]+image_data[1][z][v-2]+image_data[1][z][v-1]+image_data[1][z][v]+image_data[1][z][v+1]+image_data[1][z][v+2]+image_data[1][z+1][v]+image_data[1][z+2][v])/9.0
                    avg_b=(image_data[2][z-2][v]+image_data[2][z-1][v]+image_data[2][z][v-2]+image_data[2][z][v-1]+image_data[2][z][v]+image_data[2][z][v+1]+image_data[2][z][v+2]+image_data[2][z+1][v]+image_data[2][z+2][v])/9.0
                image_data2[0][z][v]=avg_r
                image_data2[1][z][v]=avg_g
                image_data2[2][z][v]=avg_b
    return image_data2

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

#Imagenet 2D entropy, mean filter: 5*5 1/25
def image2DEntropy55(image_data):
    image_data2=np.zeros_like(image_data)
    image_data2=meanFilter(image_data,5)
    B_entropy=0
    G_entropy=0
    R_entropy=0
    B_num = [([0] * 256) for x in range(256)]
    G_num = [([0] * 256) for x in range(256)]
    R_num = [([0] * 256) for x in range(256)]
    pmf_B = [([0] * 256) for x in range(256)]
    pmf_G = [([0] * 256) for x in range(256)]
    pmf_R = [([0] * 256) for x in range(256)]
    for i in range(224):
        for j in range(224):
            B_val=int (image_data[0][i][j])
            G_val=int (image_data[1][i][j])
            R_val=int (image_data[2][i][j])
            B_avg=int (image_data2[0][i][j])
            G_avg=int (image_data2[1][i][j])
            R_avg=int (image_data2[2][i][j])
            B_num[B_val][B_avg]=B_num[B_val][B_avg]+1
            G_num[G_val][G_avg]=G_num[G_val][G_avg]+1
            R_num[R_val][R_avg]=R_num[R_val][R_avg]+1
    for k in range(256):
        for m in range(256):
            pmf_B[k][m]=B_num[k][m]/(224.0*224.0)
            pmf_G[k][m]=G_num[k][m]/(224.0*224.0)
            pmf_R[k][m]=R_num[k][m]/(224.0*224.0)
    for k in range(256):
        for m in range(256):
            if (pmf_B[k][m]!=0):
                B_entropy=B_entropy+pmf_B[k][m]*math.log10(pmf_B[k][m])/math.log10(2)
            if (pmf_G[k][m]!=0):
                G_entropy=G_entropy+pmf_G[k][m]*math.log10(pmf_G[k][m])/math.log10(2)
            if (pmf_R[k][m]!=0):
                R_entropy=R_entropy+pmf_R[k][m]*math.log10(pmf_R[k][m])/math.log10(2)
    B_entropy=-B_entropy
    G_entropy=-G_entropy
    R_entropy=-R_entropy
    entropy=(B_entropy+G_entropy+R_entropy)/3
    return entropy

#RGB各自阶梯量化左值
def seperateScalarQuantizationLeft(image_data,step_B,step_G,step_R):
    image_data2=np.zeros_like(image_data)
    for j in range(224):
        for k in range(224):
            segment_B=np.floor(image_data[0][j][k]/step_B)
            image_data2[0][j][k]=segment_B*step_B
            segment_G=np.floor(image_data[1][j][k]/step_G)
            image_data2[1][j][k]=segment_G*step_G
            segment_R=np.floor(image_data[2][j][k]/step_R)
            image_data2[2][j][k]=segment_R*step_R
    return image_data2
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
    Total=0.0
    TP=0.0
    FP=0.0
    FN=0.0    
    for root, dirs, files in list_dirs: 
        for d in dirs: 
            print os.path.join(root, d)      
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
            if (pred_class==true_class) and (adv_class!=true_class):
                print os.path.join(root, f) 
                Total=Total+1
                ori_entropy=image2DEntropy55(original_data)
                if (ori_entropy<8.50):
                    im3=seperateScalarQuantizationLeft(original_data,128,128,128)
                elif (ori_entropy<=9.50):
                    im3=seperateScalarQuantizationLeft(original_data,64,64,64)
                else: #ori_entropy>9.50
                    im1=seperateScalarQuantizationLeft(original_data,50,50,50)
                    im2=meanFilter1(im1,5)
                    im3=chooseCloserFilter(original_data,im1,im2)
                net.blobs['data'].data[...] = im3
                ori_filtered_out = net.forward()
                ori_filtered_class = ori_filtered_out['prob'][0].argmax()
                adv_entropy=image2DEntropy55(adversarial_data)
                if (adv_entropy<8.50):
                    im6=seperateScalarQuantizationLeft(adversarial_data,128,128,128)
                elif (adv_entropy<=9.50):
                    im6=seperateScalarQuantizationLeft(adversarial_data,64,64,64)
                else: #adv_entropy>9.50
                    im4=seperateScalarQuantizationLeft(adversarial_data,50,50,50)
                    im5=meanFilter1(im4,5)
                    im6=chooseCloserFilter(adversarial_data,im4,im5)
                net.blobs['data'].data[...] = im6
                adv_filtered_out = net.forward()
                adv_filtered_class = adv_filtered_out['prob'][0].argmax()
                print("Ori predicted class is #{}.".format(pred_class))
                probs = predict(original_data, n_preds=2)
                print("Ori-filter predicted class is #{}.".format(ori_filtered_class))
                probs = predict(im3, n_preds=2)
                print("Adv predicted class is #{}.".format(adv_class))
                probs = predict(adversarial_data, n_preds=2)
                print("Adv-filter predicted class is #{}.".format(adv_filtered_class))
                probs = predict(im6, n_preds=2)
                if(ori_filtered_class!=true_class):#FP
                    FP=FP+1
                if (adv_filtered_class!=adv_class):
                    TP=TP+1
                if (adv_filtered_class==adv_class):
                    FN=FN+1
                Recall=TP/Total*100.0
                if ((TP+FP)!=0):
                    Precision=TP/(TP+FP)*100.0
                else:
                    Precision=0
                print("Total: ",Total, "TP: ",TP,"FP: ",FP,"FN: ",FN,"Recall: ",Recall,"  Precision: ",Precision)
    print("Overall results: ")
    print("Total: ",Total, "TP: ",TP,"FP: ",FP,"FN: ",FN)
    Recall=TP/Total*100.0
    Precision=TP/(TP+FP)*100.0
    print ("Recall: ",Recall)
    print ("Precision: ",Precision)


# In[5]:

# Load the labels (so we know whether 242 means 'adorable puppy' or 'garbage can')
imagenet_labels_filename = caffe_root + '/data/ilsvrc12/synset_words.txt'
try:
    labels = np.loadtxt(imagenet_labels_filename, str, delimiter='\t')
except:
    get_ipython().system(u'/opt/caffe/data/ilsvrc12/get_ilsvrc_aux.sh')
    labels = np.loadtxt(imagenet_labels_filename, str, delimiter='\t')
    
# Set Caffe to CPU mode 
caffe.set_mode_cpu()
#Load CaffeNet for FGSM detection
# Load our model! trained by the GOOGLES! <3
net = load_model()
#transformer 
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2,0,1))
transformer.set_raw_scale('data', 255)  #images in [0,255] range instead of [0,1]
transformer.set_channel_swap('data', (2,1,0))  #channels in BGR order instead of RGB    



# In[6]:

#test four ImageNet class
DDforFGSMImageNet('~/n02391049_ps',340)
DDforFGSMImageNet('~/n02510455_ps',388)
DDforFGSMImageNet('~/n02930766_ps',468)
DDforFGSMImageNet('~/n07753275_ps',953)


# In[ ]:



