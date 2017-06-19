
# coding: utf-8

# In[2]:

home_root = '/home/ll/' #change it to your own home path
nn_robust_attack_root = '/home/ll/nn_robust_attacks/' #change it to where you put the 'nn_robust_attacks' directory
import sys
sys.path.insert(0,home_root)
sys.path.insert(0,nn_robust_attack_root) 

import os
import tensorflow as tf
import numpy as np
import time
import math
import matplotlib.pyplot as plt

from setup_mnist import MNIST, MNISTModel
from li_attack import CarliniLi


# In[3]:

def generate_data(data, samples, targeted=True, start=0, inception=False):
    inputs = []
    targets = []
    for i in range(samples):
        if targeted:
            if inception:
                seq = random.sample(range(1,1001), 10)
            else:
                seq = range(data.test_labels.shape[1])

            for j in seq:
                if (j == np.argmax(data.test_labels[start+i])) and (inception == False):
                    continue
                inputs.append(data.test_data[start+i])
                targets.append(np.eye(data.test_labels.shape[1])[j])
        else:
            inputs.append(data.test_data[start+i])
            tempLabel = np.zeros_like(data.test_labels[i])
            tempIndex = data.test_labels[i].argmax()
            tempLabel[(tempIndex+1) % 10] = 1
            targets.append(tempLabel)
    inputs = np.array(inputs)
    targets = np.array(targets)
    return inputs, targets

def mnistPredicate(img, model):
    imgTobePre = np.reshape(img, (1,28,28,1))
    preList = np.squeeze(model.model.predict(imgTobePre))
    return preList.argmax()


# In[4]:

def normalization(image_data):
    image_data = np.reshape(image_data,(28,28))
    image_data2 = np.array(image_data)
    image_data2[image_data2 < -0.5] = -0.5
    image_data2[image_data2 > 0.5] = 0.5
    return image_data2

def seperateScalarQuantizationLeft(image_data, step): #for scalar quantization
    image_data2=np.array(image_data)
    image_data2 = (image_data2+0.5)*255
    image_data2 = np.floor(image_data2/step)
    image_data2 = image_data2*step
    image_data2 = image_data2/255-0.5
    return image_data2

def meanFilter1(image_data): #for smoothing spatial filter
    image_data2=np.zeros_like(image_data)
    for z in range(28):
        for v in range(28):
            if (z<2) or (z>25) or (v<2) or (v>25):
                avg_r=image_data[z][v]
            else:
                avg_r=(image_data[z-2][v]+image_data[z-1][v]+image_data[z][v-2]+image_data[z][v-1]+
                       image_data[z][v]+image_data[z][v+1]+image_data[z][v+2]+image_data[z+1][v]+image_data[z+2][v])/9.0
            image_data2[z][v]=avg_r
    return image_data2

def chooseCloserFilter(original_data,filter_data1,filter_data2): #detection filter
    result_data=np.zeros_like(original_data)
    for j in range(28):
        for k in range(28):
            a=abs(filter_data1[j][k]-original_data[j][k])
            b=abs(filter_data2[j][k]-original_data[j][k])
            if(a<b):
                result_data[j][k]=filter_data1[j][k]
            else:
                result_data[j][k]=filter_data2[j][k]
    return result_data

def expandImage(image_data):
    image_data2 = np.array(image_data)
    image_data2 = (image_data2+0.5)*255
    return image_data2

def meanFilter55forMnist(image_data):
    image_data = image_data.astype(np.float32)
    image_data2=np.zeros_like(image_data)
    for z in range(28):
        for v in range(28):
                if (z<2) or (z>25) or (v<2) or (v>25):
                    avg_r=image_data[z][v]
                else:
                    avg_r=(image_data[z-2][v-2]+image_data[z-2][v-1]+image_data[z-2][v]+image_data[z-2][v+1]+image_data[z-2][v+2]+
                           image_data[z-1][v-2]+image_data[z-1][v-1]+image_data[z-1][v]+image_data[z-1][v+1]+image_data[z-1][v+2]+
                           image_data[z][v-2]+image_data[z][v-1]+image_data[z][v]+image_data[z][v+1]+image_data[z][v+2]+
                           image_data[z+1][v-2]+image_data[z+1][v-1]+image_data[z+1][v]+image_data[z+1][v+1]+image_data[z+1][v+2]+
                           image_data[z+2][v-2]+image_data[z+2][v-1]+image_data[z+2][v]+image_data[z+2][v+1]+image_data[z+2][v+2])/25.0
                image_data2[z][v]=avg_r
    return image_data2


def image2DEntropy55_28(image_data): #computing entropy for a 28*28 MNIST image
    image_data2=np.zeros_like(image_data)
    image_data2=meanFilter55forMnist(image_data)
    B_entropy=0
    B_num = [([0] * 256) for x in range(256)]
    pmf_B = [([0] * 256) for x in range(256)]
    for i in range(28):
        for j in range(28):
            B_val=int (image_data[i][j])
            B_avg=int (image_data2[i][j])
            B_num[B_val][B_avg]=B_num[B_val][B_avg]+1
    for k in range(256):
        for m in range(256):
            pmf_B[k][m]=B_num[k][m]/(28.0*28.0)
    for k in range(256):
        for m in range(256):
            if (pmf_B[k][m]!=0):
                B_entropy=B_entropy+pmf_B[k][m]*math.log10(pmf_B[k][m])/math.log10(2)
    B_entropy=-B_entropy
    return B_entropy


# In[5]:

if __name__ == "__main__":
    with tf.Session() as sess:
        modelPath = '%smodels/mnist' % (nn_robust_attack_root)
        data, model =  MNIST(), MNISTModel(modelPath, sess)
        
        attack = CarliniLi(sess, model, max_iterations=1000)

        inputs, targets = generate_data(data, samples=1000, targeted=False, start=0, inception=False)
        
        original_classified_wrong_number = 0 #number of benign samples that are misclassified 
        disturbed_failure_number = 0 #number of samples that failed to craft corresponding adversarial samples
        test_number = 0 #number of adversarial samples that we generate
        TP = 0
        FN = 0
        FP = 0
        
        advGenTimeSum = 0 #total time for successfully crafting adversarial examples
        oriClassifyTimeSum = 0 #total time for classify the benign samples
        advClassifyTimeSum = 0 #total time for classify the adversarial samples
        oriFilteredTimeSum = 0 #totoal time that our detection filter takes to detect all the benign samples
        advFilteredTimeSum = 0 #total time that our detection filter takes to detect all the adversarial samples

        for i in range(len(targets)):
            print(i)
            
            inputIm = inputs[i:(i+1)]
            target = targets[i:(i+1)]
            
            oriCorrectLabel = data.test_labels[i].argmax()            
            
            octStart = time.time()
            oriPredicatedLabel = mnistPredicate(inputIm, model)
            octEnd = time.time()
            
            if oriPredicatedLabel != oriCorrectLabel:
                original_classified_wrong_number+=1
                continue
            
            attackStart = time.time()
            adv = attack.attack(inputIm,target)
            attackEnd = time.time()
            
            adv = normalization(adv)
            adv = np.reshape(adv, inputIm.shape)
            actStart = time.time()
            advPredicatedLabel = mnistPredicate(adv, model)
            actEnd = time.time()
            
            if advPredicatedLabel == oriCorrectLabel:
                disturbed_failure_number+=1
                continue
            
            test_number+=1
            
            advGenTime = attackEnd-attackStart
            oriClassifyTime = octEnd-octStart
            advClassifyTime = actEnd-actStart
        
            tempInput = np.reshape(inputIm,(28,28))
            tempAdv = np.reshape(adv,(28,28))

            ofctStart = time.time()
            inputForEntropy = expandImage(tempInput)
            oriEntropy = image2DEntropy55_28(inputForEntropy)
            print('oriEntropy = %f' % (oriEntropy))
            if oriEntropy < 8.5:
                inputFinal = seperateScalarQuantizationLeft(tempInput,128)
            elif oriEntropy < 9.5:
                inputFinal = seperateScalarQuantizationLeft(tempInput, 64)
            else:
                inputAfterQ = seperateScalarQuantizationLeft(tempInput,50)
                inputAfterQandF = meanFilter1(inputAfterQ)
                inputFinal = chooseCloserFilter(tempInput, inputAfterQ, inputAfterQandF)
            oriFilteredPredicatedLabel = mnistPredicate(inputFinal, model)
            ofctEnd = time.time()

            afctStart = time.time()
            advForEntropy = expandImage(tempAdv)
            advEntropy = image2DEntropy55_28(advForEntropy)
            print('advEntropy = %f' % (advEntropy))
            if advEntropy < 8.5:
                advFinal = seperateScalarQuantizationLeft(tempAdv, 128)
            elif advEntropy < 9.5:
                advFinal = seperateScalarQuantizationLeft(tempAdv, 64)
            else:
                advAfterQ = seperateScalarQuantizationLeft(tempAdv, 50)
                advAfterQandF = meanFilter1(advAfterQ)
                advFinal = chooseCloserFilter(tempAdv, advAfterQ,advAfterQandF)
            advFilteredPredicatedLabel = mnistPredicate(advFinal, model)
            afctEnd = time.time()

            oriFilteredClassifyTime=(ofctEnd-ofctStart)
            advFilterClassifyTime=(afctEnd-afctStart)
            if advPredicatedLabel != advFilteredPredicatedLabel:
                TP+=1
            else:
                FN+=1
            if oriPredicatedLabel != oriFilteredPredicatedLabel:
                FP+=1
            
            advGenTimeSum+=advGenTime
            oriClassifyTimeSum+=oriClassifyTime
            advClassifyTimeSum+=advClassifyTime
            oriFilteredTimeSum+=oriFilteredClassifyTime
            advFilteredTimeSum+=advFilterClassifyTime
            
            timeStr = "%f-%f-%f-%f-%f" % (advGenTimeSum,oriClassifyTimeSum,advClassifyTimeSum,oriFilteredTimeSum,advFilteredTimeSum)
            labelStr = '%d-%d-%d-%d' % (oriPredicatedLabel, oriFilteredPredicatedLabel, advPredicatedLabel, advFilteredPredicatedLabel)
            statisticStr = '%d-%d-%d: TP = %d, FN = %d, FP = %d' % (test_number, original_classified_wrong_number, disturbed_failure_number, TP, FN, FP)
            print(timeStr)
            print(labelStr)
            print(statisticStr)
            
        Recall = TP/(TP+FN)
        Precision = TP/(TP+FP)
        starStr = '********************************************************'
        recallStr = 'Recall = %.4f' % (Recall)
        precisionStr = 'Precision = %.4f' % (Precision)
        print(starStr)
        print(recallStr)
        print(precisionStr)
        print(starStr)

