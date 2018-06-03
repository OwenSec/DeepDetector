
# coding: utf-8

# In[1]:


nn_robust_attack_root = '~/nn_robust_attacks/'
home_root = '~/'
import sys
sys.path.insert(0,home_root)
sys.path.insert(0,nn_robust_attack_root) 

import os
import tensorflow as tf
import numpy as np
import time
import math
import scipy.misc
import matplotlib.pyplot as plt

from modified_setup_inception import ImageNet, InceptionModel
from li_attack import CarliniLi


# In[2]:


def my_generate_data2(data):
    inputs = []
    sources = []
    targets = []
    for i in range(40):
        inputs.append(data.zebraData[i])
        targets.append(data.zebraLabel[i])
        sources.append(data.zebraLabel[i].argmax())
    
    for i in range(40):
        inputs.append(data.pandaData[i])
        targets.append(data.pandaLabel[i])
        sources.append(data.pandaLabel[i].argmax())

    for i in range(20):
        inputs.append(data.cabData[i])
        targets.append(data.cabLabel[i])
        sources.append(data.cabLabel[i].argmax())
        
    inputs = np.array(inputs)
    targets = np.array(targets)
    return inputs, sources, targets


# In[3]:


def normalization(image_data):
    image_data[image_data<-0.5] = -0.5
    image_data[image_data>0.5] = 0.5

def crossMeanFilterOperations(inputDigit, start, end, coefficient):
    retDigit = np.array(inputDigit, dtype=np.float32)
    for row in range(start, end):
        for col in range(start, end):
            temp0 = inputDigit[row][col][0]
            temp1 = inputDigit[row][col][1]
            temp2 = inputDigit[row][col][2]
            for i in range(1,start+1):
                temp0+=inputDigit[row-i][col][0]
                temp0+=inputDigit[row+i][col][0]
                temp0+=inputDigit[row][col-i][0]
                temp0+=inputDigit[row][col+i][0]
                temp1+=inputDigit[row-i][col][1]
                temp1+=inputDigit[row+i][col][1]
                temp1+=inputDigit[row][col-i][1]
                temp1+=inputDigit[row][col+i][1]
                temp2+=inputDigit[row-i][col][2]
                temp2+=inputDigit[row+i][col][2]
                temp2+=inputDigit[row][col-i][2]
                temp2+=inputDigit[row][col+i][2]
            retDigit[row][col][0]= temp0/coefficient
            retDigit[row][col][1] = temp1/coefficient
            retDigit[row][col][2] = temp2/coefficient
    return retDigit

def scalarQuantization(inputDigit, interval):
    retDigit = (inputDigit+0.5)*255
    retDigit//=interval
    retDigit*=interval
    retDigit/=255.0
    retDigit-=0.5
    return retDigit

def oneDEntropy(inputDigit):
    expandDigit = np.array((inputDigit+0.5)*255,dtype=np.int16)
    f1 = np.zeros(256)
    f2 = np.zeros(256)
    f3 = np.zeros(256)
    for i in range(299):
        for j in range(299):
            f1[expandDigit[i][j][0]]+=1
            f2[expandDigit[i][j][1]]+=1
            f3[expandDigit[i][j][2]]+=1
    f1/=89401.0
    f2/=89401.0
    f3/=89401.0
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
    for j in range(299):
        for k in range(299):
            for i in range(3):
                a=abs(filter_data1[j][k][i]-original_data[j][k][i])
                b=abs(filter_data2[j][k][i]-original_data[j][k][i])
                if(a<b):
                    result_data[j][k][i]=filter_data1[j][k][i]
                else:
                    result_data[j][k][i]=filter_data2[j][k][i]
    return result_data


# In[4]:


if __name__ == "__main__":
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        data, model =  ImageNet(), InceptionModel(sess)
        img = tf.placeholder(tf.uint8, (299,299,3))
        softmax_tensor = tf.import_graph_def(
             sess.graph.as_graph_def(),
             input_map={'DecodeJpeg:0': tf.reshape(img,((299,299,3)))},
             return_elements=['softmax/logits:0'])
        
        attack = CarliniLi(sess, model, max_iterations=1000,targeted=False)

        inputs, sources, targets = my_generate_data2(data)
        
        original_classified_wrong_number = 0
        disturbed_failure_number = 0
        test_number = 0
        TTP = 0
        TP = 0
        FN = 0
        FP = 0
                
        advGenTimeSum = 0
        oriClassifyTimeSum = 0
        advClassifyTimeSum = 0
        oriFilteredTimeSum = 0
        advFilteredTimeSum = 0

        for i in range(len(targets)):
            print(i)
            
            inputIm = inputs[i:(i+1)]
            target = targets[i:(i+1)]
            
            oriCorrectLabel = sources[i]            
            
            octStart = time.time()
            oriPredicatedProb, oriPredicatedLabel = model.my_predict(inputIm, img, softmax_tensor)
            octEnd = time.time()
                        
            attackStart = time.time()
            adv = attack.attack(inputIm,target)
            attackEnd = time.time()
            
            normalization(adv)
            adv = np.reshape(adv, inputIm.shape)
            actStart = time.time()
            advPredicatedProb, advPredicatedLabel = model.my_predict(adv, img, softmax_tensor)
            actEnd = time.time()
            
            if advPredicatedLabel == oriCorrectLabel:
                print('Attack Failure!')
                disturbed_failure_number+=1
                continue
            
            test_number+=1
            
            advGenTime = attackEnd-attackStart
            oriClassifyTime = octEnd-octStart
            advClassifyTime = actEnd-actStart
        
            tempInput = np.reshape(inputIm,(299,299,3))
            tempAdv = np.reshape(adv,(299,299,3))

            ofctStart = time.time()
            oriEntropy = oneDEntropy(tempInput)
            oriEntropyStr = 'oriEntropy = %f' % (oriEntropy)
            print(oriEntropyStr)
            if oriEntropy < 4:
                inputFinal = scalarQuantization(tempInput,128)
            elif oriEntropy < 5:
                inputFinal = scalarQuantization(tempInput, 64)
            else:
                inputAfterQ = scalarQuantization(tempInput,43)
                inputAfterQandF = crossMeanFilterOperations(inputAfterQ,3,296,13)
                inputFinal = chooseCloserFilter(tempInput, inputAfterQ, inputAfterQandF)
            oriFilteredPredicatedProb, oriFilteredPredicatedLabel = model.my_predict(inputFinal, img, softmax_tensor)
            ofctEnd = time.time()

            afctStart = time.time()
            advEntropy = oneDEntropy(tempAdv)
            advEntropyStr = 'advEntropy = %f' % (advEntropy)
            print(advEntropyStr)
            if advEntropy < 4:
                advFinal = scalarQuantization(tempAdv, 128)
            elif advEntropy < 5:
                advFinal = scalarQuantization(tempAdv, 64)
            else:
                advAfterQ = scalarQuantization(tempAdv, 43)
                advAfterQandF = crossMeanFilterOperations(advAfterQ,3,296,13)
                advFinal = chooseCloserFilter(tempAdv, advAfterQ,advAfterQandF)
            advFilteredPredicatedProb, advFilteredPredicatedLabel = model.my_predict(advFinal, img, softmax_tensor)
            afctEnd = time.time()

            oriFilteredClassifyTime=(ofctEnd-ofctStart)
            advFilterClassifyTime=(afctEnd-afctStart)
            if advPredicatedLabel != advFilteredPredicatedLabel:
                TP+=1
                if advFilteredPredicatedLabel == oriCorrectLabel:
                    TTP+=1
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
            statisticStr = '%d-%d: TTP = %d, TP = %d, FN = %d, FP = %d' % (test_number, disturbed_failure_number, TTP, TP, FN, FP)
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

