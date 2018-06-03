
# coding: utf-8

# In[1]:


home_root = '~/' #change it to your own home path
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
from l2_attack import CarliniL2


# In[2]:


def generate_data(data, samples, targeted=False, start=9000, inception=False):
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
            targets.append(data.test_labels[start+i])
    inputs = np.array(inputs)
    targets = np.array(targets)
    return inputs, targets

def getProbabilities(img, model,sess):
    imgTobePre = np.reshape(img, (1,28,28,1))
    preList = np.squeeze(model.model.predict(imgTobePre))
    return max(sess.run(tf.nn.softmax(preList)))
    
def mnistPredicate(img, model):
    imgTobePre = np.reshape(img, (1,28,28,1))
    preList = np.squeeze(model.model.predict(imgTobePre))
    return preList.argmax()


# In[3]:


def normalization(image_data):
    image_data[image_data<-0.5] = -0.5
    image_data[image_data>0.5] = 0.5
    
def scalarQuantization(inputDigit, interval):
    retDigit = (inputDigit+0.5)*255
    retDigit//=interval
    retDigit*=interval
    retDigit/=255.0
    retDigit-=0.5
    return retDigit

def oneDEntropy(inputDigit):
    expandDigit = np.array((inputDigit+0.5)*255,dtype=np.int16)
    f = np.zeros(256)
    for i in range(28):
        for j in range(28):
            f[expandDigit[i][j]]+=1
    f/=784.0
    H = 0
    for i in range(256):
        if f[i] > 0:
            H+=f[i]*math.log(f[i],2)
    return -H

def crossMeanFilterOperations(inputDigit, start, end, coefficient):
    retDigit = np.array(inputDigit, dtype=np.float32)
    for row in range(start, end):
        for col in range(start, end):
            temp0 = inputDigit[row][col]
            for i in range(1,start+1):
                temp0+=inputDigit[row-i][col]
                temp0+=inputDigit[row+i][col]
                temp0+=inputDigit[row][col-i]
                temp0+=inputDigit[row][col+i]
            retDigit[row][col] = temp0/coefficient
    return retDigit

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


# In[ ]:


if __name__ == "__main__":
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        modelPath = '%smodels/mnist' % (nn_robust_attack_root)
        data, model =  MNIST(), MNISTModel(modelPath, sess)
        
        attack = CarliniL2(sess, model, batch_size=1, max_iterations=2000,confidence=0,binary_search_steps=5,initial_const=1.,learning_rate=1e-1,targeted=False)

        inputs, targets = generate_data(data, samples=1000, targeted=False, start=5500, inception=False)
        
        original_classified_wrong_number = 0 #number of benign samples that are misclassified 
        disturbed_failure_number = 0 #number of samples that failed to craft corresponding adversarial samples
        test_number = 0 #number of adversarial samples that we generate
        TTP = 0
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
            
            oriCorrectLabel = data.test_labels[i+5500].argmax()            
            
            octStart = time.time()
            oriPredicatedLabel = mnistPredicate(inputIm, model)
            oriProb = getProbabilities(inputIm,model,sess)
            octEnd = time.time()
            
            if oriPredicatedLabel != oriCorrectLabel:
                original_classified_wrong_number+=1
                continue
            
            attackStart = time.time()
            adv = attack.attack(inputIm,target)
            attackEnd = time.time()
            
            normalization(adv)
            adv = np.reshape(adv, inputIm.shape)
            actStart = time.time()
            advPredicatedLabel = mnistPredicate(adv, model)
            advProb = getProbabilities(adv,model,sess)
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

            print('probabilities = %.4f ; %.4f'%(oriProb,advProb))
            
            ofctStart = time.time()
            oriEntropy = oneDEntropy(tempInput)
            print('oriEntropy = %f' % (oriEntropy))
            if oriEntropy < 4:
                inputFinal = scalarQuantization(tempInput,128)
            elif oriEntropy < 5:
                inputFinal = scalarQuantization(tempInput, 64)
            else:
                inputAfterQ = scalarQuantization(tempInput,43)
                inputAfterQandF = crossMeanFilterOperations(inputAfterQ,3,25,13)
                inputFinal = chooseCloserFilter(tempInput, inputAfterQ, inputAfterQandF)
            oriFilteredPredicatedLabel = mnistPredicate(inputFinal, model)
            oriFilteredProb = getProbabilities(inputFinal,model,sess)
            ofctEnd = time.time()

            afctStart = time.time()
            advEntropy = oneDEntropy(tempAdv)
            print('advEntropy = %f' % (advEntropy))
            if advEntropy < 4:
                advFinal = scalarQuantization(tempAdv, 128)
            elif advEntropy < 5:
                advFinal = scalarQuantization(tempAdv, 64)
            else:
                advAfterQ = scalarQuantization(tempAdv, 43)
                advAfterQandF = crossMeanFilterOperations(advAfterQ,3,25,13)
                advFinal = chooseCloserFilter(tempAdv, advAfterQ,advAfterQandF)
            advFilteredPredicatedLabel = mnistPredicate(advFinal, model)
            advFilteredProb = getProbabilities(advFinal, model,sess)
            afctEnd = time.time()

            print('filtered probs = %.4f ; %.4f'%(oriFilteredProb,advFilteredProb))
            
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
            statisticStr = '%d-%d-%d: TP = %d, FN = %d, FP = %d, TTP = %d' % (test_number, original_classified_wrong_number, disturbed_failure_number, TP, FN, FP, TTP)
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

