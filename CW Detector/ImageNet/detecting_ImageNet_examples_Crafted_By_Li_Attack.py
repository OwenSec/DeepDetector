
# coding: utf-8

# In[1]:

#referring to detecting_MNIST_examples_Crafted_By_L2_Attack.ipynb
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

def my_generate_data2(data, samples, start=0):
    inputs = []
    sources = []
    targets = []
    for i in range(samples):
        inputs.append(data.zebraData[start+i])
        tempLabel = np.zeros_like(data.zebraLabel[i])
        tempIndex = data.zebraLabel[i].argmax()
        tempLabel[(tempIndex+100) % 1000] = 1
        targets.append((tempLabel))
        sources.append(tempIndex)
    
    for i in range(samples):
        inputs.append(data.pandaData[start+i])        
        tempLabel = np.zeros_like(data.pandaLabel[i])
        tempIndex = data.pandaLabel[i].argmax()
        tempLabel[(tempIndex+100) % 1000] = 1
        targets.append((tempLabel))
        sources.append(tempIndex)

    for i in range(samples):
        inputs.append(data.cabData[start+i])        
        tempLabel = np.zeros_like(data.cabLabel[i])
        tempIndex = data.cabLabel[i].argmax()
        tempLabel[(tempIndex+100) % 1000] = 1
        targets.append((tempLabel))
        sources.append(tempIndex)
        
    for i in range(samples):
        inputs.append(data.pineappleData[start+i])        
        tempLabel = np.zeros_like(data.pineappleLabel[i])
        tempIndex = data.pineappleLabel[i].argmax()
        tempLabel[(tempIndex+100) % 1000] = 1
        targets.append((tempLabel))
        sources.append(tempIndex)
        
    inputs = np.array(inputs)
    targets = np.array(targets)
    return inputs, sources, targets


# In[3]:

def normalization(image_data):
    image_data = np.reshape(image_data,(299,299,3))
    image_data2 = np.array(image_data)
    image_data2[image_data2<-0.5]=-0.5
    image_data2[image_data2>0.5]=0.5
    return image_data2

def seperateScalarQuantizationLeft(image_data, step):
    image_data2 = np.array(image_data)
    image_data2 = (image_data2+0.5)*255
    image_data2 = np.floor(image_data2/step)
    image_data2 = image_data2*step
    image_data2 = image_data2/255-0.5
    return image_data2

def meanFilter1(image_data):
    image_data2=np.zeros_like(image_data)
    for z in range(299):
        for v in range(299):
            if (z<2) or (z>296) or (v<2) or (v>296):
                avg_r=image_data[z][v][0]
                avg_g=image_data[z][v][1]
                avg_b=image_data[z][v][2]
            else:
                avg_r=(image_data[z-2][v][0]+image_data[z-1][v][0]+image_data[z][v-2][0]+image_data[z][v-1][0]+image_data[z][v][0]+
                       image_data[z][v+1][0]+image_data[z][v+2][0]+image_data[z+1][v][0]+image_data[z+2][v][0])/9.0
                avg_g=(image_data[z-2][v][1]+image_data[z-1][v][1]+image_data[z][v-2][1]+image_data[z][v-1][1]+image_data[z][v][1]+
                       image_data[z][v+1][1]+image_data[z][v+2][1]+image_data[z+1][v][1]+image_data[z+2][v][1])/9.0
                avg_b=(image_data[z-2][v][2]+image_data[z-1][v][2]+image_data[z][v-2][2]+image_data[z][v-1][2]+image_data[z][v][2]+
                       image_data[z][v+1][2]+image_data[z][v+2][2]+image_data[z+1][v][2]+image_data[z+2][v][2])/9.0
            image_data2[z][v]=avg_r
    return image_data2

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

def expandImage(image_data):
    image_data2 = (0.5+np.reshape(image_data,((299,299,3))))*255
    return image_data2

def meanFilter55forInceptionV3(image_data):
    image_data = image_data.astype(np.float32)
    image_data2=np.zeros_like(image_data)
    for z in range(299):
        for v in range(299):
                if (z<2) or (z>296) or (v<2) or (v>296):
                    avg_r=image_data[z][v][0]
                    avg_g=image_data[z][v][1]
                    avg_b=image_data[z][v][2]
                else:
                    avg_r=((image_data[z-2][v-2][0]+image_data[z-2][v-1][0]+image_data[z-2][v][0]+image_data[z-2][v+1][0]+image_data[z-2][v+2][0]
                           +image_data[z-1][v-2][0]+image_data[z-1][v-1][0]+image_data[z-1][v][0]+image_data[z-1][v+1][0]+image_data[z-1][v+2][0]
                           +image_data[z][v-2][0]+image_data[z][v-1][0]+image_data[z][v][0]+image_data[z][v+1][0]+image_data[z][v+2][0]
                           +image_data[z+1][v-2][0]+image_data[z+1][v-1][0]+image_data[z+1][v][0]+image_data[z+1][v+1][0]+image_data[z+1][v+2][0]
                           +image_data[z+2][v-2][0]+image_data[z+2][v-1][0]+image_data[z+2][v][0]+image_data[z+2][v+1][0]+image_data[z+2][v+2][0])/25.0)
                    avg_g=((image_data[z-2][v-2][1]+image_data[z-2][v-1][1]+image_data[z-2][v][1]+image_data[z-2][v+1][1]+image_data[z-2][v+2][1]
                           +image_data[z-1][v-2][1]+image_data[z-1][v-1][1]+image_data[z-1][v][1]+image_data[z-1][v+1][1]+image_data[z-1][v+2][1]
                           +image_data[z][v-2][1]+image_data[z][v-1][1]+image_data[z][v][1]+image_data[z][v+1][1]+image_data[z][v+2][1]
                           +image_data[z+1][v-2][1]+image_data[z+1][v-1][1]+image_data[z+1][v][1]+image_data[z+1][v+1][1]+image_data[z+1][v+2][1]
                           +image_data[z+2][v-2][1]+image_data[z+2][v-1][1]+image_data[z+2][v][1]+image_data[z+2][v+1][1]+image_data[z+2][v+2][1])/25.0)
                    avg_b=((image_data[z-2][v-2][2]+image_data[z-2][v-1][2]+image_data[z-2][v][2]+image_data[z-2][v+1][2]+image_data[z-2][v+2][2]
                           +image_data[z-1][v-2][2]+image_data[z-1][v-1][2]+image_data[z-1][v][2]+image_data[z-1][v+1][2]+image_data[z-1][v+2][2]
                           +image_data[z][v-2][2]+image_data[z][v-1][2]+image_data[z][v][2]+image_data[z][v+1][2]+image_data[z][v+2][2]
                           +image_data[z+1][v-2][2]+image_data[z+1][v-1][2]+image_data[z+1][v][2]+image_data[z+1][v+1][2]+image_data[z+1][v+2][2]
                           +image_data[z+2][v-2][2]+image_data[z+2][v-1][2]+image_data[z+2][v][2]+image_data[z+2][v+1][2]+image_data[z+2][v+2][2])/25.0)
                image_data2[z][v][0]=avg_r
                image_data2[z][v][1]=avg_g
                image_data2[z][v][2]=avg_b
    return image_data2

def image2DEntropy55(image_data):
    image_data2=np.zeros_like(image_data)
    image_data2=meanFilter55forInceptionV3(image_data)

    B_entropy=0
    G_entropy=0
    R_entropy=0
    B_num = [([0] * 256) for x in range(256)]
    G_num = [([0] * 256) for x in range(256)]
    R_num = [([0] * 256) for x in range(256)]
    pmf_B = [([0] * 256) for x in range(256)]
    pmf_G = [([0] * 256) for x in range(256)]
    pmf_R = [([0] * 256) for x in range(256)]
    for i in range(299):
        for j in range(299):
            B_val=int (image_data[i][j][0])
            G_val=int (image_data[i][j][1])
            R_val=int (image_data[i][j][2])
            B_avg=int (image_data2[i][j][0])
            G_avg=int (image_data2[i][j][1])
            R_avg=int (image_data2[i][j][2])
            B_num[B_val][B_avg]=B_num[B_val][B_avg]+1
            G_num[G_val][G_avg]=G_num[G_val][G_avg]+1
            R_num[R_val][R_avg]=R_num[R_val][R_avg]+1
    for k in range(256):
        for m in range(256):
            pmf_B[k][m]=B_num[k][m]/(299.0*299.0)
            pmf_G[k][m]=G_num[k][m]/(299.0*299.0)
            pmf_R[k][m]=R_num[k][m]/(299.0*299.0)
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


# In[1]:

if __name__ == "__main__":
    with tf.Session() as sess:
        data, model =  ImageNet(), InceptionModel(sess)
        img = tf.placeholder(tf.uint8, (299,299,3))
        softmax_tensor = tf.import_graph_def(
             sess.graph.as_graph_def(),
             input_map={'DecodeJpeg:0': tf.reshape(img,((299,299,3)))},
             return_elements=['softmax/logits:0'])
        
        attack = CarliniLi(sess, model, max_iterations=1000)

        inputs, sources, targets = my_generate_data2(data, samples=30, start=0)
        
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
            oriPredicatedLabel = model.my_predict(inputIm, img, softmax_tensor)
            octEnd = time.time()
            
            if oriPredicatedLabel != oriCorrectLabel:
                print('prediction Error!')
                original_classified_wrong_number+=1
                continue
            
            attackStart = time.time()
            adv = attack.attack(inputIm,target)
            attackEnd = time.time()
            
            adv = normalization(adv)
            adv = np.reshape(adv, inputIm.shape)
            actStart = time.time()
            advPredicatedLabel = model.my_predict(adv, img, softmax_tensor)
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
            inputForEntropy = expandImage(tempInput)
            oriEntropy = image2DEntropy55(inputForEntropy)
            oriEntropyStr = 'oriEntropy = %f' % (oriEntropy)
            print(oriEntropyStr)
            if oriEntropy < 8.5:
                inputFinal = seperateScalarQuantizationLeft(tempInput,128)
            elif oriEntropy < 9.5:
                inputFinal = seperateScalarQuantizationLeft(tempInput, 64)
            else:
                inputAfterQ = seperateScalarQuantizationLeft(tempInput,50)
                inputAfterQandF = meanFilter1(inputAfterQ)
                inputFinal = chooseCloserFilter(tempInput, inputAfterQ, inputAfterQandF)
            oriFilteredPredicatedLabel = model.my_predict(inputFinal, img, softmax_tensor)
            ofctEnd = time.time()

            afctStart = time.time()
            advForEntropy = expandImage(tempAdv)
            advEntropy = image2DEntropy55(advForEntropy)
            advEntropyStr = 'advEntropy = %f' % (advEntropy)
            print(advEntropyStr)
            if advEntropy < 8.5:
                advFinal = seperateScalarQuantizationLeft(tempAdv, 128)
            elif advEntropy < 9.5:
                advFinal = seperateScalarQuantizationLeft(tempAdv, 64)
            else:
                advAfterQ = seperateScalarQuantizationLeft(tempAdv, 50)
                advAfterQandF = meanFilter1(advAfterQ)
                advFinal = chooseCloserFilter(tempAdv, advAfterQ,advAfterQandF)
            advFilteredPredicatedLabel = model.my_predict(advFinal, img, softmax_tensor)
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
            statisticStr = '%d-%d-%d: TTP = %d, TP = %d, FN = %d, FP = %d' % (test_number, original_classified_wrong_number, disturbed_failure_number, TTP, TP, FN, FP)
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

