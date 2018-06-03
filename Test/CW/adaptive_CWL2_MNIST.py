
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
from l2_adaptive_attack import CarliniL2Adaptive


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


def normalization(image):
    image[image < -0.5] = -0.5
    image[image > 0.5] = 0.5


# In[ ]:


if __name__ == "__main__":
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        modelPath = '%smodels/mnist' % (nn_robust_attack_root)
        data, model =  MNIST(), MNISTModel(modelPath, sess)
        
        blind_attack = CarliniL2(sess, model, batch_size=1, max_iterations=2000,confidence=0,binary_search_steps=5,initial_const=1.,learning_rate=1e-1,targeted=False)
        adaptive_attack = CarliniL2Adaptive(sess, model, batch_size=1,confidence=0, max_iterations=2000,binary_search_steps=5,initial_const=1.,learning_rate=1e-1,targeted=False)

        inputs, targets = generate_data(data, samples=1000, targeted=False, start=9000, inception=False)

        total = 0
        disturbed_failure_number1 = 0
        test_number1 = 0
        disturbed_failure_number2 = 0
        test_number2 = 0

        l2_distances = []
        l2_distances2 = []
        
        
        for i in range(len(targets)):
            print(i)
            
            inputIm = inputs[i:(i+1)]
            target = targets[i:(i+1)]
            
            oriCorrectLabel = data.test_labels[i+9000].argmax()            
            
            octStart = time.time()
            oriPredicatedLabel = mnistPredicate(inputIm, model)
            oriProb = getProbabilities(inputIm,model,sess)
            octEnd = time.time()
            
            if oriPredicatedLabel != oriCorrectLabel:
                continue
            
            total+=1
            
            attackStart = time.time()
            adv = blind_attack.attack(inputIm,target)
            adv2 = adaptive_attack.attack(inputIm,target)
            attackEnd = time.time()
            
            normalization(adv)
            normalization(adv2)
            adv = np.reshape(adv, inputIm.shape)
            adv2 = np.reshape(adv2,inputIm.shape)
            actStart = time.time()
            advPredicatedLabel1 = mnistPredicate(adv, model)
            advProb1 = getProbabilities(adv,model,sess)
            advPredicatedLabel2 = mnistPredicate(adv2, model)
            advProb2 = getProbabilities(adv2,model,sess)
            actEnd = time.time()
            
            if advPredicatedLabel1 != oriCorrectLabel:
                test_number1+=1
                distortions1 = np.linalg.norm(adv-inputIm)
                l2_distances.append(distortions1)
                if advPredicatedLabel2 != oriCorrectLabel:
                    test_number2+=1
                    print('labels = ',oriCorrectLabel,' , ',advPredicatedLabel1,' , ',advPredicatedLabel2)
                    print('probs = ',oriProb,' , ',advProb1,' , ',advProb2)
                    distortions2 = np.linalg.norm(adv2-inputIm)
                    print('distortions = ',distortions1,' ; ',distortions2)                
                    l2_distances2.append(distortions2)
        
        print('succeeds = ',total,' ; ', test_number1,' ; ',test_number2)
        print(sum(l2_distances)/test_number1)
        print(sum(l2_distances2)/test_number2)

