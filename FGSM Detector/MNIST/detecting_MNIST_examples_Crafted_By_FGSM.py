
# coding: utf-8

# In[1]:

home_root = '~' #change it to your home path
cleverhans_root = '~/cleverhans/' #change it to where you store the 'cleverhans' project
import sys
sys.path.insert(0,home_root)
sys.path.insert(0,cleverhans_root) 

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import numpy as np
import keras
from keras import backend
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.platform import app
from tensorflow.python.platform import flags
from scipy import misc
import math
import time

from cleverhans.utils_mnist import data_mnist
from cleverhans.utils_tf import model_train, model_eval, model_argmax
from cleverhans.attacks import FastGradientMethod
from cleverhans.utils import cnn_model

FLAGS = flags.FLAGS #some paths need to be replaced on your own need
flags.DEFINE_string('train_dir', 'tmp/', 'Directory storing the saved model.')
flags.DEFINE_string('filename', 'mnist.ckpt', 'Filename to save model under.')
flags.DEFINE_integer('nb_epochs', 6, 'Number of epochs to train model')
flags.DEFINE_integer('batch_size', 128, 'Size of training batches')
flags.DEFINE_float('learning_rate', 0.1, 'Learning rate for training')
flags.DEFINE_integer('img_rows', 28, 'Input row dimension')
flags.DEFINE_integer('img_cols', 28, 'Input column dimension')
flags.DEFINE_integer('nb_channels', 1, 'Nb of color channels in the input.')


# In[2]:

def normalization(image_data):
    image_data2 = np.array(image_data)
    image_data2[image_data2<0] = 0
    image_data2[image_data2>1.0] = 1.0
    return image_data2

def seperateScalarQuantizationLeft(image_data, step):
    image_data2 = np.array(image_data)
    image_data2 = image_data2*255
    image_data2 = np.floor(image_data2/step)
    image_data2 = image_data2*step
    image_data2 = image_data2/255
    return image_data2

def meanFilter1(image_data):
    image_data2=np.zeros_like(image_data)
    for z in range(28):
        for v in range(28):
            if (z<2) or (z>25) or (v<2) or (v>25):
                avg_r=image_data[z][v]
            else:
                avg_r=(image_data[z-2][v]+image_data[z-1][v]+image_data[z][v-2]+image_data[z][v-1]+image_data[z][v]+
                       image_data[z][v+1]+image_data[z][v+2]+image_data[z+1][v]+image_data[z+2][v])/9.0
            image_data2[z][v]=avg_r
    return image_data2

def chooseCloserFilter(original_data,filter_data1,filter_data2):
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

    
def meanFilter55forMnist(image_data):
    image_data=image_data.astype(np.float32)
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


def image2DEntropy55_28(image_data):
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

def my_model_argmax(sess, x, predictions, samples):
    feed_dict = {x: samples}
    probabilities = sess.run(predictions, feed_dict)
    if samples.shape[0] == 1:
        return np.argmax(probabilities)
    else:
        return np.argmax(probabilities, axis=1)


# In[5]:

def main(argv=None):
    """
    MNIST cleverhans tutorial
    :return:
    """
    tf.reset_default_graph() #it is essential for restoring saved model
    
    keras.layers.core.K.set_learning_phase(0)

    # Set TF random seed to improve reproducibility
    tf.set_random_seed(1234)

    if not hasattr(backend, "tf"):
        raise RuntimeError("This tutorial requires keras to be configured"
                           " to use the TensorFlow backend.")

    # Image dimensions ordering should follow the Theano convention
    if keras.backend.image_dim_ordering() != 'tf':
        keras.backend.set_image_dim_ordering('tf')
        print("INFO: '~/.keras/keras.json' sets 'image_dim_ordering' to "
              "'th', temporarily setting to 'tf'")

    # Create TF session and set as Keras backend session
    sess = tf.Session()
    keras.backend.set_session(sess)

    # Get MNIST test data
    X_train, Y_train, X_test, Y_test = data_mnist()

    assert Y_train.shape[1] == 10.
    label_smooth = .1
    Y_train = Y_train.clip(label_smooth / 9., 1. - label_smooth)

    # Define input TF placeholder
    x = tf.placeholder(tf.float32, shape=(None, 28, 28, 1))
    y = tf.placeholder(tf.float32, shape=(None, 10))

    # Define TF model graph
    model = cnn_model()
    predictions = model(x)
    print("Defined TensorFlow model graph.")
        
    saver = tf.train.Saver()
    save_path = os.path.join(FLAGS.train_dir, FLAGS.filename)
    ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        train_params = {
            'nb_epochs': FLAGS.nb_epochs,
            'batch_size': FLAGS.batch_size,
            'learning_rate': FLAGS.learning_rate
        }
        model_train(sess, x, y, predictions, X_train, Y_train,
                args=train_params)
        saver.save(sess, save_path)

    advGenTimeStart = time.time()
    fgsm = FastGradientMethod(model, sess=sess)
    fgsm_params = {'eps': 0.1}
    adv_x = fgsm.generate(x, **fgsm_params)
    adv_x = sess.run(adv_x, feed_dict={x: X_test[0:10000]})
    advGenTimeEnd = time.time()
    advGenTime = advGenTimeEnd-advGenTimeStart

    original_classified_wrong_number = 0
    disturbed_failure_number = 0
    test_number = 0
    TTP = 0
    TP = 0
    FN = 0
    FP = 0
    
    for i in range(len(adv_x)):
        current_class = int(np.argmax(Y_test[i]))

        oriPreTimeStart = time.time()
        currentXLabel = model_argmax(sess,x,predictions,X_test[i:(i+1)])
        currentXProbList = my_model_argmax(sess,x,predictions,X_test[i:(i+1)])
        oriPreTimeEnd = time.time()
        oriPreTime = oriPreTimeEnd-oriPreTimeStart
        if currentXLabel != current_class:
            original_classified_wrong_number+=1
            continue
        
        advPreTimeStart = time.time()
        currentAdvXBeforeNormalization = np.reshape(adv_x[i:(i+1)], (FLAGS.img_rows,FLAGS.img_cols))
        currentAdvXAfterNormalization = normalization(currentAdvXBeforeNormalization)
        currentAdvXAfterNormalization = np.reshape(currentAdvXAfterNormalization,adv_x[i:(i+1)].shape)
        currentAdvXLabel = model_argmax(sess,x,predictions,currentAdvXAfterNormalization)
        currentAdvXProbList = my_model_argmax(sess,x,predictions,currentAdvXAfterNormalization)
        advPreTimeEnd = time.time()
        advPreTime = advPreTimeEnd-advPreTimeStart

        if currentAdvXLabel == currentXLabel:
            disturbed_failure_number+=1
            continue

        test_number+=1    
        
        tempX = np.reshape(X_test[i:(i+1)], (FLAGS.img_rows, FLAGS.img_cols))
        test_x = np.array(tempX)
        test_x = test_x*255
        
        oriFilteredPreTimeStart = time.time()
        currentX = np.reshape(X_test[i:(i+1)], (FLAGS.img_rows, FLAGS.img_cols))
        imageEntropy = image2DEntropy55_28(test_x)
        if imageEntropy < 8.5:
            current_x_res = seperateScalarQuantizationLeft(currentX, 128)
        elif imageEntropy < 9.5:
            current_x_res = seperateScalarQuantizationLeft(currentX, 64)
        else:
            current_x_ASQ = seperateScalarQuantizationLeft(currentX, 50)
            current_x_ASQ_AMF = meanFilter1(current_x_ASQ)
            current_x_res = chooseCloserFilter(currentX, current_x_ASQ, current_x_ASQ_AMF)
        current_x_res = np.reshape(current_x_res, X_test[i:(i+1)].shape)
        current_x_res_label = model_argmax(sess,x,predictions,current_x_res)

        
        tempX2 = np.reshape(adv_x[i:(i+1)], (FLAGS.img_rows, FLAGS.img_cols))
        test_adv_x = np.array(tempX2)
        test_adv_x = normalization(test_adv_x)
        test_adv_x = test_adv_x*255

        currentAdvX = np.reshape(adv_x[i:(i+1)], (FLAGS.img_rows, FLAGS.img_cols))
        currentAdvX = normalization(currentAdvX)
        imageEntropy2 = image2DEntropy55_28(test_adv_x)
        print('%d: %f------%f' % (i, imageEntropy,imageEntropy2))
        if imageEntropy2 < 8.5:
            current_adv_x_res = seperateScalarQuantizationLeft(currentAdvX,128)
        elif imageEntropy2 < 9.5:
            current_adv_x_res = seperateScalarQuantizationLeft(currentAdvX, 64)
        else:
            current_adv_x_ASQ = seperateScalarQuantizationLeft(currentAdvX, 50)
            current_adv_x_ASQ_AMF = meanFilter1(current_adv_x_ASQ)
            current_adv_x_res = chooseCloserFilter(currentAdvX, current_adv_x_ASQ, current_adv_x_ASQ_AMF)
        current_adv_x_res = np.reshape(current_adv_x_res, X_test[i:(i+1)].shape)
        current_adv_x_res_label = model_argmax(sess,x,predictions,current_adv_x_res)
            
        if current_adv_x_res_label != currentAdvXLabel:
            TP+=1
            if current_adv_x_res_label == current_class:
                TTP+=1
        else:
            FN+=1
        if current_x_res_label != currentXLabel:
            FP+=1
        str1 = '%d-%d-%d: TP = %d; FN = %d; FP = %d; TTP = %d' % (test_number,original_classified_wrong_number,disturbed_failure_number,TP,FN,FP,TTP)
        print(str1)
    
    Recall = TP/(TP+FN)
    Precision = TP/(TP+FP)
    tempStarStr = '********************************************************'
    recallStr = 'Recall = %.4f' % (Recall)
    precisionStr = 'Precision = %.4f' % (Precision)
    print(tempStarStr)
    print(recallStr)
    print(precisionStr)
    print(tempStarStr)


# In[7]:

if __name__ == '__main__':
    app.run()


# In[ ]:



