
# coding: utf-8

# In[1]:


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import keras
from keras import backend
import tensorflow as tf
from tensorflow.python.platform import flags

from cleverhans.utils_mnist import data_mnist
from cleverhans.utils_tf import model_train, model_eval, model_argmax
from cleverhans.attacks import FastGradientMethod
from cleverhans.utils import AccuracyReport
from cleverhans.utils_keras import cnn_model
from cleverhans.utils_keras import KerasModelWrapper

import time
import matplotlib.pyplot as plt
import math

FLAGS = flags.FLAGS


# In[2]:


def normalization(image_data):
    image_data[image_data<0] = 0
    image_data[image_data>1.0] = 1.0
    
def scalarQuantization(inputDigit, interval, left=True):
    retDigit = inputDigit*255
    retDigit//=interval
    retDigit*=interval
    if not left:
        halfInterval = interval//2
        retDigit+=(halfInterval)
    retDigit/=255.0
    return retDigit

def oneDEntropy(inputDigit):
    expandDigit = np.array(inputDigit*255,dtype=np.int16)
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
    for row in xrange(start, end):
        for col in xrange(start, end):
            temp0 = inputDigit[row][col]
            for i in range(1,start+1):
                temp0+=inputDigit[0][row-i][col]
                temp0+=inputDigit[0][row+i][col]
                temp0+=inputDigit[0][row][col-i]
                temp0+=inputDigit[0][row][col+i]
            retDigit[row][col] = temp0/coefficient
    return retDigit

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

def my_model_argmax(sess, x, predictions, samples):
    feed_dict = {x: samples}
    probabilities = sess.run(predictions, feed_dict)
    return np.reshape(probabilities,10)
#     if samples.shape[0] == 1:
#         return np.argmax(probabilities)
#     else:
#         return np.argmax(probabilities, axis=1)


# In[3]:


def mnist_tutorial(train_start=0, train_end=60000, test_start=0,
                   test_end=10000, nb_epochs=6, batch_size=128,
                   learning_rate=0.001, train_dir="/tmp",
                   filename="mnist.ckpt", load_model=False,
                   testing=False):
    keras.layers.core.K.set_learning_phase(0)
    # Object used to keep track of (and return) key accuracies
    report = AccuracyReport()
    # Set TF random seed to improve reproducibility
    tf.set_random_seed(1234)
    if not hasattr(backend, "tf"):
        raise RuntimeError("This tutorial requires keras to be configured"
                           " to use the TensorFlow backend.")
    if keras.backend.image_dim_ordering() != 'tf':
        keras.backend.set_image_dim_ordering('tf')
        print("INFO: '~/.keras/keras.json' sets 'image_dim_ordering' to "
              "'th', temporarily setting to 'tf'")
    # Create TF session and set as Keras backend session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    keras.backend.set_session(sess)
    # Get MNIST test data
    X_train, Y_train, X_test, Y_test = data_mnist(train_start=train_start,
                                                  train_end=train_end,
                                                  test_start=test_start,
                                                  test_end=test_end)
    # Use label smoothing
    assert Y_train.shape[1] == 10
    label_smooth = .1
    Y_train = Y_train.clip(label_smooth / 9., 1. - label_smooth)
    # Define input TF placeholder
    x = tf.placeholder(tf.float32, shape=(None, 28, 28, 1))
    y = tf.placeholder(tf.float32, shape=(None, 10))
    # Define TF model graph
    model = cnn_model()
    predictions = model(x)
    print("Defined TensorFlow model graph.")

    def evaluate():
        # Evaluate the accuracy of the MNIST model on legitimate test examples
        eval_params = {'batch_size': batch_size}
        acc = model_eval(sess, x, y, predictions, X_test, Y_test, args=eval_params)
        report.clean_train_clean_eval = acc
        assert X_test.shape[0] == test_end - test_start, X_test.shape
        print('Test accuracy on legitimate examples: %0.4f' % acc)
    
    train_params = {
        'nb_epochs': nb_epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'train_dir': train_dir,
        'filename': filename
    }
    
    # Train an MNIST model
    ckpt = tf.train.get_checkpoint_state(train_dir)
    ckpt_path = False if ckpt is None else ckpt.model_checkpoint_path

    rng = np.random.RandomState([2017, 8, 30])
    if load_model and ckpt_path:
        saver = tf.train.Saver()
        saver.restore(sess, ckpt_path)
        print("Model loaded from: {}".format(ckpt_path))
    else:
        print("Model was not loaded, training from scratch.")
        model_train(sess, x, y, predictions, X_train, Y_train, evaluate=evaluate,
                    args=train_params, save=True, rng=rng)

    advGenTimeStart = time.time()
    wrap = KerasModelWrapper(model)
    advGenTimeStart = time.time()
    fgsm = FastGradientMethod(wrap, sess=sess)
    fgsm_params = {'eps': 0.3,
                   'clip_min': 0.,
                   'clip_max': 1.}
    adv_x = fgsm.generate(x, **fgsm_params)
    adv_x = sess.run(adv_x, feed_dict={x: X_test[5500:]})
    advGenTimeEnd = time.time()
    advGenTime = advGenTimeEnd-advGenTimeStart

    for i in xrange(4500):
        normalization(adv_x[i:(i+1)])
    
    original_classified_wrong_number = 0
    disturbed_failure_number = 0
    test_number = 0
    TTP = 0
    TP = 0
    FN = 0
    FP = 0
    
    for i in range(len(adv_x)):
        current_class = int(np.argmax(Y_test[i+5500]))
        oriPreTimeStart = time.time()
        currentXLabel = model_argmax(sess,x,predictions,X_test[i+5500:(i+5501)])
        currentXProbList = my_model_argmax(sess,x,predictions,X_test[i+5500:(i+5501)])
        oriPreTimeEnd = time.time()
        oriPreTime = oriPreTimeEnd-oriPreTimeStart
        if currentXLabel != current_class:
            original_classified_wrong_number+=1
            continue
        
        advPreTimeStart = time.time()
        currentAdvXLabel = model_argmax(sess,x,predictions,adv_x[i:(i+1)])
        currentAdvXProbList = my_model_argmax(sess,x,predictions,adv_x[i:(i+1)])
        advPreTimeEnd = time.time()
        advPreTime = advPreTimeEnd-advPreTimeStart

        if currentAdvXLabel == currentXLabel:
            disturbed_failure_number+=1
            continue

#         fig = plt.figure('test')
#         picOne = fig.add_subplot(121)
#         picOne.imshow(X_test[i+5500:(i+5501)].reshape((28,28)), cmap='gray')
#         picTwo = fig.add_subplot(122)
#         picTwo.imshow(adv_x[i:(i+1)].reshape((28,28)), cmap='gray')
#         plt.show()

            
        test_number+=1    
   
        print('probabilities = %.4f ; %.4f'%(currentXProbList[currentXLabel],currentAdvXProbList[currentAdvXLabel]))
        
        tempX = np.reshape(X_test[i+5500:(i+5501)], (28,28))
        test_x = np.array(tempX)
        
        oriFilteredPreTimeStart = time.time()
        currentX = np.reshape(X_test[i+5500:(i+5501)], (28,28))
        imageEntropy = oneDEntropy(test_x)
        if imageEntropy < 4:
            current_x_res = scalarQuantization(currentX, 128)
        elif imageEntropy < 5:
            current_x_res = scalarQuantization(currentX, 64)
        else:
            current_x_ASQ = scalarQuantization(currentX, 43)
            current_x_ASQ_AMF = crossMeanFilterOperations(current_x_ASQ,3,25,13)
            current_x_res = chooseCloserFilter(currentX, current_x_ASQ, current_x_ASQ_AMF)
        current_x_res = np.reshape(current_x_res, X_test[0:1].shape)
        current_x_res_label = model_argmax(sess,x,predictions,current_x_res)
        current_x_res_prob = my_model_argmax(sess,x,predictions,current_x_res)

        
        tempX2 = np.reshape(adv_x[i:(i+1)], (28,28))
        test_adv_x = np.array(tempX2)
        currentAdvX = np.reshape(adv_x[i:(i+1)], (28,28))
        imageEntropy2 = oneDEntropy(test_adv_x)
        print('%d: %.2f------%.2f' % (i, imageEntropy,imageEntropy2))
        if imageEntropy2 < 4:
            current_adv_x_res = scalarQuantization(currentAdvX,128)
        elif imageEntropy2 < 5:
            current_adv_x_res = scalarQuantization(currentAdvX, 64)
        else:
            current_adv_x_ASQ = scalarQuantization(currentAdvX, 43)
            current_adv_x_ASQ_AMF = crossMeanFilterOperations(current_adv_x_ASQ,3,25,13)
            current_adv_x_res = chooseCloserFilter(currentAdvX, current_adv_x_ASQ, current_adv_x_ASQ_AMF)
        current_adv_x_res = np.reshape(current_adv_x_res, X_test[0:1].shape)
        current_adv_x_res_label = model_argmax(sess,x,predictions,current_adv_x_res)
        current_adv_x_res_prob = my_model_argmax(sess,x,predictions,current_adv_x_res)

        print('filtered Probs = %.4f ; %.4f'%(current_x_res_prob[current_x_res_label],current_adv_x_res_prob[current_adv_x_res_label]))
        
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


# In[4]:


def main(argv=None):
    mnist_tutorial(nb_epochs=FLAGS.nb_epochs,
                   batch_size=FLAGS.batch_size,
                   learning_rate=FLAGS.learning_rate,
                   train_dir=FLAGS.train_dir,
                   filename=FLAGS.filename,
                   load_model=FLAGS.load_model)


if __name__ == '__main__':
    flags.DEFINE_integer('nb_epochs', 6, 'Number of epochs to train model')
    flags.DEFINE_integer('batch_size', 128, 'Size of training batches')
    flags.DEFINE_float('learning_rate', 0.001, 'Learning rate for training')
    flags.DEFINE_string('train_dir', '/tmp', 'Directory where to save model.')
    flags.DEFINE_string('filename', 'mnist.ckpt', 'Checkpoint filename.')
    flags.DEFINE_boolean('load_model', True, 'Load saved model or train.')
    tf.app.run()

