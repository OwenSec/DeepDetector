
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
    
def boxMeanFilterOperations(inputDigit, start, end, coefficient):
    retDigit = np.array(inputDigit, dtype=np.float32)
    for row in xrange(start, end):
        for col in xrange(start, end):
            retDigit[row][col] = sum(sum(inputDigit[row-start:row+start+1,col-start:col+start+1]))/coefficient
    return retDigit

def diamondAndCrossFilterOperations(inputDigit, kernel, start, end, coefficient):
    retDigit = np.array(inputDigit, dtype=np.float32)
    for row in xrange(start, end):
        for col in xrange(start, end):
            retDigit[row][col] = sum(sum(inputDigit[row-start:row+start+1, col-start:col+start+1]*kernel))/coefficient
    return retDigit

def scalarQuantization(inputDigit, interval, left=True):
    retDigit = inputDigit*255
    retDigit//=interval
    retDigit*=interval
    if not left:
        halfInterval = interval//2
        retDigit+=(halfInterval)
    retDigit/=255.0
    return retDigit


# In[3]:


#Train with  scalar quantization left: 2,3,4,5,6,7,8,9,10
def mnist_tutorial(train_start=0, train_end=60000, test_start=0,
                   test_end=10000, nb_epochs=6, batch_size=128,
                   learning_rate=0.001, train_dir="/tmp",
                   filename="mnist.ckpt", load_model=False,
                   testing=False):
    keras.layers.core.K.set_learning_phase(0)
    report = AccuracyReport()
    tf.set_random_seed(1234)
    if not hasattr(backend, "tf"):
        raise RuntimeError("This tutorial requires keras to be configured"
                           " to use the TensorFlow backend.")
    if keras.backend.image_dim_ordering() != 'tf':
        keras.backend.set_image_dim_ordering('tf')
        print("INFO: '~/.keras/keras.json' sets 'image_dim_ordering' to "
              "'th', temporarily setting to 'tf'")

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

    # Initialize the Fast Gradient Sign Method (FGSM) attack object and graph
    wrap = KerasModelWrapper(model)
    advGenTimeStart = time.time()
    fgsm = FastGradientMethod(wrap, sess=sess)
    fgsm_params = {'eps': 0.2,
                   'clip_min': 0.,
                   'clip_max': 1.}
    adv_x = fgsm.generate(x, **fgsm_params)
    adv_x = sess.run(adv_x, feed_dict={x: X_test[:4500]})
    advGenTimeEnd = time.time()
    advGenTime = advGenTimeEnd-advGenTimeStart
    
    for i in xrange(4500):
        normalization(adv_x[i:(i+1)])

    print('adversarial examples generation time = ', advGenTime, 'seconds')
    
    intervals = [128,85,64,51,43,37,32,28,26]

    for intervalIndex in range(9):
        startTime = time.time()
        print('NBinterval =  ', intervalIndex+2, '; interval size = ', intervals[intervalIndex])
        original_classified_wrong_number = 0
        disturbed_failure_number = 0
        test_number = 0
        TTP = 0
        TP = 0
        FN = 0
        FP = 0

        for i in range(1000):
            current_class = int(np.argmax(Y_test[i]))

            currentXLabel = model_argmax(sess,x,predictions,X_test[i:(i+1)])
            if currentXLabel != current_class:
                original_classified_wrong_number+=1
                continue

            currentAdvXLabel = model_argmax(sess,x,predictions,adv_x[i:(i+1)])
            if currentAdvXLabel == currentXLabel:
                disturbed_failure_number+=1
                continue

            test_number+=1    

            currentX = np.reshape(X_test[i:(i+1)], (28,28))
            currentX = scalarQuantization(currentX, intervals[intervalIndex])
            currentX = np.reshape(currentX, X_test[i:(i+1)].shape)
            currentXFilteredLabel = model_argmax(sess,x,predictions,currentX)

            currentAdvX = np.reshape(adv_x[i:(i+1)], (28,28))
            currentAdvX = scalarQuantization(currentAdvX, intervals[intervalIndex])
            currentAdvX = np.reshape(currentAdvX, X_test[i:(i+1)].shape)
            currentAdvXFilteredLabel = model_argmax(sess,x,predictions,currentAdvX)

            if currentAdvXFilteredLabel != currentAdvXLabel:
                TP+=1
                if currentAdvXFilteredLabel == current_class:
                    TTP+=1
            else:
                FN+=1
            if currentXFilteredLabel != currentXLabel:
                FP+=1
                
            if (i+1) % 1000 == 0:
                str1 = '%d-%d-%d: TP = %d; FN = %d; FP = %d; TTP = %d' % (test_number,original_classified_wrong_number,disturbed_failure_number,TP,FN,FP,TTP)
                print(str1)

        str1 = '%d-%d-%d: TP = %d; FN = %d; FP = %d; TTP = %d' % (test_number,original_classified_wrong_number,disturbed_failure_number,TP,FN,FP,TTP)
        print(str1)
                
        endTime  = time.time()
        print('lasting ', endTime-startTime, 'seconds')
        Recall = TP/(TP+FN)
        Precision = TP/(TP+FP)
        tempStarStr = '********************************************************'
        recallStr = 'Recall = %.4f' % (Recall)
        precisionStr = 'Precision = %.4f' % (Precision)
        print(tempStarStr)
        print(recallStr)
        print(precisionStr)
        print(tempStarStr)
    
    return report

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


# In[3]:


#Train with box filters: 3,5,7,9
def mnist_tutorial(train_start=0, train_end=60000, test_start=0,
                   test_end=10000, nb_epochs=6, batch_size=128,
                   learning_rate=0.001, train_dir="/tmp",
                   filename="mnist.ckpt", load_model=False,
                   testing=False):
    """
    MNIST CleverHans tutorial
    :param train_start: index of first training set example
    :param train_end: index of last training set example
    :param test_start: index of first test set example
    :param test_end: index of last test set example
    :param nb_epochs: number of epochs to train model
    :param batch_size: size of training batches
    :param learning_rate: learning rate for training
    :param train_dir: Directory storing the saved model
    :param filename: Filename to save model under
    :param load_model: True for load, False for not load
    :param testing: if true, test error is calculated
    :return: an AccuracyReport object
    """
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
    
#     gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
#     config = tf.ConfigProto(gpu_options=gpu_options)
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

    # Initialize the Fast Gradient Sign Method (FGSM) attack object and graph
    wrap = KerasModelWrapper(model)
    advGenTimeStart = time.time()
    fgsm = FastGradientMethod(wrap, sess=sess)
    fgsm_params = {'eps': 0.2,
                   'clip_min': 0.,
                   'clip_max': 1.}
    adv_x = fgsm.generate(x, **fgsm_params)
    adv_x = sess.run(adv_x, feed_dict={x: X_test[:4500]})
    advGenTimeEnd = time.time()
    advGenTime = advGenTimeEnd-advGenTimeStart
    
    for i in xrange(4500):
        normalization(adv_x[i:(i+1)])

    print('adversarial examples generation time = ', advGenTime, 'seconds')
    
#box filter test, kernel size: 3, 5, 7, 9
    for kernelSize in xrange(3,10,2):
        startTime = time.time()
        print('box filter, size = ', kernelSize)
        original_classified_wrong_number = 0
        disturbed_failure_number = 0
        test_number = 0
        TTP = 0
        TP = 0
        FN = 0
        FP = 0

        start = (kernelSize-1)//2
        end = 28-start
        coefficient = kernelSize**2
        for i in range(4500):
            current_class = int(np.argmax(Y_test[i]))

            currentXLabel = model_argmax(sess,x,predictions,X_test[i:(i+1)])
            if currentXLabel != current_class:
                original_classified_wrong_number+=1
                continue

            currentAdvXLabel = model_argmax(sess,x,predictions,adv_x[i:(i+1)])
            if currentAdvXLabel == currentXLabel:
                disturbed_failure_number+=1
                continue

            test_number+=1    

            currentX = np.reshape(X_test[i:(i+1)], (28,28))
            currentX = boxMeanFilterOperations(currentX, start, end, coefficient)
            currentX = np.reshape(currentX, X_test[i:(i+1)].shape)
            currentXFilteredLabel = model_argmax(sess,x,predictions,currentX)

            currentAdvX = np.reshape(adv_x[i:(i+1)], (28,28))
            currentAdvX = boxMeanFilterOperations(currentAdvX, start, end, coefficient)
            currentAdvX = np.reshape(currentAdvX, X_test[i:(i+1)].shape)
            currentAdvXFilteredLabel = model_argmax(sess,x,predictions,currentAdvX)

            if currentAdvXFilteredLabel != currentAdvXLabel:
                TP+=1
                if currentAdvXFilteredLabel == current_class:
                    TTP+=1
            else:
                FN+=1
            if currentXFilteredLabel != currentXLabel:
                FP+=1
                
            if (i+1) % 1000 == 0:
                str1 = '%d-%d-%d: TP = %d; FN = %d; FP = %d; TTP = %d' % (test_number,original_classified_wrong_number,disturbed_failure_number,TP,FN,FP,TTP)
                print(str1)

        str1 = '%d-%d-%d: TP = %d; FN = %d; FP = %d; TTP = %d' % (test_number,original_classified_wrong_number,disturbed_failure_number,TP,FN,FP,TTP)
        print(str1)
                
        endTime  = time.time()
        print('lasting ', endTime-startTime, 'seconds')
        Recall = TP/(TP+FN)
        Precision = TP/(TP+FP)
        tempStarStr = '********************************************************'
        recallStr = 'Recall = %.4f' % (Recall)
        precisionStr = 'Precision = %.4f' % (Precision)
        print(tempStarStr)
        print(recallStr)
        print(precisionStr)
        print(tempStarStr)
    
    return report

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


# In[3]:


#Train with diamond filters: 3,5,7,9
def mnist_tutorial(train_start=0, train_end=60000, test_start=0,
                   test_end=10000, nb_epochs=6, batch_size=128,
                   learning_rate=0.001, train_dir="/tmp",
                   filename="mnist.ckpt", load_model=False,
                   testing=False):
    """
    MNIST CleverHans tutorial
    :param train_start: index of first training set example
    :param train_end: index of last training set example
    :param test_start: index of first test set example
    :param test_end: index of last test set example
    :param nb_epochs: number of epochs to train model
    :param batch_size: size of training batches
    :param learning_rate: learning rate for training
    :param train_dir: Directory storing the saved model
    :param filename: Filename to save model under
    :param load_model: True for load, False for not load
    :param testing: if true, test error is calculated
    :return: an AccuracyReport object
    """
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
    
#     gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
#     config = tf.ConfigProto(gpu_options=gpu_options)
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

    # Initialize the Fast Gradient Sign Method (FGSM) attack object and graph
    wrap = KerasModelWrapper(model)
    advGenTimeStart = time.time()
    fgsm = FastGradientMethod(wrap, sess=sess)
    fgsm_params = {'eps': 0.2,
                   'clip_min': 0.,
                   'clip_max': 1.}
    adv_x = fgsm.generate(x, **fgsm_params)
    adv_x = sess.run(adv_x, feed_dict={x: X_test[:4500]})
    advGenTimeEnd = time.time()
    advGenTime = advGenTimeEnd-advGenTimeStart
    
    for i in xrange(4500):
        normalization(adv_x[i:(i+1)])

    print('adversarial examples generation time = ', advGenTime, 'seconds')
    diamonds = [np.array([[0,1,0],[1,1,1],[0,1,0]]),
                         np.array([[0,0,1,0,0],
                                   [0,1,1,1,0],
                                   [1,1,1,1,1],
                                   [0,1,1,1,0],
                                   [0,0,1,0,0]]),
                        np.array([[0,0,0,1,0,0,0],
                                  [0,0,1,1,1,0,0],
                                  [0,1,1,1,1,1,0],
                                  [1,1,1,1,1,1,1],
                                  [0,1,1,1,1,1,0],
                                  [0,0,1,1,1,0,0],
                                  [0,0,0,1,0,0,0]]),
                        np.array([[0,0,0,0,1,0,0,0,0],
                                  [0,0,0,1,1,1,0,0,0],
                                  [0,0,1,1,1,1,1,0,0],
                                  [0,1,1,1,1,1,1,1,0],
                                  [1,1,1,1,1,1,1,1,1],
                                  [0,1,1,1,1,1,1,1,0],
                                  [0,0,1,1,1,1,1,0,0],
                                  [0,0,0,1,1,1,0,0,0],
                                  [0,0,0,0,1,0,0,0,0],])]
    coefficient = [5,13, 25, 41]
#diamond filter test, kernel size: 3, 5, 7, 9
    kernelIndex = -1
    for kernelSize in xrange(3,10,2):
        startTime = time.time()
        original_classified_wrong_number = 0
        disturbed_failure_number = 0
        test_number = 0
        TTP = 0
        TP = 0
        FN = 0
        FP = 0

        start = (kernelSize-1)//2
        end = 28-start
        kernelIndex+=1
        print('diamond filter')
        print(diamonds[kernelIndex])
        for i in range(4500):
            current_class = int(np.argmax(Y_test[i]))

            currentXLabel = model_argmax(sess,x,predictions,X_test[i:(i+1)])
            if currentXLabel != current_class:
                original_classified_wrong_number+=1
                continue

            currentAdvXLabel = model_argmax(sess,x,predictions,adv_x[i:(i+1)])
            if currentAdvXLabel == currentXLabel:
                disturbed_failure_number+=1
                continue

            test_number+=1    

            currentX = np.reshape(X_test[i:(i+1)], (28,28))
            currentX = diamondAndCrossFilterOperations(currentX, diamonds[kernelIndex], start, end, coefficient[kernelIndex])
            currentX = np.reshape(currentX, X_test[i:(i+1)].shape)
            currentXFilteredLabel = model_argmax(sess,x,predictions,currentX)

            currentAdvX = np.reshape(adv_x[i:(i+1)], (28,28))
            currentAdvX = diamondAndCrossFilterOperations(currentAdvX, diamonds[kernelIndex], start, end, coefficient[kernelIndex])
            currentAdvX = np.reshape(currentAdvX, X_test[i:(i+1)].shape)
            currentAdvXFilteredLabel = model_argmax(sess,x,predictions,currentAdvX)

            if currentAdvXFilteredLabel != currentAdvXLabel:
                TP+=1
                if currentAdvXFilteredLabel == current_class:
                    TTP+=1
            else:
                FN+=1
            if currentXFilteredLabel != currentXLabel:
                FP+=1
                
            if (i+1) % 1000 == 0:
                str1 = '%d-%d-%d: TP = %d; FN = %d; FP = %d; TTP = %d' % (test_number,original_classified_wrong_number,disturbed_failure_number,TP,FN,FP,TTP)
                print(str1)

        str1 = '%d-%d-%d: TP = %d; FN = %d; FP = %d; TTP = %d' % (test_number,original_classified_wrong_number,disturbed_failure_number,TP,FN,FP,TTP)
        print(str1)
                
        endTime  = time.time()
        print('lasting ', endTime-startTime, 'seconds')
        Recall = TP/(TP+FN)
        Precision = TP/(TP+FP)
        tempStarStr = '********************************************************'
        recallStr = 'Recall = %.4f' % (Recall)
        precisionStr = 'Precision = %.4f' % (Precision)
        print(tempStarStr)
        print(recallStr)
        print(precisionStr)
        print(tempStarStr)
    
    return report

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


# In[3]:


#Train with cross filters: 3,5,7,9
def mnist_tutorial(train_start=0, train_end=60000, test_start=0,
                   test_end=10000, nb_epochs=6, batch_size=128,
                   learning_rate=0.001, train_dir="/tmp",
                   filename="mnist.ckpt", load_model=False,
                   testing=False):
    """
    MNIST CleverHans tutorial
    :param train_start: index of first training set example
    :param train_end: index of last training set example
    :param test_start: index of first test set example
    :param test_end: index of last test set example
    :param nb_epochs: number of epochs to train model
    :param batch_size: size of training batches
    :param learning_rate: learning rate for training
    :param train_dir: Directory storing the saved model
    :param filename: Filename to save model under
    :param load_model: True for load, False for not load
    :param testing: if true, test error is calculated
    :return: an AccuracyReport object
    """
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
    
#     gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
#     config = tf.ConfigProto(gpu_options=gpu_options)
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

    # Initialize the Fast Gradient Sign Method (FGSM) attack object and graph
    wrap = KerasModelWrapper(model)
    advGenTimeStart = time.time()
    fgsm = FastGradientMethod(wrap, sess=sess)
    fgsm_params = {'eps': 0.2,
                   'clip_min': 0.,
                   'clip_max': 1.}
    adv_x = fgsm.generate(x, **fgsm_params)
    adv_x = sess.run(adv_x, feed_dict={x: X_test[:4500]})
    advGenTimeEnd = time.time()
    advGenTime = advGenTimeEnd-advGenTimeStart
    
    for i in xrange(4500):
        normalization(adv_x[i:(i+1)])
        
    print('adversarial examples generation time = ', advGenTime, 'seconds')
    crosses = [np.array([[0,1,0],[1,1,1],[0,1,0]]),
                         np.array([[0,0,1,0,0],
                                   [0,0,1,0,0],
                                   [1,1,1,1,1],
                                   [0,0,1,0,0],
                                   [0,0,1,0,0]]),
                        np.array([[0,0,0,1,0,0,0],
                                  [0,0,0,1,0,0,0],
                                  [0,0,0,1,0,0,0],
                                  [1,1,1,1,1,1,1],
                                  [0,0,0,1,0,0,0],
                                  [0,0,0,1,0,0,0],
                                  [0,0,0,1,0,0,0]]),
                        np.array([[0,0,0,0,1,0,0,0,0],
                                  [0,0,0,0,1,0,0,0,0],
                                  [0,0,0,0,1,0,0,0,0],
                                  [0,0,0,0,1,0,0,0,0],
                                  [1,1,1,1,1,1,1,1,1],
                                  [0,0,0,0,1,0,0,0,0],
                                  [0,0,0,0,1,0,0,0,0],
                                  [0,0,0,0,1,0,0,0,0],
                                  [0,0,0,0,1,0,0,0,0],])]
    coefficient = [5,9, 13, 17]
#diamond filter test, kernel size: 3, 5, 7, 9
    kernelIndex = -1
    for kernelSize in xrange(3,10,2):
        startTime = time.time()
        original_classified_wrong_number = 0
        disturbed_failure_number = 0
        test_number = 0
        TTP = 0
        TP = 0
        FN = 0
        FP = 0

        start = (kernelSize-1)//2
        end = 28-start
        kernelIndex+=1
        print('cross filter')
        print(crosses[kernelIndex])
        for i in range(4500):
            current_class = int(np.argmax(Y_test[i]))

            currentXLabel = model_argmax(sess,x,predictions,X_test[i:(i+1)])
            if currentXLabel != current_class:
                original_classified_wrong_number+=1
                continue

            currentAdvXLabel = model_argmax(sess,x,predictions,adv_x[i:(i+1)])
            if currentAdvXLabel == currentXLabel:
                disturbed_failure_number+=1
                continue

            test_number+=1    

            currentX = np.reshape(X_test[i:(i+1)], (28,28))
            currentX = diamondAndCrossFilterOperations(currentX, crosses[kernelIndex], start, end, coefficient[kernelIndex])
            currentX = np.reshape(currentX, X_test[i:(i+1)].shape)
            currentXFilteredLabel = model_argmax(sess,x,predictions,currentX)

            currentAdvX = np.reshape(adv_x[i:(i+1)], (28,28))
            currentAdvX = diamondAndCrossFilterOperations(currentAdvX, crosses[kernelIndex], start, end, coefficient[kernelIndex])
            currentAdvX = np.reshape(currentAdvX, X_test[i:(i+1)].shape)
            currentAdvXFilteredLabel = model_argmax(sess,x,predictions,currentAdvX)

            if currentAdvXFilteredLabel != currentAdvXLabel:
                TP+=1
                if currentAdvXFilteredLabel == current_class:
                    TTP+=1
            else:
                FN+=1
            if currentXFilteredLabel != currentXLabel:
                FP+=1
                
            if (i+1) % 1000 == 0:
                str1 = '%d-%d-%d: TP = %d; FN = %d; FP = %d; TTP = %d' % (test_number,original_classified_wrong_number,disturbed_failure_number,TP,FN,FP,TTP)
                print(str1)

        str1 = '%d-%d-%d: TP = %d; FN = %d; FP = %d; TTP = %d' % (test_number,original_classified_wrong_number,disturbed_failure_number,TP,FN,FP,TTP)
        print(str1)
                
        endTime  = time.time()
        print('lasting ', endTime-startTime, 'seconds')
        Recall = TP/(TP+FN)
        Precision = TP/(TP+FP)
        tempStarStr = '********************************************************'
        recallStr = 'Recall = %.4f' % (Recall)
        precisionStr = 'Precision = %.4f' % (Precision)
        print(tempStarStr)
        print(recallStr)
        print(precisionStr)
        print(tempStarStr)
    
    return report

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

