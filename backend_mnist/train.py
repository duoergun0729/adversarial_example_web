import numpy as np
import tensorflow as tf

from cleverhans_tutorials.tutorial_models import make_basic_cnn
from cleverhans.utils_mnist import data_mnist
from cleverhans import utils
from cleverhans.utils_tf import model_train, model_eval, tf_model_load
from cleverhans.attacks import FastGradientMethod, MadryEtAl
from scipy.fftpack import dct

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

base_url = 'digits_recognition/static/img/'

# Get MNIST test data
train_start = 0
train_end = 60000
test_start = 0
test_end = 10000
X_train, y_train, X_test, y_test = data_mnist(train_start=train_start,
                                              train_end=train_end,
                                              test_start=test_start,
                                              test_end=test_end)

img_size = 28
img_chan = 1
n_classes = 10

fgsm_params = {'eps': 0.3,
               'clip_min': 0.,
               'clip_max': 1.
               }
pgd_params = {'eps': 0.3,
              'eps_iter': .01,
              'clip_min': 0.,
              'clip_max': 1.}
train_params_clean = {
    'nb_epochs': 6,
    'batch_size': 128,
    'learning_rate': 0.001,
    'filename': 'clean_model',
    'train_dir': 'cifar/'
}
train_params_adv = {
    'nb_epochs': 6,
    'batch_size': 128,
    'learning_rate': 0.001,
    'filename': 'pgd_model',
    'train_dir': 'cifar/'
}

x = tf.placeholder(tf.float32, (None, img_size, img_size, img_chan),
                   name='x')
y = tf.placeholder(tf.float32, (None, n_classes), name='y')

report = utils.AccuracyReport()

rng = np.random.RandomState([2017, 8, 30])

sess = tf.InteractiveSession()
# sess2 = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
sess.run(tf.local_variables_initializer())

X_train_dct = dct(X_train)
X_test_dct = dct(X_test)


# model_2 = make_basic_cnn(nb_filters=64)
# fgsm2 = FastGradientMethod(model_2, sess=sess)
# adv_x_2 = fgsm2.generate(x, **fgsm_params)

def my_feature_DCT(X):
    X = dct(X)
    return X


# def make_fgsm(sess, X_data, batch_size=128):
#     """
#     Generate FGSM by running env.x_fgsm.
#     """
#     print('\nMaking adversarials via FGSM')
#
#     n_sample = X_data.shape[0]
#     n_batch = int((n_sample + batch_size - 1) / batch_size)
#     X_adv = np.empty_like(X_data)
#
#     for batch in range(n_batch):
#         print(' batch {0}/{1}'.format(batch + 1, n_batch), end='\r')
#         start = batch * batch_size
#         end = min(n_sample, start + batch_size)
#         adv = sess.run(adv_x_2, feed_dict={
#             x: X_data[start:end]
#             })
#         X_adv[start:end] = adv
#
#     return X_adv


# X_adv = make_fgsm(sess, X_train)

def load(sess, name='cifar'):
    print('\nLoading saved cifar')
    tf_model_load(sess, 'cifar/{}'.format(name))


def train(sess, predictions_adv=True, nb_filters=64, batch_size=128):

    if predictions_adv:
        model_2 = make_basic_cnn(nb_filters=nb_filters)
        preds_2 = model_2(x)
        # fgsm2 = FastGradientMethod(model_2, sess=sess)
        # adv_x_2 = fgsm2.generate(x, **fgsm_params)
        # n_sample = X_train.shape[0]
        # n_batch = int((n_sample + batch_size - 1) / batch_size)
        # X_adv = np.empty_like(X_train)
        #
        # for batch in range(n_batch):
        #     print(' batch {0}/{1}'.format(batch + 1, n_batch), end='\r')
        #     start = batch * batch_size
        #     end = min(n_sample, start + batch_size)
        #     adv = sess.run(adv_x_2, feed_dict={
        #         x: X_train[start:end]
        #     })
        #     X_adv[start:end] = adv
        pgd2 = MadryEtAl(model_2, sess=sess)
        adv_x_2 = pgd2.generate(x, **pgd_params)
        # X_adv = make_fgsm(sess, X_train)
        # dct_adv = dct(X_adv)
        # print(dct_adv.shape)
        preds_2_adv = model_2(adv_x_2)

        # load(sess, name='attack_model')

        # eval_par = {'batch_size': batch_size}
        # acc = model_eval(sess, x, y, preds_2_adv, X_test, y_test, args=eval_par)
        # print('Test accuracy of the fgsm_adv_model on adversarial examples adv_x: %0.4f\n' % acc)

        def evaluate_2():
            # Accuracy of adversarially trained cifar on legitimate test inputs
            eval_params = {'batch_size': batch_size}
            accuracy = model_eval(sess, x, y, preds_2, X_test, y_test, args=eval_params)
            print('Test accuracy on legitimate examples X_test: %0.4f' % accuracy)
            report.adv_train_clean_eval = accuracy

            # Accuracy of the adversarially trained cifar on adversarial examples
            accuracy = model_eval(sess, x, y, preds_2_adv, X_test, y_test, args=eval_params)
            print('Test accuracy on PGD adversarial examples: %0.4f' % accuracy)
            report.adv_train_adv_eval = accuracy

        # Perform and evaluate adversarial training
        model_train(sess, x, y, preds_2, X_train, y_train, save=True,
                    predictions_adv=preds_2_adv, evaluate=evaluate_2,
                    args=train_params_adv, rng=rng)

        eval_par = {'batch_size': batch_size}
        acc = model_eval(sess, x, y, preds_2_adv, X_test, y_test, args=eval_par)
        print('Test accuracy of the pgd_model on all PGD adversarial examples adv_x: %0.4f\n' % acc)


    else:
        model = make_basic_cnn(nb_filters=nb_filters)
        preds = model.get_probs(x)

        def evaluate():
            # Evaluate the accuracy of the MNIST cifar on legitimate test
            # examples
            eval_params = {'batch_size': batch_size}
            acc = model_eval(
                sess, x, y, preds, X_test, y_test, args=eval_params)
            report.clean_train_clean_eval = acc
            assert X_test.shape[0] == test_end - test_start, X_test.shape
            print('Test accuracy of the clean_model on legitimate examples X_test: %0.4f' % acc)

        model_train(sess, x, y, preds, X_train, y_train, save=True, evaluate=evaluate,
                    args=train_params_clean, rng=rng)

        pgd = MadryEtAl(model, sess=sess)
        adv_x = pgd.generate(x, **fgsm_params)
        preds_adv = model.get_probs(adv_x)

        # X_test1 = (np.array(eval(X_test[0]), dtype=np.float32)).reshape(1, 28, 28, 1)
        # print(X_test1)
        # X_adv = make_fgsm(sess, X_test)

        # Evaluate the accuracy of the dct_clean_model on adversarial examples
        eval_par = {'batch_size': batch_size}
        acc = model_eval(sess, x, y, preds_adv, X_test, y_test, args=eval_par)
        print('Test accuracy of the clean_model on adversarial examples adv_x: %0.4f\n' % acc)
        report.clean_train_adv_eval = acc
        # print(adv_x[0].shape)
        # save_img(adv_x[0])

        # load(sess2, name='clean_model')
        #
        # fgsm2 = FastGradientMethod(cifar, sess=sess2)
        # adv_x2 = fgsm2.generate(x, **fgsm_params)
        # preds_adv2 = cifar.get_probs(adv_x2)
        #
        # acc = model_eval(sess2, x, y, preds_adv2, X_test, y_test, args=eval_par)
        # print('Test accuracy of the clean_model on adversarial examples adv_x2: %0.4f\n' % acc)
        # save_img(adv_x2[0])


train(sess, predictions_adv=True, nb_filters=64, batch_size=128)
