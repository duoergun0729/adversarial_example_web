import keras
from keras.datasets import cifar10
from keras.utils import np_utils
import tensorflow as tf
from cleverhans.utils_keras import cnn_model_small, cnn_model_big
from cleverhans.utils_tf import tf_model_load, model_eval
import numpy as np
import matplotlib.pyplot as plt
from django.http import JsonResponse
import matplotlib.gridspec as gridspec
from cleverhans.attacks import FastGradientMethod, BasicIterativeMethod, MadryEtAl, DeepFool, MomentumIterativeMethod, \
    StochasticMomentumIterativeMethod
from scipy.fftpack import dct, idct
from io import BytesIO
import base64
import time
# from backend_mnist.ssim import MultiScaleSSIM
from skvideo.measure import ssim, msssim, psnr
from PIL import Image
from skimage import color
import cv2

img_size = 32
img_chan = 3
n_classes = 10

(X_train, y_train), (X_test, y_test) = cifar10.load_data()

if keras.backend.image_dim_ordering() == 'th':
    X_train = X_train.reshape(X_train.shape[0], img_chan, img_size, img_size)
    X_test = X_test.reshape(X_test.shape[0], img_chan, img_size, img_size)
else:
    X_train = X_train.reshape(X_train.shape[0], img_size, img_size, img_chan)
    X_test = X_test.reshape(X_test.shape[0], img_size, img_size, img_chan)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
# print(y_test)

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, n_classes)
Y_test = np_utils.to_categorical(y_test, n_classes)
# print(Y_test)
# print(y_test.shape)
tf.reset_default_graph()

# Define input TF placeholder
x = tf.placeholder(tf.float32, (None, img_size, img_size, img_chan), name='x')
y = tf.placeholder(tf.float32, (None, n_classes), name='y')
y_t = tf.placeholder(tf.float32, (None, n_classes), name='y_t')
x_adv = tf.placeholder(tf.float32, (None, img_size, img_size, img_chan), name='x')
epsilon = tf.placeholder(tf.float32, ())
iter_epsilon = tf.placeholder(tf.float32, ())

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.1
sess = tf.InteractiveSession(config=config)
keras.backend.set_session(sess)
sess.run(tf.global_variables_initializer())
sess.run(tf.local_variables_initializer())

# for i in range(10000):
# Y = 0.2125R + 0.7154G + 0.0721B
# im1 = X_test
# print(X_test.shape)
# im2 = im1.reshape(32, 32, 3)


fgsm_params = {'eps': epsilon,
               'clip_min': 0.,
               'clip_max': 1.}

pgd_params = {'eps': epsilon,
              'nb_iter': 10,
              'eps_iter': iter_epsilon,
              'clip_min': 0.,
              'clip_max': 1.}

bim_params = {'eps': epsilon,
              'nb_iter': 10,
              'eps_iter': iter_epsilon,
              'clip_min': 0.,
              'clip_max': 1.}

df_params = {'clip_min': 0.,
             'clip_max': 1.}

mim_params = {'eps': epsilon,
              'nb_iter': 10,
              'eps_iter': iter_epsilon,
              'clip_min': 0.,
              'clip_max': 1.}

smim_params = {'eps': epsilon,
               'nb_iter': 10,
               'eps_iter': iter_epsilon,
               'clip_min': 0.,
               'clip_max': 1.}


def load(sess, name='cifar'):
    print('\nLoading saved cifar')
    tf_model_load(sess, 'backend_cifar/cifar/{}'.format(name))


# cifar = cnn_model_small(img_rows=img_size, img_cols=img_size, channels=img_chan)
# load(sess, name='cifar_clean1')


model = cnn_model_big(img_rows=img_size, img_cols=img_size, channels=img_chan)
load(sess, name='cifar_clean(big)')

fgsm = FastGradientMethod(model, sess=sess)
adv_fgsm = fgsm.generate(x, **fgsm_params)

pgd = MadryEtAl(model, sess=sess)
adv_pgd = pgd.generate(x, **pgd_params)

bim = BasicIterativeMethod(model, sess=sess)
adv_bim = bim.generate(x, **bim_params)

mim = MomentumIterativeMethod(model, sess=sess)
adv_mim = mim.generate(x, **mim_params)

smim = StochasticMomentumIterativeMethod(model, sess=sess)
adv_smim = smim.generate(x, **smim_params)

preds_clean = model(x)
preds_fgsm = model(adv_fgsm)
preds_pgd = model(adv_pgd)
preds_bim = model(adv_bim)
preds_mim = model(adv_mim)
preds_smim = model(adv_smim)


# dct_change(X_test[0:1])
# img_change_cifar(X_test[0:1])


# def predict(sess, x_data):
#     yval = sess.run(preds, feed_dict={x: x_data})
#     arg = np.argsort(yval, axis=1).tolist()[0][::-1]
#     val = np.sort(yval, axis=1).tolist()[0][::-1]
#     return [arg, val]


def make_attack(sess, x_data, batch_size=128, ep=0.02, attack_name=None):
    n_sample = x_data.shape[0]
    n_batch = int((n_sample + batch_size - 1) / batch_size)
    x_adv = np.empty_like(x_data)
    for batch in range(n_batch):
        start = batch * batch_size
        end = min(n_sample, start + batch_size)
        feed_dict = {x: x_data[start:end], epsilon: ep, iter_epsilon: ep / 10}
        if attack_name == 'fgsm':
            adv = sess.run(adv_fgsm, feed_dict=feed_dict)
        elif attack_name == 'pgd':
            adv = sess.run(adv_pgd, feed_dict=feed_dict)
        elif attack_name == 'bim':
            adv = sess.run(adv_bim, feed_dict=feed_dict)
        # elif attack_name == 'df':
        #     adv = sess.run(adv_df, feed_dict=feed_dict)
        elif attack_name == 'mim':
            adv = sess.run(adv_mim, feed_dict=feed_dict)
        elif attack_name == 'smim':
            adv = sess.run(adv_smim, feed_dict=feed_dict)
        x_adv[start:end] = adv
    return x_adv


# adv_fgsm_1 = np.load('backend_cifar/npy/fgsm_test_big(0.05).npy')
# adv_pgd_1 = np.load('backend_cifar/npy/pgd_test_big(0.05).npy')
# adv_bim_1 = np.load('backend_cifar/npy/bim_test_big(0.05).npy')
# adv_mim_1 = np.load('backend_cifar/npy/mim_test_big(0.05).npy')
# adv_smim_1 = np.load('backend_cifar/npy/smim_test_big(0.05).npy')

# adv_fgsm_1 = np.load('backend_cifar/npy/fgsm_test_small(0.05).npy')
# adv_pgd_1 = np.load('backend_cifar/npy/pgd_test_small(0.05).npy')
# adv_bim_1 = np.load('backend_cifar/npy/bim_test_small(0.05).npy')
# adv_mim_1 = np.load('backend_cifar/npy/mim_test_small(0.05).npy')
# adv_smim_1 = np.load('backend_cifar/npy/smim_test_small(0.05).npy')
#
# eval_params = {'batch_size': 128}
# feed = {epsilon: 0.05}
#
# accuracy_fgsm = model_eval(sess, x, y, preds_clean, adv_fgsm_1, Y_test, feed=feed, args=eval_params)
# print(str(accuracy_fgsm))
# accuracy_pgd = model_eval(sess, x, y, preds_clean, adv_pgd_1, Y_test, feed=feed, args=eval_params)
# print(str(accuracy_pgd))
# accuracy_bim = model_eval(sess, x, y, preds_clean, adv_bim_1, Y_test, feed=feed, args=eval_params)
# print(str(accuracy_bim))
# accuracy_mim = model_eval(sess, x, y, preds_clean, adv_mim_1, Y_test, feed=feed, args=eval_params)
# print(str(accuracy_mim))
# accuracy_smim = model_eval(sess, x, y, preds_clean, adv_smim_1, Y_test, feed=feed, args=eval_params)
# print(str(accuracy_smim))


# start_pgd1 = time.time()
# adv_pgd_2 = make_attack(sess, X_test, attack_name='pgd')
# end_pgd1 = time.time()
# print("pgd_time1: " + str(end_pgd1 - start_pgd1))
#
# start_bim = time.time()
# adv_bim_1 = make_attack(sess, X_test, attack_name='bim')
# end_bim = time.time()
# print("bim_time: " + str(end_bim - start_bim))
#
# start_fgsm = time.time()
# adv_fgsm_1 = make_attack(sess, X_test, attack_name='fgsm')
# end_fgsm = time.time()
# print("fgsm_time: " + str(end_fgsm - start_fgsm))
#
# start_pgd = time.time()
# adv_pgd_1 = make_attack(sess, X_test, attack_name='pgd')
# end_pgd = time.time()
# print("pgd_time: " + str(end_pgd - start_pgd))
#
# start_mim = time.time()
# adv_mim_1 = make_attack(sess, X_test, attack_name='mim')
# end_mim = time.time()
# print("mim_time: " + str(end_mim - start_mim))
#
# start_smim = time.time()
# adv_smim_1 = make_attack(sess, X_test, attack_name='smim')
# end_smim = time.time()
# print("smim_time: " + str(end_smim - start_smim))

# np.save('backend_cifar/npy/fgsm_test_small(0.05).npy', adv_fgsm_1)
# np.save('backend_cifar/npy/pgd_test_small(0.05).npy', adv_pgd_1)
# np.save('backend_cifar/npy/bim_test_small(0.05).npy', adv_bim_1)
# np.save('backend_cifar/npy/mim_test_small(0.05).npy', adv_mim_1)
# np.save('backend_cifar/npy/smim_test_small(0.05).npy', adv_smim_1)

# np.save('backend_cifar/npy/fgsm_test_big(0.05).npy', adv_fgsm_1)
# np.save('backend_cifar/npy/pgd_test_big(0.05).npy', adv_pgd_1)
# np.save('backend_cifar/npy/bim_test_big(0.05).npy', adv_bim_1)
# np.save('backend_cifar/npy/mim_test_big(0.05).npy', adv_mim_1)
# np.save('backend_cifar/npy/smim_test_big(0.05).npy', adv_smim_1)

# adv_fgsm_1 = np.load('backend_cifar/npy/fgsm_test_small.npy')
# adv_pgd_1 = np.load('backend_cifar/npy/pgd_test_small.npy')
# adv_bim_1 = np.load('backend_cifar/npy/bim_test_small.npy')
# adv_mim_1 = np.load('backend_cifar/npy/mim_test_small.npy')
# adv_mimc_1 = np.load('backend_cifar/npy/mimc_test_small.npy')

#
# ssim_fgsm = 0
# ssim_pgd = 0
# ssim_bim = 0
# ssim_mim = 0

# for i in range(1000):
#     ssim_fgsm += MultiScaleSSIM(X_test[i:i + 1], adv_fgsm_1[i:i + 1])
# ssim_fgsm = ssim_fgsm / 1000
# print(ssim_fgsm)
#
# for i in range(1000):
#     ssim_pgd += MultiScaleSSIM(X_test[i:i + 1], adv_pgd_1[i:i + 1])
# ssim_pgd = ssim_pgd / 1000
# print(ssim_pgd)
#
# for i in range(1000):
#     ssim_bim += MultiScaleSSIM(X_test[i:i + 1], adv_bim_1[i:i + 1])
# ssim_bim = ssim_bim / 1000
# print(ssim_bim)
#
# for i in range(1000):
#     ssim_mim += MultiScaleSSIM(X_test[i:i + 1], adv_mim_1[i:i + 1])
# ssim_mim = ssim_mim / 1000
# print(ssim_mim)


# eval_params = {'batch_size': 128}
# feed = {epsilon: 0.3}
# accuracy_clean = model_eval(sess, x, y, preds_clean, X_test, Y_test, args=eval_params)
# print("accuracy_clean" + str(accuracy_clean))
# accuracy_fgsm = model_eval(sess, x, y, preds_fgsm, X_test, Y_test, feed=feed, args=eval_params)
# print("accuracy_fgsm" + str(accuracy_fgsm))
# accuracy_pgd = model_eval(sess, x, y, preds_pgd, X_test, Y_test, feed=feed, args=eval_params)
# print("accuracy_pgd" + str(accuracy_pgd))
# accuracy_bim = model_eval(sess, x, y, preds_bim, X_test, Y_test, feed=feed, args=eval_params)
# print("accuracy_bim" + str(accuracy_bim))
# accuracy_mim = model_eval(sess, x, y, preds_mim, X_test, Y_test, feed=feed, args=eval_params)
# print("accuracy_mim" + str(accuracy_mim))
# accuracy_mimc = model_eval(sess, x, y, preds_mimc, X_test, Y_test, feed=feed, args=eval_params)
# print("accuracy_mimc" + str(accuracy_mimc))
#
# X_test_gray = color.rgb2gray(X_test).reshape(10000, 32, 32, 1)
#
# accuracy_clean_list = []
# accuracy_fgsm_list = []
# accuracy_pgd_list = []
# accuracy_bim_list = []
# accuracy_mim_list = []
# accuracy_mimc_list = []
# ssim_fgsm_list = []
# ssim_pgd_list = []
# ssim_bim_list = []
# ssim_mim_list = []
# ssim_mimc_list = []
#
# for j in range(11):
#     print(j)
#     # j = j + 1
#     eval_params = {'batch_size': 128}
#     feed = {epsilon: j * 0.005}
#     adv_fgsm_1 = make_attack(sess, X_test, ep=j * 0.005, attack_name='fgsm')
#     adv_pgd_1 = make_attack(sess, X_test, ep=j * 0.005, attack_name='pgd')
#     adv_bim_1 = make_attack(sess, X_test, ep=j * 0.005, attack_name='bim')
#     adv_mim_1 = make_attack(sess, X_test, ep=j * 0.005, attack_name='mim')
#     adv_mimc_1 = make_attack(sess, X_test, ep=j * 0.005, attack_name='mimc')
#     print("attack over")
#     # accuracy_clean = model_eval(sess, x, y, preds_clean, X_test, Y_test, feed=feed, args=eval_params)
#     accuracy_fgsm = model_eval(sess, x, y, preds_fgsm, X_test, Y_test, feed=feed, args=eval_params)
#     accuracy_pgd = model_eval(sess, x, y, preds_pgd, X_test, Y_test, feed=feed, args=eval_params)
#     accuracy_bim = model_eval(sess, x, y, preds_bim, X_test, Y_test, feed=feed, args=eval_params)
#     accuracy_mim = model_eval(sess, x, y, preds_mim, X_test, Y_test, feed=feed, args=eval_params)
#     accuracy_mimc = model_eval(sess, x, y, preds_mimc, X_test, Y_test, feed=feed, args=eval_params)
#     print("accuracy over")
#
#     adv_fgsm_1_gray = color.rgb2gray(adv_fgsm_1).reshape(10000, 32, 32, 1)
#     adv_pgd_1_gray = color.rgb2gray(adv_pgd_1).reshape(10000, 32, 32, 1)
#     adv_bim_1_gray = color.rgb2gray(adv_bim_1).reshape(10000, 32, 32, 1)
#     adv_mim_1_gray = color.rgb2gray(adv_mim_1).reshape(10000, 32, 32, 1)
#     adv_mimc_1_gray = color.rgb2gray(adv_mimc_1).reshape(10000, 32, 32, 1)
#
#     ssim_fgsm = np.mean(ssim(X_test_gray * 255, adv_fgsm_1_gray * 255))
#     ssim_pgd = np.mean(ssim(X_test_gray * 255, adv_pgd_1_gray * 255))
#     ssim_bim = np.mean(ssim(X_test_gray * 255, adv_bim_1_gray * 255))
#     ssim_mim = np.mean(ssim(X_test_gray * 255, adv_mim_1_gray * 255))
#     ssim_mimc = np.mean(ssim(X_test_gray * 255, adv_mimc_1_gray * 255))
#     print("ssim over")
#     # accuracy_clean_list.append(accuracy_clean)
#     accuracy_fgsm_list.append(accuracy_fgsm)
#     accuracy_pgd_list.append(accuracy_pgd)
#     accuracy_bim_list.append(accuracy_bim)
#     accuracy_mim_list.append(accuracy_mim)
#     accuracy_mimc_list.append(accuracy_mimc)
#     ssim_fgsm_list.append(float(ssim_fgsm))
#     ssim_pgd_list.append(float(ssim_pgd))
#     ssim_bim_list.append(float(ssim_bim))
#     ssim_mim_list.append(float(ssim_mim))
#     ssim_mimc_list.append(float(ssim_mimc))
#     # print(accuracy_clean_list)
#     print(accuracy_fgsm_list)
#     print(accuracy_pgd_list)
#     print(accuracy_bim_list)
#     print(accuracy_mim_list)
#     print(accuracy_mimc_list)
#     print(ssim_fgsm_list)
#     print(ssim_pgd_list)
#     print(ssim_bim_list)
#     print(ssim_mim_list)
#     print(ssim_mimc_list)
#
# # print(accuracy_clean_list)
# print(accuracy_fgsm_list)
# print(accuracy_pgd_list)
# print(accuracy_bim_list)
# print(accuracy_mim_list)
# print(accuracy_mimc_list)
# print(ssim_fgsm_list)
# print(ssim_pgd_list)
# print(ssim_bim_list)
# print(ssim_mim_list)
# print(ssim_mimc_list)


X_test_gray = color.rgb2gray(X_test).reshape(10000, 32, 32, 1)
accuracy_fgsm_list = []
accuracy_pgd_list = []
accuracy_bim_list = []
accuracy_mim_list = []
accuracy_smim_list = []
ssim_fgsm_list = []
ssim_pgd_list = []
ssim_bim_list = []
ssim_mim_list = []
ssim_smim_list = []

for j in range(11):
    print(j)
    # j = j + 1
    eval_params = {'batch_size': 128}
    # feed = {epsilon: j * 0.002}
    # adv_fgsm_1 = make_attack(sess, X_test, ep=j*0.01, attack_name='fgsm')
    # adv_pgd_1 = make_attack(sess, X_test, ep=j*0.01, attack_name='pgd')
    # adv_bim_1 = make_attack(sess, X_test, ep=j*0.01, attack_name='bim')
    # adv_mim_1 = make_attack(sess, X_test, ep=j*0.01, attack_name='mim')
    # adv_smim_1 = make_attack(sess, X_test, ep=j*0.01, attack_name='smim')
    # print("attack over")


    # np.save('backend_cifar/npy/fgsm_test_small(' + str(j * 0.01) + ').npy', adv_fgsm_1)
    # np.save('backend_cifar/npy/pgd_test_small(' + str(j * 0.01) + ').npy', adv_pgd_1)
    # np.save('backend_cifar/npy/bim_test_small(' + str(j * 0.01) + ').npy', adv_bim_1)
    # np.save('backend_cifar/npy/mim_test_small(' + str(j * 0.01) + ').npy', adv_mim_1)
    # np.save('backend_cifar/npy/smim_test_small(' + str(j * 0.01) + ').npy', adv_smim_1)

    # np.save('backend_cifar/npy/fgsm_test_big(' + str(j * 0.01) + ').npy', adv_fgsm_1)
    # np.save('backend_cifar/npy/pgd_test_big(' + str(j * 0.01) + ').npy', adv_pgd_1)
    # np.save('backend_cifar/npy/bim_test_big(' + str(j * 0.01) + ').npy', adv_bim_1)
    # np.save('backend_cifar/npy/mim_test_big(' + str(j * 0.01) + ').npy', adv_mim_1)
    # np.save('backend_cifar/npy/smim_test_big(' + str(j * 0.01) + ').npy', adv_smim_1)

    # adv_fgsm_1 = np.load('backend_cifar/npy/fgsm_test_small(' + str(j * 0.01) + ').npy')
    # adv_pgd_1 = np.load('backend_cifar/npy/pgd_test_small(' + str(j * 0.01) + ').npy')
    # adv_bim_1 = np.load('backend_cifar/npy/bim_test_small(' + str(j * 0.01) + ').npy')
    # adv_mim_1 = np.load('backend_cifar/npy/mim_test_small(' + str(j * 0.01) + ').npy')
    # adv_smim_1 = np.load('backend_cifar/npy/smim_test_small(' + str(j * 0.01) + ').npy')

    adv_fgsm_1 = np.load('backend_cifar/npy/fgsm_test_big(' + str(j * 0.01) + ').npy')
    adv_pgd_1 = np.load('backend_cifar/npy/pgd_test_big(' + str(j * 0.01) + ').npy')
    adv_bim_1 = np.load('backend_cifar/npy/bim_test_big(' + str(j * 0.01) + ').npy')
    adv_mim_1 = np.load('backend_cifar/npy/mim_test_big(' + str(j * 0.01) + ').npy')
    adv_smim_1 = np.load('backend_cifar/npy/smim_test_big(' + str(j * 0.01) + ').npy')

    # accuracy_fgsm = model_eval(sess, x, y, preds_clean, adv_fgsm_1, Y_test, args=eval_params)
    # accuracy_pgd = model_eval(sess, x, y, preds_clean, adv_pgd_1, Y_test, args=eval_params)
    # accuracy_bim = model_eval(sess, x, y, preds_clean, adv_bim_1, Y_test, args=eval_params)
    # accuracy_mim = model_eval(sess, x, y, preds_clean, adv_mim_1, Y_test, args=eval_params)
    # accuracy_smim = model_eval(sess, x, y, preds_clean, adv_smim_1, Y_test, args=eval_params)
    # print("accuracy over")
    adv_fgsm_1_gray = color.rgb2gray(adv_fgsm_1).reshape(10000, 32, 32, 1)
    adv_pgd_1_gray = color.rgb2gray(adv_pgd_1).reshape(10000, 32, 32, 1)
    adv_bim_1_gray = color.rgb2gray(adv_bim_1).reshape(10000, 32, 32, 1)
    adv_mim_1_gray = color.rgb2gray(adv_mim_1).reshape(10000, 32, 32, 1)
    adv_smim_1_gray = color.rgb2gray(adv_smim_1).reshape(10000, 32, 32, 1)

    ssim_fgsm = np.mean(ssim(X_test_gray * 255, adv_fgsm_1_gray * 255))
    ssim_pgd = np.mean(ssim(X_test_gray * 255, adv_pgd_1_gray * 255))
    ssim_bim = np.mean(ssim(X_test_gray * 255, adv_bim_1_gray * 255))
    ssim_mim = np.mean(ssim(X_test_gray * 255, adv_mim_1_gray * 255))
    ssim_smim = np.mean(ssim(X_test_gray * 255, adv_smim_1_gray * 255))
    print("ssim over")
    # accuracy_fgsm_list.append(accuracy_fgsm)
    # accuracy_pgd_list.append(accuracy_pgd)
    # accuracy_bim_list.append(accuracy_bim)
    # accuracy_mim_list.append(accuracy_mim)
    # accuracy_smim_list.append(accuracy_smim)
    ssim_fgsm_list.append(float(ssim_fgsm))
    ssim_pgd_list.append(float(ssim_pgd))
    ssim_bim_list.append(float(ssim_bim))
    ssim_mim_list.append(float(ssim_mim))
    ssim_smim_list.append(float(ssim_smim))
    # print(accuracy_fgsm_list)
    # print(accuracy_pgd_list)
    # print(accuracy_bim_list)
    # print(accuracy_mim_list)
    # print(accuracy_smim_list)
    print(ssim_fgsm_list)
    print(ssim_pgd_list)
    print(ssim_bim_list)
    print(ssim_mim_list)
    print(ssim_smim_list)
print()
# print(accuracy_fgsm_list)
# print(accuracy_pgd_list)
# print(accuracy_bim_list)
# print(accuracy_mim_list)
# print(accuracy_smim_list)
print(ssim_fgsm_list)
print(ssim_pgd_list)
print(ssim_bim_list)
print(ssim_mim_list)
print(ssim_smim_list)
