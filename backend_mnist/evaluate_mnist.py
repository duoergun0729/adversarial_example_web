from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from cleverhans.utils_keras import cnn_model_big
from cleverhans_tutorials.tutorial_models import make_basic_cnn, make_basic_cnn_big
from cleverhans.utils_mnist import data_mnist
from cleverhans import utils
from cleverhans.utils_tf import tf_model_load, model_eval, model_train
from cleverhans.attacks import FastGradientMethod, BasicIterativeMethod, MadryEtAl, DeepFool, MomentumIterativeMethod, \
    StochasticMomentumIterativeMethod
from io import BytesIO
import base64
from scipy.fftpack import dct
import time
from skvideo.measure import ssim, msssim, psnr

# Get MNIST test data
train_start = 0
train_end = 60000
test_start = 0
test_end = 10000
X_train, y_train, X_test, y_test = data_mnist(train_start=train_start,
                                              train_end=train_end,
                                              test_start=test_start,
                                              test_end=test_end)
label_smooth = .1
y_train = y_train.clip(label_smooth / 9., 1. - label_smooth)

img_size = 28
img_chan = 1
n_classes = 10
x = tf.placeholder(tf.float32, (None, img_size, img_size, img_chan),
                   name='x')
y = tf.placeholder(tf.float32, (None, n_classes), name='y')
y_t = tf.placeholder(tf.float32, (None, n_classes), name='y_t')
x_adv = tf.placeholder(tf.float32, (None, img_size, img_size, img_chan),
                       name='x_adv')
epsilon = tf.placeholder(tf.float32, ())
epsilon_iter = tf.placeholder(tf.float32, ())

fgsm_params = {'eps': epsilon,
               'clip_min': 0.,
               'clip_max': 1.}

pgd_params = {'eps': epsilon,
              'nb_iter': 10,
              'eps_iter': epsilon_iter,
              'clip_min': 0.,
              'clip_max': 1.}

bim_params = {'eps': epsilon,
              'nb_iter': 10,
              'eps_iter': epsilon_iter,
              'clip_min': 0.,
              'clip_max': 1.}

df_params = {'clip_min': 0.,
             'clip_max': 1.}

mim_params = {'eps': epsilon,
              'nb_iter': 10,
              'eps_iter': epsilon_iter,
              'clip_min': 0.,
              'clip_max': 1.}

smim_params = {'eps': epsilon,
               'nb_iter': 10,
               'eps_iter': epsilon_iter,
               'clip_min': 0.,
               'clip_max': 1.}

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.1
sess = tf.InteractiveSession(config=config)
# sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
sess.run(tf.local_variables_initializer())

train_params_clean = {
    'nb_epochs': 6,
    'batch_size': 128,
    'learning_rate': 0.001,
    'filename': 'clean_model_big',
    'train_dir': 'backend_mnist/MNIST_model_final/'
}
# model_cnn = make_basic_cnn()
model_cnn = make_basic_cnn_big(nb_filters=64)
preds_clean = model_cnn(x)
rng = np.random.RandomState([2017, 8, 30])


# model_train(sess, x, y, preds_clean, X_train, y_train, save=True,
#             args=train_params_clean, rng=rng)


def load(sess, name='cifar'):
    print('\nLoading saved cifar')
    tf_model_load(sess, 'backend_mnist/MNIST_model_final/{}'.format(name))


# load(sess, name='clean_model')
load(sess, name='clean_model(big_model)')

fgsm = FastGradientMethod(model_cnn, sess=sess)
adv_fgsm = fgsm.generate(x, **fgsm_params)
adv_fgsm_target = fgsm.generate(x, **fgsm_params, y_target=y_t)

pgd = MadryEtAl(model_cnn, sess=sess)
adv_pgd = pgd.generate(x, **pgd_params)
adv_pgd_target = pgd.generate(x, **pgd_params, y_target=y_t)

bim = BasicIterativeMethod(model_cnn, sess=sess)
adv_bim = bim.generate(x, **bim_params)
adv_bim_target = bim.generate(x, **bim_params, y_target=y_t)

deepfool = DeepFool(model_cnn, sess=sess)
adv_df = deepfool.generate(x, **df_params)

mim = MomentumIterativeMethod(model_cnn, sess=sess)
adv_mim = mim.generate(x, **mim_params)

smim = StochasticMomentumIterativeMethod(model_cnn, sess=sess)
adv_smim = smim.generate(x, **smim_params)

report = utils.AccuracyReport()

# rng = np.random.RandomState([2017, 8, 30])

preds_fgsm = model_cnn(adv_fgsm)
preds_pgd = model_cnn(adv_pgd)
preds_bim = model_cnn(adv_bim)
preds_mim = model_cnn(adv_mim)
preds_smim = model_cnn(adv_smim)


def predict(sess, x_data):
    yval = sess.run(preds_clean, feed_dict={x: x_data})
    arg = np.argsort(yval, axis=1).tolist()[0][::-1]
    val = np.sort(yval, axis=1).tolist()[0][::-1]
    return [arg, val]


def make_attack(sess, x_data, batch_size=128, ep=0.1, attack_name=None):
    n_sample = x_data.shape[0]
    n_batch = int((n_sample + batch_size - 1) / batch_size)
    x_adv = np.empty_like(x_data)
    for batch in range(n_batch):
        start = batch * batch_size
        end = min(n_sample, start + batch_size)
        feed_dict = {x: x_data[start:end], epsilon: ep, epsilon_iter: ep / 10}
        if attack_name == 'fgsm':
            adv = sess.run(adv_fgsm, feed_dict=feed_dict)
        elif attack_name == 'pgd':
            adv = sess.run(adv_pgd, feed_dict=feed_dict)
        elif attack_name == 'bim':
            adv = sess.run(adv_bim, feed_dict=feed_dict)
        elif attack_name == 'df':
            adv = sess.run(adv_df, feed_dict=feed_dict)
        elif attack_name == 'mim':
            adv = sess.run(adv_mim, feed_dict=feed_dict)
        elif attack_name == 'smim':
            adv = sess.run(adv_smim, feed_dict=feed_dict)
        x_adv[start:end] = adv
    return x_adv


# adv_fgsm_1 = np.load('backend_mnist/npy/fgsm_test_big(0.1).npy')
# adv_pgd_1 = np.load('backend_mnist/npy/pgd_test_big(0.1).npy')
# adv_bim_1 = np.load('backend_mnist/npy/bim_test_big(0.1).npy')
# adv_mim_1 = np.load('backend_mnist/npy/mim_test_big(0.1).npy')
# adv_smim_1 = np.load('backend_mnist/npy/smim_test_big(0.1).npy')


# adv_fgsm_1 = np.load('backend_mnist/npy/fgsm_test_small(0.1).npy')
# adv_pgd_1 = np.load('backend_mnist/npy/pgd_test_small(0.1).npy')
# adv_bim_1 = np.load('backend_mnist/npy/bim_test_small(0.1).npy')
# adv_mim_1 = np.load('backend_mnist/npy/mim_test_small(0.1).npy')
# adv_smim_1 = np.load('backend_mnist/npy/smim_test_small(0.1).npy')

# adv_fgsm_1 = np.load('backend_mnist/npy/fgsm_test.npy')
# adv_pgd_1 = np.load('backend_mnist/npy/pgd_test.npy')
# adv_bim_1 = np.load('backend_mnist/npy/bim_test.npy')
# adv_mimc_1 = np.load('backend_mnist/npy/mim_test.npy')


# eval_params = {'batch_size': 128}
# feed = {epsilon: 0.5}
# #
# accuracy_fgsm = model_eval(sess, x, y, preds_clean, adv_fgsm_1, y_test, feed=feed, args=eval_params)
# print("accuracy_fgsm" + str(accuracy_fgsm))
# accuracy_pgd = model_eval(sess, x, y, preds_clean, adv_pgd_1, y_test, feed=feed, args=eval_params)
# print("accuracy_pgd" + str(accuracy_pgd))
# accuracy_bim = model_eval(sess, x, y, preds_clean, adv_bim_1, y_test, feed=feed, args=eval_params)
# print("accuracy_bim" + str(accuracy_bim))
# accuracy_mim = model_eval(sess, x, y, preds_clean, adv_mim_1, y_test, feed=feed, args=eval_params)
# print("accuracy_mim" + str(accuracy_mim))
# accuracy_smim = model_eval(sess, x, y, preds_clean, adv_smim_1, y_test, feed=feed, args=eval_params)
# print("accuracy_smim" + str(accuracy_smim))

# accuracy_fgsm = model_eval(sess, x, y, preds_fgsm, X_test, y_test, feed=feed, args=eval_params)
# print("accuracy_fgsm" + str(accuracy_fgsm))
# accuracy_pgd = model_eval(sess, x, y, preds_pgd, X_test, y_test, feed=feed, args=eval_params)
# print("accuracy_pgd" + str(accuracy_pgd))
# accuracy_bim = model_eval(sess, x, y, preds_bim, X_test, y_test, feed=feed, args=eval_params)
# print("accuracy_bim" + str(accuracy_bim))
# accuracy_mim = model_eval(sess, x, y, preds_mim, X_test, y_test, feed=feed, args=eval_params)
# print("accuracy_mim" + str(accuracy_mim))
# accuracy_smim = model_eval(sess, x, y, preds_smim, X_test, y_test, feed=feed, args=eval_params)
# print("accuracy_smim" + str(accuracy_smim))

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


# start_pgd = time.time()
# for i in range(20):
#     adv_pgd_1 = make_attack(sess, X_test, attack_name='pgd', ep=i*0.02)
#     accuracy_pgd = model_eval(sess, x, y, preds_pgd, X_test, y_test, args=eval_params)
#
#
# end_pgd = time.time()
# print("pgd_time: " + str(end_pgd - start_pgd))

# np.save('backend_mnist/npy/fgsm_test_small(0.1).npy', adv_fgsm_1)
# np.save('backend_mnist/npy/pgd_test_small(0.1).npy', adv_pgd_1)
# np.save('backend_mnist/npy/bim_test_small(0.1).npy', adv_bim_1)
# np.save('backend_mnist/npy/mim_test_small(0.1).npy', adv_mim_1)
# np.save('backend_mnist/npy/smim_test_small(0.1).npy', adv_smim_1)

# np.save('backend_mnist/npy/fgsm_test_big(0.1).npy', adv_fgsm_1)
# np.save('backend_mnist/npy/pgd_test_big(0.1).npy', adv_pgd_1)
# np.save('backend_mnist/npy/bim_test_big(0.1).npy', adv_bim_1)
# np.save('backend_mnist/npy/mim_test_big(0.1).npy', adv_mim_1)
# np.save('backend_mnist/npy/smim_test_big(0.1).npy', adv_smim_1)

# start_mim = time.time()
# adv_mim_1 = make_attack(sess, X_test, attack_name='mim')
# end_mim = time.time()
# print("mim_time: " + str(end_mim - start_mim))
# np.save('backend_mnist/npy/mim_test.npy', adv_mim_1)

# adv_fgsm_1 = np.load('backend_mnist/npy/fgsm_test_big.npy')
# adv_pgd_1 = np.load('backend_mnist/npy/pgd_test_big.npy')
# adv_bim_1 = np.load('backend_mnist/npy/bim_test_big.npy')
# adv_mim_1 = np.load('backend_mnist/npy/mim_test_big.npy')
# adv_mimc_1 = np.load('backend_mnist/npy/mimc_test_big1.npy')
# print(adv_fgsm_1.shape)
# adv_fgsm_1 = adv_fgsm_1.reshape(10000, 28, 28)
# print(adv_fgsm_1.shape)
# ssim_fgsm = 0
# ssim_pgd = 0
# ssim_bim = 0
# ssim_df = 0

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
# ssim_mim = 0
# for i in range(1000):
#     ssim_mim += MultiScaleSSIM(X_test[i:i + 1], adv_mim_1[i:i + 1])
# ssim_mim = ssim_mim / 1000
# print(ssim_mim)
#
# ssim_mimc = 0
# for i in range(1000):
#     ssim_mimc += MultiScaleSSIM(X_test[i:i + 1], adv_mimc_1[i:i + 1])
# ssim_mimc = ssim_mimc / 1000
# print(ssim_mimc)
#

# print(MultiScaleSSIM(X_test[0:1], adv_fgsm_1[0:1]))
# print(MultiScaleSSIM(X_test[0:1], adv_pgd_1[0:1]))
# print(MultiScaleSSIM(X_test[0:1], adv_bim_1[0:1]))
# print(MultiScaleSSIM(X_test[0:1], adv_mimc_1[0:1]))
# print(psnr(X_test[0:1] * 255, adv_mimc_1[0:1] * 255))
# print(ssim(X_test[0:1] * 255, adv_mimc_1[0:1] * 255))
# print(type(ssim(X_test[0:1] * 255, adv_mimc_1[0:1] * 255)))
# print(msssim(X_test[0:1]*255, adv_mimc_1[0:1]*255))
# print(X_test[0:1])


# def img_change(img):
#     X_tmp1 = np.empty((10, 28, 28))
#     X_tmp = 1 - img
#     X_tmp1[0] = np.squeeze(X_tmp)
#     fig = plt.figure(figsize=(1, 1))
#     gs = gridspec.GridSpec(1, 1)
#     ax = fig.add_subplot(gs[0, 0])
#     ax.imshow(X_tmp1[0], cmap='gray', interpolation='none')
#     # 去除坐标轴
#     ax.set_xticks([])
#     ax.set_yticks([])
#     # 去除边框
#     ax.spines['top'].set_visible(False)
#     ax.spines['right'].set_visible(False)
#     ax.spines['bottom'].set_visible(False)
#     ax.spines['left'].set_visible(False)
#     # 设置大小
#     fig.set_size_inches(1.5, 1.5)
#     plt.show()
#     gs.tight_layout(fig)
#     sio = BytesIO()
#     fig.savefig(sio, format='png', bbox_inches='tight', pad_inches=0.0)
#     data = base64.encodebytes(sio.getvalue()).decode()
#     return data


# img_change(X_test[0:1])
# img_change(adv_mimc_1[0:1])
#
# print(predict(sess, X_test[0:1]))
# print(predict(sess, adv_df_1))


# eval_params = {'batch_size': 128}
# feed = {epsilon: 0.15}
# accuracy_clean = model_eval(sess, x, y, preds_clean, X_test, y_test, args=eval_params)
# print("accuracy_clean" + str(accuracy_clean))
# accuracy_fgsm = model_eval(sess, x, y, preds_fgsm, X_test, y_test, feed=feed, args=eval_params)
# print("accuracy_fgsm" + str(accuracy_fgsm))
# accuracy_pgd = model_eval(sess, x, y, preds_pgd, X_test, y_test, feed=feed, args=eval_params)
# print("accuracy_pgd" + str(accuracy_pgd))
# accuracy_bim = model_eval(sess, x, y, preds_bim, X_test, y_test, feed=feed, args=eval_params)
# print("accuracy_bim" + str(accuracy_bim))
# accuracy_mim = model_eval(sess, x, y, preds_mim, X_test, y_test, feed=feed, args=eval_params)
# print("accuracy_mim" + str(accuracy_mim))
# accuracy_mimc = model_eval(sess, x, y, preds_mimc, X_test, y_test, feed=feed, args=eval_params)
# print("accuracy_mimc" + str(accuracy_mimc))

# accuracy_clean_list = []
# accuracy_fgsm_list = []
# accuracy_pgd_list = []
# accuracy_bim_list = []
# accuracy_mim_list = []
# accuracy_mimc_list = []
ssim_fgsm_list = []
ssim_pgd_list = []
ssim_bim_list = []
ssim_mim_list = []
ssim_smim_list = []

for j in range(21):
    print(j)
    # j = j + 1
    eval_params = {'batch_size': 128}
    feed = {epsilon: j * 0.01}
    adv_fgsm_1 = make_attack(sess, X_test, ep=j * 0.1, attack_name='fgsm')
    adv_pgd_1 = make_attack(sess, X_test, ep=j * 0.1, attack_name='pgd')
    adv_bim_1 = make_attack(sess, X_test, ep=j * 0.1, attack_name='bim')
    adv_mim_1 = make_attack(sess, X_test, ep=j * 0.1, attack_name='mim')
    adv_smim_1 = make_attack(sess, X_test, ep=j * 0.1, attack_name='smim')
    print("attack over")
    # accuracy_clean = model_eval(sess, x, y, preds_clean, X_test, y_test, feed=feed, args=eval_params)
    # accuracy_fgsm = model_eval(sess, x, y, preds_fgsm, X_test, y_test, feed=feed, args=eval_params)
    # accuracy_pgd = model_eval(sess, x, y, preds_pgd, X_test, y_test, feed=feed, args=eval_params)
    # accuracy_bim = model_eval(sess, x, y, preds_bim, X_test, y_test, feed=feed, args=eval_params)
    # accuracy_mim = model_eval(sess, x, y, preds_mim, X_test, y_test, feed=feed, args=eval_params)
    # accuracy_mimc = model_eval(sess, x, y, preds_smim, X_test, y_test, feed=feed, args=eval_params)
    # print("accuracy over")

    ssim_fgsm = np.mean(ssim(X_test*255, adv_fgsm_1*255))
    ssim_pgd = np.mean(ssim(X_test*255, adv_pgd_1*255))
    ssim_bim = np.mean(ssim(X_test*255, adv_bim_1*255))
    ssim_mim = np.mean(ssim(X_test*255, adv_mim_1*255))
    ssim_smim = np.mean(ssim(X_test*255, adv_smim_1*255))
    print("ssim over")
    # accuracy_clean_list.append(accuracy_clean)
    # accuracy_fgsm_list.append(accuracy_fgsm)
    # accuracy_pgd_list.append(accuracy_pgd)
    # accuracy_bim_list.append(accuracy_bim)
    # accuracy_mim_list.append(accuracy_mim)
    # accuracy_mimc_list.append(accuracy_mimc)
    ssim_fgsm_list.append(float(ssim_fgsm))
    ssim_pgd_list.append(float(ssim_pgd))
    ssim_bim_list.append(float(ssim_bim))
    ssim_mim_list.append(float(ssim_mim))
    ssim_smim_list.append(float(ssim_smim))
    # print(accuracy_clean_list)
    # print(accuracy_fgsm_list)
    # print(accuracy_pgd_list)
    # print(accuracy_bim_list)
    # print(accuracy_mim_list)
    # print(accuracy_mimc_list)
    print(ssim_fgsm_list)
    print(ssim_pgd_list)
    print(ssim_bim_list)
    print(ssim_mim_list)
    print(ssim_smim_list)

# print(accuracy_clean_list)
# print(accuracy_fgsm_list)
# print(accuracy_pgd_list)
# print(accuracy_bim_list)
# print(accuracy_mim_list)
# print(accuracy_mimc_list)
print()
print(ssim_fgsm_list)
print(ssim_pgd_list)
print(ssim_bim_list)
print(ssim_mim_list)
print(ssim_smim_list)

# accuracy_mim_list = []
#
# ssim_mim_list = []
#
# ssim_mim = 0
#
# for j in range(20):
#     # print(j)
#     j = j + 1
#     eval_params = {'batch_size': 128}
#     feed = {epsilon: j * 0.02}
#     adv_mim_1 = make_attack(sess, X_test, ep=j * 0.02, attack_name='mim')
#
#     print("attack over")
#     accuracy_mim = model_eval(sess, x, y, preds_mim, X_test, y_test, feed=feed, args=eval_params)
#
#     print("accuracy over")
#     ssim_mim = 0
#
#     for i in range(1000):
#         ssim_mim += MultiScaleSSIM(X_test[i:i + 1], adv_mim_1[i:i + 1])
#
#     ssim_mim = ssim_mim / 1000
#
#     print("ssim over")
#     accuracy_mim_list.append(accuracy_mim)
#
#     ssim_mim_list.append(float(ssim_mim))
#
#     print(accuracy_mim_list)
#
#     print(ssim_mim_list)
#
# print(accuracy_mim_list)
#
# print(ssim_mim_list)
