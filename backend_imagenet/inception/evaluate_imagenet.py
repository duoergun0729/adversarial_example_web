from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import numpy as np
from keras.utils import np_utils

import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from cleverhans.utils_tf import tf_model_load, model_eval
from cleverhans.attacks import FastGradientMethod, BasicIterativeMethod, MadryEtAl, MomentumIterativeMethod, \
    StochasticMomentumIterativeMethod
from io import BytesIO
import base64
from tensorflow.contrib.slim.nets import inception
# from tensorflow.contrib.slim.nets import resnet_v2
# from backend_imagenet.inception.inception_v4 import inception_v4, inception_v4_arg_scope
from backend_imagenet.inception.inception_resnet2 import inception_resnet_v2, inception_resnet_v2_arg_scope
import tensorflow.contrib.slim as slim
import os
import json
import PIL
import glob
import time

img_list = []


def timage():
    for files in glob.glob('backend_imagenet/cifar/n01440764/*.JPEG'):
        img = PIL.Image.open(files)
        wide = img.width > img.height
        new_w = 299 if wide else int(img.width * 299 / img.height)
        new_h = 299 if not wide else int(img.height * 299 / img.width)
        img = img.resize((new_w, new_h)).crop((0, 0, 299, 299))
        img = (np.asarray(img) / 255.0).astype(np.float32).reshape(1, 299, 299, 3)
        print(img.shape)
        img_list.append(img)
    img_tuple = tuple(img_list)
    imgs_all = np.vstack(img_tuple)
    print(imgs_all.shape)
    np.save("backend_imagenet/inception/val.npy", imgs_all)


# timage()

# a = np.array([1])
#
# for i in range(999):
#     a = np.vstack((a, [1]))
# print(a.shape)
# Y_test = np_utils.to_categorical(a, 1000)
# print(Y_test)
# np.save("backend_imagenet/inception/y_test.npy", Y_test)

tf.logging.set_verbosity(tf.logging.ERROR)
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.8
sess = tf.InteractiveSession(config=config)
# sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
sess.run(tf.local_variables_initializer())


#
# sess_defense = tf.InteractiveSession(config=config)
# # sess_defense = tf.InteractiveSession()
# sess_defense.run(tf.global_variables_initializer())
# sess_defense.run(tf.local_variables_initializer())
#
# sess_res2 = tf.InteractiveSession(config=config)
# # sess_res2 = tf.InteractiveSession()
# sess_res2.run(tf.global_variables_initializer())
# sess_res2.run(tf.local_variables_initializer())

# sess_v4 = tf.InteractiveSession(config=config)
# # sess_res2 = tf.InteractiveSession()
# sess_v4.run(tf.global_variables_initializer())
# sess_v4.run(tf.local_variables_initializer())

# sess_v2 = tf.InteractiveSession(config=config)
# # sess_res2 = tf.InteractiveSession()
# sess_v2.run(tf.global_variables_initializer())
# sess_v2.run(tf.local_variables_initializer())


class InceptionModel(object):
    """Model class for CleverHans library."""

    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.built = False

    def __call__(self, x_input):
        """Constructs cifar and return probabilities for given input."""
        reuse = True if self.built else None
        with slim.arg_scope(inception.inception_v3_arg_scope()):
            _, end_points = inception.inception_v3(
                x_input, num_classes=self.num_classes, is_training=False,
                reuse=reuse)
        self.built = True
        output = end_points['Predictions']
        # Strip off the extra reshape op at the output
        probs = output.op.inputs[0]
        return probs


class Inceptionv2Model(object):
    """Model class for CleverHans library."""

    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.built = False

    def __call__(self, x_input):
        """Constructs cifar and return probabilities for given input."""
        reuse = True if self.built else None
        with slim.arg_scope(inception.inception_v2_arg_scope()):
            _, end_points = inception.inception_v2(
                x_input, num_classes=self.num_classes, is_training=False,
                reuse=reuse)
        self.built = True
        output = end_points['Predictions']
        # Strip off the extra reshape op at the output
        probs = output.op.inputs[0]
        return probs


class Resnetv2Model(object):
    """Model class for CleverHans library."""

    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.built = False

    def __call__(self, x_input):
        """Constructs cifar and return probabilities for given input."""
        reuse = True if self.built else None
        with slim.arg_scope(inception_resnet_v2_arg_scope()):
            _, end_points = inception_resnet_v2(
                x_input, num_classes=self.num_classes, is_training=False,
                reuse=reuse)
        self.built = True
        output = end_points['Predictions']
        # Strip off the extra reshape op at the output
        probs = output.op.inputs[0]
        return probs


# class Inceptionv4Model(object):
#     """Model class for CleverHans library."""
#
#     def __init__(self, num_classes):
#         self.num_classes = num_classes
#         self.built = False
#
#     def __call__(self, x_input):
#         """Constructs cifar and return probabilities for given input."""
#         reuse = True if self.built else None
#         with slim.arg_scope(inception_v4_arg_scope()):
#             _, end_points = inception_v4(
#                 x_input, num_classes=self.num_classes, is_training=False,
#                 reuse=reuse)
#         self.built = True
#         output = end_points['Predictions']
#         # Strip off the extra reshape op at the output
#         probs = output.op.inputs[0]
#         return probs


eps = 2.0 * 16 / 255.0
img_size = 299
img_chan = 3
n_classes = 1000
num_classes = 1001

x = tf.placeholder(tf.float32, (None, img_size, img_size, img_chan), name='x')
y = tf.placeholder(tf.float32, (None, n_classes), name='y')
y_t = tf.placeholder(tf.float32, (None, n_classes), name='y_t')
x_adv = tf.placeholder(tf.float32, (None, img_size, img_size, img_chan), name='x')
epsilon = tf.placeholder(tf.float32, ())

data_dir = 'backend_imagenet/cifar/'

model = InceptionModel(num_classes)
# cifar = Resnetv2Model(num_classes)
# model_v4 = Inceptionv4Model(num_classes)
# model_v2 = Inceptionv2Model(num_classes)

with open(os.path.join(data_dir, 'imagenet_zh_cn.json'), encoding='utf-8') as f:
    imagenet_labels = json.load(f)

# with open('./backend_imagenet/cifar/imagenet.json') as f:
#     imagenet_labels = json.load(f)

fgsm_params = {'eps': epsilon,
               'clip_min': 0.,
               'clip_max': 1.}

pgd_params = {'eps': epsilon,
              'nb_iter': 10,
              'eps_iter': .001,
              'clip_min': 0.,
              'clip_max': 1.}

bim_params = {'eps': epsilon,
              'nb_iter': 10,
              'eps_iter': .001,
              'clip_min': 0.,
              'clip_max': 1.}

mim_params = {'eps': epsilon,
              'nb_iter': 10,
              'eps_iter': .001,
              'clip_min': 0.,
              'clip_max': 1.}

smim_params = {'eps': epsilon,
               'nb_iter': 10,
               'eps_iter': .001,
               'clip_min': 0.,
               'clip_max': 1.}

fgsm = FastGradientMethod(model)
adv_fgsm = fgsm.generate(x, **fgsm_params)

pgd = MadryEtAl(model)
adv_pgd = pgd.generate(x, **pgd_params)

bim = BasicIterativeMethod(model)
adv_bim = bim.generate(x, **bim_params)

mim = MomentumIterativeMethod(model)
adv_mim = mim.generate(x, **mim_params)

smim = StochasticMomentumIterativeMethod(model)
adv_smim = smim.generate(x, **smim_params)

preds = model(x)
preds_fgsm = model(adv_fgsm)
preds_pgd = model(adv_pgd)
preds_bim = model(adv_bim)
preds_mim = model(adv_mim)

# restore_vars = [
#     var for var in tf.global_variables()
#     if var.name.startswith('InceptionResnetV2/')
# ]
# start = time.time()
# saver1 = tf.train.Saver(restore_vars)
# sess.run(tf.global_variables_initializer())
# saver1.restore(sess, os.path.join(data_dir, 'inception_resnet_v2.ckpt'))
# end = time.time()
# print("res time" + str(end - start))


restore_vars = [
    var for var in tf.global_variables()
    if var.name.startswith('InceptionV3/')
]

start = time.time()
saver = tf.train.Saver(restore_vars)
sess.run(tf.global_variables_initializer())
saver.restore(sess, os.path.join(data_dir, 'inception_v3.ckpt'))
end = time.time()
print("incv3 time" + str(end - start))


def predict(sess, x_data):
    yval = sess.run(preds, feed_dict={x: x_data})
    arg = np.argsort(yval, axis=1).tolist()[0][::-1]
    val = np.sort(yval, axis=1).tolist()[0][::-1]
    return [arg, val]


def make_attack(sess, x_data, batch_size=1, ep=0.1, attack_name=None):
    n_sample = x_data.shape[0]
    n_batch = int((n_sample + batch_size - 1) / batch_size)
    x_adv = np.empty_like(x_data)
    for batch in range(n_batch):
        start = batch * batch_size
        end = min(n_sample, start + batch_size)
        feed_dict = {x: x_data[start:end], epsilon: ep}
        if attack_name == 'fgsm':
            adv = sess.run(adv_fgsm, feed_dict=feed_dict)
        elif attack_name == 'pgd':
            adv = sess.run(adv_pgd, feed_dict=feed_dict)
        elif attack_name == 'bim':
            adv = sess.run(adv_bim, feed_dict=feed_dict)
        elif attack_name == 'mim':
            adv = sess.run(adv_mim, feed_dict=feed_dict)
        elif attack_name == 'smim':
            adv = sess.run(adv_smim, feed_dict=feed_dict)
        x_adv[start:end] = adv
    return x_adv


def num_to_name(input_list):
    output_result = []
    for item in input_list:
        output_result = output_result + [imagenet_labels[item]]
    return output_result


global imgs

x_test = np.load("backend_imagenet/inception/val.npy")
y_test = np.load("backend_imagenet/inception/y_test.npy")

# print("attack1 start")
# adv_fgsm1 = make_attack(sess, x_test, attack_name="fgsm")
# np.save("backend_imagenet/inception/fgsm_v3(0.1).npy", adv_fgsm1)
# print("attack2 start")
# adv_pgd1 = make_attack(sess, x_test, attack_name="pgd")
# np.save("backend_imagenet/inception/pgd_v3(0.1).npy", adv_pgd1)
# print("attack3 start")
# adv_bim1 = make_attack(sess, x_test, attack_name="bim")
# np.save("backend_imagenet/inception/bim_v3(0.1).npy", adv_bim1)
# print("attack4 start")
# adv_mim1 = make_attack(sess, x_test, attack_name="mim")
# np.save("backend_imagenet/inception/mim_v3(0.1).npy", adv_mim1)
# print("attack5 start")
# adv_smim1 = make_attack(sess, x_test, attack_name="smim")
# np.save("backend_imagenet/inception/smim_v3(0.1).npy", adv_smim1)

# print("attack1 start")
# adv_fgsm1 = make_attack(sess, x_test, attack_name="fgsm")
# np.save("backend_imagenet/inception/fgsm_v2(0.1).npy", adv_fgsm1)
# print("attack2 start")
# adv_pgd1 = make_attack(sess, x_test, attack_name="pgd")
# np.save("backend_imagenet/inception/pgd_v2(0.1).npy", adv_pgd1)
# print("attack3 start")
# adv_bim1 = make_attack(sess, x_test, attack_name="bim")
# np.save("backend_imagenet/inception/bim_v2(0.1).npy", adv_bim1)
# print("attack4 start")
# adv_mim1 = make_attack(sess, x_test, attack_name="mim")
# np.save("backend_imagenet/inception/mim_v2(0.1).npy", adv_mim1)
# print("attack5 start")
# adv_smim1 = make_attack(sess, x_test, attack_name="smim")
# np.save("backend_imagenet/inception/smim_v2(0.1).npy", adv_smim1)

# adv_fgsm1 = np.load("backend_imagenet/inception/fgsm_v3(0.1).npy")
# adv_pgd1 = np.load("backend_imagenet/inception/pgd_v3(0.1).npy")
# adv_bim1 = np.load("backend_imagenet/inception/bim_v3(0.1).npy")
# adv_mim1 = np.load("backend_imagenet/inception/mim_v3(0.1).npy")
# adv_smim1 = np.load("backend_imagenet/inception/smim_v3(0.1).npy")

adv_fgsm1 = np.load("backend_imagenet/inception/fgsm_v2(0.1).npy")
adv_pgd1 = np.load("backend_imagenet/inception/pgd_v2(0.1).npy")
adv_bim1 = np.load("backend_imagenet/inception/bim_v2(0.1).npy")
adv_mim1 = np.load("backend_imagenet/inception/mim_v2(0.1).npy")
adv_smim1 = np.load("backend_imagenet/inception/smim_v2(0.1).npy")

eval_params = {'batch_size': 4}
acc_fgsm = model_eval(sess, x, y, preds, X_test=adv_fgsm1, Y_test=y_test, args=eval_params)
print(acc_fgsm)
acc_pgd = model_eval(sess, x, y, preds, X_test=adv_pgd1, Y_test=y_test, args=eval_params)
print(acc_pgd)
acc_bim = model_eval(sess, x, y, preds, X_test=adv_bim1, Y_test=y_test, args=eval_params)
print(acc_bim)
acc_mim = model_eval(sess, x, y, preds, X_test=adv_mim1, Y_test=y_test, args=eval_params)
print(acc_mim)
acc_smim = model_eval(sess, x, y, preds, X_test=adv_smim1, Y_test=y_test, args=eval_params)
print(acc_smim)
# adv_fgsm = make_attack(sess, x_test, attack_name="fgsm")
# np.save("backend_imagenet/inception/fgsm_v3(0.03).npy", adv_fgsm)
# adv_pgd = make_attack(sess, x_test, attack_name="pgd")
# np.save("backend_imagenet/inception/pgd_v3(0.03).npy", adv_pgd)
# adv_bim = make_attack(sess, x_test, attack_name="bim")
# np.save("backend_imagenet/inception/bim_v3(0.03).npy", adv_bim)
# adv_mim = make_attack(sess, x_test, attack_name="mim")
# np.save("backend_imagenet/inception/mim_v3(0.03).npy", adv_mim)
# adv_smim = make_attack(sess, x_test, attack_name="smim")
# np.save("backend_imagenet/inception/smim_v3(0.03).npy", adv_smim)

# v3_fgsm = np.load("backend_imagenet/inception/inceptionv3_fgsm.npy")
# v3_pgd = np.load("backend_imagenet/inception/inceptionv3_pgd.npy")
# v3_bim = np.load("backend_imagenet/inception/inceptionv3_bim.npy")
# v3_mim = np.load("backend_imagenet/inception/inceptionv3_mim.npy")

# r2_fgsm = np.load("backend_imagenet/inception/resnetv2_fgsm.npy")
# r2_pgd = np.load("backend_imagenet/inception/resnetv2_pgd.npy")
# r2_bim = np.load("backend_imagenet/inception/resnetv2_bim.npy")
# r2_mim = np.load("backend_imagenet/inception/resnetv2_mim.npy")

# img_change_imagenet(r2_fgsm[0:1])
# start_acc = time.time()
# acc = model_eval(sess, x, y, preds, X_test=r2_fgsm, Y_test=y_test, args=eval_params)
# end_acc = time.time()
# print("fgsm")
# print(end_acc - start_acc)
# print(acc)
#
# start_acc = time.time()
# acc = model_eval(sess, x, y, preds, X_test=r2_pgd, Y_test=y_test, args=eval_params)
# end_acc = time.time()
# print("pgd")
# print(end_acc - start_acc)
# print(acc)
#
# start_acc = time.time()
# acc = model_eval(sess, x, y, preds, X_test=r2_bim, Y_test=y_test, args=eval_params)
# end_acc = time.time()
# print("bim")
# print(end_acc - start_acc)
# print(acc)
#
# start_acc = time.time()
# acc = model_eval(sess, x, y, preds, X_test=r2_mim, Y_test=y_test, args=eval_params)
# end_acc = time.time()
# print("mim")
# print(end_acc - start_acc)
# print(acc)



# start_acc = time.time()
# acc = model_eval(sess_v4, x, y, preds, X_test=x_test, Y_test=y_test, args=eval_params)
# end_acc = time.time()
# print(end_acc - start_acc)
# print(acc)
# start_acc = time.time()
# acc = model_eval(sess_v4, x, y, preds, X_test=r2_fgsm, Y_test=y_test, args=eval_params)
# end_acc = time.time()
# print("fgsm")
# print(end_acc - start_acc)
# print(acc)
#
# start_acc = time.time()
# acc = model_eval(sess_v4, x, y, preds, X_test=r2_pgd, Y_test=y_test, args=eval_params)
# end_acc = time.time()
# print("pgd")
# print(end_acc - start_acc)
# print(acc)
#
# start_acc = time.time()
# acc = model_eval(sess_v4, x, y, preds, X_test=r2_bim, Y_test=y_test, args=eval_params)
# end_acc = time.time()
# print("bim")
# print(end_acc - start_acc)
# print(acc)
#
# start_acc = time.time()
# acc = model_eval(sess_v4, x, y, preds, X_test=r2_mim, Y_test=y_test, args=eval_params)
# end_acc = time.time()
# print("mim")
# print(end_acc - start_acc)
# print(acc)
