import tensorflow as tf
import numpy as np
from cleverhans.utils_tf import tf_model_load
from cleverhans.attacks import FastGradientMethod, BasicIterativeMethod, MadryEtAl


def load(sess, name='cifar', dir_name='cifar'):
    print('\nLoading saved cifar "' + str(name) + '"')
    with sess.as_default():
        tf_model_load(sess, dir_name + name)


def predict(sess, x, predictions, X_test, num=10):
    with sess.as_default():
        yval = sess.run(predictions, feed_dict={x: X_test})
    arg = np.argsort(yval, axis=1).tolist()[0][::-1][0:num]
    val = np.sort(yval, axis=1).tolist()[0][::-1][0:num]
    return [arg, val]

#
# def make_attack(sess, x, y_t, epsilon, x_data, cifar, attack_name=None, target=-1, eps=0.3):
#     fgsm = FastGradientMethod(cifar, sess=sess)
#     adv_fgsm = fgsm.generate(x, **fgsm_params)
#     adv_fgsm_target = fgsm.generate(x, **fgsm_params, y_target=y_t)
#
#     pgd = MadryEtAl(cifar, sess=sess)
#     adv_pgd = pgd.generate(x, **pgd_params)
#     adv_pgd_target = pgd.generate(x, **pgd_params, y_target=y_t)
#
#     bim = BasicIterativeMethod(cifar, sess=sess)
#     adv_bim = bim.generate(x, **bim_params)
#     adv_bim_target = bim.generate(x, **bim_params, y_target=y_t)
#     adv = None
#     if target is not -1:
#         y_target = np.zeros((1, 10))
#         y_target[np.arange(1), target] = 1
#         feed_dict_target = {x: x_data, y_t: y_target, epsilon: eps}
#         if attack_name == 'fgsm':
#             adv = sess.run(adv_fgsm_target, feed_dict=feed_dict_target)
#         elif attack_name == 'pgd':
#             adv = sess.run(adv_pgd_target, feed_dict=feed_dict_target)
#         elif attack_name == 'bim':
#             adv = sess.run(adv_bim_target, feed_dict=feed_dict_target)
#     else:
#         feed_dict = {x: x_data, epsilon: eps}
#         if attack_name == 'fgsm':
#             adv = sess.run(adv_fgsm, feed_dict=feed_dict)
#         elif attack_name == 'pgd':
#             adv = sess.run(adv_pgd, feed_dict=feed_dict)
#         elif attack_name == 'bim':
#             adv = sess.run(adv_bim, feed_dict=feed_dict)
#     return adv


