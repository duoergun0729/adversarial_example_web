from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from cleverhans.utils_tf import tf_model_load
from cleverhans.attacks import FastGradientMethod, BasicIterativeMethod, MadryEtAl, MomentumIterativeMethod, \
    StochasticMomentumIterativeMethod
from io import BytesIO
import base64
from tensorflow.contrib.slim.nets import inception
import tensorflow.contrib.slim as slim
import os
import json
import PIL
from backend_imagenet.inception.inception_resnet2 import inception_resnet_v2, inception_resnet_v2_arg_scope
import time

g_imagenet = tf.Graph()
tf.logging.set_verbosity(tf.logging.ERROR)
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.05
sess = tf.InteractiveSession(graph=g_imagenet, config=config)
# sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
sess.run(tf.local_variables_initializer())


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


img_size = 299
img_chan = 3
num_classes = 1001

x = tf.placeholder(tf.float32, (None, img_size, img_size, img_chan), name='x')
y = tf.placeholder(tf.float32, (None, num_classes), name='y')
y_t = tf.placeholder(tf.float32, (None, num_classes), name='y_t')
x_adv = tf.placeholder(tf.float32, (None, img_size, img_size, img_chan), name='x')
epsilon = tf.placeholder(tf.float32, ())
iter_epsilon = tf.placeholder(tf.float32, ())
data_dir = 'model/imagenet'

model = InceptionModel(num_classes)
preds = model(x)
# model_resnet = Resnetv2Model(num_classes)
# preds_res = model_resnet(x)
# cifar = Resnetv2Model(num_classes)
# preds = cifar(x)

with open(os.path.join(data_dir, 'imagenet_zh_cn.json'), encoding='utf-8') as f:
    imagenet_labels = json.load(f)


def name_to_num(name):
    if name == '不选择' or name == '':
        return -1
    for i in range(len(imagenet_labels)):
        if name == imagenet_labels[i]:
            return i + 1
    return -1


fgsm_params = {'eps': epsilon,
               'clip_min': 0.,
               'clip_max': 1.,
               }

pgd_params = {'eps': epsilon,
              'eps_iter': iter_epsilon,
              'nb_iter': 10,
              'clip_min': 0.,
              'clip_max': 1.}

bim_params = {'eps': epsilon,
              'eps_iter': iter_epsilon,
              'nb_iter': 10,
              'clip_min': 0.,
              'clip_max': 1.}

mim_params = {'eps': epsilon,
              'eps_iter': iter_epsilon,
              'nb_iter': 10,
              'clip_min': 0.,
              'clip_max': 1.}

smim_params = {'eps': epsilon,
               'eps_iter': iter_epsilon,
               'nb_iter': 10,
               'clip_min': 0.,
               'clip_max': 1.}


fgsm = FastGradientMethod(model)
adv_fgsm = fgsm.generate(x, **fgsm_params)
adv_fgsm_target = fgsm.generate(x, **fgsm_params, y_target=y_t)

pgd = MadryEtAl(model)
adv_pgd = pgd.generate(x, **pgd_params)
adv_pgd_target = pgd.generate(x, **pgd_params, y_target=y_t)

bim = BasicIterativeMethod(model)
adv_bim = bim.generate(x, **bim_params)
adv_bim_target = bim.generate(x, **bim_params, y_target=y_t)

mim = MomentumIterativeMethod(model)
adv_mim = mim.generate(x, **mim_params)
adv_mim_target = mim.generate(x, **mim_params, y_target=y_t)

smim = StochasticMomentumIterativeMethod(model)
adv_smim = smim.generate(x, **smim_params)
adv_smim_target = smim.generate(x, **smim_params, y_target=y_t)

restore_vars = [
    var for var in tf.global_variables()
    if var.name.startswith('InceptionV3/')
]
with sess.as_default():
    with g_imagenet.as_default():
        saver = tf.train.Saver(restore_vars)
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, os.path.join(data_dir, 'inception_v3.ckpt'))
# saver = tf.train.Saver(restore_vars)
# sess.run(tf.global_variables_initializer())
# saver.restore(sess, os.path.join(data_dir, 'inception_v3.ckpt'))
print("inceptionv3 over")


# saver2 = tf.train.Saver(restore_vars)
# sess_defense.run(tf.global_variables_initializer())
# saver2.restore(sess_defense, os.path.join(data_dir, 'adv_inception_v3.ckpt'))
#
# print("adv_inceptionv3 over")


# restore_vars1 = [
#     var for var in tf.global_variables()
#     if var.name.startswith('InceptionResnetV2/')
# ]


# print("resnet2 start")
# start = time.time()
# sess.run(tf.global_variables_initializer())
# saver = tf.train.Saver(restore_vars1)
# saver.restore(sess, os.path.join(data_dir, 'inception_resnet_v2.ckpt'))
# print("resnet2 over")
# end = time.time()
# print("time" + str(end - start))


def img_change_imagenet(img):
    fig = plt.figure(figsize=(1, 1))
    gs = gridspec.GridSpec(1, 1)
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(img[0], interpolation='none')
    # 去除坐标轴
    ax.set_xticks([])
    ax.set_yticks([])
    # 去除边框
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    # 设置大小
    fig.set_size_inches(1.5, 1.5)
    gs.tight_layout(fig)
    sio = BytesIO()
    fig.savefig(sio, format='png', bbox_inches='tight', pad_inches=0.0)
    data = base64.encodebytes(sio.getvalue()).decode()
    return data


def predict(sess, x_data):
    yval = sess.run(preds, feed_dict={x: x_data})
    arg = np.argsort(yval, axis=1).tolist()[0][::-1]
    val = np.sort(yval, axis=1).tolist()[0][::-1]
    return [arg, val]


def make_attack(sess, x_data, attack_name=None, target=-1, eps=0.3):
    adv = None
    if target is not -1:
        y_target = np.zeros((1, 1001))
        y_target[np.arange(1), target] = 1
        feed_dict_target = {x: x_data, y_t: y_target, epsilon: eps, iter_epsilon: eps / 10}
        if attack_name == 'fgsm':
            adv = sess.run(adv_fgsm_target, feed_dict=feed_dict_target)
        elif attack_name == 'pgd':
            adv = sess.run(adv_pgd_target, feed_dict=feed_dict_target)
        elif attack_name == 'bim':
            adv = sess.run(adv_bim_target, feed_dict=feed_dict_target)
        elif attack_name == 'mim':
            adv = sess.run(adv_mim_target, feed_dict=feed_dict_target)
        elif attack_name == 'smim':
            adv = sess.run(adv_smim_target, feed_dict=feed_dict_target)
    else:
        feed_dict = {x: x_data, epsilon: eps, iter_epsilon: eps / 10}
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
    return adv


def num_to_name(input_list):
    output_result = []
    for item in input_list:
        output_result = output_result + [imagenet_labels[item - 1]]
    return output_result


global imgs


@csrf_exempt
def upload_imagenet(request):
    global imgs
    is_jpeg = True
    image = request.FILES.get("file")
    print(os.path.splitext(image.name)[1])
    if os.path.splitext(image.name)[1] == '.png':
        is_jpeg = False
    print(is_jpeg)
    img = PIL.Image.open(image)
    if not is_jpeg:
        png = img.convert('RGB')
        img = png
    wide = img.width > img.height
    new_w = img_size if wide else int(img.width * img_size / img.height)
    new_h = img_size if not wide else int(img.height * img_size / img.width)
    img = img.resize((new_w, new_h)).crop((0, 0, img_size, img_size))
    imgs = (np.asarray(img) / 255.0).astype(np.float32)
    response = {}
    return JsonResponse(response)


@csrf_exempt
def drawinput_imagenet(request):
    response = {}
    global imgs
    if imgs is None:
        return JsonResponse(response)
    imgs = imgs.reshape(1, 299, 299, 3)
    fgsm_disturb = float(request.POST.get('fgsm_disturb'))
    pgd_disturb = float(request.POST.get('pgd_disturb'))
    bim_disturb = float(request.POST.get('bim_disturb'))
    mim_disturb = float(request.POST.get('mim_disturb'))
    smim_disturb = float(request.POST.get('smim_disturb'))
    fgsm_target = str(request.POST.get('fgsm_target'))
    pgd_target = str(request.POST.get('pgd_target'))
    bim_target = str(request.POST.get('bim_target'))
    mim_target = str(request.POST.get('mim_target'))
    smim_target = str(request.POST.get('smim_target'))
    print(fgsm_target)
    # target = int(request.POST.get('target'))
    # input_fgsm = float(request.POST.get('input_fgsm'))
    # input_pgd = float(request.POST.get('input_pgd'))
    # input_bim = float(request.POST.get('input_bim'))
    # input_mim = float(request.POST.get('input_mim'))

    x_adv_fgsm = make_attack(sess, imgs, attack_name='fgsm', target=name_to_num(fgsm_target),
                             eps=fgsm_disturb)
    x_adv_pgd = make_attack(sess, imgs, attack_name='pgd', target=name_to_num(pgd_target),
                            eps=pgd_disturb)
    x_adv_bim = make_attack(sess, imgs, attack_name='bim', target=name_to_num(bim_target),
                            eps=bim_disturb)
    x_adv_mim = make_attack(sess, imgs, attack_name='mim', target=name_to_num(mim_target),
                            eps=mim_disturb)
    x_adv_smim = make_attack(sess, imgs, attack_name='smim', target=name_to_num(smim_target),
                             eps=smim_disturb)

    # noise_fgsm = x_adv_fgsm - imgs
    # noise_pgd = x_adv_pgd - imgs
    # noise_bim = x_adv_bim - imgs
    # noise_mim = x_adv_mim - imgs

    # print(imgs)
    # print(x_adv_fgsm - imgs)
    # print(x_adv_pgd - imgs)
    # print(x_adv_bim - imgs)
    # print(x_adv_mim - imgs)

    output_clean_1 = predict(sess, imgs)
    output_fgsm_1 = predict(sess, x_adv_fgsm)
    output_pgd_1 = predict(sess, x_adv_pgd)
    output_bim_1 = predict(sess, x_adv_bim)
    output_mim_1 = predict(sess, x_adv_mim)
    output_smim_1 = predict(sess, x_adv_smim)

    # output_clean_2 = predict(sess_defense, imgs)
    # output_fgsm_2 = predict(sess_defense, x_adv_fgsm)
    # output_pgd_2 = predict(sess_defense, x_adv_pgd)
    # output_bim_2 = predict(sess_defense, x_adv_bim)
    # output_mim_2 = predict(sess_defense, x_adv_mim)

    data_clean = img_change_imagenet(imgs)
    data_fgsm = img_change_imagenet(x_adv_fgsm)
    data_pgd = img_change_imagenet(x_adv_pgd)
    data_bim = img_change_imagenet(x_adv_bim)
    data_mim = img_change_imagenet(x_adv_mim)
    data_smim = img_change_imagenet(x_adv_smim)

    src_clean = 'data:image/png;base64,' + str(data_clean)
    src_fgsm = 'data:image/png;base64,' + str(data_fgsm)
    src_pgd = 'data:image/png;base64,' + str(data_pgd)
    src_bim = 'data:image/png;base64,' + str(data_bim)
    src_mim = 'data:image/png;base64,' + str(data_mim)
    src_smim = 'data:image/png;base64,' + str(data_smim)

    # data_clean_noise = img_change_imagenet(imgs)
    # data_fgsm_noise = img_change_imagenet(noise_fgsm * 255)
    # data_pgd_noise = img_change_imagenet(noise_pgd * 255)
    # data_bim_noise = img_change_imagenet(noise_bim * 255)
    # data_mim_noise = img_change_imagenet(noise_mim * 255)

    # src_clean_noise = 'data:image/png;base64,' + str(data_clean_noise)
    # src_fgsm_noise = 'data:image/png;base64,' + str(data_fgsm_noise)
    # src_pgd_noise = 'data:image/png;base64,' + str(data_pgd_noise)
    # src_bim_noise = 'data:image/png;base64,' + str(data_bim_noise)
    # src_mim_noise = 'data:image/png;base64,' + str(data_mim_noise)

    echarts_attack = []
    # echarts_defense = []
    echarts_clean_dict = {'name': num_to_name(output_clean_1[0][0:5]), 'value': output_clean_1[1][0:5]}
    echarts_attack.append(echarts_clean_dict)
    echarts_fgsm_dict = {'name': num_to_name(output_fgsm_1[0][0:5]), 'value': output_fgsm_1[1][0:5]}
    echarts_attack.append(echarts_fgsm_dict)
    echarts_pgd_dict = {'name': num_to_name(output_pgd_1[0][0:5]), 'value': output_pgd_1[1][0:5]}
    echarts_attack.append(echarts_pgd_dict)
    echarts_bim_dict = {'name': num_to_name(output_bim_1[0][0:5]), 'value': output_bim_1[1][0:5]}
    echarts_attack.append(echarts_bim_dict)
    echarts_mim_dict = {'name': num_to_name(output_mim_1[0][0:5]), 'value': output_mim_1[1][0:5]}
    echarts_attack.append(echarts_mim_dict)
    echarts_smim_dict = {'name': num_to_name(output_smim_1[0][0:5]), 'value': output_smim_1[1][0:5]}
    echarts_attack.append(echarts_smim_dict)

    # echarts_clean_dict = {'name': num_to_name(output_clean_2[0][0:5]), 'value': output_clean_2[1][0:5]}
    # echarts_defense.append(echarts_clean_dict)
    # echarts_fgsm_dict = {'name': num_to_name(output_clean_2[0][0:5]), 'value': output_fgsm_2[1][0:5]}
    # echarts_defense.append(echarts_fgsm_dict)
    # echarts_pgd_dict = {'name': num_to_name(output_clean_2[0][0:5]), 'value': output_pgd_2[1][0:5]}
    # echarts_defense.append(echarts_pgd_dict)
    # echarts_bim_dict = {'name': num_to_name(output_clean_2[0][0:5]), 'value': output_bim_2[1][0:5]}
    # echarts_defense.append(echarts_bim_dict)

    response['echarts'] = echarts_attack
    # response['echarts_defense'] = echarts_defense
    response['name'] = ['clean', 'fgsm', 'pgd', 'bim', 'mim', 'smim']
    # response['name'] = ['clean', 'fgsm', 'pgd', 'bim', 'mim']
    # response['img'] = [src_clean, src_fgsm, src_pgd, src_bim, src_mim]
    response['img'] = [src_clean, src_fgsm, src_pgd, src_bim, src_mim, src_smim]
    # response['imgnoise'] = [src_clean_noise, src_fgsm_noise, src_pgd_noise, src_bim_noise, src_mim_noise]
    response['attack_result'] = [
        str(imagenet_labels[output_clean_1[0][0] - 1]) + "<br>(" + '%.2f' % (output_clean_1[1][0] * 100) + "%)",
        str(imagenet_labels[output_fgsm_1[0][0] - 1]) + "<br>(" + '%.2f' % (output_fgsm_1[1][0] * 100) + "%)",
        str(imagenet_labels[output_pgd_1[0][0] - 1]) + "<br>(" + '%.2f' % (output_pgd_1[1][0] * 100) + "%)",
        str(imagenet_labels[output_bim_1[0][0] - 1]) + "<br>(" + '%.2f' % (output_bim_1[1][0] * 100) + "%)",
        str(imagenet_labels[output_mim_1[0][0] - 1]) + "<br>(" + '%.2f' % (output_mim_1[1][0] * 100) + "%)",
        str(imagenet_labels[output_smim_1[0][0] - 1]) + "<br>(" + '%.2f' % (output_smim_1[1][0] * 100) + "%)"]

    # response['defense_result'] = [
    #     str(imagenet_labels[output_clean_2[0][0] - 1]) + "<br>(" + '%.2f' % (output_clean_2[1][0] * 100) + "%)",
    #     str(imagenet_labels[output_fgsm_2[0][0] - 1]) + "<br>(" + '%.2f' % (output_fgsm_2[1][0] * 100) + "%)",
    #     str(imagenet_labels[output_pgd_2[0][0] - 1]) + "<br>(" + '%.2f' % (output_pgd_2[1][0] * 100) + "%)",
    #     str(imagenet_labels[output_bim_2[0][0] - 1]) + "<br>(" + '%.2f' % (output_bim_2[1][0] * 100) + "%)", ]
    # str(imagenet_labels[output_mim_2[0][0] - 1]) + "(" + '%.4f' % output_mim_2[1][0] + ")"]

    return JsonResponse(response)
