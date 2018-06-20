from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from cleverhans_tutorials.tutorial_models import make_basic_cnn_big, make_basic_cnn

from cleverhans.utils_tf import tf_model_load
from cleverhans.attacks import FastGradientMethod, BasicIterativeMethod, MadryEtAl, MomentumIterativeMethod, \
    StochasticMomentumIterativeMethod
from io import BytesIO
import base64

g_mnist = tf.Graph()
tf.logging.set_verbosity(tf.logging.ERROR)
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.05
sess_mnist = tf.InteractiveSession(graph=g_mnist, config=config)
# sess = tf.InteractiveSession()
sess_mnist.run(tf.global_variables_initializer())
sess_mnist.run(tf.local_variables_initializer())

img_size = 28
img_chan = 1
n_classes = 10
num_classes = 10

x = tf.placeholder(tf.float32, (None, img_size, img_size, img_chan), name='x')
y = tf.placeholder(tf.float32, (None, n_classes), name='y')
y_t = tf.placeholder(tf.float32, (None, num_classes), name='y_t')
x_adv = tf.placeholder(tf.float32, (None, img_size, img_size, img_chan), name='x')
epsilon = tf.placeholder(tf.float32, ())
iter_epsilon = tf.placeholder(tf.float32, ())

# model = make_basic_cnn_big(img_rows=img_size, img_cols=img_size, channels=img_chan)
model = make_basic_cnn()
preds = model(x)

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


with sess_mnist.as_default():
    with g_mnist.as_default():
        tf.global_variables_initializer().run()
        saver = tf.train.Saver()
        saver.restore(sess_mnist, 'model/mnist/clean_model')
# saver = tf.train.Saver()
# saver.restore(sess_mnist, 'model/mnist/clean_model')
# def load(sess_mnist, name='mnist'):
#     print('\nLoading saved mnist')
#     tf_model_load(sess_mnist, 'model/mnist/{}'.format(name))
#
#
# load(sess_mnist, name='clean_model')


def img_change_mnist(img):
    X_tmp1 = np.empty((10, 28, 28))
    X_tmp = 1 - img
    X_tmp1[0] = np.squeeze(X_tmp)
    fig = plt.figure(figsize=(1, 1))
    gs = gridspec.GridSpec(1, 1)
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(X_tmp1[0], cmap='gray', interpolation='none')
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
    # plt.show()
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
        y_target = np.zeros((1, 10))
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


def name_to_num(name):
    if name == '不选择' or name == '':
        return -1
    for i in range(10):
        if name == str(i):
            return i
    return -1


@csrf_exempt
def drawinput_mnist(request):
    response = {}
    input = ((255 - np.array(eval(request.POST.get('inputs')), dtype=np.float32)) / 255.0).reshape(1, 28, 28, 1)
    multipleSelection = request.POST.getlist('multipleSelection', [])
    print(multipleSelection)
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

    x_adv_fgsm = make_attack(sess_mnist, input, attack_name='fgsm', target=name_to_num(fgsm_target),
                             eps=fgsm_disturb)
    x_adv_pgd = make_attack(sess_mnist, input, attack_name='pgd', target=name_to_num(pgd_target),
                            eps=pgd_disturb)
    x_adv_bim = make_attack(sess_mnist, input, attack_name='bim', target=name_to_num(bim_target),
                            eps=bim_disturb)
    x_adv_mim = make_attack(sess_mnist, input, attack_name='mim', target=name_to_num(mim_target),
                            eps=mim_disturb)
    x_adv_smim = make_attack(sess_mnist, input, attack_name='smim', target=name_to_num(smim_target),
                             eps=smim_disturb)

    # noise_fgsm = x_adv_fgsm - input
    # noise_pgd = x_adv_pgd - input
    # noise_bim = x_adv_bim - input
    # noise_mim = x_adv_mim - input

    # print(input)
    # print(x_adv_fgsm - input)
    # print(x_adv_pgd - input)
    # print(x_adv_bim - input)
    # print(x_adv_mim - input)

    output_clean_1 = predict(sess_mnist, input)
    output_fgsm_1 = predict(sess_mnist, x_adv_fgsm)
    output_pgd_1 = predict(sess_mnist, x_adv_pgd)
    output_bim_1 = predict(sess_mnist, x_adv_bim)
    output_mim_1 = predict(sess_mnist, x_adv_mim)
    output_smim_1 = predict(sess_mnist, x_adv_smim)

    # output_clean_2 = predict(sess_defense, input)
    # output_fgsm_2 = predict(sess_defense, x_adv_fgsm)
    # output_pgd_2 = predict(sess_defense, x_adv_pgd)
    # output_bim_2 = predict(sess_defense, x_adv_bim)
    # output_mim_2 = predict(sess_defense, x_adv_mim)

    data_clean = img_change_mnist(input)
    data_fgsm = img_change_mnist(x_adv_fgsm)
    data_pgd = img_change_mnist(x_adv_pgd)
    data_bim = img_change_mnist(x_adv_bim)
    data_mim = img_change_mnist(x_adv_mim)
    data_smim = img_change_mnist(x_adv_smim)

    src_clean = 'data:image/png;base64,' + str(data_clean)
    src_fgsm = 'data:image/png;base64,' + str(data_fgsm)
    src_pgd = 'data:image/png;base64,' + str(data_pgd)
    src_bim = 'data:image/png;base64,' + str(data_bim)
    src_mim = 'data:image/png;base64,' + str(data_mim)
    src_smim = 'data:image/png;base64,' + str(data_smim)

    # data_clean_noise = img_change_mnist(input)
    # data_fgsm_noise = img_change_mnist(noise_fgsm * 255)
    # data_pgd_noise = img_change_mnist(noise_pgd * 255)
    # data_bim_noise = img_change_mnist(noise_bim * 255)
    # data_mim_noise = img_change_mnist(noise_mim * 255)

    # src_clean_noise = 'data:image/png;base64,' + str(data_clean_noise)
    # src_fgsm_noise = 'data:image/png;base64,' + str(data_fgsm_noise)
    # src_pgd_noise = 'data:image/png;base64,' + str(data_pgd_noise)
    # src_bim_noise = 'data:image/png;base64,' + str(data_bim_noise)
    # src_mim_noise = 'data:image/png;base64,' + str(data_mim_noise)

    echarts_attack = []
    # echarts_defense = []
    echarts_clean_dict = {'name': output_clean_1[0][0:5], 'value': output_clean_1[1][0:5]}
    echarts_attack.append(echarts_clean_dict)
    echarts_fgsm_dict = {'name': output_fgsm_1[0][0:5], 'value': output_fgsm_1[1][0:5]}
    echarts_attack.append(echarts_fgsm_dict)
    echarts_pgd_dict = {'name': output_pgd_1[0][0:5], 'value': output_pgd_1[1][0:5]}
    echarts_attack.append(echarts_pgd_dict)
    echarts_bim_dict = {'name': output_bim_1[0][0:5], 'value': output_bim_1[1][0:5]}
    echarts_attack.append(echarts_bim_dict)
    echarts_mim_dict = {'name': output_mim_1[0][0:5], 'value': output_mim_1[1][0:5]}
    echarts_attack.append(echarts_mim_dict)
    echarts_smim_dict = {'name': output_smim_1[0][0:5], 'value': output_smim_1[1][0:5]}
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
        str(output_clean_1[0][0]) + "<br>(" + '%.2f' % (output_clean_1[1][0] * 100) + "%)",
        str(output_fgsm_1[0][0]) + "<br>(" + '%.2f' % (output_fgsm_1[1][0] * 100) + "%)",
        str(output_pgd_1[0][0]) + "<br>(" + '%.2f' % (output_pgd_1[1][0] * 100) + "%)",
        str(output_bim_1[0][0]) + "<br>(" + '%.2f' % (output_bim_1[1][0] * 100) + "%)",
        str(output_mim_1[0][0]) + "<br>(" + '%.2f' % (output_mim_1[1][0] * 100) + "%)",
        str(output_smim_1[0][0]) + "<br>(" + '%.2f' % (output_smim_1[1][0] * 100) + "%)"]

    # response['defense_result'] = [
    #     str(imagenet_labels[output_clean_2[0][0] - 1]) + "<br>(" + '%.2f' % (output_clean_2[1][0] * 100) + "%)",
    #     str(imagenet_labels[output_fgsm_2[0][0] - 1]) + "<br>(" + '%.2f' % (output_fgsm_2[1][0] * 100) + "%)",
    #     str(imagenet_labels[output_pgd_2[0][0] - 1]) + "<br>(" + '%.2f' % (output_pgd_2[1][0] * 100) + "%)",
    #     str(imagenet_labels[output_bim_2[0][0] - 1]) + "<br>(" + '%.2f' % (output_bim_2[1][0] * 100) + "%)", ]
    # str(imagenet_labels[output_mim_2[0][0] - 1]) + "(" + '%.4f' % output_mim_2[1][0] + ")"]

    return JsonResponse(response)


@csrf_exempt
def check(request):
    response = {'check': True}
    return JsonResponse(response)
