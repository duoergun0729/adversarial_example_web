import keras
from keras.datasets import cifar10
from keras import backend
from keras.utils import np_utils
import tensorflow as tf
from cleverhans import utils
import numpy as np
from cleverhans.utils_keras import cnn_model_small, cnn_model_big
from cleverhans.attacks import FastGradientMethod, BasicIterativeMethod, MadryEtAl
from cleverhans.utils_tf import model_train, model_eval, batch_eval


if not hasattr(backend, "tf"):
    raise RuntimeError("This tutorial requires keras to be configured"
                       " to use the TensorFlow backend.")
img_rows = 32
img_cols = 32
nb_classes = 10
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
print("load_data over")
if keras.backend.image_dim_ordering() == 'th':
    X_train = X_train.reshape(X_train.shape[0], 3, img_rows, img_cols)
    X_test = X_test.reshape(X_test.shape[0], 3, img_rows, img_cols)
else:
    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 3)
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 3)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)


# train_params = {
#         'nb_epochs': 10,
#         'batch_size': 128,
#         'learning_rate': 0.001,
#         'filename': 'cifar_clean01',
#         'train_dir': 'backend_cifar/cifar/'
#     }
train_params_adv = {
    'nb_epochs': 10,
    'batch_size': 128,
    'learning_rate': 0.001,
    'filename': 'cifar_pgd0.03',
    'train_dir': 'backend_cifar/cifar/'
}

fgsm_params = {'eps': 0.03,
               'clip_min': 0.,
               'clip_max': 1.}

pgd_params = {'eps': 0.03,
              'eps_iter': .01,
              'clip_min': 0.,
              'clip_max': 1.}

bim_params = {'eps': 0.03,
              'eps_iter': .05,
              'clip_min': 0.,
              'clip_max': 1.}

x = tf.placeholder(tf.float32, shape=(None, 32, 32, 3))
y = tf.placeholder(tf.float32, shape=(None, 10))

report = utils.AccuracyReport()

rng = np.random.RandomState([2017, 8, 30])

sess = tf.InteractiveSession()
keras.backend.set_session(sess)
sess.run(tf.global_variables_initializer())
sess.run(tf.local_variables_initializer())

model_2 = cnn_model_small(img_rows=32, img_cols=32, channels=3)
predictions_2 = model_2(x)

fgsm = FastGradientMethod(model_2, sess=sess)
adv_x_2_fgsm = fgsm.generate(x, **fgsm_params)

pgd = MadryEtAl(model_2, sess=sess)
adv_x_2_pgd = pgd.generate(x, **pgd_params)

bim = BasicIterativeMethod(model_2, sess=sess)
adv_x_2_bim = bim.generate(x, **bim_params)

preds_2_adv_fgsm = model_2(adv_x_2_fgsm)
preds_2_adv_pgd = model_2(adv_x_2_pgd)
preds_2_adv_bim = model_2(adv_x_2_bim)


def evaluate_2():
    # Evaluate the accuracy of the adversarialy trained CIFAR10 cifar on
    # legitimate test examples
    eval_params = {'batch_size': 128}
    accuracy = model_eval(sess, x, y, predictions_2, X_test, Y_test, args=eval_params)
    print('Test accuracy on legitimate test examples: ' + str(accuracy))
    report.adv_train_clean_eval = accuracy

    # Evaluate the accuracy of the adversarially trained CIFAR10 cifar on
    # adversarial examples
    accuracy_adv = model_eval(sess, x, y, preds_2_adv_pgd, X_test,
                              Y_test, args=eval_params)
    print('Test accuracy on adversarial examples: ' + str(accuracy_adv))
    report.adv_train_adv_eval = accuracy_adv


# Perform adversarial training
model_train(sess, x, y, predictions_2, X_train, Y_train, save=True,
            predictions_adv=preds_2_adv_pgd, evaluate=evaluate_2,
            args=train_params_adv, rng=rng)

eval_params = {'batch_size': 128}

# Evaluate the accuracy of the CIFAR10 cifar on adversarial examples
accuracy_fgsm = model_eval(sess, x, y, preds_2_adv_fgsm, X_test, Y_test,
                      args=eval_params)
print('Test accuracy on adversarial_fgsm examples: ' + str(accuracy_fgsm))
accuracy_pgd = model_eval(sess, x, y, preds_2_adv_pgd, X_test, Y_test,
                      args=eval_params)
print('Test accuracy on adversarial_pgd examples: ' + str(accuracy_pgd))
accuracy_bim = model_eval(sess, x, y, preds_2_adv_bim, X_test, Y_test,
                      args=eval_params)
print('Test accuracy on adversarial_bim examples: ' + str(accuracy_bim))

