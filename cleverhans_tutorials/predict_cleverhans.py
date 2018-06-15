import numpy as np
import tensorflow as tf
from tensorflow.python.platform import flags
import logging

# from digits_recognition.utils_mnist import data_mnist
from cleverhans_tutorials.tutorial_models import make_basic_cnn
from cleverhans.utils_mnist import data_mnist
from cleverhans import utils_tf, utils
from cleverhans.utils_tf import model_train, model_eval, tf_model_load


#加载数据
# print('\nLoading MNIST')
# mnist = tf.keras.datasets.mnist
# (X_train, y_train), (X_test, y_test) = mnist.load_data()

# Get MNIST test data
train_start = 0
train_end = 60000
test_start = 0
test_end = 10000
X_train, y_train, X_test, y_test = data_mnist(train_start=train_start,
                                              train_end=train_end,
                                              test_start=test_start,
                                              test_end=test_end)

# Use label smoothing
# assert y_train.shape[1] == 10
label_smooth = .1
y_train = y_train.clip(label_smooth / 9., 1. - label_smooth)

img_size = 28
img_chan = 1
n_classes = 10
# Define input TF placeholder
x = tf.placeholder(tf.float32, (None, img_size, img_size, img_chan),
                           name='x')
y = tf.placeholder(tf.float32, (None, n_classes), name='y')
training = tf.placeholder_with_default(False, (), name='mode')
x_adv = tf.placeholder(tf.float32, (None, img_size, img_size, img_chan),
                           name='x_adv')

model = make_basic_cnn()
preds = model.get_probs(x)

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
sess.run(tf.local_variables_initializer())

# Object used to keep track of (and return) key accuracies
    #用于跟踪（和返回）键精度的对象
report = utils.AccuracyReport()

model_path = "models/mnist"
    # Train an MNIST cifar

rng = np.random.RandomState([2017, 8, 30])


def evaluate(batch_size=128, ):  # 如果属实，请完成单元测试的准确性报告以验证性能是否足够
    # Evaluate the accuracy of the MNIST cifar on legitimate test
    # examples  评估MNIST模型在合法测试样本上的准确性
    eval_params = {'batch_size': batch_size}
    acc = utils_tf.model_eval(  # X_test: numpy array with training inputs； Y_test: numpy array with training outputs
        sess, x, y, preds, X_test, y_test, args=eval_params)
    report.clean_train_clean_eval = acc
    assert X_test.shape[0] == test_end - test_start, X_test.shape
    print('Test accuracy on legitimate examples: %0.4f' % acc)


train_params = {
        'nb_epochs': 6,
        'batch_size': 128,
        'learning_rate': 0.001
}


def clean_train(sess, load=True, name='cifar'):
    if load:
        print('\nLoading saved cifar')
        tf_model_load(sess, 'cifar/{}'.format(name))
    else:
        model_train(sess, x, y, preds, X_train, y_train, evaluate=evaluate,
                args=train_params, rng=rng)


clean_train(sess, load=True, name='clean_model')


def predict(sess, X_data, batch_size=128):

    print('\nPredicting')
    n_classes = preds.get_shape().as_list()[1]
    n_sample = X_data.shape[0]
    n_batch = int((n_sample+batch_size-1) / batch_size)
    yval = np.empty((n_sample, n_classes))

    for batch in range(n_batch):
        print(' batch {0}/{1}'.format(batch + 1, n_batch), end='\r')
        start = batch * batch_size
        end = min(n_sample, start + batch_size)
        y_batch = sess.run(preds, feed_dict={x: X_data[start:end]})
        yval[start:end] = y_batch
    print()
    return yval


result = predict(sess, X_train[0:2])
print(result)







