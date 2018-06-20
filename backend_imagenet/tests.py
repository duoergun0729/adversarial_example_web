import tensorflow as tf


def SSIM(x, y, model, eps=0.3, nb_iter=30, eps_iter=0.01, momentum=1):
    s = 0
    epsilon = tf.random_uniform(tf.shape(x), -eps, eps)
    epsilon = tf.clip_by_value(epsilon, -eps, eps)
    adv_x = x + epsilon
    for i in range(nb_iter):
        preds = model(adv_x)
        loss = tf.nn.softmax_cross_entropy_with_logits(logits=preds, labels=y)
        grad, _ = tf.gradients(loss, adv_x)
        s = s * momentum + grad
        adv_x = adv_x + tf.clip_by_value(eps_iter * tf.sign(s), -eps, eps)
        adv_x = tf.stop_gradient(adv_x)
    return adv_x



