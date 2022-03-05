# based on https://github.com/wzell/mann/blob/master/models/maximum_mean_discrepancy.py
import tensorflow as tf
from keras import backend as K

def gaussian_kernel(x1, x2, beta=1.0):
    # r = tf.transpose(x1)
    # r = tf.expand_dims(r, 2)
    # r = x1.transpose(0, 'x', 1)
    r=x1
    return tf.reduce_sum(K.exp(-beta * K.square(r - x2)), axis=-1)

def MMD(x1, x2, beta):
    """
    maximum mean discrepancy (MMD) based on Gaussian kernel
    function for keras models (theano or tensorflow backend)

    - Gretton, Arthur, et al. "A kernel method for the two-sample-problem."
    Advances in neural information processing systems. 2007.
    """
    x1x1 = gaussian_kernel(x1, x1, beta)
    x1x2 = gaussian_kernel(x1, x2, beta)
    x2x2 = gaussian_kernel(x2, x2, beta)
    diff = tf.reduce_mean(x1x1) - 2 * tf.reduce_mean(x1x2) + tf.reduce_mean(x2x2)
    return diff

# def MMD(x1, x2, beta):
#     """
#     maximum mean discrepancy (MMD) based on Gaussian kernel
#     function for keras models (theano or tensorflow backend)
#
#     - Gretton, Arthur, et al. "A kernel method for the two-sample-problem."
#     Advances in neural information processing systems. 2007.
#     """
#     x1x1 = gaussian_kernel(x1, x1, beta)
#     x1x2 = gaussian_kernel(x1, x2, beta)
#     x2x2 = gaussian_kernel(x2, x2, beta)
#     diff = x1x1.mean() - 2 * x1x2.mean() + x2x2.mean()
#     return diff
#
# def gaussian_kernel(x1, x2, beta=1.0):
#     # r = x1.dimshuffle(0, 'x', 1)
#     r = tf.transpose(x1)
#     r = tf.expand_dims(r, 2)
#     return K.exp(-beta * K.square(r - x2).sum(axis=-1))