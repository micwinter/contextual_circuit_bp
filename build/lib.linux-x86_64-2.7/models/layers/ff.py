import numpy as np
import tensorflow as tf
from models.layers.activations import activations
from models.layers.normalizations import normalizations


def resnet_layer(
        bottom,
        layer_weights,
        name,
        normalization=None):
    ln = '%s_branch' % layer_name
    in_layer = self.conv_layer(
            bottom,
            int(bottom.get_shape()[-1]),
            layer_weights[0],
            batchnorm=[ln],
            name=ln)
    rlayer = tf.identity(bottom)
    if normalization is not None:
        nm = normalizations()[normalization]
    if activation is not None:
        ac = activations()[activation]
    for idx, lw in enumerate(layer_weights):
        ln = '%s_%s' % (name, idx)
        rlayer = self.conv_layer(
            rlayer,
            int(in_layer.get_shape()[-1]),
            lw,
            name=ln)
        rlayer = nm(ac(rlayer))
    return rlayer + bottom


def conv_layer(
        bottom,
        out_channels,
        name,
        in_channels=None,
        filter_size=3,
        stride=[1, 1, 1, 1],
        padding='SAME'):
    with tf.variable_scope(name):
        if in_channels is None:
            in_channels = int(bottom.get_shape()[-1])
        filt, conv_biases = get_conv_var(
            filter_size=filter_size,
            in_channels=in_channels,
            out_channels=out_channels,
            name=name)
        conv = tf.nn.conv2d(bottom, filt, stride, padding=padding)
        bias = tf.nn.bias_add(conv, conv_biases)
        return relu

def fc_layer(bottom, out_channels, name, in_channels=None):
    with tf.variable_scope(name):
        if in_channels is None:
            in_channels = int(bottom.get_shape()[-1])
        weights, biases = self.get_fc_var(in_channels, out_channels, name)

        x = tf.reshape(bottom, [-1, in_channels])
        fc = tf.nn.bias_add(tf.matmul(x, weights), biases)

        return fc

def get_conv_var(
        filter_size,
        in_channels,
        out_channels,
        name,
        init_type='xavier'):
    if init_type == 'xavier':
        weight_init = [
            [filter_size, filter_size, in_channels, out_channels],
            tf.contrib.layers.xavier_initializer_conv2d(uniform=False)]
    else:
        weight_init = tf.truncated_normal(
            [filter_size, filter_size, in_channels, out_channels],
            0.0, 0.001)
    bias_init = tf.truncated_normal([out_channels], .0, .001)
    filters = self.get_var(weight_init, name, 0, name + "_filters")
    biases = self.get_var(bias_init, name, 1, name + "_biases")

    return filters, biases

def get_fc_var(
        in_size,
        out_size,
        name,
        init_type='xavier'):
    if init_type == 'xavier':
        weight_init = [
            [in_size, out_size],
            tf.contrib.layers.xavier_initializer(uniform=False)]
    else:
        weight_init = tf.truncated_normal(
            [in_size, out_size], 0.0, 0.001)
    bias_init = tf.truncated_normal([out_size], .0, .001)
    weights = get_var(weight_init, name, 0, name + "_weights")
    biases = get_var(bias_init, name, 1, name + "_biases")

    return weights, biases

def get_var(
        self,
        initial_value,
        name,
        idx,
        var_name,
        in_size=None,
        out_size=None):
    if self.data_dict is not None and name in self.data_dict:
        value = self.data_dict[name][idx]
    else:
        value = initial_value

    if self.trainable:
        # get_variable, change the boolian to numpy
        if type(value) is list:
            var = tf.get_variable(
                name=var_name, shape=value[0], initializer=value[1])
        else:
            var = tf.get_variable(name=var_name, initializer=value)
    else:
        var = tf.constant(value, dtype=tf.float32, name=var_name)
    self.var_dict[(name, idx)] = var
    return var
