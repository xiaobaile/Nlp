import tensorflow as tf


def soft_plus(x, name="soft_plus"):
    with tf.variable_scope(name):
        return tf.nn.softplus


def swish(x, name="swish"):
    with tf.variable_scope(name):
        return tf.nn.sigmoid(x * 1.0) * x


def leaky_relu(x, leak=0.2, name="leaky_relu"):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
    return f1 * x + f2 * tf.abs(x)


def cube(x, name="cube_act"):
    with tf.variable_scope(name):
        return tf.pow(x, 3)


def penalized_tanh(x, name="penalized_tanh"):
    with tf.variable_scope(name):
        alpha = 0.25
        return tf.maximum(tf.tanh(x), alpha * tf.tanh(x))


def cosper(x, name="cosper_act"):
    with tf.variable_scope(name):
        return tf.cos(x) - x


def minsin(x, name="minsin_act"):
    with tf.variable_scope(name):
        return tf.minimum(x, tf.sin(x))


def tanhrev(x, name="tanhprev"):
    with tf.variable_scope(name):
        return tf.pow(tf.atan(x), 2) - x


def maxsig(x, name="maxsig_act"):
    with tf.variable_scope(name):
        return tf.maximum(x, tf.sigmoid(x))


def maxtanh(x, name="maxtanh_act"):
    with tf.variable_scope(name):
        return tf.maximum(x, tf.tanh(x))


def get_activation(active_type="swish", **kwargs):
    mp = {
        "sigmoid": tf.nn.sigmoid,
        "tanh": tf.nn.tanh,
        "softsign": tf.nn.softsign,
        "relu": tf.nn.relu,
        "leaky_relu": leaky_relu,
        "elu": tf.nn.elu,
        "selu": tf.nn.selu,
        "swish": swish,
        "sin": tf.sin,
        "cube": cube,
        "penalized_tanh": penalized_tanh,
        "cosper": cosper,
        "minsin": minsin,
        "tanhrev": tanhrev,
        "maxsig": maxsig,
        "maxtanh": maxtanh,
        "softplus": tf.nn.softplus,
    }

    assert active_type in mp, "%s is not in activation list"
    return mp[active_type]
