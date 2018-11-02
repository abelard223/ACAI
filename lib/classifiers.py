import tensorflow as tf


class ClassifierOps(object):
    def __init__(self, loss, output, extra_loss=None):
        """

        :param loss: loss ops of classifier
        :param output: predict result op of classifier
        :param extra_loss:
        """
        self.loss = loss
        self.output = output
        self.extra_loss = extra_loss
        # Inputs, train op (used for data augmentation).
        self.x = None
        self.label = None
        self.train_op = None


def single_layer_classifier(h, l, nclass, scope='classifier', reuse=False, smoothing=None):
    """

    :param h: hidden placeholder
    :param l: label placeholder
    :param nclass: number of class
    :param scope: scope name for this module
    :param reuse:
    :param smoothing:
    :return:
    """
    # Here can reuse=True or reuse=Flase
    with tf.variable_scope(scope, reuse=reuse):
        # [b, 4, 4, 16] => [b, -1]
        h0 = tf.layers.flatten(h)
        # => [b, 10]
        logits = tf.layers.dense(h0, nclass)
        # => [b]
        output = tf.argmax(logits, 1)
        if smoothing:
            l -= abs(smoothing) * (l - 1. / nclass)
        loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=l)

    return ClassifierOps(loss=loss, output=output)
