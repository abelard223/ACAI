#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

from absl import app
from absl import flags

import tensorflow as tf
from lib import data, layers, train, utils, classifiers, eval

FLAGS = flags.FLAGS


class AEBaseline(train.AE):

    def model(self, latent, depth, scales):
        """

        :param latent:
        :param depth:
        :param scales:
        :return:
        """
        print('self.nclass', self.nclass)
        # [b, 32, 32, 1]
        x = tf.placeholder(tf.float32, [None, self.height, self.width, self.colors], 'x')
        # [b, 10]
        l = tf.placeholder(tf.float32, [None, self.nclass], 'label')
        # [?, 4, 4, 16]
        h = tf.placeholder(tf.float32, [None, self.height >> scales, self.width >> scales, latent], 'h')
        # [b, 32, 32, 1] => [b, 4, 4, 16]
        encode = layers.encoder(x, scales, depth, latent, 'ae_encoder')
        # [b, 4, 4, 16] => [b, 32, 32, 1]
        decode = layers.decoder(h, scales, depth, self.colors, 'ae_decoder')
        # [b, 4, 4, 16] => [b, 32, 32, 1], auto-reuse
        ae = layers.decoder(encode, scales, depth, self.colors, 'ae_decoder')
        #
        loss = tf.losses.mean_squared_error(x, ae)

        utils.HookReport.log_tensor(loss, 'loss')
        utils.HookReport.log_tensor(tf.sqrt(loss) * 127.5, 'rmse')

        # we only use encode to acquire representation and wont use classification to backprop encoder
        # hence we will stop_gradient(encoder)
        xops = classifiers.single_layer_classifier(tf.stop_gradient(encode), l, self.nclass)
        xloss = tf.reduce_mean(xops.loss)
        # record classification loss on latent
        utils.HookReport.log_tensor(xloss, 'classify_latent')

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = tf.train.AdamOptimizer(FLAGS.lr).minimize(loss + xloss, tf.train.get_global_step())

        ops = train.AEOps(x, h, l, encode, decode, ae, train_op, classify_latent=xops.output)

        n_interpolations = 16
        n_images_per_interpolation = 16

        def gen_images():
            return self.make_sample_grid_and_save(ops, interpolation=n_interpolations, height=n_images_per_interpolation)

        recon, inter, slerp, samples = tf.py_func(gen_images, [], [tf.float32]*4)
        tf.summary.image('reconstruction', tf.expand_dims(recon, 0))
        tf.summary.image('interpolation', tf.expand_dims(inter, 0))
        tf.summary.image('slerp', tf.expand_dims(slerp, 0))
        tf.summary.image('samples', tf.expand_dims(samples, 0))

        return ops


def main(argv):
    print(FLAGS.flag_values_dict())


    batch = FLAGS.batch
    dataset = data.get_dataset(FLAGS.dataset, dict(batch_size=batch))
    # latent: [?, 4, 4, 16]
    scales = int(round(math.log(dataset.width // FLAGS.latent_width, 2)))
    model = AEBaseline(
        dataset,
        FLAGS.train_dir,
        latent=FLAGS.latent, # channels of latent
        depth=FLAGS.depth, # channels of first convolution
        scales=scales)
    model.train()


if __name__ == '__main__':
    import  os

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    flags.DEFINE_string('train_dir', './logs','Folder where to save training data.')
    flags.DEFINE_float('lr', 0.0001, 'Learning rate.')
    flags.DEFINE_integer('batch', 64, 'Batch size.')
    flags.DEFINE_string('dataset', 'mnist32', 'Data to train on.')
    flags.DEFINE_integer('total_kimg', 1 << 14, 'Training duration in samples.')
    flags.DEFINE_integer('depth', 64, 'Depth of first for convolution.')
    flags.DEFINE_integer('latent', 16, 'Latent depth = depth multiplied by latent_width ** 2.')
    flags.DEFINE_integer('latent_width', 4, 'Width of the latent space.')
    app.run(main)
