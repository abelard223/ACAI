import  math, os
from    absl import app, flags
import  tensorflow as tf
from    lib import data, layers, train, utils, classifiers


FLAGS = flags.FLAGS

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'



class AEDropout(train.AE):

    def model(self, latent, depth, scales, dropout):

        x = tf.placeholder(tf.float32, [None, self.height, self.width, self.colors], 'x')
        l = tf.placeholder(tf.float32, [None, self.nclass], 'label')
        h = tf.placeholder(tf.float32, [None, self.height >> scales, self.width >> scales, latent], 'h')

        encode = layers.encoder(x, scales, depth, latent, 'ae_encoder')
        # [b, 4, 4, 16] => [b, 16*16]
        encode_train = tf.layers.flatten(encode)
        # [b, 16*16]
        encode_train = tf.nn.dropout(encode_train, dropout)
        # [b, 4, 4, 16]
        encode_train = tf.reshape(encode_train, tf.shape(encode))

        decode = layers.decoder(h, scales, depth, self.colors, 'ae_decoder')
        # no dropout
        ae = layers.decoder(encode, scales, depth, self.colors, 'ae_decoder')
        # with dropout
        ae_train = layers.decoder(encode_train, scales, depth, self.colors, 'ae_decoder')
        loss = tf.losses.mean_squared_error(x, ae_train)

        utils.HookReport.log_tensor(loss, 'loss')
        utils.HookReport.log_tensor(tf.sqrt(loss) * 127.5, 'rmse')

        xops = classifiers.single_layer_classifier(tf.stop_gradient(encode), l, self.nclass)
        xloss = tf.reduce_mean(xops.loss)
        utils.HookReport.log_tensor(xloss, 'classify_latent')

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = tf.train.AdamOptimizer(FLAGS.lr).minimize(loss + xloss, tf.train.get_global_step())

        ops = train.AEOps(x, h, l, encode, decode, ae, train_op, classify_latent=xops.output)

        def gen_images():
            return self.make_sample_grid_and_save(ops)

        recon, inter, slerp, samples = tf.py_func(gen_images, [], [tf.float32]*4)

        tf.summary.image('reconstruction', tf.expand_dims(recon, 0))
        tf.summary.image('interpolation', tf.expand_dims(inter, 0))
        tf.summary.image('slerp', tf.expand_dims(slerp, 0))
        tf.summary.image('samples', tf.expand_dims(samples, 0))

        return ops


def main(argv):
    del argv  # Unused.
    batch = FLAGS.batch
    dataset = data.get_dataset(FLAGS.dataset, dict(batch_size=batch))
    scales = int(round(math.log(dataset.width // FLAGS.latent_width, 2)))
    model = AEDropout(
        dataset,
        FLAGS.train_dir,
        dropout=FLAGS.dropout,
        latent=FLAGS.latent,
        depth=FLAGS.depth,
        scales=scales)
    model.train()


if __name__ == '__main__':
    flags.DEFINE_string('train_dir', './logs', 'Folder where to save training data.')
    flags.DEFINE_float('lr', 0.0001, 'Learning rate.')
    flags.DEFINE_integer('batch', 64, 'Batch size.')
    flags.DEFINE_string('dataset', 'lines32', 'Data to train on.')
    flags.DEFINE_integer('total_kimg', 1 << 14, 'Training duration in samples.')

    flags.DEFINE_integer('depth', 64, 'Depth of first for convolution.')
    flags.DEFINE_integer('latent', 16, 'Latent depth=depth multiplied by latent_width ** 2.')
    flags.DEFINE_integer('latent_width', 4, 'Width of the latent space.')
    flags.DEFINE_float('dropout', 0.5, 'Probability to keep value.')



    app.run(main)
