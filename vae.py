import  math, os
from    absl import app
from    absl import flags
import  tensorflow as tf
from    lib import data, layers, train, utils, classifiers



FLAGS = flags.FLAGS

class VAE(train.AE):

    def model(self, latent, depth, scales, beta):
        """

        :param latent: hidden/latent channel number
        :param depth: channel number for factor
        :param scales: factor
        :param beta: beta for KL divergence
        :return:
        """
        # x is rescaled to [-1, 1] in data argumentation phase
        x = tf.placeholder(tf.float32, [None, self.height, self.width, self.colors], 'x')
        l = tf.placeholder(tf.float32, [None, self.nclass], 'label')
        # [32>>3, 32>>3, latent_depth]
        h = tf.placeholder(tf.float32, [None, self.height >> scales, self.width >> scales, latent], 'h')

        def encoder(x):
            return layers.encoder(x, scales, depth, latent, 'vae_enc')

        def decoder(h):
            return layers.decoder(h, scales, depth, self.colors, 'vae_dec')

        # [b, 4, 4, 16]
        encode = encoder(x)

        with tf.variable_scope('vae_u_std'):
            encode_shape = tf.shape(encode)
            # [b, 16*16]
            encode_flat = tf.layers.flatten(encode)
            # not run-time shape, 16*16
            latent_dim = encode_flat.get_shape()[-1]
            # dense:[16*16, 16*16]
            # mean
            q_mu = tf.layers.dense(encode_flat, latent_dim)
            # dense: [16*16, 16*16]
            log_q_sigma_sq = tf.layers.dense(encode_flat, latent_dim)

        # [b, 16*16], log square
        # variance
        # => [b, 4*4*16]
        q_sigma = tf.sqrt(tf.exp(log_q_sigma_sq))

        # N(u, std^2)
        q_z = tf.distributions.Normal(loc=q_mu, scale=q_sigma)
        q_z_sample = q_z.sample()
        # [b, 4*4*16] => [b, 4, 4, 16]
        q_z_sample_reshaped = tf.reshape(q_z_sample, encode_shape)
        # [b, 32, 32, 1]
        p_x_given_z_logits = decoder(q_z_sample_reshaped)
        # [b, 32, 32, 1]
        p_x_given_z = tf.distributions.Bernoulli(logits=p_x_given_z_logits)

        # for VAE, h stands for sampled value with Guassian(u, std^2)
        # -1~1
        ae = 2*tf.nn.sigmoid(p_x_given_z_logits) - 1
        decode = 2*tf.nn.sigmoid(decoder(h)) - 1

        # compute kl divergence
        # there is a closed form of KL between two Guassian distributions
        # please refer to here:
        # https://stats.stackexchange.com/questions/7440/kl-divergence-between-two-univariate-gaussians
        loss_kl = 0.5*tf.reduce_sum(-log_q_sigma_sq - 1 + tf.exp(log_q_sigma_sq) + q_mu**2)
        loss_kl = loss_kl/tf.to_float(tf.shape(x)[0])

        # rescale to [0, 1], convenient for Bernoulli distribution
        x_bernoulli = 0.5*(x + 1)
        # can use reconstruction or use density estimation
        loss_ll = tf.reduce_sum(p_x_given_z.log_prob(x_bernoulli))
        loss_ll = loss_ll/tf.to_float(tf.shape(x)[0])

        #
        elbo = loss_ll - beta*loss_kl

        utils.HookReport.log_tensor(loss_kl, 'kl_divergence')
        utils.HookReport.log_tensor(loss_ll, 'log_likelihood')
        utils.HookReport.log_tensor(elbo, 'elbo')

        xops = classifiers.single_layer_classifier(tf.stop_gradient(encode), l, self.nclass, scope='classifier')
        xloss = tf.reduce_mean(xops.loss)
        utils.HookReport.log_tensor(xloss, 'classify_loss_on_h')

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        ae_vars = tf.global_variables('vae_enc') + tf.global_variables('vae_dec') + tf.global_variables('vae_u_std')
        xl_vars = tf.global_variables('classifier')
        with tf.control_dependencies(update_ops):
            train_ae = tf.train.AdamOptimizer(FLAGS.lr).minimize(- elbo, var_list=ae_vars)
            train_xl = tf.train.AdamOptimizer(FLAGS.lr).minimize(xloss, tf.train.get_global_step(), var_list=xl_vars)

        ops = train.AEOps(x, h, l, q_z_sample_reshaped, decode, ae, tf.group(train_ae, train_xl),
                          classify_latent=xops.output)

        n_interpolations = 16
        n_images_per_interpolation = 16

        def gen_images():
            return self.make_sample_grid_and_save( ops, interpolation=n_interpolations,
                height=n_images_per_interpolation)

        recon, inter, slerp, samples = tf.py_func( gen_images, [], [tf.float32]*4)
        tf.summary.image('reconstruction', tf.expand_dims(recon, 0))
        tf.summary.image('interpolation', tf.expand_dims(inter, 0))
        tf.summary.image('slerp', tf.expand_dims(slerp, 0))
        tf.summary.image('samples', tf.expand_dims(samples, 0))

        return ops


def main(argv):

    batch = FLAGS.batch
    dataset = data.get_dataset(FLAGS.dataset, dict(batch_size=batch))
    scales = int(round(math.log(dataset.width // FLAGS.latent_width, 2)))
    model = VAE(
        dataset,
        FLAGS.train_dir,
        latent=FLAGS.latent,
        depth=FLAGS.depth,
        scales=scales,
        beta=FLAGS.beta)
    model.train()


if __name__ == '__main__':
    flags.DEFINE_string('train_dir', './logs', 'Folder where to save training data.')
    flags.DEFINE_float('lr', 0.0001, 'Learning rate.')
    flags.DEFINE_integer('batch', 64, 'Batch size.')
    flags.DEFINE_string('dataset', 'mnist32', 'Data to train on.')
    flags.DEFINE_integer('total_kimg', 1 << 14, 'Training duration in samples.')

    flags.DEFINE_integer('depth', 64, 'Depth of first for convolution.')
    flags.DEFINE_integer('latent', 16, 'Latent depth=depth multiplied by latent_width ** 2.')
    flags.DEFINE_integer('latent_width', 4, 'Width of the latent space.')
    flags.DEFINE_float('beta', 1.0, 'ELBO KL term scale.')

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    tf.logging.set_verbosity(tf.logging.INFO)
    app.run(main)
