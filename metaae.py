import  tensorflow as tf
from    lib import train, utils, classifiers, data
import  math
from    tensorflow import flags

FLAGS = tf.flags.FLAGS



class MetaAE(train.AE):

    def get_weights(self, c, factor, h_c, name):
        """

        :param c: channel of first conv
        :param factor: enlarge channel layer by layer, on factor
        :param h_c: channel of hidden
        :param name: scope name, we set reuse=tf.AUTO_REUSE
        :return:
        """
        # save all variable
        vars = []
        # kernel size
        k = 3
        myinit = tf.contrib.layers.xavier_initializer()

        # print(factor, type(factor))
        # assert factor is 3 will ERROR!!!
        assert factor == 3

        if name == 'encoder':
            with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
                # layer1
                # [b, d, d, img_c] => [b, d, d, c]
                vars.append(tf.get_variable('w1', [1, 1, self.colors, c], dtype=tf.float32, initializer=myinit))
                vars.append(tf.get_variable('b1', [c], dtype=tf.float32, initializer=tf.initializers.zeros()))

                # layer2, factor=0
                vars.append(tf.get_variable('w2', [3, 3, c, c<<0], dtype=tf.float32, initializer=myinit))
                vars.append(tf.get_variable('b2', [c], dtype=tf.float32, initializer=tf.initializers.zeros()))
                vars.append(tf.get_variable('w3', [3, 3, c<<0, c<<0], dtype=tf.float32, initializer=myinit))
                vars.append(tf.get_variable('b3', [c], dtype=tf.float32, initializer=tf.initializers.zeros()))
                # x = tf.nn.avg_pool(x, [1, 2, 2, 1], [1, 2, 2, 1], 'VALID')

                # layer3, factor=1
                vars.append(tf.get_variable('w4', [3, 3, c<<0, c<<1], dtype=tf.float32, initializer=myinit))
                vars.append(tf.get_variable('b4', [c<<1], dtype=tf.float32, initializer=tf.initializers.zeros()))
                vars.append(tf.get_variable('w5', [3, 3, c<<1, c<<1], dtype=tf.float32, initializer=myinit))
                vars.append(tf.get_variable('b5', [c<<1], dtype=tf.float32, initializer=tf.initializers.zeros()))
                # x = tf.nn.avg_pool(x, [1, 2, 2, 1], [1, 2, 2, 1], 'VALID')


                # layer4, factor=2
                vars.append(tf.get_variable('w6', [3, 3, c<<1, c<<2], dtype=tf.float32, initializer=myinit))
                vars.append(tf.get_variable('b6', [c<<2], dtype=tf.float32, initializer=tf.initializers.zeros()))
                vars.append(tf.get_variable('w7', [3, 3, c<<2, c<<2], dtype=tf.float32, initializer=myinit))
                vars.append(tf.get_variable('b7', [c<<2], dtype=tf.float32, initializer=tf.initializers.zeros()))
                # x = tf.nn.avg_pool(x, [1, 2, 2, 1], [1, 2, 2, 1], 'VALID')

                # layer5
                vars.append(tf.get_variable('w8', [3, 3, c<<2, c<<3], dtype=tf.float32, initializer=myinit))
                vars.append(tf.get_variable('b8', [c<<3], dtype=tf.float32, initializer=tf.initializers.zeros()))

                # layer6
                vars.append(tf.get_variable('w9', [3, 3, c<<3, h_c], dtype=tf.float32, initializer=myinit))
                vars.append(tf.get_variable('b9', [h_c], dtype=tf.float32, initializer=tf.initializers.zeros()))

        elif name is 'decoder':
            with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
                # layer1
                vars.append(tf.get_variable('w1', [3, 3, h_c, c << 2], dtype=tf.float32, initializer=myinit))
                vars.append(tf.get_variable('b1', [c << 2], dtype=tf.float32, initializer=tf.initializers.zeros()))
                vars.append(tf.get_variable('w2', [3, 3, c << 2, c << 2], dtype=tf.float32, initializer=myinit))
                vars.append(tf.get_variable('b2', [c << 2], dtype=tf.float32, initializer=tf.initializers.zeros()))
                # tf.batch_to_space(tf.tile(x, [n ** 2, 1, 1, 1]), [[0, 0], [0, 0]], n)

                # layer2
                vars.append(tf.get_variable('w3', [3, 3, c << 2, c << 1], dtype=tf.float32, initializer=myinit))
                vars.append(tf.get_variable('b3', [c << 1], dtype=tf.float32, initializer=tf.initializers.zeros()))
                vars.append(tf.get_variable('w4', [3, 3, c << 1, c << 1], dtype=tf.float32, initializer=myinit))
                vars.append(tf.get_variable('b4', [c << 1], dtype=tf.float32, initializer=tf.initializers.zeros()))
                # tf.batch_to_space(tf.tile(x, [n ** 2, 1, 1, 1]), [[0, 0], [0, 0]], n)

                # layer3
                vars.append(tf.get_variable('w5', [3, 3, c << 1, c << 0], dtype=tf.float32, initializer=myinit))
                vars.append(tf.get_variable('b5', [c << 0], dtype=tf.float32, initializer=tf.initializers.zeros()))
                vars.append(tf.get_variable('w6', [3, 3, c << 0, c << 0], dtype=tf.float32, initializer=myinit))
                vars.append(tf.get_variable('b6', [c << 0], dtype=tf.float32, initializer=tf.initializers.zeros()))
                # tf.batch_to_space(tf.tile(x, [n ** 2, 1, 1, 1]), [[0, 0], [0, 0]], n)

                # layer4
                vars.append(tf.get_variable('w7', [3, 3, c, c], dtype=tf.float32, initializer=myinit))
                vars.append(tf.get_variable('b7', [c], dtype=tf.float32, initializer=tf.initializers.zeros()))

                # layer5
                vars.append(tf.get_variable('w8', [3, 3, c, self.colors], dtype=tf.float32, initializer=myinit))
                vars.append(tf.get_variable('b8', [self.colors], dtype=tf.float32, initializer=tf.initializers.zeros()))
        else:
            raise NotImplementedError

        return vars

    def forward_encoder(self, x, vars):
        """

        :param x:
        :return:
        """
        idx = 0

        # layer1
        op = tf.nn.conv2d(x, vars[idx + 0], strides=(1,1,1,1), padding='SAME')
        op = tf.nn.bias_add(op, vars[idx + 1])
        idx += 2

        # layer2/3/4, factor=0,1,2
        for idx in range(2, 14, 4): # step=4
            op = tf.nn.conv2d(op, vars[idx], strides=(1,1,1,1), padding='SAME')
            op = tf.nn.bias_add(op, vars[idx + 1])
            op = tf.nn.leaky_relu(op)


            op = tf.nn.conv2d(op, vars[idx + 2], strides=(1,1,1,1), padding='SAME')
            op = tf.nn.bias_add(op, vars[idx + 3])
            op = tf.nn.leaky_relu(op)

            op = tf.nn.avg_pool(op, [1, 2, 2, 1], [1, 2, 2, 1], 'VALID')

        # layer5
        idx = 14
        op = tf.nn.conv2d(op, vars[idx], strides=(1,1,1,1), padding='SAME')
        op = tf.nn.bias_add(op, vars[idx + 1])
        op = tf.nn.leaky_relu(op)
        idx += 2

        # layer6
        op = tf.nn.conv2d(op, vars[idx], strides=(1,1,1,1), padding='SAME')
        op = tf.nn.bias_add(op, vars[idx + 1])
        idx += 2

        # print(idx, len(vars))
        assert idx == len(vars)

        return op


    def forward_decoder(self, h, vars):
        """

        :param x:
        :return:
        """
        idx = 0

        op = h
        # layer1/2/3, factor=2,1,0
        for idx in range(0, 12, 4): # step=4
            op = tf.nn.conv2d(op, vars[idx], strides=(1,1,1,1), padding='SAME')
            # print(vars[idx].name, vars[idx+1].name)
            op = tf.nn.bias_add(op, vars[idx + 1])
            op = tf.nn.leaky_relu(op)


            op = tf.nn.conv2d(op, vars[idx + 2], strides=(1,1,1,1), padding='SAME')
            op = tf.nn.bias_add(op, vars[idx + 3])
            op = tf.nn.leaky_relu(op)

            op = tf.batch_to_space(tf.tile(op, [2 ** 2, 1, 1, 1]), [[0, 0], [0, 0]], 2)

        idx = 12
        # layer4
        op = tf.nn.conv2d(op, vars[idx + 0], strides=(1,1,1,1), padding='SAME')
        op = tf.nn.bias_add(op, vars[idx + 1])
        op = tf.nn.leaky_relu(op)
        idx += 2
        # layer5
        op = tf.nn.conv2d(op, vars[idx + 0], strides=(1,1,1,1), padding='SAME')
        op = tf.nn.bias_add(op, vars[idx + 1])
        idx += 2

        assert idx == len(vars)

        return op

    def forward_ae(self, x, vars):
        """

        :param x:
        :return:
        """
        assert len(vars) is (18+16)

        vars_encoder = vars[:18]
        vars_decoder = vars[18:]

        op = self.forward_encoder(x, vars_encoder)
        op = self.forward_decoder(op, vars_decoder)

        return op

    def model(self, latent, depth, scales):
        """

        :param latent: latent channel
        :param depth: basic channel number
        :param scale: channel factor
        :return:
        """
        # get hidden features maps dim/height/width and channel number
        h_d, h_c = self.height>>scales, latent

        print('h_d:', h_d, 'h_c:', h_c, 'ch:', depth, 'scales:', scales, 'batchsz:', FLAGS.batch)

        # [b, 32, 32, 1]
        x = tf.placeholder(tf.float32, [None, self.height, self.width, self.colors], name='x')
        # [b, 10]
        l = tf.placeholder(tf.float32, [None, self.nclass], name='label')
        # [b, 4, 4, 16]
        h = tf.placeholder(tf.float32, [None, h_d, h_d, h_c], name='h')

        # meta batch size
        task_num = 8
        update_num = 5
        update_lr = 0.05
        meta_lr = 1e-3
        # 2 of [b/2, 32, 32, 1]
        x_tasks = tf.split(x, num_or_size_splits=2, axis=0)
        # => 2 of task_num of [b/2/task_num, 32, 1]
        x_tasks = list(map(lambda x: tf.split(x, num_or_size_splits=task_num, axis=0), x_tasks))

        # merge 2 variables list into a list
        # this is 1st time to get these variables, so it will create and return
        vars = self.get_weights(depth, scales, latent, 'encoder') + self.get_weights(depth, scales, latent, 'decoder')

        def task_metalearn(task_input):
            """
            create single task op, we need call this func multiple times to create a bunches of ops
            for several tasks
            NOTICE: this function will use outer `vars`.
            """
            x_spt, x_qry = task_input
            preds_qry, losses_qry, accs_qry = [], [], []

            pred_spt = self.forward_ae(x_spt, vars)
            loss_spt = tf.losses.mean_squared_error(labels=x_spt, predictions=pred_spt)

            grads = tf.gradients(loss_spt, vars)
            # if FLAGS.stop_grad:
            #     grads = [tf.stop_gradient(grad) for grad in grads]
            fast_weights = list(map(lambda x:x[0] - update_lr * x[1], zip(vars, grads)))
            pred_qry = self.forward_ae(x_qry, fast_weights)
            preds_qry.append(pred_qry)
            loss_qry = tf.losses.mean_squared_error(labels=x_qry, predictions=pred_qry)
            losses_qry.append(loss_qry)

            for _ in range(update_num - 1):
                pred_spt = self.forward_ae(x_spt, fast_weights)
                loss_spt = tf.losses.mean_squared_error(labels=x_spt, predictions=pred_spt)
                grads = tf.gradients(loss_spt, fast_weights)
                # if FLAGS.stop_grad:
                #     grads = [tf.stop_gradient(grad) for grad in grads]
                fast_weights = list(map(lambda x: x[0] - update_lr * x[1], zip(fast_weights, grads)))
                pred_qry = self.forward_ae(x_qry, fast_weights)
                preds_qry.append(pred_qry)
                loss_qry = tf.losses.mean_squared_error(labels=x_qry, predictions=pred_qry)
                losses_qry.append(loss_qry)


            task_output = [pred_spt, preds_qry, loss_spt, losses_qry]

            # task_accuracya = tf.contrib.metrics.accuracy(tf.argmax(tf.nn.softmax(task_outputa), 1), tf.argmax(labela, 1))
            # for j in range(num_updates):
            #     task_accuraciesb.append(
            #         tf.contrib.metrics.accuracy(tf.argmax(tf.nn.softmax(task_outputbs[j]), 1), tf.argmax(labelb, 1)))
            # task_output.extend([task_accuracya, task_accuraciesb])

            return task_output


        out_dtype = [tf.float32, [tf.float32] * update_num, tf.float32, [tf.float32] * update_num]
        # out_dtype.extend([tf.float32, [tf.float32]*update_num])
        pred_spt, preds_qry, loss_spt, losses_qry = \
            tf.map_fn(task_metalearn, elems=x_tasks, dtype=out_dtype, parallel_iterations=task_num)


        self.loss_spt = tf.reduce_sum(loss_spt) / tf.to_float(task_num)
        self.losses_qry = [tf.reduce_sum(losses_qry[j]) / tf.to_float(task_num) for j in range(update_num)]
        self.pred_spt, self.preds_qry = pred_spt, preds_qry
        del pred_spt, preds_qry, loss_spt, losses_qry
        # self.total_accuracy1 = tf.reduce_sum(accuraciesa) / tf.to_float(FLAGS.meta_batch_size)
        # self.total_accuracies2 = [tf.reduce_sum(accuraciesb[j]) / tf.to_float(FLAGS.meta_batch_size) for j in range(num_updates)]
        # self.pretrain_op = tf.train.AdamOptimizer(self.meta_lr).minimize(loss_spt)

        optimizer = tf.train.AdamOptimizer(meta_lr)
        gvs = optimizer.compute_gradients(self.losses_qry[update_num-1])
        # gvs = [(tf.clip_by_value(grad, -10, 10), var) for grad, var in gvs]
        meta_op = optimizer.apply_gradients(gvs, global_step=tf.train.get_or_create_global_step())


        # utils.HookReport.log_tensor(self.loss_spt, 'loss')
        # # utils.HookReport.log_tensor(tf.sqrt(loss_spt) * 127.5, 'rmse')


        for i in range(update_num):
            # print(losses_qry[i])
            utils.HookReport.log_tensor(self.losses_qry[i], 'loss_qry%d'%i)
            utils.HookReport.log_tensor(tf.sqrt(self.losses_qry[i]) * 127.5, 'rmse%d'%i)

        # we only use encode to acquire representation and wont use classification to backprop encoder
        # hence we will stop_gradient(encoder)
        encoder_op = self.forward_encoder(x, vars[:18])
        decoder_op = self.forward_decoder(h, vars[18:])
        ae_op = self.forward_ae(x, vars)
        xops = classifiers.single_layer_classifier(tf.stop_gradient(encoder_op), l, self.nclass)
        xloss = tf.reduce_mean(xops.loss)
        # record classification loss on latent
        utils.HookReport.log_tensor(xloss, 'classify_latent')

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = tf.group([meta_op])

        ops = train.AEOps(x, h, l, encoder_op, decoder_op, ae_op, train_op, classify_latent=xops.output)

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
    model = MetaAE(
        dataset,
        FLAGS.train_dir,
        latent=FLAGS.latent, # channels of latent
        depth=FLAGS.depth, # channels of first convolution
        scales=scales)
    model.train()


if __name__ == '__main__':
    import  os

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    tf.logging.set_verbosity(tf.logging.INFO)

    flags.DEFINE_string('train_dir', './logs','Folder where to save training data.')
    flags.DEFINE_float('lr', 0.001, 'Learning rate.')
    flags.DEFINE_integer('batch', 64, 'Batch size.')
    flags.DEFINE_string('dataset', 'mnist32', 'Data to train on.')
    flags.DEFINE_integer('total_kimg', 1 << 14, 'Training duration in samples.')
    flags.DEFINE_integer('depth', 16, 'Depth of first for convolution.')
    flags.DEFINE_integer('latent', 8, 'Latent depth = depth multiplied by latent_width ** 2.')
    flags.DEFINE_integer('latent_width', 4, 'Width of the latent space.')
    tf.app.run(main)