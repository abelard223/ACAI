import tensorflow as tf





def Dense(x, x_dim, y_dim, name, reuse=None):

    with tf.variable_scope(name, reuse=reuse):
        w = tf.get_variable('weight', [x_dim, y_dim])
        b = tf.get_variable('bias', [y_dim])
        y = tf.add(tf.matmul(x, w), b)
    return y


def Encoder(x, name):

    with tf.variable_scope(name, reuse=None):
        x = tf.nn.relu(Dense(x, 784, 1000, 'layer1', reuse=False))
        x = tf.nn.relu(Dense(x, 1000, 1000, 'layer2', reuse=False))
        x = Dense(x, 1000, 10, 'layer3', reuse=False)
    return x

def Decoder(x, name, reuse=None):

    with tf.variable_scope(name, reuse=reuse):
        x = tf.nn.relu(Dense(x, 10, 1000, 'layer1', reuse=False))
        x = tf.nn.relu(Dense(x, 1000, 1000, 'layer2', reuse=False))
        x = tf.nn.sigmoid(Dense(x, 1000, 784, 'layer3', reuse=False))
    return x

def build_network(x):
    batchsz = 32
    x_ph = tf.placeholder(tf.float32, [batchsz, 784], name='input')
    z_ph = tf.placeholder(tf.float32, [1, 10], name='z')

    x = Encoder(x_ph, 'Encoder')
    x_hat = Decoder(x, 'Decoder1', reuse=None)
    x_hat2 = Decoder(z_ph, 'Decoder2', reuse=True)

    # ...


