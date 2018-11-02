import tensorflow as tf





a = tf.placeholder(tf.float32, [])
b = tf.placeholder(tf.float32, [])


d = a + b
c = tf.placeholder(tf.float32, [])
d = tf.add_n((d, d))

with tf.Session() as sess:
    res = sess.run(d, feed_dict={
        a: 1,
        b: 2
    })

    print(res)

