import tensorflow as tf


def conv_net(input_tensor, reuse=False, output_size=2):

    number_outputs = [32, 64, 128, 256, 2]
    kernel_sizes = [7, 5, 3, 2, 2]

    with tf.name_scope("model"):
        net = input_tensor
        init_op = tf.contrib.layers.xavier_initializer_conv2d()

        for layer_i, values in enumerate(zip(number_outputs, kernel_sizes)):

            amount_outputs, kernel_size = values

            with tf.variable_scope("conv{}".format(layer_i)) as scope:
                net = tf.contrib.layers.conv2d(net,
                                               amount_outputs,
                                               [kernel_size, kernel_size],
                                               activation_fn=tf.nn.relu,
                                               padding='SAME',
                                               weights_initializer=init_op,
                                               scope=scope,
                                               reuse=reuse)

                net = tf.contrib.layers.max_pool2d(net, [2, 2], padding='SAME')
        net = tf.contrib.layers.flatten(net)

        with tf.variable_scope('output') as scope:
            net = tf.contrib.layers.fully_connected(net,
                                                    output_size,
                                                    reuse=reuse,
                                                    scope=scope,
                                                    activation_fn=None)
    return net


def contrastive_loss(x1, x2, y, margin):
    with tf.name_scope("contrastive-loss"):
        l2 = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(x1, x2)), 1, keepdims=True))
        tmp = y * tf.square(l2)
        tmp2 = (1 - y) * tf.square(tf.maximum((margin - l2), 0))
        return tf.reduce_mean(tmp + tmp2) / 2
