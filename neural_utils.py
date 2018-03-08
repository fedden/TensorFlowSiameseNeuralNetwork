import tensorflow as tf


def conv_net(input_tensor, reuse=False, output_size=2):

    number_outputs = [32, 64, 128, 256, 2]
    kernel_sizes = [7, 5, 3, 1, 1]

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


################################################################################
# !TODO: maybe try changing the network back to this definition and check that
#        graphing capacity hasn't been lost.
def old_network(input, reuse=False):
    with tf.name_scope("model"):
        with tf.variable_scope("conv1") as scope:
            net = tf.contrib.layers.conv2d(input, 32, [7, 7], activation_fn=tf.nn.relu, padding='SAME',
                                           weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(), scope=scope, reuse=reuse)
            net = tf.contrib.layers.max_pool2d(net, [2, 2], padding='SAME')

        with tf.variable_scope("conv2") as scope:
            net = tf.contrib.layers.conv2d(net, 64, [5, 5], activation_fn=tf.nn.relu, padding='SAME',
                                           weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(), scope=scope, reuse=reuse)
            net = tf.contrib.layers.max_pool2d(net, [2, 2], padding='SAME')

        with tf.variable_scope("conv3") as scope:
            net = tf.contrib.layers.conv2d(net, 128, [3, 3], activation_fn=tf.nn.relu, padding='SAME',
                                           weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(), scope=scope, reuse=reuse)
            net = tf.contrib.layers.max_pool2d(net, [2, 2], padding='SAME')

        with tf.variable_scope("conv4") as scope:
            net = tf.contrib.layers.conv2d(net, 256, [1, 1], activation_fn=tf.nn.relu, padding='SAME',
                                           weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(), scope=scope, reuse=reuse)
            net = tf.contrib.layers.max_pool2d(net, [2, 2], padding='SAME')

        with tf.variable_scope("conv5") as scope:
            net = tf.contrib.layers.conv2d(net, 2, [1, 1], activation_fn=None, padding='SAME',
                                           weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(), scope=scope, reuse=reuse)
            net = tf.contrib.layers.max_pool2d(net, [2, 2], padding='SAME')

        net = tf.contrib.layers.flatten(net)

    return net
################################################################################


def contrastive_loss(model1, model2, y, margin):
    with tf.name_scope("contrastive-loss"):
        d = tf.sqrt(tf.reduce_sum(
            tf.pow(model1 - model2, 2), 1, keep_dims=True))
        tmp = y * tf.square(d)
        tmp2 = (1 - y) * tf.square(tf.maximum((margin - d), 0))
        return tf.reduce_mean(tmp + tmp2) / 2


def contrastive_loss_2(left_encoding, right_encoding, labels, margin=0.2):
    with tf.name_scope("contrastive-loss"):
        distance = tf.reduce_sum(tf.square(left_encoding - right_encoding), 1)
        sqrt_distance = tf.sqrt(distance)

        loss = labels * tf.square(tf.maximum(0., margin - sqrt_distance))
        loss += (1 - labels) * distance
        return 0.5 * tf.reduce_mean(loss)
