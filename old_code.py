import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import shutil
import os
from tensorflow.examples.tutorials.mnist import input_data
from dataset import next_batch, load_data_from_directory, unison_shuffle


batch_size = 64
train_iter = 5000
step = 100
dataset_folder = './sign_dataset/'

image_height = 28
image_width = 28
image_channels = 1
target_shape = (image_height, image_width, image_channels)



def get_mnist():
    mnist = input_data.read_data_sets("MNIST_data/")
    return mnist


def mynet(input, reuse=False):
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


def contrastive_loss(model1, model2, y, margin):
    with tf.name_scope("contrastive-loss"):
        d = tf.sqrt(tf.reduce_sum(
            tf.pow(model1 - model2, 2), 1, keep_dims=True))
        tmp = y * tf.square(d)
        tmp2 = (1 - y) * tf.square(tf.maximum((margin - d), 0))
        return tf.reduce_mean(tmp + tmp2) / 2


# Create the output directory.
shutil.rmtree('img', ignore_errors=True)
os.makedirs('img')
try:
    os.remove('train.log')
except OSError:
    pass

batch_generator = next_batch(batch_size,
                             data_directory=dataset_folder,
                             target_shape=target_shape,
                             probability=0.1)

c = ['#ff0000', '#ffff00', '#00ff00', '#00ffff', '#0000ff',
     '#ff00ff', '#990000', '#999900', '#009900', '#009999']


left = tf.placeholder(tf.float32, [None, image_width, image_height, image_channels], name='left')
right = tf.placeholder(tf.float32, [None, image_width, image_height, image_channels], name='right')
with tf.name_scope("similarity"):
    # 1 if same, 0 if different
    label = tf.placeholder(tf.int32, [None, 1], name='label')
    label = tf.to_float(label)
margin = 0.2

left_output = mynet(left, reuse=False)

right_output = mynet(right, reuse=True)

loss = contrastive_loss(left_output, right_output, label, margin)

global_step = tf.Variable(0, trainable=False)

train_step = tf.train.MomentumOptimizer(
    0.01, 0.99, use_nesterov=True).minimize(loss, global_step=global_step)

images, labels, labels_to_class_list = \
    load_data_from_directory(data_directory=dataset_folder,
                             target_shape=target_shape,
                             grayscale=True)
amount_classes = len(np.unique(labels))

images, labels = unison_shuffle(images, labels)

amount_test_images = 100
test_images = images[:amount_test_images]
test_labels = labels[:amount_test_images]
del images
del labels

saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # setup tensorboard
    tf.summary.scalar('step', global_step)
    tf.summary.scalar('loss', loss)
    for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name, var)
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter('train.log', sess.graph)

    # train iter
    for i, data in enumerate(batch_generator):
        b_l, b_r, b_sim = data

        _, l, summary_str = sess.run([train_step, loss, merged],
                                     feed_dict={left: b_l, right: b_r, label: b_sim})

        writer.add_summary(summary_str, i)
        print(("#%d - Loss" % i, l))
        print()

        if (i + 1) % step == 0:
            # generate test
            feat = sess.run(left_output, feed_dict={left: test_images})

            labels = test_labels
            # plot result
            f = plt.figure(figsize=(16, 9))
            for j in range(10):
                plt.plot(feat[labels == j, 0].flatten(), feat[labels == j, 1].flatten(),
                         '.', c=c[j], alpha=0.8)
            plt.legend(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
            plt.savefig('img/%d.jpg' % (i + 1))

    saver.save(sess, "model/model.ckpt")
