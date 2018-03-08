import tensorflow as tf
from neural_utils import conv_net, contrastive_loss


class SiameseNetwork(object):

    def __init__(self,
                 input_image_shape,
                 output_encoding_size=2,
                 optimiser=None,
                 margin=0.2):
        """ init the model with hyper-parameters etc """
        image_height, image_width, image_channels = input_image_shape
        batch_list_shape = [None, image_height, image_width, image_channels]

        # Inputs.
        self.left_input = tf.placeholder(tf.float32,
                                         batch_list_shape,
                                         name='left')
        self.right_input = tf.placeholder(tf.float32,
                                          batch_list_shape,
                                          name='right')
        with tf.name_scope("similarity"):
            # 1 if same, 0 if different
            self.labels = tf.placeholder(tf.int32, [None, 1], name='label')
            self.labels = tf.to_float(self.labels)

        # Outputs.
        self.left_output = conv_net(self.left_input,
                                    output_encoding_size,
                                    reuse=False)
        self.right_output = conv_net(self.right_input,
                                     output_encoding_size,
                                     reuse=True)

        # Train ops.
        self.loss = contrastive_loss(self.left_output,
                                     self.right_output,
                                     self.labels,
                                     margin)

        self.global_step = tf.Variable(0, trainable=False)
        if optimiser is None:
            optimiser = tf.train.MomentumOptimizer(0.01,
                                                   0.99,
                                                   use_nesterov=True)
        self.train_op = optimiser.minimize(self.loss,
                                           global_step=self.global_step)

        self.saver = tf.train.Saver()
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        # Set up tensorboard
        tf.summary.scalar('step', self.global_step)
        tf.summary.scalar('loss', self.loss)
        for var in tf.trainable_variables():
            tf.summary.histogram(var.op.name, var)
        self.merged = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter('train.log', self.sess.graph)
        self.iter = 0

    def inference(self, x):
        """Here the net computes an embedding for the input image(s)."""
        return self.sess.run(self.left_output, feed_dict={self.left_input: x})

    def optimise(self, batch_left_input, batch_right_input, batch_similar):
        """Here the net is optimised through the contrastive loss."""
        feed_dict = {
            self.left_input: batch_left_input,
            self.right_input: batch_right_input,
            self.labels: batch_similar
        }
        tensors = [self.train_op, self.loss, self.merged]
        _, loss, summary_str = self.sess.run(tensors, feed_dict=feed_dict)
        self.writer.add_summary(summary_str, self.iter)
        self.iter += 1
        return loss

    def save(self, path="model/model.ckpt"):
        self.saver.save(self.sess, path)

    def close(self):
        self.sess.close()
