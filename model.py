import numpy as np
import tensorflow as tf
from neural_utils import conv_net, contrastive_loss


class SiameseNetwork(object):

    def __init__(self,
                 input_image_shape,
                 output_encoding_size=2,
                 optimiser=None,
                 margin=0.2,
                 learning_rate=0.01,
                 gradient_clip_amount=None):
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
            self.labels = tf.placeholder(tf.float32, [None, 1], name='label')

        # Outputs.
        self.left_output = conv_net(input_tensor=self.left_input,
                                    output_size=output_encoding_size,
                                    reuse=False)
        self.right_output = conv_net(input_tensor=self.right_input,
                                     output_size=output_encoding_size,
                                     reuse=True)

        # Train ops.
        self.loss = contrastive_loss(self.left_output,
                                     self.right_output,
                                     self.labels,
                                     margin)
        self.debug_op = tf.add_check_numerics_ops()

        self.global_step = tf.Variable(0, trainable=False)
        if optimiser is None:
            optimiser = tf.train.MomentumOptimizer(learning_rate,
                                                   0.99,
                                                   use_nesterov=True)

        gradients, variables = zip(*optimiser.compute_gradients(self.loss))
        if gradient_clip_amount is not None:
            gradients, _ = tf.clip_by_global_norm(gradients, gradient_clip_amount)
        self.train_op = optimiser.apply_gradients(zip(gradients, variables),
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

        try:
            loss = self.sess.run(self.loss, feed_dict=feed_dict)
            tensors = [self.train_op, self.debug_op, self.merged]
            _, _, summary_str = self.sess.run(tensors, feed_dict=feed_dict)

        except tf.errors.InvalidArgumentError as e:
            print('final loss:', loss)
            print('\ntensor name: {}, tensor inputs:{}\n'.format(e.op.name, e.op.inputs[0]))
            print(e.message)
            print("Data (isfinite):",
                  np.isfinite(batch_left_input).all(),
                  np.isfinite(batch_right_input).all(),
                  np.isfinite(batch_similar).all())

        self.writer.add_summary(summary_str, self.iter)
        self.iter += 1
        return loss

    def save(self, path="model/model.ckpt"):
        self.saver.save(self.sess, path)

    def close(self):
        self.sess.close()
