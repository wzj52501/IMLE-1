import tensorflow as tf
import numpy as np
import os
import layer as ly
from vgg19 import VGG19


class GLANN(op_base):
    def __init__(self,FLAGS):
        op_base.__init__(self,FLAGS)

    def G(self,z):
        with tf.variable_scope(name,reuse = tf.AUTO_REUSE):
            x = ly.fc(z,32*32*128,name = 'G_fc_0')
            x = ly.batch_normal(x,name = 'G_bn_0')
            x = ly.relu(x)

            x = tf.reshape(x,shape = [self.FLAGS.batch_size, 32, 32, 128])
            x = ly.conv2d(x,128,name = 'G_conv2d_0')
            x = ly.batch_normal(x,name = 'G_bn_1')
            x = ly.relu(x)

            x = ly.deconv2d(x,64,name = 'G_deconv2d_0')
            x = ly.batch_normal(x,name = 'G_bn_2')
            x = ly.relu(x)

            x = ly.deconv2d(x,32,name = 'G_deconv2d_1')
            x = ly.batch_normal(x,name = 'G_bn_3')
            x = ly.relu(x)

            ### (256,256,3)
            x = ly.deconv2d(x,3,name = 'G_deconv2d_2')
            x = ly.batch_normal(x,name = 'G_bn_4')
            x = ly.tanh(x)

            return x



    def T(self,n,name = 'T'):

        with tf.variable_scope(name,reuse = tf.AUTO_REUSE):
            n = tf.reshape(n,shape = [self.FLAGS.batch_size,self.FLAGS.n_deep * self.FLAGS.n_dim])
            n = ly.fc(n,512,name = 'T_0')
            n = ly.batch_normal(n,name = 'T_bn_0')
            n = ly.relu(n)

            n = ly.fc(n, self.FLAGS.n_deep * self.FLAGS.input_deep ,name = 'T_1')
            n = ly.batch_normal(n,name = 'T_bn_1')
            n = ly.relu(n)

            n = tf.reshape(n,shape = [self.FLAGS.batch_size,self.FLAGS.n_deep,self.FLAGS.input_deep])

            return n

    def minimizer_z(self,input_z,n_group):
        n_group_shape = tf.get_shape().as_list()
        loss = tf.square( tf.concat( [ input_z for i in range(n_group_shape[1]) ], axis = 1 ) - n_group )
        min_value = tf.reduce_min(loss,axis = 1)
        min_arg = tf.argmin(loss,axis = 1)

        def get_arg():
            with tf.control_dependencies([min_arg]):
                return tf.identity(mean), tf.identity(variance)
        min_n =
        return min_value


    def graph(self):

        self.input_z = tf.placeholder(tf.float32,shape = [self.FLAGS.batch_size,self.FLAGS.input_deep])
        self.input_n = tf.placeholder(tf.float32,shape = [self.FLAGS.batch_size,self.FLAGS.n_deep,self.FLAGS.n_dim])

        self.label = tf.placeholder(tf.float32,shape = [self.FLAGS.batch_size,self.FLAGS.image_height,self.FLAGS.image_weight,self.FLAGS.channels])
        self.output = tf.placeholder(tf.float32,shape = [self.FLAGS.batch_size,self.FLAGS.image_height,self.FLAGS.image_weight,self.FLAGS.channels])

        # self.n_group = tf.concat( [ tf.squeeze(cell) for cell in tf.split(self.input_n,self.FLAGS.n_deep,axis = 1) ],axis = 0)

        self.T_n = self.T(self.input_n,'T')
        self.min_z,self.min_z_loss = self.minimizer_z(self.input_z,self.T_n)
        self.t_loss = tf.reduce_sum(self.min_z)

        self.fake = self.G(self.T_n)
        self.vgg = VGG19()






