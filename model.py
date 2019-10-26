import tensorflow as tf
import numpy as np
import os
import layer as ly
from vgg19 import VGG19
from op_base import op_base


class GLANN(op_base):
    def __init__(self,args,sess):
        op_base.__init__(self,args)
        self.sess = sess
        self.summaries = []

    def G(self,z,name):
        with tf.variable_scope(name,reuse = tf.AUTO_REUSE):
            x = ly.fc(z,32*32*128,name = 'G_fc_0')
            x = ly.batch_normal(x,name = 'G_bn_0')
            x = ly.relu(x)

            x = tf.reshape(x,shape = [self.batch_size, 32, 32, 128])
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
            n = tf.reshape(n,shape = [self.batch_size,self.n_deep * self.n_dim])
            n = ly.fc(n,512,name = 'T_0')
            n = ly.batch_normal(n,name = 'T_bn_0')
            n = ly.relu(n)

            n = ly.fc(n, self.n_deep * self.input_deep ,name = 'T_1')
            n = ly.batch_normal(n,name = 'T_bn_1')
            n = ly.relu(n)

            n = tf.reshape(n,shape = [self.batch_size,self.n_deep,self.input_deep])

            return n

    def minimizer_z(self,input_z,n_group):
        n_group_shape = tf.get_shape().as_list()
        loss = tf.square( tf.concat( [ input_z for i in range(n_group_shape[1]) ], axis = 1 ) - n_group )
        min_value = tf.reduce_min(loss,axis = 1)
        min_arg = tf.argmin(loss,axis = 1)

        def get_arg():
            with tf.control_dependencies([min_arg]):
                return tf.identity(mean), tf.identity(variance)
        # min_n =
        return min_value

    def get_vars(self, name, scope=None):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=name)

    def average_gradients(self,grad_group):
        average_list = []
        for cell in zip(*grad_group):
            cell_grads = []
            for grad, var in cell:
                grad = tf.expend_dims(grad,axis = 0)
                cell_grads.append(grad)

            ave_grad = tf.reduce_mean(cell_grads,axis = 0)
            average_list.append((ave_grad,cell[0][1]))

        return average_list


    def graph(self,input_z,input_n,label,g_opt = None, t_opt = None):

        # self.input_z = tf.placeholder(tf.float32,shape = [self.batch_size,self.input_deep])
        # self.input_n = tf.placeholder(tf.float32,shape = [self.batch_size,self.n_deep,self.n_dim])
        #
        # self.label = tf.placeholder(tf.float32,shape = [self.batch_size,self.image_height,self.image_weight,self.channels])
        # self.output = tf.placeholder(tf.float32,shape = [self.batch_size,self.image_height,self.image_weight,self.channels])

        T_n = self.T(input_n,'T')
        min_z,min_z_loss = minimizer_z(input_z,T_n)
        t_loss = tf.reduce_sum(min_z)

        fake = self.G(T_n,'G')
        vgg = VGG19()
        vgg_loss = tf.reduce_mean(vgg(min_z) - vgg(label))


        t_grad = t_opt.compute_gradients(t_loss, self.get_vars('T'))
        g_grad = t_opt.compute_gradients(vgg_loss, self.get_vars('G'))

        return t_loss, t_grad, vgg_loss, g_grad,

    def train(self):

        ## data
        input_z, input_n, label = ''
        ## lr
        global_steps = tf.get_variable(name='global_step', shape=[], initializer=tf.constant_initializer(0),
                                       trainable=False)
        decay_change_batch_num = 350.0
        decay_steps = (self.train_data_num / self.batch_size / self.gpu_nums) * decay_change_batch_num

        lr = tf.train.exponential_decay(self.lr,
                                        global_steps,
                                        decay_steps,
                                        0.1,
                                        staircase=True)

        self.summaries.append(tf.summary.scalar('lr',lr))

        ## opt
        g_opt = tf.train.AdamOptimizer(lr)
        t_opt = tf.train.AdamOptimizer(lr)

        ## graph
        t_mix_grads = []
        g_mix_grads = []
        t_loss_mix = []
        vgg_loss_mix = []
        for i in range(self.gpu_nums):
            with tf.device('%s:%s' %( self.train_utils,i) ):
                with tf.name_scope('distributed_%s' % i):
                    t_loss, t_grad, vgg_loss, g_grad = self.graph(input_z,input_n,label)
                    t_mix_grads.append(t_grad)
                    t_loss_mix.append(t_loss)
                    g_mix_grads.append(g_grad)
                    t_loss_mix.append(vgg_loss)


        ### loss
        t_ave_loss = tf.reduce_mean(t_loss_mix,axis = 0)
        vgg_ave_loss = tf.reduce_mean(vgg_loss_mix,axis = 0)
        self.summaries.append(tf.summary.scalar('t_loss',t_ave_loss))
        self.summaries.append(tf.summary.scalar('vgg_loss',vgg_ave_loss))

        ### grad_op
        t_ave_grad, g_ave_grad = self.average_gradients(t_mix_grads) , self.average_gradients(g_mix_grads)
        self.summaries.append(tf.summary.scalar('t_grad',t_ave_grad))
        self.summaries.append(tf.summary.scalar('g_grad',g_ave_grad))
        t_grad_op, g_grad_op = t_opt.apply_gradients(t_ave_grad,global_step=global_steps),  g_opt.apply_gradients(g_ave_grad,global_step=global_steps)

        ### variable_op
        MOVING_AVERAGE_DECAY = 0.9
        t_variable_average = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY)
        g_variable_average = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY)

        t_var_op = t_variable_average.apply(self.get_vars('T'))
        g_var_op = t_variable_average.apply(self.get_vars('G'))

        t_group = tf.groups(t_grad_op,t_var_op)
        g_group = tf.groups(g_grad_op,g_var_op)


        ## init
        init = tf.global_variables_initializer()












