import tensorflow as tf
import numpy as np
import os
import layer as ly
from vgg19 import VGG19
from op_base import op_base
from util import *


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
        n_group_shape = n_group.get_shape().as_list()
        z_pad = tf.expand_dims(input_z,axis = 1)
        pad_top = (self.n_deep - 1) // 2
        pad_bottom = self.n_deep - 1 - pad_top
        z_pad = tf.pad(z_pad,[[0,0],[pad_top, pad_bottom],[0,0]],"SYMMETRIC")
        loss = tf.reduce_sum( tf.square(z_pad - n_group) , axis = -1)
        min_loss = tf.reduce_min(loss,axis = -1)
        print(min_loss.shape)
        min_arg = tf.argmin(loss,axis = -1)

        ### sess.run
        # min_arg = tf.Session().run(min_arg)
        # min_value_list = []
        # for batch_index,batch_value in enumerate(list(min_arg)):
        #     min_value_list.append(n_group[batch_index,batch_value,:])
        # min_value = tf.concat(min_value_list)

        ## identity
        def calu_minarg():
            with tf.control_dependencies([min_arg]):
                min_value_list = []
                for batch_index, batch_value in enumerate(list(min_arg)):
                    min_value_list.append(n_group[batch_index, batch_value, :])
                return tf.identity(min_value_list)
        min_value = calu_minarg()

        return min_value, min_loss

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


    def graph(self,label,g_opt = None, t_opt = None):

        # self.input_z = tf.placeholder(tf.float32,shape = [self.batch_size,self.input_deep])
        # self.input_n = tf.placeholder(tf.float32,shape = [self.batch_size,self.n_deep,self.n_dim])
        #
        # self.label = tf.placeholder(tf.float32,shape = [self.batch_size,self.image_height,self.image_weight,self.channels])
        # self.output = tf.placeholder(tf.float32,shape = [self.batch_size,self.image_height,self.image_weight,self.channels])

        input_z = tf.random_normal(shape = [self.batch_size,self.input_deep],mean = 0.,stddev = 0.2)
        input_n = tf.random_normal(shape = [self.batch_size,self.n_deep,self.n_dim],mean = 0.,stddev = 0.2)

        T_n = self.T(input_n,'T')
        min_z,min_z_loss = self.minimizer_z(input_z,T_n)
        t_loss = tf.reduce_sum(min_z_loss)

        fake = self.G(T_n,'G')
        vgg = VGG19()
        vgg_loss = tf.reduce_mean(vgg(min_z) - vgg(label))


        t_grad = t_opt.compute_gradients(t_loss, self.get_vars('T'))
        g_grad = g_opt.compute_gradients(vgg_loss, self.get_vars('G'))

        return t_loss, t_grad, vgg_loss, g_grad

    def make_data_queue(self):
        images_label, image_names = load_image(self)
        ### train with noise
        # tf.random_normal()
        ### eval with image
        # real image

        input_queue = tf.train.slice_input_producer([images_label, image_names], num_epochs=self.epoch, shuffle=False)
        label, name = tf.train.batch(input_queue, batch_size=self.batch_size, num_threads=2,
                                      capacity=64,
                                      allow_smaller_final_batch=False)

        return label, name

    def main(self):
        label, name = self.make_data_queue()
        self.start(label,name)



    def start(self,label_image,label_name):

        ## lr
        global_steps = tf.get_variable(name='global_step', shape=[], initializer=tf.constant_initializer(0),
                                       trainable=False)
        decay_change_batch_num = 350.0
        train_data_num = 3000 * self.epoch
        decay_steps = (train_data_num / self.batch_size / self.gpu_nums) * decay_change_batch_num

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
                    t_loss, t_grad, vgg_loss, g_grad = self.graph(label_image, g_opt = g_opt, t_opt = t_opt)
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
        init_local = tf.local_variables_initializer()
        sess.run([init,init_local])

        ## summary init
        summary_writer = tf.summary.FileWriter(self.summary_dir, self.sess.graph)
        summary_op = tf.summary.merge(self.summaries)

        ## queue init
        coord = tf.train.Coordinator()
        thread = tf.train.start_queue_runners(sess = self.sess)

        ### train
        saver = tf.train.Saver(max_to_keep = 1)
        step = 1
        if(need_train):
            try:
                while not coord.should_stop():
                    _t, _g = self.sess.run([t_group,g_group])
                if(step % 10 == 0):
                    summary_str = self.sess.run(summary_op)
                    summary_writer.add_summary(summary_str,step)
                if(step % 100 == 0):
                    saver.save(self.sess,os.path.join(self.model_save_path,'model_%s.ckpt' % step))
                step += 1

            except tf.errors.OutOfRangeError:
                print('finish thread')
            finally:
                coord.request_stop()

        coord.join(thread)
        print('thread break')














