import tensorflow as tf
import numpy as np
import os
import layer as ly
from vgg19 import VGG19
from op_base import op_base
from util import *


class IMLE(op_base):
    def __init__(self,args,sess):
        op_base.__init__(self,args)
        self.sess = sess
        self.sess_arg = tf.Session()
        self.summaries = []

    def init_sess(self,sess,init):
        sess.run(init)

    def G(self,z,name = 'G',is_training = True):
        with tf.variable_scope(name,reuse = tf.AUTO_REUSE):
            # x = ly.fc(z,32*32*128,name = 'G_fc_0')
            x = ly.fc(z,8*8*32,name = 'G_fc_0')
            x = tf.reshape(x, shape=[-1, 8, 8, 32])


            x = ly.conv2d(x, 32, name='G_conv2d_0') ### 8, 8, 32
            x = ly.batch_normal(x, name='G_bn_0', is_training=is_training)
            x = ly.relu(x)


            x = ly.conv2d(x, 64, name='G_conv2d_1') ### 8, 8, 64
            x = ly.batch_normal(x, name='G_bn_1', is_training=is_training)
            x = ly.relu(x)

            x = ly.conv2d(x, 128, name='G_conv2d_2') ### 8, 8, 128
            x = ly.batch_normal(x, name='G_bn_2', is_training=is_training)
            x = ly.relu(x)

            x = ly.conv2d(x, 256, name='G_conv2d_3') ### 8, 8, 256
            x = ly.batch_normal(x, name='G_bn_3', is_training=is_training)
            x = ly.relu(x)

            x = ly.deconv2d(x,128,name = 'G_deconv2d_0') ### 16,16, 128
            x = ly.batch_normal(x,name = 'G_bn_4',is_training = is_training)
            x = ly.relu(x)

            x = ly.deconv2d(x,64,name = 'G_deconv2d_1') ### 32,32, 64
            x = ly.batch_normal(x,name = 'G_bn_5',is_training = is_training)
            x = ly.relu(x)

            x = ly.deconv2d(x,32,name = 'G_deconv2d_2') ### 64,64,32
            x = ly.batch_normal(x,name = 'G_bn_6',is_training = is_training)
            x = ly.relu(x)

            x = ly.deconv2d(x,16,name = 'G_deconv2d_3') ### 128,128,16
            x = ly.batch_normal(x,name = 'G_bn_7',is_training = is_training)
            x = ly.relu(x)


            x = ly.deconv2d(x,3,name = 'G_deconv2d_4') ### (256,256,3)
            x = ly.batch_normal(x,name = 'G_bn_8',is_training = is_training)
            x = tf.nn.tanh(x)

            return x



    def T(self,n,name = 'T',is_training = True):

        with tf.variable_scope(name,reuse = tf.AUTO_REUSE):
            n = tf.reshape(n,shape = [self.batch_size,self.n_deep * self.n_dim])
            n = ly.fc(n,512,name = 'T_0')
            n = ly.batch_normal(n,name = 'T_bn_0',is_training = is_training)
            n = ly.relu(n)

            n = ly.fc(n, self.n_deep * self.input_deep ,name = 'T_1')
            n = ly.batch_normal(n,name = 'T_bn_1',is_training = is_training)
            n = ly.relu(n)

            n = tf.reshape(n,shape = [self.batch_size,self.n_deep,self.input_deep])

            return n

    def get_vars(self, name, scope=None):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=name)

    def average_gradients(self,grad_group):
        average_list = []
        for cell in zip(*grad_group):
            ### mix discribution cell
            cell_grads = []
            for grad, var in cell:
                grad = tf.expand_dims(grad,axis = 0)
                cell_grads.append(grad)
            ave_grad = tf.reduce_mean(tf.concat(cell_grads,axis = 0),axis = 0)
            ave_vars = cell[0][1]
            average_list.append((ave_grad,ave_vars))

        return average_list


    # def minimizer_z(self,input_z,n_group):
    #     n_group_shape = n_group.get_shape().as_list()
    #     z_pad = tf.expand_dims(input_z,axis = 1)
    #     pad_top = self.n_deep // 2
    #     pad_bottom = self.n_deep - 1 - pad_top
    #     z_pad = tf.pad(z_pad,[[0,0],[pad_top, pad_bottom],[0,0]],"SYMMETRIC")
    #     loss = tf.reduce_sum( tf.square(z_pad - n_group) , axis = -1)
    #     min_loss = tf.reduce_min(loss,axis = -1)
    #     min_arg = tf.argmin(loss,axis = -1)
    #
    #     def add_twice_dim(mix_data):
    #         data, index = mix_data[0], mix_data[1]
    #         return data[index, :], index
    #     min_value = tf.map_fn(add_twice_dim , (n_group,min_arg))[0]
    #
    #
    #     return min_value, min_loss


    # def graph(self,label,g_opt = None, t_opt = None,is_training = True):
        #
        # input_z = tf.random_normal(shape = [self.batch_size,self.input_deep],mean = 0.,stddev = 0.2)
        # input_n = tf.random_normal(shape = [self.batch_size,self.n_deep,self.n_dim],mean = 0.,stddev = 0.2)
        #
        # T_n = self.T(input_n,'T',is_training = is_training)
        # min_n,min_n_loss = self.minimizer_z(input_z,T_n)
        # t_loss = tf.reduce_sum(min_n_loss)
        #
        # fake = self.G(min_n,'G',is_training = is_training)
        # vgg = VGG19()
        # vgg_loss = tf.reduce_mean(vgg(fake) - vgg(label))
        #
        # t_grad = t_opt.compute_gradients(t_loss, self.get_vars('T'))
        # g_grad = g_opt.compute_gradients(vgg_loss, self.get_vars('G'))
        #
        # return t_loss, t_grad, vgg_loss, g_grad

    def graph(self,label,g_opt,is_training = True):
        input_z = tf.random_normal(shape=[self.batch_size, self.input_deep], mean=0., stddev=0.2)
        fake = self.G(input_z,'G',is_training = is_training)
        def minest_fake(one_real_image):
            cell_image = tf.expand_dims(one_real_image,axis = 0)
            mix_image = tf.concat([ cell_image for i in range(self.batch_size) ],axis = 0)
            minest_index = tf.argmin( tf.reduce_sum(tf.square(mix_image - fake ),axis = [1,2,3]), axis = -1 ) ## index
            return fake[minest_index]
        min_fake = tf.map_fn(minest_fake , label)
        g_loss = tf.reduce_sum( tf.reduce_mean( tf.square(min_fake - label), axis = [1,2,3] ))

        g_grad = g_opt.compute_gradients(g_loss, self.get_vars('G'))
        return g_loss, g_grad

    def make_data_queue(self):
        images_label, image_names = load_image(self)

        input_queue = tf.train.slice_input_producer([images_label, image_names], num_epochs=self.epoch, shuffle=False)
        label, name = tf.train.batch(input_queue, batch_size=self.batch_size, num_threads=2,
                                      capacity=64,
                                      allow_smaller_final_batch=False)

        return label, name

    def main(self):
        label, name = self.make_data_queue()
        self.start(label,name)



    def start(self,label_image,label_name,need_train = True):

        ## lr
        global_steps = tf.get_variable(name='global_step', shape=[], initializer=tf.constant_initializer(0),
                                       trainable=False)
        decay_change_batch_num = 200.0
        train_data_num = 2000 * self.epoch
        decay_steps = (train_data_num / self.batch_size / self.gpu_nums) * decay_change_batch_num

        lr = tf.train.exponential_decay(self.lr,
                                        global_steps,
                                        decay_steps,
                                        0.1,
                                        staircase=True)

        self.summaries.append(tf.summary.scalar('lr',lr))

        ## opt
        g_opt = tf.train.AdamOptimizer(lr)

        ## graph
        g_mix_grads = []
        g_loss_mix = []
        for i in range(self.gpu_nums):
            with tf.device('%s:%s' %( self.train_utils,i) ):
                with tf.name_scope('distributed_%s' % i):
                    g_loss, g_grad = self.graph(label_image, g_opt = g_opt)
                    g_mix_grads.append(g_grad)
                    g_loss_mix.append(g_loss)

                    # tf.get_variable_scope().reuse_variables()

        ### loss
        g_ave_loss = tf.reduce_mean(g_loss_mix,axis = 0)
        self.summaries.append(tf.summary.scalar('g_loss',g_ave_loss))

        ### grad_op
        g_ave_grad = self.average_gradients(g_mix_grads)

        g_grad_op = g_opt.apply_gradients(g_ave_grad,global_step=global_steps)

        ### variable_op
        MOVING_AVERAGE_DECAY = 0.9
        g_variable_average = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY)

        g_var_op = g_variable_average.apply(self.get_vars('G'))

        g_group = tf.group(g_grad_op,g_var_op)

        ## init
        self.init_sess(self.sess,[tf.global_variables_initializer(),tf.local_variables_initializer()])

        ## summary init
        summary_writer = tf.summary.FileWriter(self.summary_dir, self.sess.graph)
        summary_op = tf.summary.merge(self.summaries)

        ## queue init
        coord = tf.train.Coordinator()
        thread = tf.train.start_queue_runners(sess = self.sess)

        ### train
        saver = tf.train.Saver(max_to_keep = 1)
        step = 1
        print('start train')
        if(need_train):
            try:
                while not coord.should_stop():
                    print('start %s' % step)
                    _g = self.sess.run([g_group])
                if(step % 10 == 0):

                    print('update summary')
                    summary_str = self.sess.run(summary_op)
                    summary_writer.add_summary(summary_str,step)
                if(step % 100 == 0):
                    print('update model')
                    saver.save(self.sess,os.path.join(self.model_save_path,'model_%s.ckpt' % step))
                step += 1

            except tf.errors.OutOfRangeError:
                print('finish thread')
            finally:
                coord.request_stop()

        coord.join(thread)
        print('thread break')















