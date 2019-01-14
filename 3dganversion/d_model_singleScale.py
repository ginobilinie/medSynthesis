import tensorflow as tf
import numpy as np
from skimage.transform import resize

from d_scale_model import DScaleModel
from loss_functions import adv_loss
import constants as c


from __future__ import division
import os
import time
from glob import glob
import tensorflow as tf
import numpy as np
from six.moves import xrange

from utils import *
from loss_functions import *
from scipy.misc import imsave

class MR2CT(object):
    def __init__(self, sess, batch_size=10, depth_MR=32, height_MR=32,
                 width_MR=32, depth_CT=32, height_CT=24,
                 width_CT=24, l_num=2, wd=0.0005, checkpoint_dir=None, path_patients_h5=None, learning_rate=2e-8):
        """
        Args:
            sess: TensorFlow session
            batch_size: The size of batch. Should be specified before training.
            output_size: (optional) The resolution in pixels of the images. [64]
            y_dim: (optional) Dimension of dim for y. [None]
            z_dim: (optional) Dimension of dim for Z. [100]
            gf_dim: (optional) Dimension of gen filters in first conv layer. [64]
            df_dim: (optional) Dimension of discrim filters in first conv layer. [64]
            gfc_dim: (optional) Dimension of gen units for for fully connected layer. [1024]
            dfc_dim: (optional) Dimension of discrim units for fully connected layer. [1024]
            c_dim: (optional) Dimension of image color. For grayscale input, set to 1. [3]
        """
        self.sess = sess
        self.l_num=l_num
        self.wd=wd
        self.learning_rate=learning_rate
        self.batch_size=batch_size       
        self.depth_MR=depth_MR
        self.height_MR=height_MR
        self.width_MR=width_MR
        self.depth_CT=depth_CT
        self.height_CT=height_CT
        self.width_CT=width_CT
        self.checkpoint_dir = checkpoint_dir
        self.data_generator = Generator_3D_patches(path_patients_h5,self.batch_size)
        self.build_model()

    def build_model(self):
        self.inputMR=tf.placeholder(tf.float32, shape=[None, self.depth_MR, self.height_MR, self.width_MR, 1])
        self.CT_GT=tf.placeholder(tf.float32, shape=[None, self.depth_CT, self.height_CT, self.width_CT, 1])
        batch_size_tf = tf.shape(self.inputMR)[0]  #variable batchsize so we can test here
        self.train_phase = tf.placeholder(tf.bool, name='phase_train')
        self.G = self.generator(self.inputMR,batch_size_tf)
        print 'shape output G ',self.G.get_shape()
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.g_loss=lp_loss(self.G, self.CT_GT, self.l_num, batch_size_tf)
        print 'learning rate ',self.learning_rate
        self.g_optim =tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.g_loss)
        self.merged = tf.merge_all_summaries()
        self.writer = tf.train.SummaryWriter("./summaries", self.sess.graph)
        self.saver = tf.train.Saver()


    def discriminator(self,inputMR,batch_size_tf):        
        
        ######## FCN for the 32x32x32 to 24x24x24 ####################################        
        conv1_a = conv_op_3d_bn(inputMR, name="conv1_a", kh=5, kw=5, kz=5,  n_out=48, dh=1, dw=1, dz=1, wd=self.wd, padding='VALID',train_phase=self.train_phase)#30
        conv2_a = conv_op_3d_bn(conv1_a, name="conv2_a", kh=3, kw=3, kz=3,  n_out=96, dh=1, dw=1, dz=1, wd=self.wd, padding='SAME',train_phase=self.train_phase)
        pool1 = mpool_op_3d(conv2_a, name="mpool1", kh=2, kw=2,kz=2, dh=2, dw=2, dz=2)
        conv3_a = conv_op_3d_bn(pool1, name="conv3_a", kh=3, kw=3, kz=3,  n_out=128, dh=1, dw=1, dz=1, wd=self.wd, padding='SAME',train_phase=self.train_phase)#28
        conv4_a = conv_op_3d_bn(conv3_a, name="conv4_a", kh=3, kw=3, kz=3,  n_out=96, dh=1, dw=1, dz=1, wd=self.wd, padding='VALID',train_phase=self.train_phase)
        pool2 = mpool_op_3d(conv4_a, name="mpool1", kh=2, kw=2,kz=2, dh=2, dw=2, dz=2)
        conv5_a = conv_op_3d_bn(pool2, name="conv5_a", kh=3, kw=3, kz=3,  n_out=48, dh=1, dw=1, dz=1, wd=self.wd, padding='SAME',train_phase=self.train_phase)#26
        fc6=fullyconnect_op(conv5_a, n_outputs=512)
        dp6=dropout(fc6,dropout_rate=0.5, train_phase=self.train_phase, name='Dropout')
        fc7=fullyconnect_op(dp6, n_outputs=128)
        dp7=dropout(fc7,dropout_rate=0.5, train_phase=self.train_phase, name='Dropout')
        fc8=fullyconnect_op(dp7, n_outputs=2)
        # here we should define loss for fc8 and train it...
        loss=fc8
        return loss




    def train(self, config):
        path_test='/home/dongnie/warehouse/prostate/ganData64to24Test'
        print 'global_step ', self.global_step.name
        print 'trainable vars '
        for v in tf.trainable_variables():
            print v.name

        if self.load(self.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")
            self.sess.run(tf.initialize_all_variables())

        start = self.global_step.eval() # get last global_step
        print("Start from:", start)
        

        for it in range(start,config.iterations):

            X,y=self.data_generator.next()
            

            # Update G network
            _, loss_eval, layer_out_eval = self.sess.run([self.g_optim, self.g_loss, self.MR_16_downsampled],
                        feed_dict={ self.inputMR: X, self.CT_GT:y, self.train_phase: True })
            self.global_step.assign(it).eval() # set and update(eval) global_step with index, i
            

            if it%config.show_every==0:#show loss every show_every its
                print 'it ',it,'loss ',loss_eval
                print 'layer min ', np.min(layer_out_eval)
                print 'layer max ', np.max(layer_out_eval)
                print 'layer mean ', np.mean(layer_out_eval)
             #    print 'trainable vars ' 
                # for v in tf.trainable_variables(): 
                #     print v.name 
                #     data_var=self.sess.run(v) 
                #     grads = tf.gradients(self.g_loss, v) 
                #     var_grad_val = self.sess.run(grads, feed_dict={self.inputMR: X, self.CT_GT:y }) 
                #     print 'grad min ', np.min(var_grad_val) 
                #     print 'grad max ', np.max(var_grad_val) 
                #     print 'grad mean ', np.mean(var_grad_val) 
                #     #print 'shape ',data_var.shape 
                #     print 'filter min ', np.min(data_var) 
                #     print 'filter max ', np.max(data_var) 
                #     print 'filter mean ', np.mean(data_var)    
                    #self.writer.add_summary(summary, it)
                            # print 'trainable vars ' 

            
            if it%config.test_every==0 and it!=0:#==0:#test one subject                

                mr_test_itk=sitk.ReadImage(os.path.join(path_test,'prostate_1to1_MRI.nii'))
                ct_test_itk=sitk.ReadImage(os.path.join(path_test,'prostate_1to1_CT.nii'))
                mrnp=sitk.GetArrayFromImage(mr_test_itk)
                #mu=np.mean(mrnp)
                #mrnp=(mrnp-mu)/(np.max(mrnp)-np.min(mrnp))
                ctnp=sitk.GetArrayFromImage(ct_test_itk)
                print mrnp.dtype
                print ctnp.dtype
                ct_estimated=self.test_1_subject(mrnp,ctnp,[32,32,32],[24,24,24],[5,5,2])
                psnrval=psnr(ct_estimated,ctnp)
                print ct_estimated.dtype
                print ctnp.dtype
                print 'psnr= ',psnrval
                volout=sitk.GetImageFromArray(ct_estimated)
                sitk.WriteImage(volout,'ct_estimated_{}'.format(it)+'.nii.gz')

            if it%config.save_every==0:#save weights every save_every iterations
                self.save(self.checkpoint_dir, it)

            #Note, we should better return global loss here
            return global_loss
