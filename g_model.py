'''
This class implements a 3D FCN for the task of generating CT from MRI
By Roger Trullo and Dong Nie
Oct., 2016
'''

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
import collections
import datetime

class MR2CT(object):
    def __init__(self, sess, batch_size=10, height_MR=64,width_MR=64, height_CT=48,
                 width_CT=48, l_num=2, wd=0.0005, checkpoint_dir=None, path_patients_h5=None, learning_rate=2e-8,lr_step=30000,
                 lam_lp=1, lam_gdl=1, lam_adv=1, alpha=2):

    
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
        self.lam_lp=lam_lp
        self.lam_gdl=lam_gdl
        self.lam_adv=lam_adv
        self.alpha=alpha
        self.lr_step=lr_step
        self.l_num=l_num
        self.wd=wd
        self.learning_rate=learning_rate
        self.batch_size=batch_size       
        self.height_MR=height_MR
        self.width_MR=width_MR
        self.height_CT=height_CT
        self.width_CT=width_CT
        self.checkpoint_dir = checkpoint_dir
        self.data_generator = Generator_2D_slices(path_patients_h5,self.batch_size)
        self.build_model()

    def build_model(self)：
	    with tf.device('/gpu:0'):
			self.inputMR=tf.placeholder(tf.float32, shape=[None, self.height_MR, self.width_MR, 5])#5 chans input
			self.CT_GT=tf.placeholder(tf.float32, shape=[None, self.height_CT, self.width_CT, 1])
			batch_size_tf = tf.shape(self.inputMR)[0]  #variable batchsize so we can test here
			self.train_phase = tf.placeholder(tf.bool, name='phase_train')
			self.G, self.layer = self.generator(self.inputMR,batch_size_tf)
			print 'G shape ',self.G.get_shape
			self.D, self.D_logits = self.discriminator(self.CT_GT)#real CT data
			self.D_, self.D_logits_ = self.discriminator(self.G, reuse=True)#fake generated CT data
			self.d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.D_logits, tf.ones_like(self.D)))
			self.d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.D_logits_, tf.zeros_like(self.D_)))
			self.d_loss=self.d_loss_real+self.d_loss_fake
			self.global_step = tf.Variable(0, name='global_step', trainable=False)
			self.g_loss, self.lpterm, self.gdlterm, self.bceterm=self.combined_loss_G(batch_size_tf)
			t_vars = tf.trainable_variables()
			self.d_vars = [var for var in t_vars if 'd_' in var.name]
			self.g_vars = [var for var in t_vars if 'g_' in var.name]
			self.d_optim = tf.train.AdamOptimizer(self.learning_rate, beta1=0.5) \
			                  .minimize(self.d_loss, var_list=self.d_vars)
			self.g_optim = tf.train.AdamOptimizer(self.learning_rate, beta1=0.5) \
			                  .minimize(self.g_loss, var_list=self.g_vars, global_step=self.global_step)
			print 'shape output G ',self.G.get_shape()
			#print 'shape output D ',self.D.get_shape()
			print 'learning rate ',self.learning_rate
			#self.learning_rate_tensor = tf.train.exponential_decay(self.learning_rate, self.global_step,                                     #self.lr_step, 0.1, staircase=True)
			#self.g_optim = tf.train.GradientDescentOptimizer(self.learning_rate_tensor).minimize(self.g_loss, global_step=self.global_step)
			#self.g_optim = tf.train.MomentumOptimizer(self.learning_rate_tensor, 0.9).minimize(self.g_loss, global_step=self.global_step)
			self.merged = tf.merge_all_summaries()
			self.writer = tf.train.SummaryWriter("./summaries", self.sess.graph)
			self.saver = tf.train.Saver()


    def generator(self,inputMR,batch_size_tf):
               
        ######## FCN for the 32x32x32 to 24x24x24 ###################################
        print 'input shape, ',inputMR.get_shape() 
        conv1_a = conv_op_bn(inputMR, name="g_conv1_a", kh=7, kw=7, n_out=128, dh=1, dw=1, wd=self.wd, padding='VALID',train_phase=self.train_phase)#30
        conv2_a = conv_op_bn(conv1_a, name="g_conv2_a", kh=5, kw=5, n_out=128, dh=1, dw=1, wd=self.wd, padding='VALID',train_phase=self.train_phase)
        conv3_a = conv_op_bn(conv2_a, name="g_conv3_a", kh=3, kw=3, n_out=256, dh=1, dw=1, wd=self.wd, padding='VALID',train_phase=self.train_phase)#28
        conv4_a = conv_op_bn(conv3_a, name="g_conv4_a", kh=3, kw=3, n_out=256, dh=1, dw=1, wd=self.wd, padding='VALID',train_phase=self.train_phase)#28
        conv5_a = conv_op_bn(conv4_a, name="g_conv5_a", kh=3, kw=3, n_out=128, dh=1, dw=1, wd=self.wd, padding='VALID',train_phase=self.train_phase)
        
        conv6_a = conv_op_bn(conv5_a, name="g_conv6_a", kh=3, kw=3, n_out=128, dh=1, dw=1, wd=self.wd, padding='SAME',train_phase=self.train_phase)#26
        conv7_a = conv_op_bn(conv6_a, name="g_conv7_a", kh=3, kw=3, n_out=128, dh=1, dw=1, wd=self.wd, padding='SAME',train_phase=self.train_phase)
        conv8_a = conv_op_bn(conv7_a, name="g_conv8_a", kh=3, kw=3, n_out=64, dh=1, dw=1, wd=self.wd, padding='SAME',train_phase=self.train_phase)
                #conv7_a = conv_op_3d_bn(conv6_a, name="conv7_a", kh=3, kw=3, n_out=1, dh=1, dw=1, wd=self.wd, padding='SAME',train_phase=self.train_phase)#24
        conv9_a = conv_op(conv8_a, name="g_conv9_a", kh=3, kw=3, n_out=1, dh=1, dw=1, wd=self.wd, padding='SAME',activation=False)#24 I modified it here,dong
        print 'conv9a shape, ',conv9_a.get_shape()
        #self.MR_16_downsampled=conv7_a#JUST FOR TEST
        return conv9_a,conv9_a


    def discriminator(self, inputCT, reuse=False):
    	if reuse:
	        tf.get_variable_scope().reuse_variables()
        print 'ct shape ',inputCT.get_shape()
        h0=conv_op_bn(inputCT, name="d_conv_dis_1_a", kh=5, kw=5, n_out=32, dh=1, dw=1, wd=self.wd, padding='VALID',train_phase=self.train_phase)
        print 'h0 shape ',h0.get_shape()
        m0=mpool_op(h0, 'pool0', kh=2, kw=2, dh=2, dw=2)
        print 'm0 shape ',m0.get_shape()
        h1 = conv_op_bn(m0, name="d_conv2_dis_a", kh=5, kw=5, n_out=64, dh=1, dw=1, wd=self.wd, padding='VALID',train_phase=self.train_phase)
        print 'h1 shape ',h1.get_shape()
        m1=mpool_op(h1, 'pool1', kh=2, kw=2, dh=2, dw=2)
        print 'mi shape ',m1.get_shape()
        h2 = conv_op_bn(m1, name="d_conv3_dis_a", kh=5, kw=5, n_out=128, dh=1, dw=1, wd=self.wd, padding='VALID',train_phase=self.train_phase)#28
        h3 = conv_op_bn(h2, name="d_conv4_dis_a", kh=5, kw=5, n_out=256, dh=1, dw=1, wd=self.wd, padding='VALID',train_phase=self.train_phase)
        fc1=fullyconnected_op(h3, name="d_fc1", n_out=512, wd=self.wd, activation=True)
        fc2=fullyconnected_op(fc1, name="d_fc2", n_out=128, wd=self.wd, activation=True)
        fc3=fullyconnected_op(fc2, name="d_fc3", n_out=1, wd=self.wd, activation=False)
        return tf.nn.sigmoid(fc3), fc3




    def train(self, config):
    	path_test='/home/dongnie/warehouse/prostate/ganData64to24Test'
        print 'global_step ', self.global_step.name
        print 'lr_step ',self.lr_step
        print 'trainable vars '
        for v in tf.trainable_variables():
            print v.name

        
        if self.load(self.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")
            self.sess.run(tf.initialize_all_variables())

        self.sess.graph.finalize()
        
        start = self.global_step.eval() # get last global_step
        print("Start from:", start)
        
        
        

        for it in range(start,config.iterations):

            X,y=self.data_generator.next()


            # Update D network
            _, loss_eval_D, = self.sess.run([self.d_optim, self.d_loss],
                        feed_dict={ self.inputMR: X, self.CT_GT:y, self.train_phase: True })
            #### maybe we need to get a different batch???########

            # Update G network
            _, loss_eval_G, lp_eval,gdl_eval,bce_eval, layer_out_eval = self.sess.run([self.g_optim, 
                                    self.g_loss, self.lpterm, self.gdlterm, self.bceterm, self.layer],
                                    feed_dict={ self.inputMR: X, self.CT_GT:y, self.train_phase: True })
            
            

            if it%config.show_every==0:#show loss every show_every its
                #curr_lr=self.sess.run(self.learning_rate_tensor)
                #print 'lr= ',curr_lr
                print 'time ',datetime.datetime.now(),' it ',it,'loss D bce ',loss_eval_D
                print 'loss total G ',loss_eval_G
                print 'loss lp G ',lp_eval
                print 'loss gdl G',gdl_eval
                print 'loss bce G ',bce_eval


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
                ct_estimated=self.test_1_subject(mrnp,ctnp,[64,64,5],[48,48,1],[2,5,5])
                psnrval=psnr(ct_estimated,ctnp)
                print ct_estimated.dtype
                print ctnp.dtype
                print 'psnr= ',psnrval
                volout=sitk.GetImageFromArray(ct_estimated)
                sitk.WriteImage(volout,'ct_estimated_{}'.format(it)+'.nii.gz')

            if it%config.save_every==0:#save weights every save_every iterations
                self.save(self.checkpoint_dir, it)

    def evaluate(self,patch_MR):
        """ patch_MR is a np array of shape [H,W,nchans]
        """
        patch_MR=np.expand_dims(patch_MR,axis=0)#[1,H,W,nchans]
        #patch_MR=np.expand_dims(patch_MR,axis=4)#[1,H,W,nchans]
        #patch_MR=patch_MR.astype(np.float32)

        patch_CT_pred, MR16_eval= self.sess.run([self.G,self.layer],
                        feed_dict={ self.inputMR: patch_MR, self.train_phase: False})

        patch_CT_pred=np.squeeze(patch_CT_pred)#[Z,H,W]
        #imsave('mr32.png',np.squeeze(MR16_eval[0,:,:,2]))
        #imsave('ctpred.png',np.squeeze(patch_CT_pred[0,:,:,0]))
        #print 'mean of layer  ',np.mean(MR16_eval)
        #print 'min ct estimated ',np.min(patch_CT_pred)
        #print 'max ct estimated ',np.max(patch_CT_pred)
        #print 'mean of ctpatch estimated ',np.mean(patch_CT_pred)
        return patch_CT_pred


    def test_1_subject(self,MR_image,CT_GT,MR_patch_sz,CT_patch_sz,step):
        """
            receives an MR image and returns an estimated CT image of the same size
        """
        matFA=MR_image
        matSeg=CT_GT
        dFA=MR_patch_sz
        dSeg=CT_patch_sz

        eps=1e-5
        [row,col,leng]=matFA.shape
        margin1=int((dFA[0]-dSeg[0])/2)
        margin2=int((dFA[1]-dSeg[1])/2)
        margin3=int((dFA[2]-dSeg[2])/2)
        cubicCnt=0
        marginD=[margin1,margin2,margin3]
        print 'matFA shape is ',matFA.shape
        matFAOut=np.zeros([row+2*marginD[0],col+2*marginD[1],leng+2*marginD[2]])
        print 'matFAOut shape is ',matFAOut.shape
        matFAOut[marginD[0]:row+marginD[0],marginD[1]:col+marginD[1],marginD[2]:leng+marginD[2]]=matFA

        matFAOut[0:marginD[0],marginD[1]:col+marginD[1],marginD[2]:leng+marginD[2]]=matFA[0:marginD[0],:,:] #we'd better flip it along the first dimension
        matFAOut[row+marginD[0]:matFAOut.shape[0],marginD[1]:col+marginD[1],marginD[2]:leng+marginD[2]]=matFA[row-marginD[0]:matFA.shape[0],:,:] #we'd better flip it along the 1st dimension

        matFAOut[marginD[0]:row+marginD[0],0:marginD[1],marginD[2]:leng+marginD[2]]=matFA[:,0:marginD[1],:] #we'd better flip it along the 2nd dimension
        matFAOut[marginD[0]:row+marginD[0],col+marginD[1]:matFAOut.shape[1],marginD[2]:leng+marginD[2]]=matFA[:,col-marginD[1]:matFA.shape[1],:] #we'd better to flip it along the 2nd dimension

        matFAOut[marginD[0]:row+marginD[0],marginD[1]:col+marginD[1],0:marginD[2]]=matFA[:,:,0:marginD[2]] #we'd better flip it along the 3rd dimension
        matFAOut[marginD[0]:row+marginD[0],marginD[1]:col+marginD[1],marginD[2]+leng:matFAOut.shape[2]]=matFA[:,:,leng-marginD[2]:matFA.shape[2]]


        matOut=np.zeros((matSeg.shape[0],matSeg.shape[1],matSeg.shape[2]))
        used=np.zeros((matSeg.shape[0],matSeg.shape[1],matSeg.shape[2]))+eps
        #fid=open('trainxxx_list.txt','a');
        print 'last i ',row-dSeg[0]
        for i in range(0,row-dSeg[0]+1,step[0]):
            print 'i ',i
            for j in range(0,col-dSeg[1]+1,step[1]):
                for k in range(0,leng-dSeg[2]+1,step[2]):
                    volSeg=matSeg[i:i+dSeg[0],j:j+dSeg[1],k:k+dSeg[2]]
                    #print 'volSeg shape is ',volSeg.shape
                    volFA=matFAOut[i:i+dSeg[0]+2*marginD[0],j:j+dSeg[1]+2*marginD[1],k:k+dSeg[2]+2*marginD[2]]
                    #print 'volFA shape is ',volFA.shape
                    #mynet.blobs['dataMR'].data[0,0,...]=volFA
                    #mynet.forward()
                    #temppremat = mynet.blobs['softmax'].data[0].argmax(axis=0) #Note you have add softmax layer in deploy prototxt
                    temppremat=self.evaluate(volFA)
                    if len(temppremat.shape)==2:
                        temppremat=np.expand_dims(temppremat,axis=2)
                    #print 'patchout shape ',temppremat.shape
                    #temppremat=volSeg
                    matOut[i:i+dSeg[0],j:j+dSeg[1],k:k+dSeg[2]]=matOut[i:i+dSeg[0],j:j+dSeg[1],k:k+dSeg[2]]+temppremat;
                    used[i:i+dSeg[0],j:j+dSeg[1],k:k+dSeg[2]]=used[i:i+dSeg[0],j:j+dSeg[1],k:k+dSeg[2]]+1;
        matOut=matOut/used
        return matOut


            
    def save(self, checkpoint_dir, step):
        model_name = "MR2CT.model"
        
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoints...")

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
            return True
        else:
            return False

    def combined_loss_G(self,batch_size_tf):
        """
        Calculates the sum of the combined adversarial, lp and GDL losses in the given proportion. Used
        for training the generative model.

        @param gen_frames: A list of tensors of the generated frames at each scale.
        @param gt_frames: A list of tensors of the ground truth frames at each scale.
        @param d_preds: A list of tensors of the classifications made by the discriminator model at each
                        scale.
        @param lam_adv: The percentage of the adversarial loss to use in the combined loss.
        @param lam_lp: The percentage of the lp loss to use in the combined loss.
        @param lam_gdl: The percentage of the GDL loss to use in the combined loss.
        @param l_num: 1 or 2 for l1 and l2 loss, respectively).
        @param alpha: The power to which each gradient term is raised in GDL loss.

        @return: The combined adversarial, lp and GDL losses.

        """


        lpterm=lp_loss(self.G, self.CT_GT, self.l_num, batch_size_tf)
        gdlterm=gdl_loss(self.G, self.CT_GT, self.alpha,batch_size_tf)
        bceterm=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.D_logits_, tf.ones_like(self.D_)))
        loss_=self.lam_lp*lpterm + self.lam_gdl*gdlterm + self.lam_adv*bceterm
        tf.add_to_collection('losses', loss_)
        loss = tf.add_n(tf.get_collection('losses'), name='total_loss')

        return loss, lpterm, gdlterm, bceterm
