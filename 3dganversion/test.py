
import numpy as np
import tensorflow as tf

import SimpleITK as sitk
from utils import *

"""
	@author: roger
	This code runs the test manually and iis just to verify			

"""


train_phase=tf.placeholder(tf.bool, name='phase_train')#This is for testing!

def generator(inputMR,batch_size_tf,wd):
	
	######## FCN for the 32x32x32 to 24x24x24 ####################################        
	conv1_a = conv_op_3d_bn(inputMR, name="conv1_a", kh=9, kw=9, kz=9,  n_out=32, dh=1, dw=1, dz=1, wd=wd, padding='VALID',train_phase=train_phase)#30
	conv2_a = conv_op_3d_bn(conv1_a, name="conv2_a", kh=3, kw=3, kz=3,  n_out=32, dh=1, dw=1, dz=1, wd=wd, padding='SAME',train_phase=train_phase)
	conv3_a = conv_op_3d_bn(conv2_a, name="conv3_a", kh=3, kw=3, kz=3,  n_out=32, dh=1, dw=1, dz=1, wd=wd, padding='SAME',train_phase=train_phase)#28
	conv4_a = conv_op_3d_bn(conv3_a, name="conv4_a", kh=3, kw=3, kz=3,  n_out=32, dh=1, dw=1, dz=1, wd=wd, padding='SAME',train_phase=train_phase)#28
	conv5_a = conv_op_3d_bn(conv4_a, name="conv5_a", kh=9, kw=9, kz=9,  n_out=64, dh=1, dw=1, dz=1, wd=wd, padding='VALID',train_phase=train_phase)

	conv6_a = conv_op_3d_bn(conv5_a, name="conv6_a", kh=3, kw=3, kz=3,  n_out=64, dh=1, dw=1, dz=1, wd=wd, padding='SAME',train_phase=train_phase)#26
	conv7_a = conv_op_3d_bn(conv6_a, name="conv7_a", kh=3, kw=3, kz=3,  n_out=64, dh=1, dw=1, dz=1, wd=wd, padding='SAME',train_phase=train_phase)
	conv8_a = conv_op_3d_bn(conv7_a, name="conv8_a", kh=3, kw=3, kz=3,  n_out=32, dh=1, dw=1, dz=1, wd=wd, padding='SAME',train_phase=train_phase)
	        #conv7_a = conv_op_3d_bn(conv6_a, name="conv7_a", kh=3, kw=3, kz=3,  n_out=1, dh=1, dw=1, dz=1, wd=wd, padding='SAME',train_phase=train_phase)#24
	conv9_a = conv_op_3d_norelu(conv8_a, name="conv9_a", kh=3, kw=3, kz=3,  n_out=1, dh=1, dw=1, dz=1, wd=wd, padding='SAME')#24 I modified it here,dong
    #MR_16_downsampled=conv7_a#JUST FOR TEST
	return conv9_a




def evaluate(sess,patch_MR):
	""" patch_MR is a np array of shape [H,W,nchans]
	"""
	patch_MR=np.expand_dims(patch_MR,axis=0)#[1,H,W,nchans]
	patch_MR=np.expand_dims(patch_MR,axis=4)#[1,H,W,nchans]
	#patch_MR=patch_MR.astype(np.float32)

	patch_CT_pred= sess.run(G, feed_dict={inputMR: patch_MR, train_phase: False})

	patch_CT_pred=np.squeeze(patch_CT_pred)#[Z,H,W]
	#imsave('mr32.png',np.squeeze(MR16_eval[0,:,:,2]))
	#imsave('ctpred.png',np.squeeze(patch_CT_pred[0,:,:,0]))
	#print 'mean of layer  ',np.mean(MR16_eval)
	#print 'min ct estimated ',np.min(patch_CT_pred)
	#print 'max ct estimated ',np.max(patch_CT_pred)
	#print 'mean of ctpatch estimated ',np.mean(patch_CT_pred)
	return patch_CT_pred




def test_1_subject(sess,MR_image,CT_GT,MR_patch_sz,CT_patch_sz,step):
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
	            temppremat=evaluate(sess, volFA)
	            #print 'patchout shape ',temppremat.shape
	            #temppremat=volSeg
	            matOut[i:i+dSeg[0],j:j+dSeg[1],k:k+dSeg[2]]=matOut[i:i+dSeg[0],j:j+dSeg[1],k:k+dSeg[2]]+temppremat;
	            used[i:i+dSeg[0],j:j+dSeg[1],k:k+dSeg[2]]=used[i:i+dSeg[0],j:j+dSeg[1],k:k+dSeg[2]]+1;
	matOut=matOut/used
	return matOut


def load(self, checkpoint_dir):
    print(" [*] Reading checkpoints...")

    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        saver.restore(sess, ckpt.model_checkpoint_path)
        return True
    else:
        return False



path_test='/home/dongnie/warehouse/prostate/ganData64to24Test'

model = "./checkpoint"
checkpoint_file = tf.train.latest_checkpoint(model)


wd=0.0005

inputMR=tf.placeholder(tf.float32, shape=[None, 32, 32, 32, 1])
batch_size_tf = tf.shape(inputMR)[0]  #variable batchsize so we can test here
G=generator(inputMR,batch_size_tf,wd)
saver = tf.train.Saver()

with tf.Session() as sess:
	sess.run(tf.initialize_all_variables())

	ckpt = tf.train.latest_checkpoint(model)   
	print(ckpt)
	saver.restore(sess, ckpt) # restore all variables

	mr_test_itk=sitk.ReadImage(os.path.join(path_test,'prostate_1to1_MRI.nii'))
	ct_test_itk=sitk.ReadImage(os.path.join(path_test,'prostate_1to1_CT.nii'))
	mrnp=sitk.GetArrayFromImage(mr_test_itk)
	#mu=np.mean(mrnp)
	#mrnp=(mrnp-mu)/(np.max(mrnp)-np.min(mrnp))
	ctnp=sitk.GetArrayFromImage(ct_test_itk)
	print mrnp.dtype
	print ctnp.dtype
	ct_estimated=test_1_subject(sess,mrnp,ctnp,[32,32,32],[16,16,16],[2,5,5])
	psnrval=psnr(ct_estimated,ctnp)
	print ct_estimated.dtype
	print ctnp.dtype
	print 'psnr= ',psnrval
	volout=sitk.GetImageFromArray(ct_estimated)
	sitk.WriteImage(volout,'ct_estimated_test_script.nii.gz')