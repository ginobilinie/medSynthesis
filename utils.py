import tensorflow as tf
import numpy as np
import os
import SimpleITK as sitk
import h5py
from tensorflow.contrib.layers.python.layers import batch_norm as batch_norm

ignore_label=0

'''
This class implements data generator, some evaluation metrics, the testing function and so on.
By Roger Trullo and Dong Nie
Oct., 2016
'''


def psnr(ct_generated,ct_GT):
    print ct_generated.shape
    print ct_GT.shape

    mse=np.sqrt(np.mean((ct_generated-ct_GT)**2))
    print 'mse ',mse
    max_I=np.max([np.max(ct_generated),np.max(ct_GT)])
    print 'max_I ',max_I
    return 20.0*np.log10(max_I/mse)

def dice(im1, im2,organid):
    """
    Computes the Dice coefficient, a measure of set similarity.
    Parameters
    ----------
    im1 : array-like, bool
        Any array of arbitrary size. If not boolean, will be converted.
    im2 : array-like, bool
        Any other array of identical size. If not boolean, will be converted.
    Returns
    -------
    dice : float
        Dice coefficient as a float on range [0,1].
        Maximum similarity = 1
        No similarity = 0
        
    Notes
    -----
    The order of inputs for `dice` is irrelevant. The result will be
    identical if `im1` and `im2` are switched.
    """
    im1=im1==organid
    im2=im2==organid
    im1 = np.asarray(im1).astype(np.bool)
    im2 = np.asarray(im2).astype(np.bool)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    # Compute Dice coefficient
    intersection = np.logical_and(im1, im2)

    return 2. * intersection.sum() / (im1.sum() + im2.sum())


def Generator_2D_slices(path_patients,batchsize):
    #path_patients='/home/dongnie/warehouse/CT_patients/test_set/'
    print path_patients
    patients = os.listdir(path_patients)#every file  is a hdf5 patient
    while True:
        
        for idx,namepatient in enumerate(patients):
            print namepatient            
            f=h5py.File(os.path.join(path_patients,namepatient))
            dataMRptr=f['dataMR']
            dataMR=dataMRptr.value
            
            dataCTptr=f['dataCT']
            dataCT=dataCTptr.value

            dataMR=np.squeeze(dataMR)
            dataCT=np.squeeze(dataCT)

            #print 'mr shape h5 ',dataMR.shape#B,H,W,C
            #print 'ct shape h5 ',dataCT.shape#B,H,W
            
            shapedata=dataMR.shape
            #Shuffle data
            idx_rnd=np.random.choice(shapedata[0], shapedata[0], replace=False)
            dataMR=dataMR[idx_rnd,...]
            dataCT=dataCT[idx_rnd,...]
            modulo=np.mod(shapedata[0],batchsize)
################## always the number of samples will be a multiple of batchsz##########################3            
            if modulo!=0:
                to_add=batchsize-modulo
                inds_toadd=np.random.randint(0,dataMR.shape[0],to_add)
                X=np.zeros((dataMR.shape[0]+to_add,dataMR.shape[1],dataMR.shape[2],dataMR.shape[3]))#dataMR
                X[:dataMR.shape[0],...]=dataMR
                X[dataMR.shape[0]:,...]=dataMR[inds_toadd]                
                
                y=np.zeros((dataCT.shape[0]+to_add,dataCT.shape[1],dataCT.shape[2]))#dataCT
                y[:dataCT.shape[0],...]=dataCT
                y[dataCT.shape[0]:,...]=dataCT[inds_toadd]
                
            else:
                X=np.copy(dataMR)                
                y=np.copy(dataCT)

            #X = np.expand_dims(X, axis=3)    
            X=X.astype(np.float32)
            y=np.expand_dims(y, axis=3)#B,H,W,C
            y=y.astype(np.float32)
            #y[np.where(y==5)]=0
            print 'y shape ', y.shape                   
            for i_batch in xrange(int(X.shape[0]/batchsize)):
                yield (X[i_batch*batchsize:(i_batch+1)*batchsize,...],  y[i_batch*batchsize:(i_batch+1)*batchsize,...])



def Generator_3D_patches(path_patients,batchsize):
    #path_patients='/home/dongnie/warehouse/CT_patients/test_set/'
    print path_patients
    patients = os.listdir(path_patients)#every file  is a hdf5 patient
    while True:
        
        for idx,namepatient in enumerate(patients):
            print namepatient            
            f=h5py.File(os.path.join(path_patients,namepatient))
            dataMRptr=f['dataMR']
            dataMR=dataMRptr.value
            #dataMR=np.squeeze(dataMR)
            
            dataCTptr=f['dataCT']
            dataCT=dataCTptr.value
            #dataCT=np.squeeze(dataCT)

            dataMR=np.squeeze(dataMR)
            dataCT=np.squeeze(dataCT)
            print 'mr shape h5 ',dataMR.shape

            
            shapedata=dataMR.shape
            #Shuffle data
            idx_rnd=np.random.choice(shapedata[0], shapedata[0], replace=False)
            dataMR=dataMR[idx_rnd,...]
            dataCT=dataCT[idx_rnd,...]
            modulo=np.mod(shapedata[0],batchsize)
################## always the number of samples will be a multiple of batchsz##########################3            
            if modulo!=0:
                to_add=batchsize-modulo
                inds_toadd=np.random.randint(0, dataMR.shape[0], to_add)
                X=np.zeros((dataMR.shape[0]+to_add, dataMR.shape[1], dataMR.shape[2], dataMR.shape[3]))#dataMR
                X[:dataMR.shape[0],...]=dataMR
                X[dataMR.shape[0]:,...]=dataMR[inds_toadd]                
                
                y=np.zeros((dataCT.shape[0]+to_add, dataCT.shape[1], dataCT.shape[2], dataCT.shape[3]))#dataCT
                y[:dataCT.shape[0],...]=dataCT
                y[dataCT.shape[0]:,...]=dataCT[inds_toadd]
                
            else:
                X=np.copy(dataMR)                
                y=np.copy(dataCT)

            X = np.expand_dims(X, axis=4)     
            X=X.astype(np.float32)
            y=np.expand_dims(y, axis=4)
            y=y.astype(np.float32)
            
            print 'y shape ', y.shape
            print 'X shape ', X.shape                 
            for i_batch in xrange(int(X.shape[0]/batchsize)):
                yield (X[i_batch*batchsize:(i_batch+1)*batchsize,...],  y[i_batch*batchsize:(i_batch+1)*batchsize,...])


def get_test_slices(path_test):
    """
    Gets a clip from the test dataset.

    @param test_batch_size: The number of clips.
    @param num_rec_out: The number of outputs to predict. Outputs > 1 are computed recursively,
                        using the previously-generated frames as input. Default = 1.

    @return: An array of shape:
             [test_batch_size, c.TEST_HEIGHT, c.TEST_WIDTH, (3 * (c.HIST_LEN + num_rec_out))].
             A batch of frame sequences with values normalized in range [-1, 1].
    """
    patients = os.listdir(path_test)#every file  is a hdf5 patient
    idx=np.random.choice(len(patients), 1, replace=False)
    f=h5py.File(os.path.join(path_test,patients[idx]))
    dataMRptr=f['dataMR']
    dataMR=dataMRptr.value
            
    dataCTptr=f['dataCT']
    dataCT=dataCTptr.value
    dataCT=np.expand_dims(dataCT,3)
    return [dataMR,dataCT]


#name, shape=shape,dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer_conv2d()

def conv_op(input_op, name, kw, kh, n_out, dw, dh,wd,padding,activation=True):
    n_in = input_op.get_shape()[-1].value
    shape=[kh, kw, n_in, n_out]
    with tf.variable_scope(name):
    	kernel=_variable_with_weight_decay("w", shape, wd)
        conv = tf.nn.conv2d(input_op, kernel, (1, dh, dw, 1), padding=padding)
        bias_init_val = tf.constant(0.0, shape=[n_out], dtype=tf.float32)
        biases = tf.get_variable(initializer=bias_init_val, trainable=True, name='b')
        z = tf.nn.bias_add(conv, biases)
        if activation:
            z=tf.nn.relu(z, name='Activation')
        return z

def conv_op_bn(input_op, name, kw, kh, n_out, dw, dh, wd, padding,train_phase):
    n_in = input_op.get_shape()[-1].value
    shape=[kh, kw, n_in, n_out]
    scope_bn=name+'_bn'
    with tf.variable_scope(name):
        kernel=_variable_with_weight_decay("w", shape, wd)
        conv = tf.nn.conv2d(input_op, kernel, (1, dh, dw, 1), padding=padding)
        bias_init_val = tf.constant(0.0, shape=[n_out], dtype=tf.float32)
        biases = tf.get_variable(initializer=bias_init_val, trainable=True, name='b')
        out_conv = tf.nn.bias_add(conv, biases)
        z=batch_norm_layer(out_conv,train_phase,scope_bn)
        #activation = tf.nn.relu(z, name='Activation')
        return z


def mpool_op(input_op, name, kh, kw, dh, dw):
    return tf.nn.max_pool(input_op,
                          ksize=[1, kh, kw, 1],
                          strides=[1, dh, dw, 1],
                          padding='SAME',
                          name=name)


def conv_op_3d(input_op, name, kw, kh, kz, n_out, dw, dh, dz, wd, padding):
    n_in = input_op.get_shape()[-1].value
    shape=[kz, kh, kw, n_in, n_out]
    with tf.variable_scope(name):
        kernel=_variable_with_weight_decay("w", shape, wd)
        conv = tf.nn.conv3d(input_op, kernel, (1, dz, dh, dw, 1), padding=padding)
        bias_init_val = tf.constant(0.0, shape=[n_out], dtype=tf.float32)
        biases = tf.get_variable(initializer=bias_init_val, trainable=True, name='b')
        z = tf.nn.bias_add(conv, biases)
        activation = tf.nn.relu(z, name='Activation')
        return activation


def conv_op_3d_bn(input_op, name, kw, kh, kz, n_out, dw, dh, dz, wd, padding,train_phase):
    n_in = input_op.get_shape()[-1].value
    shape=[kz, kh, kw, n_in, n_out]
    scope_bn=name+'_bn'
    with tf.variable_scope(name):
        kernel=_variable_with_weight_decay("w", shape, wd)
        conv = tf.nn.conv3d(input_op, kernel, (1, dz, dh, dw, 1), padding=padding)
        bias_init_val = tf.constant(0.0, shape=[n_out], dtype=tf.float32)
        biases = tf.get_variable(initializer=bias_init_val, trainable=True, name='b')
        out_conv = tf.nn.bias_add(conv, biases)
        z=batch_norm_layer(out_conv,train_phase,scope_bn)
        #activation = tf.nn.relu(z, name='Activation')
        return z

def conv_op_3d_norelu(input_op, name, kw, kh, kz, n_out, dw, dh, dz, wd, padding):
    n_in = input_op.get_shape()[-1].value
    shape=[kz, kh, kw, n_in, n_out]
    with tf.variable_scope(name):
        kernel=_variable_with_weight_decay("w", shape, wd)
        conv = tf.nn.conv3d(input_op, kernel, (1, dz, dh, dw, 1), padding=padding)
        bias_init_val = tf.constant(0.0, shape=[n_out], dtype=tf.float32)
        biases = tf.get_variable(initializer=bias_init_val, trainable=True, name='b')
        z = tf.nn.bias_add(conv, biases)
        #activation = tf.nn.relu(z, name='Activation')
        return z

def deconv_op_3d(input_op, name, kw, kh, kz, n_out, wd, batchsize):
    n_in = input_op.get_shape()[-1].value
    shape=[kz, kh, kw, n_in, n_out]

    zin=input_op.get_shape()[1].value
    hin=input_op.get_shape()[2].value
    win=input_op.get_shape()[3].value
    output_shape=[batchsize, 2*zin, 2*hin, 2*win, n_out]
    with tf.variable_scope(name):
        kernel=_variable_with_weight_decay("w", shape, wd)
        conv =  tf.nn.conv3d_transpose(input_op, kernel, output_shape,strides=[1, 2, 2, 2, 1], padding='SAME')
        bias_init_val = tf.constant(0.0, shape=[n_out], dtype=tf.float32)
        biases = tf.get_variable(initializer=bias_init_val, trainable=True, name='b')
        z = tf.nn.bias_add(conv, biases)
        activation = tf.nn.relu(z, name='Activation')
        return activation

def conv_op_norelu(input_op, name, kw, kh, n_out, dw, dh,wd):
    n_in = input_op.get_shape()[-1].value
    shape=[kh, kw, n_in, n_out]
    with tf.variable_scope(name):
        kernel=_variable_with_weight_decay("w", shape, wd)
        conv = tf.nn.conv2d(input_op, kernel, (1, dh, dw, 1), padding='SAME')
        bias_init_val = tf.constant(0.0, shape=[n_out], dtype=tf.float32)
        biases = tf.get_variable(initializer=bias_init_val, trainable=True, name='b')
        z = tf.nn.bias_add(conv, biases)      
        return z
    
def deconv_op(input_op, name, kw, kh, n_out, wd, batchsize):
    n_in = input_op.get_shape()[-1].value
    shape=[kh, kw, n_out, n_in]
    #batchsize=input_op.get_shape()[0].value
    hin=input_op.get_shape()[1].value
    win=input_op.get_shape()[2].value
    output_shape=[batchsize, 2*hin, 2*win, n_out]
    with tf.variable_scope(name):
        kernel = _variable_with_weight_decay("w", shape, wd)
        conv =  tf.nn.conv2d_transpose(input_op, kernel, output_shape,strides=[1, 2, 2, 1], padding='SAME')
        bias_init_val = tf.constant(0.0, shape=[n_out], dtype=tf.float32)
        biases = tf.get_variable(initializer=bias_init_val, trainable=True, name='b')
        z = tf.nn.bias_add(conv, biases)
        activation = tf.nn.relu(z, name='Activation')
        return activation

def deconv_op_norelu(input_op, name, kw, kh, n_out, wd):
    n_in = input_op.get_shape()[-1].value
    shape=[kh, kw, n_out, n_in]
    batchsize=input_op.get_shape()[0].value
    hin=input_op.get_shape()[1].value
    win=input_op.get_shape()[2].value
    output_shape=[batchsize, 2*hin, 2*win, n_out]
    with tf.variable_scope(name):
        kernel = _variable_with_weight_decay("w", shape, wd)
        conv =  tf.nn.conv2d_transpose(input_op, kernel, output_shape,strides=[1, 2, 2, 1], padding='SAME')
        bias_init_val = tf.constant(0.0, shape=[n_out], dtype=tf.float32)
        biases = tf.get_variable(initializer=bias_init_val, trainable=True, name='b')
        z = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())
        return z


def concatenate_op(input_op1,input_op2,name):
    return tf.concat(3,[input_op1,input_op2],name=name)


def upsample_op(input_op,name):
    height=input_op.get_shape()[1].value
    width=input_op.get_shape()[2].value
    return tf.image.resize_nearest_neighbor(input_op, size=[2*height, 2*width],name=name)

#here I writhe 3d pooling
def mpool_op_3d(input_op, name, kh, kw, kz, dh, dw, dz):
    return tf.nn.max_pool_3d(input_op,
                          ksize=[1, kh, kw, kz, 1],
                          strides=[1, dh, dw, dz, 1],
                          padding='SAME',
                          name=name)



def fullyconnected_op(input_op, name, n_out, wd, activation=True):
    im_shape = input_op.get_shape().as_list()
    assert len(im_shape) > 1, "Input Tensor shape must be at least 2-D: batch, ninputs"
    n_inputs = int(np.prod(im_shape[1:])) #units at lower layer
    shape=[n_inputs, n_out]
    with tf.variable_scope(name):
        W=_variable_with_weight_decay("w", shape, wd)
        print W.name
        bias_init_val = tf.constant(0.0, shape=[n_out], dtype=tf.float32)
        biases = tf.get_variable(initializer=bias_init_val, trainable=True, name ='b')
        if len(im_shape) > 2: #we have to flatten it then
            x = tf.reshape(input_op, [-1, n_inputs])
        else:
            x=input_op
        z = tf.matmul(x, W)+biases
        if activation:
            z = tf.nn.relu(z)

    return z

def _variable_with_weight_decay(name, shape, wd):
  """Helper to create an initialized Variable with weight decay.
  Note that the Variable is initialized with a truncated normal distribution.
  A weight decay is added only if one is specified.
  Args:
    name: name of the variable
    shape: list of ints
    stddev: standard deviation of a truncated Gaussian
    wd: add L2Loss weight decay multiplied by this float. If None, weight
        decay is not added for this Variable.
  Returns:
    Variable Tensor
  """
  #tf.contrib.layers.xavier_initializer_conv2d()
  [fan_in, fan_out]=get_fans(shape)
  initializer=xavier_init(fan_in, fan_out)
  
  var=tf.get_variable(name, shape=shape,dtype=tf.float32, initializer=initializer)
  weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
  tf.add_to_collection('losses', weight_decay)

  return var


def get_fans(shape):
    receptive_field_size = np.prod(shape[:-2])
    fan_in = shape[-2] * receptive_field_size
    fan_out = shape[-1] * receptive_field_size
        
    return fan_in, fan_out


def xavier_init(n_inputs, n_outputs, uniform=True):

    """Set the parameter initialization using the method described.
    This method is designed to keep the scale of the gradients roughly the same
    in all layers.
    Xavier Glorot and Yoshua Bengio (2010):
           Understanding the difficulty of training deep feedforward neural
           networks. International conference on artificial intelligence and
           statistics.
    Args:
    n_inputs: fan_in
    n_outputs: fan_out
    uniform: If true use a uniform distribution, otherwise use a normal.
    Returns:
    An initializer.
    """
    if uniform:
        # 6 was used in the paper.
        init_range = np.sqrt(6.0 / (n_inputs + n_outputs))
        return tf.random_uniform_initializer(-init_range, init_range)
    else:
        # 3 gives us approximately the same limits as above since this repicks
        # values greater than 2 standard deviations from the mean.
        stddev = np.sqrt(3.0 / (n_inputs + n_outputs))
        return tf.truncated_normal_initializer(stddev=stddev)

def batch_norm_layer(x,train_phase,scope_bn):
    outputs = tf.contrib.layers.batch_norm(x, is_training=train_phase, center=False, scale=False, activation_fn=tf.nn.relu, updates_collections=None, scope='batch_norm')
    return outputs
    # def batch_norm_layer(x,train_phase,scope_bn):
#     bn_train = batch_norm(x, decay=0.999, center=True, scale=True,
#     updates_collections=None,
#     is_training=True,
#     reuse=None, # is this right?
#     trainable=True,
#     scope=scope_bn)
#     bn_inference = batch_norm(x, decay=0.999, center=True, scale=True,
#     updates_collections=None,
#     is_training=False,
#     reuse=True, # is this right?
#     trainable=True,
#     scope=scope_bn)
#     z = tf.cond(train_phase, lambda: bn_train, lambda: bn_inference)
#     return z