import tensorflow as tf
import numpy as np


'''
All kinds of loss functions
By Roger Trullo and Dong Nie
Oct., 2016
'''

def lp_loss(ct_generated, gt_ct, l_num, batch_size_tf):
    """
    Calculates the sum of lp losses between the predicted and ground truth images.

    @param ct_generated: The predicted ct
    @param gt_ct: The ground truth ct
    @param l_num: 1 or 2 for l1 and l2 loss, respectively).

    @return: The lp loss.
    """
    lp_loss=tf.reduce_sum(tf.abs(ct_generated - gt_ct)**l_num)/(2*tf.cast(batch_size_tf,tf.float32))
    #print 'lp_loss ',gt_ct.get_shape()
    tf.add_to_collection('losses', lp_loss)

    loss = tf.add_n(tf.get_collection('losses'), name='total_loss')
    return loss

def gdl_loss(gen_CT, gt_CT, alpha, batch_size_tf):
    """
    Calculates the sum of GDL losses between the predicted and ground truth images.

    @param gen_CT: The predicted CTs.
    @param gt_CT: The ground truth images
    @param alpha: The power to which each gradient term is raised.
    @param batch_size_tf batch size
    @return: The GDL loss.
    """
    # calculate the loss for each scale

    # create filters [-1, 1] and [[1],[-1]] for diffing to the left and down respectively.
    pos = tf.constant(np.identity(1), dtype=tf.float32)
    neg = -1 * pos
    filter_x = tf.expand_dims(tf.pack([neg, pos]), 0)  # [-1, 1]
    filter_y = tf.pack([tf.expand_dims(pos, 0), tf.expand_dims(neg, 0)])  # [[1],[-1]]
    strides = [1, 1, 1, 1]  # stride of (1, 1)
    padding = 'SAME'

    gen_dx = tf.abs(tf.nn.conv2d(gen_CT, filter_x, strides, padding=padding))
    gen_dy = tf.abs(tf.nn.conv2d(gen_CT, filter_y, strides, padding=padding))
    gt_dx = tf.abs(tf.nn.conv2d(gt_CT, filter_x, strides, padding=padding))
    gt_dy = tf.abs(tf.nn.conv2d(gt_CT, filter_y, strides, padding=padding))

    grad_diff_x = tf.abs(gt_dx - gen_dx)
    grad_diff_y = tf.abs(gt_dy - gen_dy)

    gdl=tf.reduce_sum((grad_diff_x ** alpha + grad_diff_y ** alpha))/tf.cast(batch_size_tf,tf.float32)

    return gdl








def combined_loss(gen_CT, gt_CT, d_preds, lam_adv=1, lam_lp=1, lam_gdl=1, l_num=2, alpha=2):
    """
    Computes the weighted sum of the combined adversarial, lp and GDL losses.

    @param gen_CT: The predicted CTs.
    @param gt_CT: The ground truth images
    @param d_preds: classifications made by the discriminator mode.
    @param lam_adv: The weight of the adversarial loss.
    @param lam_lp: The weight of the lp loss.
    @param lam_gdl: The weight of the GDL loss.
    @param l_num: 1 or 2 for l1 and l2 loss, respectively).
    @param alpha: The power to which each gradient term is raised in GDL loss.

    @return: The combined adversarial, lp and GDL losses.
    """
    batch_size = tf.shape(gen_CT[0])[0]  # variable batch size as a tensor

    loss = lam_lp * lp_loss(gen_CT, gt_CT, l_num)
    loss += lam_gdl * gdl_loss(gen_CT, gt_CT, alpha)
    if c.ADVERSARIAL: loss += lam_adv * adv_loss(d_preds, tf.ones([batch_size, 1]))

    return loss



def cross_entropy_Discriminator(logits_D,gt_D):
    """
    logits_D is the output of the discriminator [batch_size,1]
    gt_D should be all ones for real data, and all zeros for fake-
    generated (output of generator) data[batch_size,1]"""

    bce=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits_D, gt_D))
    return bce



def bce_loss(preds, targets):
    """
    Calculates the sum of binary cross-entropy losses between predictions and ground truths.

    @param preds: A 1xN tensor. The predicted classifications of each frame.
    @param targets: A 1xN tensor The target labels for each frame. (Either 1 or -1). Not "truths"
                    because the generator passes in lies to determine how well it confuses the
                    discriminator.

    @return: The sum of binary cross-entropy losses.
    """
    return tf.squeeze(-1 * (tf.matmul(targets, log10(preds), transpose_a=True) +
                            tf.matmul(1 - targets, log10(1 - preds), transpose_a=True)))


def adv_loss(preds, labels):
    """
    Computes the sum of BCE losses between the predicted classifications and true labels.

    @param preds: The predicted classifications.
    @param labels: The true labels.

    @return: The adversarial loss.
    """
    # calculate the loss for each scale
    scale_losses = []
    for i in xrange(len(preds)):
        loss = bce_loss(preds[i], labels)
        scale_losses.append(loss)

    # condense into one tensor and avg
    return tf.reduce_mean(tf.pack(scale_losses))
