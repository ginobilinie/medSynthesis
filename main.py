import os
import tensorflow as tf
import pprint
from g_model import MR2CT

pp = pprint.PrettyPrinter()

flags = tf.app.flags
flags.DEFINE_integer("iterations", 500000, "Epoch to train [50000]")
flags.DEFINE_float("learning_rate", 1e-6, "Learning rate of for SGD [1e-6]")
flags.DEFINE_integer("batch_size", 10, "The size of batch images [10]")
flags.DEFINE_integer("show_every", 100, "The size of batch images [100]")
flags.DEFINE_integer("save_every", 2000, "The size of batch images [2000]")
flags.DEFINE_integer("test_every", 10, "test every [10000] iterations the subject")
flags.DEFINE_integer("l_num", 2, "The l norm value, either 1 or 2 [2]")
flags.DEFINE_integer("lr_step", 30000, "The step to decrease lr [lr_step]")
flags.DEFINE_integer("sizeMR", 64, "The size of MR patch [32]")
flags.DEFINE_integer("input_slices", 5, "The numer of slice in the MR patch [5]")
flags.DEFINE_integer("sizeCT", 48, "The size of CT patch [24]")
flags.DEFINE_float("frac_crop", 0.5, "The fraction to crop the patch [0.5] will convert from 64 to 32")
flags.DEFINE_float("wd", 0.0005, "weight decay [0.0005] ")
flags.DEFINE_float("lam_lp", 1.0, "weight lp loss [1.0] ")
flags.DEFINE_float("lam_gdl", 1.0, "weight gdl loss [1.0] ")
flags.DEFINE_float("lam_adv", 0.05, "weight adv loss [0.05] ")
flags.DEFINE_float("alpha", 2, "alpha for the gdl loss [2] ")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("path_patients_h5", "/raid/dongnie/brainData",
	 "Directory where the h5 files are located ['/raid/dongnie/brainData']")
FLAGS = flags.FLAGS

'''
This is the main entrance which you run the medical image sysnthesis task.
By Roger Trullo and Dong Nie
Oct., 2016
'''

def main(_):
    pp.pprint(flags.FLAGS.__flags)

    #if not os.path.exists(FLAGS.checkpoint_dir):
    #    os.makedirs(FLAGS.checkpoint_dir)
    config = tf.ConfigProto(allow_soft_placement=True)
    with tf.Session(config=config) as sess:
        with tf.device('/gpu:0'):
            gen_model = MR2CT(sess, batch_size=FLAGS.batch_size, height_MR=FLAGS.sizeMR, width_MR=FLAGS.sizeMR,
                    height_CT=FLAGS.sizeCT, width_CT=FLAGS.sizeCT, l_num=FLAGS.l_num, wd=FLAGS.wd,
                    checkpoint_dir=FLAGS.checkpoint_dir, path_patients_h5=FLAGS.path_patients_h5, learning_rate=FLAGS.learning_rate,
                    lr_step=FLAGS.lr_step,lam_lp=FLAGS.lam_lp, lam_gdl=FLAGS.lam_gdl, lam_adv=FLAGS.lam_adv, alpha=FLAGS.alpha)
            gen_model.train(FLAGS)
if __name__ == '__main__':
    tf.app.run()
