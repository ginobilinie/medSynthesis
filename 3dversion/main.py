import os
import tensorflow as tf
import pprint
from g_model import MR2CT

pp = pprint.PrettyPrinter()

flags = tf.app.flags
flags.DEFINE_integer("iterations", 100000, "Epoch to train [10000]")
flags.DEFINE_float("learning_rate", 2e-7, "Learning rate of for SGD [2e-7]")
flags.DEFINE_integer("batch_size", 20, "The size of batch images [10]")
flags.DEFINE_integer("show_every", 100, "The size of batch images [10]")
flags.DEFINE_integer("save_every", 1000, "The size of batch images [1000]")
flags.DEFINE_integer("test_every", 5000, "test every [5000] iterations the subject")
flags.DEFINE_integer("l_num", 2, "The l norm value, either 1 or 2 [2]")
flags.DEFINE_integer("sizeMR", 32, "The size of MR patch [32]")
flags.DEFINE_integer("input_slices", 5, "The numer of slice in the MR patch [5]")
flags.DEFINE_integer("sizeCT", 24, "The size of CT patch [24]")
flags.DEFINE_float("frac_crop", 0.5, "The fraction to crop the patch [0.5] will convert from 64 to 32")
flags.DEFINE_float("wd", 0.0005, "weight decay [0.0005] ")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("path_patients_h5", "/home/dongnie/warehouse/prostate/MR32CT24_gan3d",
	 "Directory where the h5 files are located ['/home/dongnie/warehouse/prostate/MR32CT24_gan3d']")
FLAGS = flags.FLAGS


def main(_):
    pp.pprint(flags.FLAGS.__flags)

    #if not os.path.exists(FLAGS.checkpoint_dir):
    #    os.makedirs(FLAGS.checkpoint_dir)

    with tf.Session() as sess:
    	gen_model = MR2CT(sess, batch_size=FLAGS.batch_size, depth_MR=FLAGS.sizeMR, height_MR=FLAGS.sizeMR, width_MR=FLAGS.sizeMR,
                    depth_CT=FLAGS.sizeCT, height_CT=FLAGS.sizeCT, width_CT=FLAGS.sizeCT, l_num=FLAGS.l_num, wd=FLAGS.wd,
                    checkpoint_dir=FLAGS.checkpoint_dir, path_patients_h5=FLAGS.path_patients_h5, learning_rate=FLAGS.learning_rate)
    	gen_model.train(FLAGS)
if __name__ == '__main__':
    tf.app.run()