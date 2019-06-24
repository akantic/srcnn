import tensorflow as tf
from model import SRCNN
from train import train
from test import test

flags = tf.app.flags
config = flags.FLAGS

flags.DEFINE_integer("epoch", 1500, "Number of epoch")
flags.DEFINE_integer("image_size", 33, "The size of image input")
flags.DEFINE_integer("label_size",  21, "The size of image output")
flags.DEFINE_integer("f1", 9, "f1")
flags.DEFINE_integer("f2", 1, "f2")
flags.DEFINE_integer("f3", 5, "f3")
flags.DEFINE_integer("c", 3, "Image color channels")
flags.DEFINE_boolean("train", False, "Train or not")
flags.DEFINE_integer("scale", 3, "Image scale factor")
flags.DEFINE_integer("stride", 21, "Stride")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Checkpoint directory")
flags.DEFINE_float("learning_rate", 1e-4 , "Learning rate")
flags.DEFINE_string("image_name", "butterfly_GT.bmp", "image to use for testing")

def main(_): 
    with tf.Session() as sess:
        
        if config.train:
            srcnn = SRCNN(sess, image_size = config.image_size, label_size = config.label_size, c = config.c)
            train(srcnn, config)
        else:
            diff = config.image_size - config.label_size
            srcnn = SRCNN(sess, image_size = config.image_size + diff, label_size = config.image_size, c = config.c)
            test(srcnn, config)

if __name__=='__main__':
    tf.app.run() 
