import tensorflow as tf
import os

class SRCNN(object):

    def __init__(self,
                 sess,
                 image_size,
                 label_size,
                 c):
        self.sess = sess
        self.image_size = image_size
        self.label_size = label_size
        self.c = c

        self.images = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, self.c], name='images')
        self.labels = tf.placeholder(tf.float32, [None, self.label_size, self.label_size, self.c], name='labels')
        
        self.pred = self.model()
        self.loss = tf.reduce_mean(tf.square(self.labels - self.pred))
        self.saver = tf.train.Saver()

    def model(self):
        weights = {
            'w1': tf.Variable(tf.random_normal([9, 9, self.c, 64], stddev=1e-3), name='w1'),
            'w2': tf.Variable(tf.random_normal([1, 1, 64, 32], stddev=1e-3), name='w2'),
            'w3': tf.Variable(tf.random_normal([5, 5, 32, self.c], stddev=1e-3), name='w3')
        }

        biases = {
            'b1': tf.Variable(tf.zeros([64], name='b1')),
            'b2': tf.Variable(tf.zeros([32], name='b2')),
            'b3': tf.Variable(tf.zeros([self.c], name='b3'))
        }

        conv1 = tf.nn.relu(tf.nn.conv2d(self.images, weights['w1'], strides=[1,1,1,1], padding='VALID') + biases['b1'])
        conv2 = tf.nn.relu(tf.nn.conv2d(conv1, weights['w2'], strides=[1,1,1,1], padding='VALID') + biases['b2'])
        conv3 = tf.nn.conv2d(conv2, weights['w3'], strides=[1,1,1,1], padding='VALID') + biases['b3'] 
        return conv3  

    def load(self, checkpoint_dir):
        print("\nReading Checkpoints...")
        checkpoint_dir = os.path.join(checkpoint_dir, "srcnn")
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_path = str(ckpt.model_checkpoint_path) 
            self.saver.restore(self.sess, os.path.join(os.getcwd(), ckpt_path))
            print("\n Checkpoint Loading Success! %s\n"% ckpt_path)
        else:
            print("\n! Checkpoint Loading Failed \n")

    def save(self, checkpoint_dir, step):
        model_name = "SRCNN.model"
        checkpoint_dir = os.path.join(checkpoint_dir, "srcnn")

        if not os.path.exists(checkpoint_dir):
             os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)