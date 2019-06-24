import tensorflow as tf
import time
import os
import numpy as np

from utils import prepare_train_data

def train(model, config):
    input_, label_ = prepare_train_data(config)

    model.train_op = tf.train.AdamOptimizer(learning_rate=config.learning_rate).minimize(model.loss)
    tf.initialize_all_variables().run()

    counter = 0
    time_ = time.time()

    model.load("checkpoint")

    print("Starting to train on {} images".format(input_.shape))
    for ep in range(config.epoch):
        batch_i = len(input_) // config.batch_size
        for idx in range(0, batch_i):
            batch_images = input_[idx * config.batch_size : (idx + 1) * config.batch_size]
            batch_labels = label_[idx * config.batch_size : (idx + 1) * config.batch_size]
            counter += 1
            _, err = model.sess.run([model.train_op, model.loss], feed_dict={model.images: batch_images, model.labels: batch_labels})
            if counter % 100 == 0:
                print("Epoch: [%2d], step: [%2d], time: [%4.4f], loss: [%.8f]" % ((ep+1), counter, time.time()-time_, err))
            if counter % 1000 == 0:
                model.save("checkpoint", counter)
