import tensorflow as tf
import time
import os
import numpy as np

from utils import prepare_test_data, merge, imsave, psnr, postprocess

def test(model, config):
    input_, label_, nx, ny, original_image = prepare_test_data(config, config.image_name)

    tf.initialize_all_variables().run()

    print("Now Start Testing...")
    model.load("checkpoint")
    result = model.pred.eval({model.images: input_})
    #print(label_[1] - result[1])
    print(result.shape, np.squeeze(result).shape)
    
    predicted_image = merge(result, [nx, ny], original_image.shape)
    #image_LR = merge(input_, [nx, ny], self.c_dim)
    #checkimage(image_LR)

    img = postprocess(predicted_image)
    imsave(img, 'result/' + "srcnn-" + config.image_name)
    print("PSNR", psnr(img, original_image))
