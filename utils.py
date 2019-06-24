import cv2
import numpy as np
import tensorflow as tf
import os 
import glob
import h5py
import math
import scipy.misc
import scipy.ndimage
from PIL import Image

def modcrop(img, scale =3):
    if len(img.shape) == 3:
        h, w, _ = img.shape
        h = h - np.mod(h, scale)
        w = w - np.mod(w, scale)
        img = img[0:h, 0:w, :]
    else:
        h, w = img.shape
        h = h - np.mod(h, scale)
        w = w - np.mod(w, scale)
        img = img[0:h, 0:w]

    return img

def imread(path):
    img = cv2.imread(path)
    return img

def imsave(image, path):
    cv2.imwrite(os.path.join(os.getcwd(), path), image)

def preprocess(path, scale = 3, save=False):
    img = imread(path)

    label_ = modcrop(img, scale)

    bicbuic_img = scipy.misc.imresize(label_, (1./scale), interp="bicubic")
    input_ = scipy.misc.imresize(bicbuic_img, scale/1., interp="bicubic")

    # bicbuic_img = cv2.resize(label_,None,fx = 1.0/scale ,fy = 1.0/scale, interpolation = cv2.INTER_CUBIC)# Resize by scaling factor
    # input_ = cv2.resize(bicbuic_img,None,fx = scale ,fy=scale, interpolation = cv2.INTER_CUBIC)# Resize by scaling factor

    if (save):
        imsave(input_, "result/bicubic-" + path.split("/")[-1])
        print("Bicubic PSNR", psnr(input_, label_))
    return input_, label_

def postprocess(img):
    x = img < 0
    img[x] = 0
    x = img > 1
    img[x] = 1.0
    img *= 255.
    return img.astype(np.uint8)

def psnr(noisy_image, original_image):
    im1 = cv2.cvtColor(noisy_image.astype(np.uint8), cv2.COLOR_BGR2YCR_CB)
    im2 = cv2.cvtColor(original_image, cv2.COLOR_BGR2YCR_CB)
  
    y1 = im1[:,:,0]
    y2 = im2[:,:,0]

    imdiff = y2-y1
    rmse = math.sqrt(np.mean(imdiff**2))
    psnr = 20*math.log10(255./rmse)
    return psnr

def psnr_two(original_image, noisy_image):
    checkimage(original_image)
    checkimage(noisy_image)
    h, w, c = original_image.shape
    print("MAX: ", np.max(noisy_image), np.max(original_image * 255.))
    mse = np.sum((original_image * 255. - noisy_image) ** 2)/(h*w*c)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

def merge(images, size, shape):
    print("[merge] Starting to merge the final image, sub_images size: {} into original shape: {}".format(size, shape))

    h, w = images.shape[1], images.shape[2]
    img = np.zeros((w*size[1], h*size[0], shape[2]))
    for idx, image in enumerate(images):        
        i = idx % size[0]
        j = idx // size[0]
        img[j * h : j * h + h,i * w : i * w + w, :] = image

    img = img[0:shape[0], 0:shape[1], :]
    return img

def prepare_train_data(config):
    data_dir = os.path.join(os.getcwd(), "Train") 
    data = glob.glob(os.path.join(data_dir, "*.*")) 
    print(data)
    sub_input_sequence = []
    sub_label_sequence = []

    padding = abs(config.image_size - config.label_size) / 2 

    for i in range(len(data)):
        input_, label_ = preprocess(data[i], config.scale) 

        if len(input_.shape) == 3: 
            h, w, c = input_.shape
        else:
            h, w = input_.shape 

        for x in range(0, h - config.image_size + 1, config.stride):
            for y in range(0, w - config.image_size + 1, config.stride):
                sub_input = input_[x: x + config.image_size, y: y + config.image_size] 
                sub_label = label_[x + int(padding): x + int(padding) + config.label_size, y + int(padding): y + int(padding) + config.label_size]
                sub_input = sub_input.reshape([config.image_size, config.image_size, config.c])
                sub_label = sub_label.reshape([config.label_size, config.label_size, config.c])
                sub_input =  sub_input / 255.0
                sub_label =  sub_label / 255.0
                sub_input_sequence.append(sub_input)
                sub_label_sequence.append(sub_label)

    arrinput = np.asarray(sub_input_sequence)
    arrlabel = np.asarray(sub_label_sequence)

    print(arrinput.shape)
    print(arrlabel.shape)

    return arrinput, arrlabel

def prepare_test_data(config, image_name):
    data_dir = os.path.join(os.getcwd(), "Test")
    data = glob.glob(os.path.join(data_dir, image_name))

    print(data)
    sub_input_sequence = []
    sub_label_sequence = []
    help_sequence = []

    input_, label_, = preprocess(data[0], config.scale, True) 
    
    if len(input_.shape) == 3:
        h, w, c = input_.shape
    else:
        h, w = input_.shape

    print(input_.shape)
    nx, ny = 0, 0
    padding = abs(config.image_size - config.label_size) / 2 
    diff = config.image_size - config.label_size
    for x in range(0, h - config.image_size + 1, config.stride + diff):
        for y in range(0, w - config.image_size + 1, config.stride + diff):
            sub_input = input_[x: x + config.image_size, y: y + config.image_size] # 33 * 33
            sub_input = np.pad(sub_input, ((6,6),(6,6),(0,0)), "edge")

            sub_input = sub_input.reshape([config.image_size+12, config.image_size+12, config.c])

            sub_label = label_[x + int(padding): x + int(padding) + config.label_size, y + int(padding): y + int(padding) + config.label_size] # 21 * 21
            sub_label = sub_label.reshape([config.label_size, config.label_size, config.c])

            sub_input =  sub_input / 255.0
            sub_label =  sub_label / 255.0

            sub_input_sequence.append(sub_input)
            sub_label_sequence.append(sub_label)


            if x+2*config.image_size > h:
                sub_input = input_[x + config.image_size: h, y : y + config.image_size] # 33 * 33
                sub_input = np.pad(sub_input, ((0,(x+2*config.image_size)-h), (0,0),(0,0)), "edge")
                sub_input = np.pad(sub_input, ((6,6),(6,6),(0,0)), "edge")
                sub_input =  sub_input / 255.0
                help_sequence.append(sub_input)

            if y+2*config.image_size > w:
                sub_input = input_[x: x + config.image_size, y + config.image_size: w] # 33 * 33
                sub_input = np.pad(sub_input, ((0,0),(0,(y+2*config.image_size)-w),(0,0)), "edge")
                sub_input = np.pad(sub_input, ((6,6),(6,6),(0,0)), "edge")
                sub_input =  sub_input / 255.0
                sub_input_sequence.append(sub_input)

            if (x+2*config.image_size > h) and (y+2*config.image_size > w):
                sub_input = input_[x + config.image_size: h, y + config.image_size: w] # 33 * 33
                sub_input = np.pad(sub_input, ((0,(x+2*config.image_size)-h),(0,(y+2*config.image_size)-w),(0,0)), "edge")
                sub_input = np.pad(sub_input, ((6,6),(6,6),(0,0)), "edge")
                sub_input =  sub_input / 255.0
                help_sequence.append(sub_input)
                sub_label_sequence.append(sub_label)

    sub_input_sequence.extend(help_sequence)
    arrinput = np.asarray(sub_input_sequence) 
    arrlabel = np.asarray(sub_label_sequence) 
    print(arrinput.shape)
    nx = math.ceil(w / config.image_size)
    ny = math.ceil(h / (config.image_size))
    print(nx, ny)
    return arrinput, arrlabel, nx, ny, label_

