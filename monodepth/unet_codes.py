#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import time
import os
import sys
import smtplib
import cv2
import numpy as np
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import random
import pandas as pd
import skimage

sys.path.append('/dstor/Users/Jake/tankercv/codes/augmentations')
import tanker_aug as augs

from keras.applications.resnet50 import ResNet50
from keras.callbacks import LearningRateScheduler
from keras.models import load_model
from keras.callbacks import ModelCheckpoint
from keras.layers import concatenate, Conv2DTranspose
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Input, BatchNormalization, ZeroPadding2D
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.applications.vgg19 import VGG19
from keras.applications.vgg16 import VGG16
from keras.applications.xception import Xception
from keras.applications.resnet50 import ResNet50
from keras.utils import to_categorical
from collections import Counter
import keras.optimizers

from keras.callbacks import Callback
from keras.callbacks import EarlyStopping
from keras.utils import plot_model
import glob, os


# In[ ]:


def dice_coef(y_true, y_pred, smooth=1):
    intersection = K.sum(y_true * y_pred, axis=[1,2,3])
    union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3])
    return K.mean( (2. * intersection + smooth) / (union + smooth), axis=0)

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


def dice_coef(y_true, y_pred, smooth = 1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_np(y_true, y_pred, smooth = 1):
    y_true_f = np.array(y_true).flatten()
    y_pred_f = np.array(y_pred).flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


# In[ ]:


import numpy as np
import pylab
def normalize(A):
    return (A - np.min(A)) / (np.max(A) - np.min(A))
    
def standardize(A): 
    return (A - np.mean(A)) / np.std(A)

def crop(img, hoff, woff, h_crop = 325, w_crop = 250):
    h, w, c = img.shape
    start_h = int(h/2 - w/2 + hoff)
    return img[start_h : start_h + h_crop, woff: woff + w_crop]

import random

import cv2
from random import randint as roll

#12, 25, 50 or 100 for full
scale = 12
img_rows = 325
img_cols = 250
def preprocess(img, augment = True):
    img = CLAHE3d(np.uint16(img))
    img = standardize(img)
    if augment:
        y_shift = roll(-scale, scale)
        x_shift = roll(0,scale)
        img = crop(img, y_shift, x_shift, img_rows, img_cols)
        img = add_gaussian_noise(img , 0.01)
        return normalize(img)
    else:
        img = crop(img, 0, 0, img_rows, img_cols)
        return normalize(img)


def show_img(img):
    pylab.imshow(img)
    pylab.colorbar()
    pylab.show()

def add_gaussian_noise(image_in, noise_sigma):
    temp_image = np.float64(np.copy(image_in))
    h = temp_image.shape[0]
    w = temp_image.shape[1]
    noise = np.random.randn(h, w) * noise_sigma
    noisy_image = np.zeros(temp_image.shape, np.float64)
    if len(temp_image.shape) == 2:
        noisy_image = temp_image + noise
    else:
        noisy_image[:,:,0] = temp_image[:,:,0] + noise
        noisy_image[:,:,1] = temp_image[:,:,1] + noise
        noisy_image[:,:,2] = temp_image[:,:,2] + noise
    return noisy_image
def CLAHE3d(img):
    #print("CLAHE")
    #print(img.shape)
    img = np.uint16(img)

    ret = []
    for i in range(len(img)):
        red_hist = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        green_hist =cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4,4))
        blue_hist =cv2.createCLAHE(clipLimit=8.0, tileGridSize=(2,2))
        red = normalize(red_hist.apply(img[i]))
        green = normalize(green_hist.apply(img[i]))
        blue = normalize(blue_hist.apply(img[i]))
        ret.append((np.stack((red,green,blue), axis = 2)))
    return ret


#import skimage
def random_blend(image, min_shapes, max_shapes, min_size, max_size, alpha, rseed):

    shapes, labels = skimage.draw.random_shapes(np.shape(image), max_shapes, min_shapes, min_size, 
                                        max_size, random_seed = rseed, num_channels = 1, 
                                        allow_overlap = True)
    
    np.random.seed(rseed)
    alpha_val = np.random.randint(alpha[0]*100, alpha[1]*100)/100

    shapes = np.squeeze(shapes)
    
    #Mask applied differently for grayscale or color
    if np.size(np.shape(image))==3:
        mask = np.zeros_like(image)
        for i in range(image.shape[2]):
            mask[:,:,i] = shapes.copy()        
    else:
        mask = shapes
    #scale to roughly match image
    mask = mask.astype(int)*np.max(image)/np.max(shapes)
    #masked = np.add(alpha_val * mask, (1-alpha_val)*image)
    masked = cv2.convertScaleAbs(image*(1-alpha_val)+mask*alpha_val)
    return masked

def hist_equalize(image, a = 2, b = 8):
    clahe = cv2.createCLAHE(clipLimit = a, tileGridSize = (b,b))
    image = clahe.apply(image)
    if np.ndim(image)==2:
        image = image[...,np.newaxis]
    return image

#background_path = '/raid/Jake/data/background_only'
#boom_path       = '/raid/Jake/data/boom_masks_gray/'

#make these global
#global background_list, boom_list
#background_list = [i for i in glob.glob(background_path + '/*.{}'.format('jpg'))]
#boom_list       = [i for i in glob.glob(boom_path + '/*.{}'.format('jpg'))]

#background_path_hs = '/raid/Jake/data/half_size/background_only'
#boom_path_hs       = '/raid/Jake/data/half_size/boom_masks_gray/'

#make these global
#global background_list_hs, boom_list_hs
#background_list_hs = [i for i in glob.glob(background_path_hs + '/*.{}'.format('jpg'))]
#boom_list_hs       = [i for i in glob.glob(boom_path_hs + '/*.{}'.format('jpg'))]



def overlay(image, fploc, boom, blur_size = 21, shadow = True, overlay_dir = '.', overlay_mask = True):

    #read in plane image and make mask
    plane = image.astype('uint8')
    
    #Threshold image
    bin_pln = ((plane>16) + (plane<5))
    #Close and open to get rid of some noisy pixels
    kernel = np.ones((3,3),np.uint8)

    binary = cv2.morphologyEx(bin_pln.astype('uint8'), cv2.MORPH_OPEN, kernel)
    kernel = np.ones((7,7),np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    #Find and fill contours to get segmented image 
    img_unused, contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cont_img = np.zeros_like(binary)
    cv2.drawContours(cont_img, contours, -1, (255), thickness = cv2.FILLED, hierarchy = hierarchy, maxLevel = 1 )
    
    #Erode the binary image to remove some of the outline
    binary = cont_img
    kernel = np.ones((3,3), np.uint8)
    binary = cv2.erode(binary,kernel,iterations = 1)
    inv_bin = cv2.bitwise_not(binary)
        
    #read in random background and boom
    sz = np.max(np.shape(plane))
    background_list = [i for i in glob.glob(overlay_dir + '/bkg/*.{}'.format('jpg'))]
    
    if sz > 1500:
        blur = 7   
    else:
        blur = 3
    
    c = len(background_list) - 1
    background = cv2.UMat(cv2.imread(background_list[np.random.randint(c)], cv2.IMREAD_GRAYSCALE))
    

    #mask boom
    th, bin_boom = cv2.threshold(boom, 15, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5,5), np.uint8)
    bin_boom = cv2.erode(bin_boom,kernel,iterations = 1)
    inv_bin_boom = cv2.bitwise_not(bin_boom)
    
    #get outer contour of binary image
    img_unused, contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cont_img = np.zeros_like(plane)
    cv2.drawContours(cont_img, contours, -1, (255), 3)
    cont_img = cv2.GaussianBlur(cont_img, (blur,blur), 0)
    
    
    
    #add in the random shading
    if shadow:
        plane = random_blend(plane, 2,6,20,600,[.1, .5], int(time.time()))
        dim = np.random.randint(400,1000)/1000
        plane = plane * dim
        plane = plane.astype('uint8')
        
    layer1 = cv2.add(cv2.bitwise_and(plane, plane, mask = binary), 
                     cv2.bitwise_and(background, background, mask = inv_bin))
    
    #smooth edges of plane by blurring just the outer contour
    img1 = cv2.UMat.get(layer1)
    img2 = cv2.GaussianBlur(img1, (11,11), 0)
    alpha = cont_img/255
    blended = cv2.UMat(cv2.convertScaleAbs(img1*(1-alpha)+img2*alpha))
    
    #Combine the boom into plane/background image
    layer2 = cv2.add(cv2.bitwise_and(boom, boom, mask = bin_boom), 
         cv2.bitwise_and(blended, blended, mask = inv_bin_boom))
    
    #this is the output for the plane image
    out = cv2.UMat.get(layer2)
    if np.ndim(out)==2:
        out = out[...,np.newaxis]

    #Enlarge location spots
    fploc = fploc.astype('float')
    layer1 = cv2.GaussianBlur(fploc, (blur_size,blur_size), 0)
    if np.max(layer1) > 0:
        layer1 = layer1*(255/(np.max(layer1)))
    layer1 = layer1.astype('uint8')

    #Overlay binary boom onto location spots to be consistent with occlusion
    if overlay_mask:
        layer2 = cv2.bitwise_and(layer1, layer1, mask = inv_bin_boom)
        layer2 = cv2.UMat.get(layer2)
    else:
        layer2 = layer1

    #plt.figure()
    #plt.imshow(layer2) 
    if np.ndim(layer2)==2:
        layer2 = layer2[...,np.newaxis]
        #print(np.max(layer2))
    return np.array(out), np.array(layer2)

# In[ ]:


def gen(img_dir, mask_dir, img_augs, overlay_dir = '.', batch_sz = 8, shadow = True, blur_size = 25, do_overlay = True):
    
    #blur_size = 25 #size of Gaussian kernel applied to mask
    
    img_names = [i for i in glob.glob(img_dir + '/*.{}'.format('jpg'))]
    mask_names = [i for i in glob.glob(mask_dir + '/*.{}'.format('jpg'))]
    
    tst_img_sz = np.shape(cv2.imread(img_names[0], cv2.IMREAD_GRAYSCALE))
    IMG = np.zeros((batch_sz,tst_img_sz[0], tst_img_sz[1],1))
    MASK = np.zeros((batch_sz,tst_img_sz[0], tst_img_sz[1],1))
    
    boom_list = [i for i in glob.glob(overlay_dir + '/boom/*.{}'.format('jpg'))]
    b = len(boom_list) - 1
    boom = cv2.UMat(cv2.imread(boom_list[np.random.randint(b)], cv2.IMREAD_GRAYSCALE))
    
    while True:
        #print("Looped")
        np.random.seed(0)
        np.random.shuffle(img_names)
        np.random.seed(0)
        np.random.shuffle(mask_names)
        count = 0
        for i, m in zip(img_names, mask_names):
            #open image
            #print(img_names[:5],mask_names[:5])
            assert(i.split('/')[-1] == m.split('/')[-1])
            img = cv2.imread(i, cv2.IMREAD_GRAYSCALE)[...,np.newaxis]
            mask = cv2.imread(m, cv2.IMREAD_GRAYSCALE)[...,np.newaxis]/255
            #print('sum of mask = ' + str(np.sum(mask)))
            #Applying keras augmentations to image before overlay
            #augment
            seed = random.randint(0,10000)
            #print(seed)
            #aug = img_augs.get_random_transform(img.shape, seed)
            
            
            #img = img_augs.apply_transform(img, aug)
            img = img_augs.random_transform(img, seed=seed)
            mask = img_augs.random_transform(mask, seed=seed)
            #mask = img_augs.apply_transform(img2, aug)
           
            #overlay
            if do_overlay:
                img, mask = overlay(img, mask, boom, blur_size, shadow, overlay_dir = overlay_dir)
            else:
                
                mask = cv2.GaussianBlur(mask, (blur_size,blur_size), 0)
                mask = mask[...,np.newaxis]
                if np.max(mask)>0:
                    mask = ((mask/np.max(mask)))
            #a = random.randint(0,600)/100
            #b = random.randint(2,128)
            #img = hist_equalize(img, a, b)
            

            
            IMG[count,:,:,:]=img
            MASK[count,:,:,:]=mask
            count += 1
    
            if count == batch_sz:
                count = 0
                yield IMG, MASK


def gen_multimask(img_dir, mask_dir0, mask_dir1, mask_dir2, mask_dir3, mask_dir4, img_augs, overlay_dir = '.', batch_sz = 8, shadow = True, blur_size = 25, do_overlay = True, overlay_mask = True):
    
    #blur_size = 25 #size of Gaussian kernel applied to mask
    
    img_names = [i for i in sorted(glob.glob(img_dir + '/*.{}'.format('jpg')))]
    mask_names0 = [i for i in sorted(glob.glob(mask_dir0 + '/*.{}'.format('jpg')))]
    mask_names1 = [i for i in sorted(glob.glob(mask_dir1 + '/*.{}'.format('jpg')))]
    mask_names2 = [i for i in sorted(glob.glob(mask_dir2 + '/*.{}'.format('jpg')))]
    mask_names3 = [i for i in sorted(glob.glob(mask_dir3 + '/*.{}'.format('jpg')))]
    mask_names4 = [i for i in sorted(glob.glob(mask_dir4 + '/*.{}'.format('jpg')))]
    
    #print(mask_names0[:30])
    #print(mask_names1[:30])
    
    tst_img_sz = np.shape(cv2.imread(img_names[0], cv2.IMREAD_GRAYSCALE))
    IMG = np.zeros((batch_sz,tst_img_sz[0], tst_img_sz[1],1))
    MASK0 = np.zeros((batch_sz,tst_img_sz[0], tst_img_sz[1],1))
    MASK1 = np.zeros((batch_sz,tst_img_sz[0], tst_img_sz[1],1))
    MASK2 = np.zeros((batch_sz,tst_img_sz[0], tst_img_sz[1],1))
    MASK3 = np.zeros((batch_sz,tst_img_sz[0], tst_img_sz[1],1))
    MASK4 = np.zeros((batch_sz,tst_img_sz[0], tst_img_sz[1],1))
    
    boom_list = [i for i in glob.glob(overlay_dir + '/boom/*.{}'.format('jpg'))]
    b = len(boom_list) - 1

    
    while True:
        #print("Looped")
        np.random.seed(0)
        rng_state = np.random.get_state()
        np.random.shuffle(img_names)
        np.random.set_state(rng_state)
        np.random.shuffle(mask_names0)

        count = 0
        for i, m0 in zip(img_names, mask_names0):
            boom = cv2.UMat(cv2.imread(boom_list[np.random.randint(b)], cv2.IMREAD_GRAYSCALE))
            #open image
            #print(img_names[:5],mask_names[:5])
            assert(i.split('/')[-1] == m0.split('/')[-1])
            
            tmp = m0.split('/')
            msks = ['mask_1', 'mask_2', 'mask_3', 'mask_4']
            ms = []
            for idx, m in enumerate(msks):
                tmp[-2] = m
                ms.append('/'.join(tmp))
            
            #print(i, m0, ms)
            img = cv2.imread(i, cv2.IMREAD_GRAYSCALE)[...,np.newaxis]
            mask0 = cv2.imread(m0, cv2.IMREAD_GRAYSCALE)[...,np.newaxis]/255
            mask1 = cv2.imread(ms[0], cv2.IMREAD_GRAYSCALE)[...,np.newaxis]/255
            mask2 = cv2.imread(ms[1], cv2.IMREAD_GRAYSCALE)[...,np.newaxis]/255
            mask3 = cv2.imread(ms[2], cv2.IMREAD_GRAYSCALE)[...,np.newaxis]/255
            mask4 = cv2.imread(ms[3], cv2.IMREAD_GRAYSCALE)[...,np.newaxis]/255
            #print('sum of mask = ' + str(np.sum(mask)))
            #Applying keras augmentations to image before overlay
            #augment
            seed = random.randint(0,10000)
            #print(seed)
            #aug = img_augs.get_random_transform(img.shape, seed)
            
            
            #img = img_augs.apply_transform(img, aug)
            img = img_augs.random_transform(img, seed=seed)
            mask0 = img_augs.random_transform(mask0, seed=seed)
            mask1 = img_augs.random_transform(mask1, seed=seed)
            mask2 = img_augs.random_transform(mask2, seed=seed)
            mask3 = img_augs.random_transform(mask3, seed=seed)
            mask4 = img_augs.random_transform(mask4, seed=seed)
            #mask = img_augs.apply_transform(img2, aug)
           
            #overlay
            if do_overlay:
                img2, mask0 = overlay(img, mask0, boom, blur_size, shadow, overlay_dir = overlay_dir, overlay_mask = overlay_mask)
                foo, mask1 = overlay(img, mask1, boom, blur_size, shadow, overlay_dir = overlay_dir, overlay_mask = overlay_mask)
                foo, mask2 = overlay(img, mask2, boom, blur_size, shadow, overlay_dir = overlay_dir, overlay_mask = overlay_mask)
                foo, mask3 = overlay(img, mask3, boom, blur_size, shadow, overlay_dir = overlay_dir, overlay_mask = overlay_mask)
                foo, mask4 = overlay(img, mask4, boom, blur_size, shadow, overlay_dir = overlay_dir, overlay_mask = overlay_mask)
            else:
                print('Functionality not added yet, must do overlay!!')
                mask0 = cv2.GaussianBlur(mask0, (blur_size,blur_size), 0)
                mask0 = mask0[...,np.newaxis]
                if np.max(mask)>0:
                    mask = ((mask/np.max(mask)))
            #a = random.randint(0,600)/100
            #b = random.randint(2,128)
            #img = hist_equalize(img, a, b)
            

            
            IMG[count,:,:,:]=img2
            MASK0[count,:,:,:]=mask0
            MASK1[count,:,:,:]=mask1
            MASK2[count,:,:,:]=mask2
            MASK3[count,:,:,:]=mask3
            MASK4[count,:,:,:]=mask4
            count += 1
    
            if count == batch_sz:
                count = 0
                yield IMG, MASK0, MASK1, MASK2, MASK3, MASK4

#make sure input dimensionality is in nice powers of 2
#some layers added to make things work for 1080 in 1st dimension. Remove for more general net

def get_unet(img_rows = 512, img_cols = 512):
    inputs = Input((img_rows, img_cols,1))
    #added in by JAL
    #This part pads the image to make it divide evenly for 4 1/2 size pool layers (2^4 = 16)
    #The addition of a conv layer (conv9) at the end of the U-net with a carefully chosen kernel then 
    #resizes the network output to match the input size. 
    pad_rows = (16*np.ceil(img_rows/16) - img_rows).astype('int32')
    pad_cols = (16*np.ceil(img_cols/16) - img_cols).astype('int32')
    
    kernel_rows = (pad_rows + 1.0).astype('int32')
    kernel_cols = (pad_cols + 1.0).astype('int32')
    
    
    
    padded = ZeroPadding2D(padding=((0, pad_rows), (0, pad_cols)), data_format=None)(inputs)
    
    bn = BatchNormalization()(padded)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same',kernel_initializer = 'he_normal')(bn)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same',kernel_initializer = 'he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    pool1 = BatchNormalization()(pool1)
    
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same',kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same',kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    pool2 = BatchNormalization()(pool2)

    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same',kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same',kernel_initializer = 'he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    pool3 = BatchNormalization()(pool3)

    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same',kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same',kernel_initializer = 'he_normal')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    pool4 = BatchNormalization()(pool4)

    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same',kernel_initializer = 'he_normal')(pool4)
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same',kernel_initializer = 'he_normal')(conv5)
    conv5 = BatchNormalization()(conv5)

    up6 = concatenate([Conv2DTranspose(256, 2, strides=2, padding='same',kernel_initializer = 'he_normal')(conv5), conv4])
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same',kernel_initializer = 'he_normal')(up6)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same',kernel_initializer = 'he_normal')(conv6)
    conv6 = BatchNormalization()(conv6)

    up7 = concatenate([Conv2DTranspose(128, 2, strides=2, padding='same',kernel_initializer = 'he_normal')(conv6), conv3])
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same',kernel_initializer = 'he_normal')(up7)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same',kernel_initializer = 'he_normal')(conv7)
    conv7 = BatchNormalization()(conv7)

    up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same',kernel_initializer = 'he_normal')(conv7), conv2])
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same',kernel_initializer = 'he_normal')(up8)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same',kernel_initializer = 'he_normal')(conv8)
    conv8 = BatchNormalization()(conv8)

    up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same',kernel_initializer = 'he_normal')(conv8), conv1])
    #added in by JAL
    conv9 = Conv2D(32, (kernel_rows, kernel_cols), activation='relu', padding='valid',kernel_initializer = 'he_normal')(up9)
    
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same',kernel_initializer = 'he_normal')(conv9)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same',kernel_initializer = 'he_normal')(conv9)
    conv9 = BatchNormalization()(conv9)
    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

    model = Model(inputs=[inputs], outputs=[conv10])

    #model.compile(optimizer=optimizers.Adam(lr=1e-5), loss=dice_coef_loss, metrics=[dice_coef])

    return model


# In[ ]:


def test_network(model, test_gen):
    times = 0
    for x,y in test_gen:
        #Need to binarize image then find contours cv2.findContours,
        #then find the moment of each contour. Will probably need to iterate. 
        for i in range(len(x)):
            yout = model.predict(x)
        if times < 3:

            img = np.squeeze(yout[0])
            truth = np.squeeze(y[0]).astype('uint8')

            th, img = cv2.threshold(img, .95, 255, cv2.THRESH_BINARY)

            img = img.astype('uint8')

            #filter out small peaks
            kernel = np.ones((9,9), np.uint8)
            opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
            img = opening

            #find centers of prediction
            img_unused, contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            centers = np.zeros_like(img)
            center_loc = np.zeros((np.max(np.shape(contours)),2))
            count = 0
            for c in contours:
                 # calculate moments for each contour
                M = cv2.moments(c)

                # calculate x,y coordinate of center
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                center_loc[count,:]=[cY, cX]
                centers[cY,cX] = 255
                count += 1

            #now get centers of truth and compare, hopefully they are in the same order
            img_unused, contours, hierarchy = cv2.findContours(cv2.GaussianBlur(truth, (11,11), 0), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            center_loc_truth = np.zeros((np.max(np.shape(contours)),2))
            count = 0
            for c in contours:
                 # calculate moments for each contour
                M = cv2.moments(c)

                # calculate x,y coordinate of center
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                center_loc_truth[count,:]=[cY, cX]
                count += 1


            f, axarr = plt.subplots(2,2, figsize=(15,8))

            a = (cv2.GaussianBlur(centers, (21,21), 0)*.4)+(np.squeeze(x[0]/255)*.6)
            axarr[0,0].imshow(np.squeeze(x[0]), cmap = 'gray')
            axarr[0,1].imshow(np.squeeze(y[0]),cmap = 'gray')
            axarr[1,1].imshow(cv2.GaussianBlur(centers, (21,21), 0),cmap = 'gray')
            axarr[1,0].imshow(a,cmap = 'gray')    


            if len(center_loc)==len(center_loc_truth):
                center_loc_0 = center_loc[center_loc[:,0].argsort()]
                center_loc_truth_0 = center_loc_truth[center_loc_truth[:,0].argsort()]
                mae0 = mean_absolute_error(center_loc_0, center_loc_truth_0)

                center_loc_1 = center_loc[center_loc[:,1].argsort()]
                center_loc_truth_1 = center_loc_truth[center_loc_truth[:,1].argsort()]
                mae1 = mean_absolute_error(center_loc_1, center_loc_truth_1)
                if mae1<mae0:
                    difference = center_loc_1 - center_loc_truth_1
                    print('Pixel error for each feature point: \n', difference)
                    print('MAE across feature points = ' + str(mae1))
                else:
                    difference = center_loc_0 - center_loc_truth_0
                    print('Pixel error for each feature point: \n', difference)
                    print('MAE across feature points = ' + str(mae0))
            else:
                print(str(len(center_loc_truth))+ ' total truth feature points, ' +str(len(center_loc)) + ' feature points mapped.')


            #print(predLoc, truthLoc)
            #print("Err (x,y): ", truthLoc[0] - predLoc[0],  truthLoc[1] - predLoc[1])
            plt.show()
            #do_once = False
        times += 1
        if times > 3:
            break

def feature_points_from_prediction(image, model_prediction, kernel_size = 9, show_plots = True, color = (0,255,0)):
    x   = image
    prediction = np.squeeze(model_prediction)
    img = np.squeeze(model_prediction)

    #using a large kernel to remove any smaller high points, then
    #subtracting that image from the original. The remaining image is only the finer predictions.
    kernel = np.ones((31,31), np.uint8)
    opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    img2 = opening
    img = img-img2

    x_0 = cv2.cvtColor(np.squeeze(x).astype('uint8'),cv2.COLOR_GRAY2RGB)

    th, img = cv2.threshold(img, .4, 255, cv2.THRESH_BINARY)

    img = img.astype('uint8')

    #filter out small peaks
    kernel = np.ones((kernel_size,kernel_size), np.uint8)
    opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    erosion = cv2.erode(img,kernel,iterations = 1)
    img = opening

    #find centers of prediction
    img_unused, contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


    centers = np.zeros_like(img)
    centers = cv2.cvtColor(centers,cv2.COLOR_GRAY2RGB)
    center_loc = np.zeros((np.max(np.shape(contours)),2))
    count = 0
    for c in contours:
         # calculate moments for each contour
        M = cv2.moments(c)
        if M["m00"]>2:
        # calculate x,y coordinate of center
            
            cX = np.round(M["m10"] / M["m00"],2)
            cY = np.round(M["m01"] / M["m00"],2)
            cXr = int(M["m10"] / M["m00"])
            cYr = int(M["m01"] / M["m00"])
            if (abs(cY)+abs(cX)) != 0:
                center_loc[count,:]=[cY, cX]
                cv2.drawMarker(centers, (cXr,cYr),color, cv2.MARKER_TILTED_CROSS, 20, 3)
                cv2.drawMarker(x_0, (cXr,cYr),color, cv2.MARKER_SQUARE, 15, 5)
                #centers[cY,cX] = 255
                count += 1
    #center_loc = center_loc[center_loc!=0]
    if show_plots:
        f, axarr = plt.subplots(1,2, figsize=(28,14))
        axarr[0].imshow(np.squeeze(x_0), cmap = 'gray')
        axarr[0].title.set_text('Network Input')
        axarr[1].imshow(prediction, cmap = 'gray')
        axarr[1].title.set_text('Filtered Predictions')
        f.tight_layout()
        plt.show()

    return center_loc#, np.squeeze(x_0)            
# In[ ]:


            
class custom_callback(Callback):
    def __init__(self, times, save_file, test_gen):
        self.times = times
        self.file = save_file
        self.dice = 1e-10
        self.gen = test_gen
        return
        
    def on_epoch_end(self, epoch, logs={}):
        times = 0
        dice = []
        for x,y in self.gen:
            #Need to binarize image then find contours cv2.findContours,
            #then find the moment of each contour. Will probably need to iterate. 
            for i in range(len(x)):
                yout = self.model.predict(x)
                dice.append(dice_coef_np(y[i], yout[i]))
            if times < 3:

                img = np.squeeze(yout[0])
                truth = np.squeeze(y[0]).astype('uint8')

                th, img = cv2.threshold(img, .999, 255, cv2.THRESH_BINARY)

                img = img.astype('uint8')

                #filter out small peaks
                kernel = np.ones((13,13), np.uint8)
                opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
                img = opening

                #find centers of prediction
                img_unused, contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                centers = np.zeros_like(img)
                center_loc = np.zeros((np.max(np.shape(contours)),2))
                count = 0
                for c in contours:
                     # calculate moments for each contour
                    M = cv2.moments(c)

                    # calculate x,y coordinate of center
                    if M["m00"] != 0:
                        cX = int(M["m10"] / M["m00"])
                        cY = int(M["m01"] / M["m00"])
                    else:
                        cX, cY = 0, 0 
                    center_loc[count,:]=[cY, cX]
                    centers[cY,cX] = 255
                    count += 1

                #now get centers of truth and compare, hopefully they are in the same order
                img_unused, contours, hierarchy = cv2.findContours(cv2.GaussianBlur(truth, (13,13), 0), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                center_loc_truth = np.zeros((np.max(np.shape(contours)),2))
                count = 0
                for c in contours:
                     # calculate moments for each contour
                    M = cv2.moments(c)

                    # calculate x,y coordinate of center
                    if M["m00"] != 0:
                        cX = int(M["m10"] / M["m00"])
                        cY = int(M["m01"] / M["m00"])
                    else:
                        cX, cY = 0, 0 
                    center_loc_truth[count,:]=[cY, cX]
                    count += 1


                f, axarr = plt.subplots(2,2, figsize=(15,8))

                a = (cv2.GaussianBlur(centers, (21,21), 0)*.4)+(np.squeeze(x[0]/255)*.6)
                axarr[0,0].imshow(np.squeeze(x[0]), cmap = 'gray')
                axarr[0,1].imshow(np.squeeze(yout[0]),cmap = 'gray')
                axarr[1,1].imshow(cv2.GaussianBlur(centers, (21,21), 0),cmap = 'gray')
                axarr[1,0].imshow(a,cmap = 'gray')    


                if len(center_loc)==len(center_loc_truth):
                    center_loc_0 = center_loc[center_loc[:,0].argsort()]
                    center_loc_truth_0 = center_loc_truth[center_loc_truth[:,0].argsort()]
                    if len(center_loc_0)>0:
                        mae0 = mean_absolute_error(center_loc_0, center_loc_truth_0)

                    center_loc_1 = center_loc[center_loc[:,1].argsort()]
                    center_loc_truth_1 = center_loc_truth[center_loc_truth[:,1].argsort()]
                    if len(center_loc_1)>0:
                        mae1 = mean_absolute_error(center_loc_1, center_loc_truth_1)
                        if mae1<mae0:
                            difference = center_loc_1 - center_loc_truth_1
                            print('Pixel error for each feature point: \n', difference)
                            print('MAE across feature points = ' + str(mae1))
                        else:
                            difference = center_loc_0 - center_loc_truth_0
                            print('Pixel error for each feature point: \n', difference)
                            print('MAE across feature points = ' + str(mae0))
                else:
                    print(str(len(center_loc_truth))+ ' total truth feature points, ' +str(len(center_loc)) + ' feature points mapped.')


                #print(predLoc, truthLoc)
                #print("Err (x,y): ", truthLoc[0] - predLoc[0],  truthLoc[1] - predLoc[1])
                plt.show()
                #do_once = False
            times += 1
            if times > 3:
                break
        #print("MAE (x,y)", np.mean(MAE_x), np.mean(MAE_y))
        if self.dice < np.mean(dice):
            self.model.save(self.file, overwrite=True)
            self.dice = np.mean(dice)
        self.model.save(self.backup, overwrite = True)
        #if self.MAEXY > (np.mean(MAE_x) + np.mean(MAE_y)):
            #self.model.save(self.file, overwrite=True)
            #self.MAEXY = np.mean(MAE_x) + np.mean(MAE_y)            

            
class custom_callback_features(Callback):
    def __init__(self, times, save_file, test_gen, backup = 'backup.h5'):
        self.times = times
        self.file = save_file
        self.dice = 1e-10
        self.gen = test_gen
        self.backup = backup
        return
        
    def on_epoch_end(self, epoch, logs={}):
        times = 0
        dice = []
        for x,y in self.gen:
            #Need to binarize image then find contours cv2.findContours,
            #then find the moment of each contour. Will probably need to iterate. 
            for i in range(len(x)):
                yout = self.model.predict(x)
                dice.append(dice_coef_np(y[i], yout[i]))
            if times < 3:
                
                net_in = np.squeeze(x[0])
                raw_out = np.squeeze(yout[0])
                truth = np.squeeze(y[0]).astype('uint8')
                
                
                f, axarr = plt.subplots(1,2, figsize=(28,14))
                axarr[0].imshow(np.squeeze(raw_out), cmap = 'gray')
                axarr[0].title.set_text('Raw Network Output')
                axarr[1].imshow(truth, cmap = 'gray')
                axarr[1].title.set_text('Truth')
                f.tight_layout()
                plt.show()
                
                features, foo = feature_points_from_prediction(net_in, raw_out, kernel_size = 1, show_plots = True)
                
                #now get centers of truth and compare, hopefully they are in the same order
                th, truth_bin = cv2.threshold(truth, .2, 255, cv2.THRESH_BINARY)
                img_unused, contours, hierarchy = cv2.findContours(truth_bin, 
                                                       cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                center_loc_truth = np.zeros((np.max(np.shape(contours)),2))
                count = 0
                for c in contours:
                     # calculate moments for each contour
                    M = cv2.moments(c)

                    # calculate x,y coordinate of center
                    if M["m00"] != 0:
                        cX = int(M["m10"] / M["m00"])
                        cY = int(M["m01"] / M["m00"])
                    else:
                        cX, cY = 0, 0 
                    center_loc_truth[count,:]=[cY, cX]
                    count += 1
                if features.size:
                    if len(features)<30:
                        columns = ['Pred_ Feature_Y', 'Pred_Feature_X']
                        predictions = pd.DataFrame(columns = columns, 
                                                   index = np.arange(0,len(features),1))
                        predictions[:] = features
                        columns = ['Truth_ Feature_Y', 'Truth_Feature_X']
                        truth_loc = pd.DataFrame(columns = columns, 
                                                 index = np.arange(0,len(center_loc_truth),1))
                        truth_loc[:] = center_loc_truth
                        print(predictions)
                        print(truth_loc)
                
            times += 1
            if times > 3:
                break
        if self.dice < np.mean(dice):
            self.model.save(self.file, overwrite=True)
            self.dice = np.mean(dice)
        self.model.save(self.backup, overwrite = True)
        #print("MAE (x,y)", np.mean(MAE_x), np.mean(MAE_y))
        #if self.MAEXY > (np.mean(MAE_x) + np.mean(MAE_y)):
            #self.model.save(self.file, overwrite=True)
            #self.MAEXY = np.mean(MAE_x) + np.mean(MAE_y)            

            

            
            
'''             
                
def overlay_segment(image, fploc, blur_size = 21, shadow = True):
    #print('test')
    b = len(boom_list)
    c = len(background_list)
    #read in plane image and make mask
    plane = image.astype('uint8')
    
    #Threshold image
    bin_pln = ((plane>16) + (plane<5))
    #Close an open to get rid of some noisy pixels
    kernel = np.ones((3,3),np.uint8)
    binary = cv2.morphologyEx(bin_pln.astype('uint8'), cv2.MORPH_OPEN, kernel)
    kernel = np.ones((7,7),np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    #Find and fill contours to get segmented image 
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cont_img = np.zeros_like(binary)
    cv2.drawContours(cont_img, contours, -1, (255), thickness = cv2.FILLED, hierarchy = hierarchy, maxLevel = 1 )
    
    #Erode the binary image to remove some of the outline
    binary = cont_img
    kernel = np.ones((5,5), np.uint8)
    binary = cv2.erode(binary,kernel,iterations = 1)
    inv_bin = cv2.bitwise_not(binary)
        
    #read in random background and boom
    sz = np.max(np.shape(plane))
    if sz > 1500:
        blur = 7
        background = cv2.UMat(cv2.imread(background_list[np.random.randint(c)], cv2.IMREAD_GRAYSCALE))
        boom = cv2.UMat(cv2.imread(boom_list[np.random.randint(b)], cv2.IMREAD_GRAYSCALE))
    else:
        blur = 3
        background = cv2.UMat(cv2.imread(background_list_hs[np.random.randint(c)], cv2.IMREAD_GRAYSCALE))
        boom = cv2.UMat(cv2.imread(boom_list_hs[np.random.randint(b)], cv2.IMREAD_GRAYSCALE))        
    
    #mask boom
    th, bin_boom = cv2.threshold(boom, 35, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5,5), np.uint8)
    bin_boom = cv2.erode(bin_boom,kernel,iterations = 1)
    inv_bin_boom = cv2.bitwise_not(bin_boom)
    
    #get outer contour of binary image
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cont_img = np.zeros_like(plane)
    cv2.drawContours(cont_img, contours, -1, (255), 3)
    cont_img = cv2.GaussianBlur(cont_img, (blur,blur), 0)
    
    
    #add in the random shading
    if shadow:
        plane = random_blend(plane, 2,6,20,600,[.1, .5], int(time.time()))
        dim = np.random.randint(400,1000)/1000
        plane = plane * dim
        plane = plane.astype('uint8')
        
    #Combine background and plane    
    layer1 = cv2.add(cv2.bitwise_and(plane, plane, mask = binary), 
                     cv2.bitwise_and(background, background, mask = inv_bin))
    
    #smooth edges of plane by blurring just the outer contour
    img1 = cv2.UMat.get(layer1)
    img2 = cv2.GaussianBlur(img1, (11,11), 0)
    alpha = cont_img/255
    blended = cv2.UMat(cv2.convertScaleAbs(img1*(1-alpha)+img2*alpha))
    
    #Comine the boom into plane/background image
    layer2 = cv2.add(cv2.bitwise_and(boom, boom, mask = bin_boom), 
         cv2.bitwise_and(blended, blended, mask = inv_bin_boom))
    
    #this is the output for the plane image
    out = cv2.UMat.get(layer2)
    if np.ndim(out)==2:
        out = out[...,np.newaxis]

    #Overlay binary boom onto binary to be consistent with occlusion
    layer2 = cv2.bitwise_and(binary, binary, mask = inv_bin_boom)
    layer2 = cv2.UMat.get(layer2)
    
    
    if np.max(layer2)>0:
        layer2 = ((layer2/np.max(layer2))).astype('uint8')
    if np.ndim(layer2)==2:
        layer2 = layer2[...,np.newaxis]
    return np.array(out), np.array(layer2)



def gen_segment(img_dir, mask_dir, img_augs, batch_sz = 8, shadow = True):
    
    blur_size = 3 #size of Gaussian kernel applied to mask
    
    img_names = [i for i in glob.glob(img_dir + '/*.{}'.format('jpg'))]
    mask_names = [i for i in glob.glob(mask_dir + '/*.{}'.format('jpg'))]
    
    tst_img_sz = np.shape(cv2.imread(img_names[0], cv2.IMREAD_GRAYSCALE))
    IMG = np.zeros((batch_sz,tst_img_sz[0], tst_img_sz[1],1))
    MASK = np.zeros((batch_sz,tst_img_sz[0], tst_img_sz[1],1))
    
    while True:
        #print("Looped")
        np.random.seed(0)
        np.random.shuffle(img_names)
        np.random.seed(0)
        np.random.shuffle(mask_names)
        count = 0
        for i, m in zip(img_names, mask_names):
            #open image
            #print(img_names[:5],mask_names[:5])
            assert(i.split('/')[-1] == m.split('/')[-1])
            img = cv2.imread(i, cv2.IMREAD_GRAYSCALE)[...,np.newaxis]
            mask = cv2.imread(m, cv2.IMREAD_GRAYSCALE)[...,np.newaxis]
            
            #Applying keras augmentations to image before overlay
            #augment
            seed = random.randint(0,10000)
            #print(seed)
            #aug = img_augs.get_random_transform(img.shape, seed)
            
            
            #img = img_augs.apply_transform(img, aug)
            img = img_augs.random_transform(img, seed=seed)
            mask = img_augs.random_transform(mask, seed=seed)
            #mask = img_augs.apply_transform(img2, aug)
            
            
            #overlay
            img, mask = overlay_segment(img, mask, blur_size, shadow)
            
            if False:
                a = random.randint(0,400)/100
                b = random.randint(2,24)
                img = hist_equalize(img, a, b)            
            
            IMG[count,:,:,:]=img
            MASK[count,:,:,:]=mask
            count += 1
    
            if count == batch_sz:
                count = 0
                yield IMG, MASK
                
                class custom_callback_segment(Callback):
    def __init__(self, times, save_file, test_gen, backup = 'backup.h5'):
        self.times = times
        self.file = save_file
        self.dice = 1e-10
        self.gen = test_gen
        self.backup = backup
        return
        
    def on_epoch_end(self, epoch, logs={}):
        times = 0
        dice = []
        for x,y in self.gen:
            #Need to binarize image then find contours cv2.findContours,
            #then find the moment of each contour. Will probably need to iterate. 
            for i in range(len(x)):
                yout = self.model.predict(x)
                dice.append(dice_coef_np(y[i], yout[i]))
            if times < 3:
                
                net_in = np.squeeze(x[0])
                raw_out = np.squeeze(yout[0])
                truth = np.squeeze(y[0]).astype('uint8')
                
                
                f, axarr = plt.subplots(1,2, figsize=(28,14))
                axarr[0].imshow(np.squeeze(raw_out), cmap = 'gray')
                axarr[0].title.set_text('Raw Network Output')
                axarr[1].imshow(truth, cmap = 'gray')
                axarr[1].title.set_text('Truth')
                f.tight_layout()
                plt.show()
                
                
            times += 1
            if times > 3:
                break
        if self.dice < np.mean(dice):
            self.model.save(self.file, overwrite=True)
            self.dice = np.mean(dice)
        self.model.save(self.backup, overwrite = True)
        #print("MAE (x,y)", np.mean(MAE_x), np.mean(MAE_y))
        #if self.MAEXY > (np.mean(MAE_x) + np.mean(MAE_y)):
            #self.model.save(self.file, overwrite=True)
            #self.MAEXY = np.mean(MAE_x) + np.mean(MAE_y)      
                
'''               





