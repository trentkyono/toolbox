#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Bundling all the Augmentation codes together. 

#import skimage
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os, os.path
import shutil
import glob
import cv2
import pandas as pd

# In[ ]:


# generate random crops so the location changes depending on the seed. 
# suggested useage:
#n = num_samples
#seeds = np.arange[1,n,1]
#for i in seeds:
#        t_reference, t_cropped = random_crop(image, final_size, truth_position, seeds[i])
#        np.append(t_reference, axis = 0)
#        np.append(t_cropped, axis = 2)

def random_crop(image, final_size, truth_pos, rseed):
    #find how much needs to be cropped off
    orig_size  = np.shape(image)
    crop_total = np.subtract(orig_size,final_size)
    # to manually set the offset, just feed in a reasonable array for x1,y1
    if np.size(rseed)==2:
        y1 = rseed[0]
        y2 = crop_total[0]-x1
        x1 = rseed[1]
        x2 = crop_total[1]-y1
    else:      
        #set random seed for repeatability
        np.random.seed(rseed)
        #get a random crop from each axis
        y1 = np.random.randint(0,crop_total[0])
        y2 = crop_total[0] - y1
        x1 = np.random.randint(0,crop_total[1])
        x2 = crop_total[1] - x1
    #crop the thing
    cropped = skimage.util.crop(image, ((y1, y2), (x1, x2)))
    #print(np.where(cropped>0)) #for use with test image (single 1 on a field of zeros)
    #now need to worry about the reference location
    new_loc = np.subtract(truth_pos,[y1,x1])
    #print([x1,y1])
    #print(new_loc)
    return new_loc, cropped


# In[ ]:


# generate random rotations depending on the seed. 
# suggested useage:
#n = num_samples
#seeds = np.arange[1,n,1]
#for i in seeds:
#        t_reference, t_cropped = random_crop(image, [min_angle, max_angle], truth_position, seeds[i])
#        np.append(t_reference, axis = 0)
#        np.append(t_rotated, axis = 2)

def random_rotate(image, rotation_angle, reference, rseed):
    
    if np.size(rotation_angle)==1:
        rot_angle = rotation_angle
    else:
        np.random.seed(rseed)
        rot_angle = np.random.randint(rotation_angle[0]*100, rotation_angle[1]*100)/100
    
    row,col = [image.shape[0], image.shape[1]]
    center=tuple(np.array([col,row])/2)
    rot_mat = cv2.getRotationMatrix2D(center,rot_angle,1.0)
    rotated_pic = cv2.warpAffine(image, rot_mat, (col,row))
    # to find the position of the reference:
    # calculate with sine and cosine
    #first get center of array
    y_cent = (image.shape[0]-1)/2
    x_cent = (image.shape[1]-1)/2
    offset = np.subtract(reference, [y_cent, x_cent])
    #print(offset)
    #make rotation matrix, not sure why the angle needs to be negative here
    #c,s = np.cos(np.radians(-rot_angle)), np.sin(np.radians(-rot_angle))
    #R_mat = np.array(((c,-s), (s,c)))

    # rotate the points about the center of the array
    rotated = np.matmul(np.matrix.transpose(offset),rot_mat[:2,:2])
    #print(np.add(rotated, [y_cent, x_cent]))
    new_reference = np.add(rotated, [y_cent, x_cent])
    return rotated_pic, new_reference


# In[ ]:


# generate random obscurations depending on the seed. 
# suggested useage:
#n = num_samples
#seeds = np.arange[1,n,1]
#for i in seeds:
#        t_reference, t_cropped = random_cut(image, min_shapes, max_shapes, 
#                min_size, max_size, truth_position, seeds[i])
#        np.append(t_reference, axis = 0)
#        np.append(t_rotated, axis = 2)

def random_cut(image, min_shapes, max_shapes, min_size, max_size, reference, rseed):

    shapes, labels = skimage.draw.random_shapes(np.shape(image), max_shapes, min_shapes, min_size, 
                                    max_size, random_seed = rseed, num_channels = 1, 
                                    allow_overlap = True)
    shapes = np.squeeze(shapes)
    shapes = shapes > 254
    #Mask applied differently for grayscale or color
    if np.size(np.shape(image))==3:
        mask = np.zeros_like(image)
        for i in range(image.shape[2]):
            mask[:,:,i] = shapes.copy()        
    else:
        mask = shapes   
    masked = np.multiply(mask, image)
    return masked, reference


# In[ ]:


# generate random obscurations with random opacity depending on the seed. 
# suggested useage:
#n = num_samples
#seeds = np.arange[1,n,1]
#for i in seeds:
#        t_reference, t_cropped = random_cut(image, min_shapes, max_shapes, 
#                min_size, max_size, truth_position, seeds[i])
#        np.append(t_reference, axis = 0)
#        np.append(t_rotated, axis = 2)

def random_blend(image, min_shapes, max_shapes, min_size, max_size, alpha, reference, rseed):

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
    masked = np.add(alpha_val * mask, (1-alpha_val)*image)
    return masked, reference


# In[ ]:


# resize image and rescale coordinate location to follow
# suggested useage:
#        t_reference, t_resized = random_cut(image, reference, fractional size)
#fractional size is whatever fraction of the original you would like the new image to be. 

def resize_image(image, reference, fractional_size):
    new_reference = np.rint(np.multiply(reference,fractional_size)).astype(int)
    sz = np.rint(np.multiply(np.shape(image),fractional_size)).astype('int16')

    resized_im = cv2.resize(image, (sz[1], sz[0]),
                           interpolation = cv2.INTER_AREA )
    
    return resized_im, new_reference



def split_folder(folder, test_split = 0.15, isolate = False, shuffle = False, seed = 0):
    
    #function takes data folder and copies the data into 3 different folder. One for train, one for validation, 
    #and one for test. It renames the files so they will be read in correctly and paired with the right csv data.
    #the csv has a record of which original file is where. Split is repeatable with seed. 
    
    #isolate takes 10% of the data and tucks it away for network testing
    iso_perc = .1
    
    val_dir = folder + '/val_data'
    train_dir = folder + '/train_data'
    test_dir = folder + '/test_data'

    if not os.path.exists(val_dir):
        os.mkdir(val_dir)
    if not os.path.exists(train_dir):
        os.mkdir(train_dir)
    if not os.path.exists(test_dir) and isolate:
        os.mkdir(test_dir)

    #os.chdir(folder)
    csv_files = [i for i in glob.glob(folder + '/*.{}'.format('csv'))]
    jpg_files = [i for i in sorted(glob.glob(folder + '/*.{}'.format('jpg')))]
    num_files = len(jpg_files)
    strt = 0
    
    if isolate:
        num_val = int(num_files-(num_files-(iso_perc*num_files))*test_split)
        strt = np.rint(iso_perc*num_files).astype(int)
    else:
        num_val = np.rint((1-test_split)*len(jpg_files)).astype(int)
        
    ind = np.arange(len(jpg_files))
    if shuffle:
        np.random.seed(seed)
        np.random.shuffle(ind)

    [shutil.copy(jpg_files[ind[j]], (train_dir + "/trainimg" + '{num:0{width}}'.format(num = j, width=6) + ".jpg")) for j in range(strt,num_val)]
    [shutil.copy(jpg_files[ind[j + num_val]], (val_dir + "/valimg" + '{num:0{width}}'.format(num = j, width=6) + ".jpg")) for j in range(strt, len(jpg_files) - num_val)]

    metadata = pd.read_csv(csv_files[0])
    
    if isolate:
        [shutil.copy(jpg_files[ind[j]], (test_dir + "/testimg" + '{num:0{width}}'.format(num = j, width=6) + ".jpg")) for j in range(strt)]
        isolate_meta = metadata.loc[ind[:strt],:]
        isolate_meta.to_csv(test_dir + '/test_metadata.csv')
    
    train_meta = metadata.loc[ind[strt:num_val],:]
    val_meta  = metadata.loc[ind[num_val:],:]    

    train_meta.to_csv(train_dir + '/train_metadata.csv')
    val_meta.to_csv(val_dir + '/val_metadata.csv')
        
    return train_dir, val_dir, test_dir if isolate else (train_dir, val_dir)


def apply_boom(image, reference, folder, rseed):
    
    #might be necessary to cross rerence the mask if the binarization isn't acting reliably
    #mask_folder = r'C:\Users\zx420d\Documents\Tanker\Data\Boom_Masks\Boom_masks_binary'
    #jpg_mask = [i for i in glob.glob(mask_folder + '/*.{}'.format('jpg'))]
    
    jpg_boom = [i for i in glob.glob(folder + '/*.{}'.format('jpg'))]
    
    #which boom mask to use:
    np.random.seed(rseed)
    ind = np.random.randint(0,len(jpg_boom),1)
    
    #get boom image
    boom_img = skimage.io.imread(jpg_boom[ind[0]])
    frac_size = np.size(image, axis = 0)/np.size(boom_img, axis = 0)
    if frac_size != 1:
        boom_img = skimage.transform.resize(boom_img, 
                                         np.rint(np.multiply(np.shape(boom_img),frac_size)), 
                                         anti_aliasing = True, 
                                         mode = 'constant',
                                         clip = True)
    
    #threshold to make mask (binary image wasn't forming good mask)
    thresh = skimage.filters.threshold_otsu(boom_img)
    binary = boom_img > thresh
    #combine into new image
    new_img = np.multiply(boom_img, binary) + np.multiply(image, np.invert(binary))
    
    return new_img, reference

#function to rescale the reference location in terms of array percentage, and rescale image from 0-1. 
def rescale_inputs(image, reference):
    xsz = np.size(image, axis = 1)
    ysz = np.size(image, axis = 0)
    new_ref = np.divide(reference-np.array([ysz, xsz])/2, np.array([ysz, xsz])/2)
    image = image/np.max(image)
    return image, new_ref

def overlay(background_list, boom_list, image):
    b = len(boom_list)
    c = len(background_list)
    
    #read in plane image and make mask
    plane = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #plane = cv2.UMat(cv2.imread(image_path, cv2.IMREAD_GRAYSCALE))
    th, bin_pln = cv2.threshold(plane, 10, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5,5),np.uint8)
    binary = cv2.morphologyEx(bin_pln, cv2.MORPH_OPEN, kernel)
    inv_bin = cv2.bitwise_not(binary)
    
    #read in random background and boom
    background = cv2.UMat(cv2.imread(background_list[np.random.randint(c)], cv2.IMREAD_GRAYSCALE))
    boom = cv2.UMat(cv2.imread(boom_list[np.random.randint(b)], cv2.IMREAD_GRAYSCALE))
    
    #mask boom
    th, bin_boom = cv2.threshold(boom, 35, 255, cv2.THRESH_BINARY)
    inv_bin_boom = cv2.bitwise_not(bin_boom)
    
    #layer things
    layer1 = cv2.add(cv2.bitwise_and(plane, plane, mask = binary), 
                     cv2.bitwise_and(background, background, mask = inv_bin))

    layer2 = cv2.add(cv2.bitwise_and(boom, boom, mask = bin_boom), 
                     cv2.bitwise_and(layer1, layer1, mask = inv_bin_boom))
    
    return cv2.UMat.get(layer2)
