import os
import glob
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn
from keras.models import load_model

sys.path.append('/raid/Jake/codes/augmentations')
sys.path.append('/raid/Jake/codes/unet')
import tanker_aug as augs
import unet_codes as uf



def predict_on_directory(source_dir, model, feature_point, save_name):
    #list all the data files
    csv_files = [i for i in glob.glob(source_dir + '/*.{}'.format('csv'))]
    metadata = pd.read_csv(csv_files[0])
    jpg_files = [i for i in sorted(glob.glob(source_dir + '/*.{}'.format('jpg')))]
    
    test = cv2.imread(jpg_files[1], 0)
    shpe = np.shape(test)
    
    col_names = ['pixel_x_' + str(feature_point), 'pixel_y_' + str(feature_point)]
    
    #make empty 
    columns = ['Feature_num', 
           'File', 
           'File_num',
           'Truth_x', 'Truth_y',
           'best_pix_x', 'best_pix_y',
           'centroid_x', 'centroid_y',
           'MaxPixVal',
           'RMSE_best_pix']
    feature_test = pd.DataFrame(columns = columns, index = np.arange(0,len(jpg_files),1))
    
    ii = 0
    for i in range(len(jpg_files)):

        truth = metadata[col_names].iloc[i] #this is x,y or col,row

        image = np.zeros([1,shpe[0],shpe[1],1])
        image[0,:,:,0] = cv2.imread(jpg_files[i], 0)
        image[0,:,:,0] = cv2.GaussianBlur(image[0,:,:,0], (3,3), 0)
        prediction = model.predict(image)

        #now what to do with the prediction
        kernel = np.ones((5,5),np.uint8)
        opening = cv2.morphologyEx(np.squeeze(prediction), cv2.MORPH_OPEN, kernel)
        
        (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(opening[:,:]) #finds 'most confident' pixel
        centroids = uf.feature_points_from_prediction(image, prediction, kernel_size = 1, show_plots = False)
        if not centroids.size>0:
            centroids = [0,0]
        
        #need to add things into the dataframe
        feature_test['Feature_num'].iloc[i+ii]=feature_point
        feature_test['File'].iloc[i+ii]=jpg_files[i]
        feature_test['File_num'].iloc[i+ii] =i
        feature_test['Truth_x'].iloc[i+ii]=truth[0]
        feature_test['Truth_y'].iloc[i+ii]=truth[1]
        feature_test['best_pix_x'].iloc[i+ii]=maxLoc[0]
        feature_test['best_pix_y'].iloc[i+ii]=maxLoc[1]
        feature_test['centroid_x'].iloc[i+ii]=centroids[1]
        feature_test['centroid_y'].iloc[i+ii]=centroids[0]
        feature_test['MaxPixVal'].iloc[i+ii]=prediction[0,maxLoc[1],maxLoc[0],0]

        if not np.mod(i,200):
            print('On file: ' + str(i))

    max_pix_x_diff = feature_test['best_pix_x']-feature_test['Truth_x']
    max_pix_y_diff = feature_test['best_pix_y']-feature_test['Truth_y']

    total_diff = (max_pix_x_diff**2 + max_pix_y_diff**2)**(0.5)
    feature_test['RMSE_best_pix'] = total_diff
    
    if save_name:
        feature_test.to_csv(save_name)
    else:
        feature_test.to_csv('feature_' + str(feature_point) + '_challenge_predictions.csv')
    return feature_test


def label_predictions(source_dir, out_dir, predictions_df, threshold = 0.0):
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    jpg_files = [i for i in sorted(glob.glob(source_dir + '/*.{}'.format('jpg')))]
    max_prediction_value = predictions_df['MaxPixVal'].max()
    for i in range(len(jpg_files)):
        image = cv2.imread(jpg_files[i], 0)
        g_color = 255*(predictions_df['MaxPixVal'].iloc[i]/max_prediction_value)
        r_color = 255 -g_color
        image = cv2.cvtColor(image,cv2.COLOR_GRAY2RGB)
        if predictions_df['MaxPixVal'].iloc[i]>=threshold:
            cv2.drawMarker(image, (predictions_df['best_pix_x'].iloc[i],predictions_df['best_pix_y'].iloc[i]),
                           (r_color, g_color, 0), cv2.MARKER_TILTED_CROSS, 20,5)
        cv2.drawMarker(image, (predictions_df['Truth_x'].iloc[i],predictions_df['Truth_y'].iloc[i]), (0, 255, 0),
                       cv2.MARKER_SQUARE, 60,3)
        cv2.imwrite(out_dir + 'frame_' + '{num:0{width}}'.format(num = i, width=6) + '.jpg', image)
        if not np.mod(i,200):
            print('On file: ' + str(i))
        #plt.imshow(image, cmap = 'gray')
        
        
        
def highlight_predictions(source_dir, out_dir, predictions_df, column_names, start = 20, end = 8000, 
                          color = (0,255,0), marker=cv2.MARKER_TILTED_CROSS, marker_size = 60):
    '''
    Column names in the form [Col_x_pixels, Col_y_pixels]
    color as (r,g,b)
    marker as cv2.MARKER_TILTED_CROSS, etc
    '''
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    jpg_files = [i for i in sorted(glob.glob(source_dir + '/*.{}'.format('jpg')))]
    for i in range(start, end):
        image = cv2.imread(jpg_files[i], 1)
        #image = cv2.cvtColor(image,cv2.COLOR_GRAY2RGB)
        xval = int(predictions_df[column_names[0]].iloc[i])
        yval = int(predictions_df[column_names[1]].iloc[i])
        cv2.drawMarker(image, (xval,yval), color,
                       marker, marker_size,5)
        cv2.imwrite(out_dir + '/frame_' + '{num:0{width}}'.format(num = i, width=6) + '.jpg', image)
        if not np.mod(i,200):
            print('On file: ' + str(i))
        #plt.imshow(image, cmap = 'gray')
        
        

def make_movie(source_dir, out_dir, out_file, start_frm = 0, num_frms = [], fps = 15, stride = 1):
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    img_array = []
    files = sorted(glob.glob(source_dir + '/*.{}'.format('jpg')))
    if not num_frms:
        num_frms = len(files)
    
    for i in range(start_frm, start_frm + num_frms):
        if not np.mod(i,stride):
            filename = files[i]
            img = cv2.imread(filename,1)
            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            height, width, layers = img.shape
            size = (width,height)
            img_array.append(img)


    out = cv2.VideoWriter(out_dir + out_file + '.avi',cv2.VideoWriter_fourcc(*'DIVX'), fps, size)

    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()
    print('Movie Written')
    
    
def show_training(log_name):
    history = pd.read_csv(log_name)
    #print(pd.melt(history, ['epoch']))
    pt = sn.relplot(x ="epoch", y = "value", hue = 'variable',kind = "line", 
                data = pd.melt(history, ['epoch']))