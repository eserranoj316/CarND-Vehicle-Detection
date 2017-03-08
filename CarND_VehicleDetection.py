import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from sklearn.cross_validation import train_test_split
from sklearn.utils import shuffle

from tqdm import tqdm
import cv2
import glob
import time
import os
import sys
import pickle
#functions from lessons and some other written utils
from function_utils import *
# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip

def pipeline(image):
    global ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins
    box_list = []
    for scale in [1,1.5,2]:
        box_list += find_cars(image, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)
    #e = draw_boxes(img, boxes, color=(0, 0, 255), thick=6)
    #plt.imshow(e)
    heat = np.zeros_like(image[:,:,0]).astype(np.float)
    # Add heat to each box in box list
    heat = add_heat(heat,box_list)
    # Apply threshold to help remove false positives
    heat = apply_threshold(heat,1)
    # Visualize the heatmap when displaying    
    heatmap = np.clip(heat, 0, 255)
    # Find final boxes from heatmap using label function
    labels = label(heatmap)
    draw_img = draw_labeled_bboxes(np.copy(image), labels)
    return draw_img

def process_image(image):
    result = pipeline(image)
    return result

if __name__ == "__main__":
    usePickledSvc = False
    if usePickledSvc == False:
        # Read in cars and notcars
        images = glob.glob('non-vehicles_smallset/not*/*.jpeg') + glob.glob('vehicles_smallset/cars*/*.jpeg')
        cars = []
        notcars = []
        for image in images:
            if 'image' in image or 'extra' in image:
                notcars.append(image)
            else:
                cars.append(image)
        #834
        cars += glob.glob('vehicles/GTI_Far/*.png')
        #909
        cars += glob.glob('vehicles/GTI_Left/*.png')
        #419
        cars += glob.glob('vehicles/GTI_MiddleClose/*.png')
        #664
        cars += glob.glob('vehicles/GTI_Right/*.png')
        #5966
        cars += glob.glob('vehicles/KITTI_extracted/*.png')
        
        notcars += glob.glob('non-vehicles/GTI/*.png')
        notcars += glob.glob('non-vehicles/Extras/*.png')
        
        #shuffle car and notcar list
        cars = shuffle(cars)
        notcars = shuffle(notcars)
        
        color_space = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
        orient = 9  # HOG orientations
        pix_per_cell = 8 # HOG pixels per cell
        cell_per_block = 2 # HOG cells per block
        hog_channel = "ALL" # Can be 0, 1, 2, or "ALL"
        spatial_size = (32, 32) # Spatial binning dimensions
        hist_bins = 32    # Number of histogram bins
        spatial_feat = True # Spatial features on or off
        hist_feat = True # Histogram features on or off
        hog_feat = True # HOG features on or off
        
        #Get Hog features for single car and notcar for display 
        carimage = mpimg.imread(cars[1000])
        carimage = convert_color(carimage, conv='RGB2YCrCb')
        #carimage = convert_color(carimage, conv='RGB2LUV')
        plt.imshow(carimage)
        #extracting hog features per channel
        f,carimage1 = get_hog_features(carimage[:,:,0], orient, pix_per_cell, cell_per_block, vis=True, feature_vec=True)
        f,carimage2 = get_hog_features(carimage[:,:,1], orient, pix_per_cell, cell_per_block, vis=True, feature_vec=True)
        f,carimage3 = get_hog_features(carimage[:,:,2], orient, pix_per_cell, cell_per_block, vis=True, feature_vec=True)
        
        notcarimage = mpimg.imread(notcars[2000])
        notcarimage = convert_color(notcarimage, conv='RGB2YCrCb')
        plt.imshow(notcarimage)
        #extracting hog features per channel
        f,notcarimage1 = get_hog_features(notcarimage[:,:,0], orient, pix_per_cell, cell_per_block, vis=True, feature_vec=True)
        f,notcarimage2 = get_hog_features(notcarimage[:,:,1], orient, pix_per_cell, cell_per_block, vis=True, feature_vec=True)
        f,notcarimage3 = get_hog_features(notcarimage[:,:,2], orient, pix_per_cell, cell_per_block, vis=True, feature_vec=True)
        #display hog features extracted via plt.imshow  
        
        #get the features of all cars
        car_features = extract_features(cars, color_space=color_space, 
                                spatial_size=spatial_size, hist_bins=hist_bins, 
                                orient=orient, pix_per_cell=pix_per_cell, 
                                cell_per_block=cell_per_block, 
                                hog_channel=hog_channel, spatial_feat=spatial_feat, 
                                hist_feat=hist_feat, hog_feat=hog_feat)
        
        out = open("pickle_carfeatures.p", "wb" )
        pickle.dump(car_features,out)
        out.close()
        
        notcar_features = extract_features(notcars, color_space=color_space, 
                                spatial_size=spatial_size, hist_bins=hist_bins, 
                                orient=orient, pix_per_cell=pix_per_cell, 
                                cell_per_block=cell_per_block, 
                                hog_channel=hog_channel, spatial_feat=spatial_feat, 
                                hist_feat=hist_feat, hog_feat=hog_feat)
        
        out = open("pickle_notcarfeatures.p", "wb" )
        pickle.dump(notcar_features,out)
        out.close()
        X = np.vstack((car_features, notcar_features)).astype(np.float64)                        
        # Fit a per-column scaler
        X_scaler = StandardScaler().fit(X)
        # Apply the scaler to X
        scaled_X = X_scaler.transform(X)
        # Define the labels vector
        y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))
        
        # Split up data training(80) and test(20) sets
        # each set should be balance between car and not car 
        
        num = int(len(car_features) * 0.80)
        car_train   = scaled_X[0:num]
        car_train_Y = y[0:num]
        car_test    =    scaled_X[num:len(car_features)]
        car_test_Y  = y[num:len(car_features)]
        
        num = int(len(notcar_features) * 0.80)
        notcar_train = scaled_X[len(car_features):(len(car_features) + num)]
        notcar_train_Y = y[len(car_features):(len(car_features) + num)]
        notcar_test =  scaled_X[(len(car_features) + num):]
        notcar_test_Y = y[(len(car_features) + num):]
        
        X_train = np.vstack((car_train, notcar_train)).astype(np.float64)                        
        y_train = np.hstack((car_train_Y, notcar_train_Y))
        X_train, y_train = shuffle(X_train, y_train)
        
        X_test = np.vstack((car_test, notcar_test)).astype(np.float64)                        
        y_test = np.hstack((car_test_Y, notcar_test_Y))
        X_test, y_test = shuffle(X_test, y_test)
        
        
        print('Using:',orient,'orientations',pix_per_cell,
            'pixels per cell and', cell_per_block,'cells per block')
        print('Feature vector length:', len(X_train[0]))
        # Use a linear SVC 
        svc = LinearSVC()
        # Check the training time for the SVC
        t=time.time()
        svc.fit(X_train, y_train)
        t2 = time.time()
        print(round(t2-t, 2), 'Seconds to train SVC...')
        # Check the score of the SVC
        print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
        # Check the prediction time for a single sample
        t=time.time()
        #save the svc and features extraction parameters
        out = open("svc_pickle.p", "wb" )
        pickle.dump(svc,out)
        #svc = dist_pickle["svc"]
        #X_scaler = dist_pickle["scaler"]
        pickle.dump(X_scaler,out)
        #orient = dist_pickle["orient"]
        pickle.dump(orient,out)
        #pix_per_cell = dist_pickle["pix_per_cell"]
        pickle.dump(pix_per_cell,out)
        #cell_per_block = dist_pickle["cell_per_block"]
        pickle.dump(cell_per_block,out)
        #spatial_size = dist_pickle["spatial_size"]
        pickle.dump(spatial_size,out)
        #hist_bins = dist_pickle["hist_bins"]
        pickle.dump(hist_bins,out)
        out.close()
    else:
        color_space = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
        hog_channel = "ALL" # Can be 0, 1, 2, or "ALL"
        spatial_feat = True # Spatial features on or off
        hist_feat = True # Histogram features on or off
        hog_feat = True # HOG features on or off
        inFile = open("svc_pickle_ed.p", "rb" )
        svc = pickle.load(inFile)
        X_scaler = pickle.load(inFile)
        orient = pickle.load(inFile)
        pix_per_cell = pickle.load(inFile)
        cell_per_block = pickle.load(inFile) 
        spatial_size = pickle.load(inFile)
        hist_bins = pickle.load(inFile)
        inFile.close()
    # Uncomment the following line if you extracted training
    # data from .png images (scaled 0 to 1 by mpimg) and the
    # image you are searching is a .jpg (scaled 0 to 255)
    #image = image.astype(np.float32)/255
    #image_shape = image.shape
    #y_start_stop = [390,image_shape[0]]
    #windows = slide_window(image, x_start_stop=[None, None], y_start_stop=y_start_stop, 
    #                    xy_window=(int(64), int(64)), xy_overlap=(0.75, 0.75))
    #hot_windows = search_windows(image, windows, svc, X_scaler, color_space=color_space, 
    #                        spatial_size=spatial_size, hist_bins=hist_bins, 
    #                        orient=orient, pix_per_cell=pix_per_cell, 
    #                        cell_per_block=cell_per_block, 
    #                        hog_channel=hog_channel, spatial_feat=spatial_feat, 
    #                        hist_feat=hist_feat, hog_feat=hog_feat)                       
    
    #window_img = draw_boxes(draw_image,windows, color=(0, 0, 255), thick=6)                    
    #plt.imshow(window_img)
    
    #window_img = draw_boxes(draw_image,hot_windows, color=(0, 0, 255), thick=6)                    
    #plt.imshow(window_img)
    image = mpimg.imread('test_images/test6.jpg')
    boxlist = []
    for scale in [1,1.5]:
        ystart = 390
        ystop = 656
        boxlist += find_cars(image, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)
    e = draw_boxes(img, boxlist, color=(0, 0, 255), thick=6)
    plt.imshow(e)
    
    white_output = "P5_Final.mp4"
    clip1 = VideoFileClip("project_video.mp4")
    white_clip = clip1.fl_image(process_image)
    white_clip.write_videofile(white_output, audio=False)


