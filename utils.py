#!/usr/bin/env python
# coding: utf-8

# In[77]:


import cv2
import numpy as np
from skimage import morphology
import matplotlib.pyplot as plt
import copy
#np.set_printoptions(threshold='nan')


# In[97]:
#calculate the heatmap
def returnHeatmap(feature_conv, weight_softmax, class_idx):
    # generate the class activation maps upsample to 256x256
    size_upsample = (224, 224)
    bz, nc, h, w = feature_conv.shape
    output_cam = []
    for idx in class_idx:
        cam = weight_softmax[idx].dot(feature_conv.reshape((nc, h*w)))
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)
        output_cam.append(cv2.resize(cam_img, size_upsample))
    return output_cam

# In[102]:

#extract interest points
def surf(input):
    

    surf = cv2.xfeatures2d.SURF_create(400)
    kp,desc_query = surf.detectAndCompute(input,None)

    pts = []
    for i in range(len(kp)):
        pts.append([kp[i].pt[0], kp[i].pt[1]])

    pts = np.array(pts)
    pts = pts.astype(int)
    #print(pts)
    minx, miny = np.min(pts, axis=0)
    maxx, maxy = np.max(pts, axis=0)
    print(minx, miny, maxx, maxy)
    #draw bounding box from surf
    #surf_image=cv2.drawKeypoints(input,kp, input)
    #img = cv2.rectangle(input, (minx, miny), (maxx, maxy), (255,0,0))
    #cv2.imwrite('tmp/surf.jpg', img)
    #cv2.imwrite('tmp/surf_image.jpg', surf_image)
    corner = [minx, miny, maxx, maxy]
    return  corner, kp
    #cv2.imshow('sp',img)
    #cv2.waitKey(0)


# In[ ]:

#calculate the topk maximum value (not usedi in final version)
def topKmax(input, k):
    flatten = input.flatten()
    max_index = np.argsort(flatten)
    max_loc = []
    for i in range(k):
        max_loc.append(np.unravel_index(max_index[-1-i], input.shape))
        
    return max_loc


# In[ ]:

#calculate the IOU (not used in final version)
def IOU(box1, box2):
    xi1 = max(box1[0],box2[0])
    yi1 = max(box1[1],box2[1])
    xi2 = min(box1[2],box2[2])
    yi2 = min(box1[3],box2[3])
    
    box1_area = (box1[2]-box1[0])*(box1[3]-box1[1])
    box2_area = (box2[2]-box2[0])*(box2[3]-box2[1])
    union_area = box1_area + box2_area - inter_area
    
    iou = inter_area / union_area
    
    return iou


# In[ ]:

#detect if the central point exits
def central_point_detection(point, box):
    
    if point[0] > box[0] and point[0] < box[1] and point[1] > box[2] and point[1] < box[3]:
        return True
    else:
        return False

# In[133]:

#find the central point of heatmap
def calculate_value(input, point):
    height, width = input.shape
    sum = 0
    for i in range(width):
        sum += input[point[0]][i]
    for j in range(height):
        sum += input[j][point[1]]
    sum -= input[point[0]][point[1]]
    return sum


# In[131]:


def find_central_point(input):
    height, width = input.shape
    maximum = 0
    maxi_loc = []
    sum = 0
    for i in range(height):
        for j in range(width):
            sum = calculate_value(input, [i, j])
            if sum > maximum:
                maximum = sum
                maxi_loc = [i, j]
            sum = 0
    print(maximum)            
    return maxi_loc


#cross filter
def cross_filter(input):
    height, width = input.shape
    output = np.zeros((height, width))
    maximum = 0
    maxi_loc = []
    sum = 0
    for i in range(height):
        for j in range(width):
            vert = input[:, j]
            horiz = input[i,:]

            sum =  np.sum(vert) + np.sum(horiz) - input[i][j]
            output[i][j] = sum
            if sum > maximum:
                maximum = sum
                maxi_loc = [i, j]
            sum = 0
    print(maximum)
    
    output = output - np.min(output)
    output = output/np.max(output)
    output = np.uint8(255 * output)
    
    return output, maxi_loc
#sum filter
def sum_filter(input, kernel_size=300):
    height, width = input.shape
    print(input.shape)
    num_pad = int(kernel_size / 2)
    padded = np.pad(input, ((num_pad, num_pad), (num_pad, num_pad)), 'constant', constant_values=0)
    output = np.zeros((height, width))
    kernel = np.ones((kernel_size, kernel_size))
    maximum = 0
    maxi_loc = []

    for h in range(height):
        for w in range(width):
            vert_start = h
            vert_end = h + kernel_size
            horiz_start = w
            horiz_end = w + kernel_size
            
            a_slice = padded[vert_start:vert_end, horiz_start:horiz_end]
            output[h,w] = np.sum(np.multiply(a_slice, kernel))
            if output[h,w] > maximum:
                maximum = output[h,w]
                maxi_loc = [h,w]
    
    output = output - np.min(output)
    output = output/np.max(output)
    output = np.uint8(255 * output)
    
    #cv2.imwrite('tmp/accumulate_around_pixels.jpg', output)
            
    return output, maxi_loc

#find the left-top and right-bottom point
def find_corner(input, threshold):
    x_array, y_array = np.where(input>threshold)

    x_min = np.min(x_array)
    x_max = np.max(x_array)
    y_min = np.min(y_array)
    y_max = np.max(y_array)

    return [x_min, y_min], [x_max, y_max]