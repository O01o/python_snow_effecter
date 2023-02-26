import numpy as np
import cv2
import math

import datetime

# import matplotlib.pyplot as plt

def gauss(x, a=1, mu=0, sigma=30):
    return a * np.exp(-(x - mu)**2 / (2*sigma**2))

def dist(y=0, x=0):
    return math.sqrt(y**2 + x**2)

def include_range(min, max):
    return int(min)+1, int(max)+1

def linalg_mapping(a, x):
    # a = [[1,2,3],[4,5,6],[7,8,9]]
    # print(a[0][2], a[0])
    # x = [1,2,3]
    y = []
    for i in range(len(a)):
        yi = 0
        if len(x) != len(a[i]):
            return 0
        for j in range(len(x)):
            yij = x[j] * a[i][j]
            # print(yij, x[j], a[i][j])
            yi += yij
        # print()
        yi = round(yi, 3)
        y.append(yi)
    
    return y[0], y[1]


def pixel_warping_effect(src, center=(100,100), size=0, rad=0):
    dst = src.copy()
    height, width, colors = src.shape
    print(height, width, height*width)
    
    power = size / 3
    move = [power*np.sin(np.radians(rad)), power*np.cos(np.radians(rad))]
    move = np.round(move, 3)
    print(move)
    start = [center[0]-size, center[1]-size]
    finish = [center[0]+size, center[1]+size]
    
    '''
    map = np.zeros([height, width, 2])
    for y in range(height):
        for x in range(width):
            map[y, x, :] = [y, x]
    
    for y in range(start[0], finish[0]):
        for x in range(start[1], finish[1]):
            map[y, x, :] += [
                move[1] * gauss(dist(center[0]-y, center[1]-x)),
                move[0] * gauss(dist(center[0]-y, center[1]-x))
            ]
            map_int = map.copy()
            map_int = map.astype(int)
    '''
    
    '''
    black square
    '''
    # dst[start[0]:finish[0], start[1]:finish[0], :] = 0
    
    '''
    make gaussian warping map
    '''
    for i in range(size*2):
        for j in range(size*2):
            y = start[0] + i
            x = start[1] + j
            ydash = int(y + (move[1] * gauss(dist(size-i, size-j))))
            xdash = int(x + (move[0] * gauss(dist(size-i, size-j))))
            dst[y, x, :] = src[ydash, xdash, :]
            
    '''
    simple pixel warping
    '''
    '''
    for y in range(start[0], finish[0]):
        for x in range(start[1], finish[1]):
            dst[map_int[y,x,0], map_int[y,x,1], :] = src[y, x, :]
    '''
    
    '''
    complement color from around pixel
    (+ simple pixel warping)
    '''
    '''
    for y in range(start[0], finish[0]):
        for x in range(start[1], finish[1]):
            count = 0
            if (dst[y, x, :] == 0).any():
                pixel = np.zeros([3])
                if (dst[y-1, x-1, :] != 0).any():
                    pixel += dst[y-1, x-1, :]
                    count += 1
                if (dst[y-1, x, :] != 0).any(): 
                    pixel += dst[y-1, x, :]
                    count += 1
                if (dst[y-1, x+1, :] != 0).any(): 
                    pixel += dst[y-1, x+1, :]
                    count += 1
                if (dst[y, x-1, :] != 0).any(): 
                    pixel += dst[y, x-1, :]
                    count += 1
                if (dst[y, x+1, :] != 0).any(): 
                    pixel += dst[y, x+1, :]
                    count += 1
                if (dst[y+1, x-1, :] != 0).any(): 
                    pixel += dst[y+1, x-1, :]
                    count += 1
                if (dst[y+1, x, :] != 0).any(): 
                    pixel += dst[y+1, x, :]
                    count += 1
                if (dst[y+1, x+1, :] != 0).any(): 
                    pixel += dst[y+1, x+1, :]
                    count += 1
                if count != 0:            
                    pixel /= count
                    dst[y, x, :] = pixel
    print(count)
    '''
    
    '''
    homography complement pixel warping
    '''
    '''
    count = 0
    for y in range(start[0], finish[0]-2):
        for x in range(start[1], finish[1]-2):
            count += 1
            print(count)
            quad = np.float32([
                map[y,x,:], map[y+1,x,:], map[y,x+1,:], map[y+1,x+1,:]
            ])
            quad_int = np.array([
                map_int[y,x,:], map_int[y+1,x,:], 
                map_int[y,x+1,:], map_int[y+1,x+1,:]
            ])
            quad_dash = np.float32([
                [0,0], [1,0], [0,1], [1,1]
            ])
            homography = cv2.getPerspectiveTransform(quad, quad_dash)
            
            xy_minmax = np.array([
                np.amin(quad, axis=0),
                np.amax(quad, axis=0)
            ])
            min_y, max_y = include_range(xy_minmax[0,0], xy_minmax[1,0])
            min_x, max_x = include_range(xy_minmax[0,1], xy_minmax[1,1])
            
            node_list = []            
            for y in range(min_y, max_y):
                for x in range(min_x, max_x):
                    node_list.append([y,x,1])
                    
            for node in node_list:
                py, px = linalg_mapping(homography, node)
                if py >= 0 and py <= 1 and px >= 0 and px <= 1:
                    try:
                        dst[node[0], node[1], :] = (
                            (1-py) * (1-px) * src[quad_int[0,0], quad_int[0,1], :] + 
                            py * (1-px) * src[quad_int[1,0], quad_int[0,1], :] + 
                            (1-py) * px * src[quad_int[0,0], quad_int[1,1], :] + 
                            py * px * src[quad_int[1,0], quad_int[1,1], :]
                        )
                    except IndexError:
                        pass
                    
            print(count, node_list, quad_int)
    '''
    
    '''
    plot
    '''
    '''
    scope = 0 # +:scale up, -:scale down
    for y in range(center[0]-(size-scope), center[0]+(size-scope)):
        for x in range(center[1]-(size-scope), center[1]+(size-scope)):
            
            plt.plot(map[y,x,1], map[y,x,0], marker='.')
    plt.show()
    '''
    
    
    return dst