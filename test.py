import numpy as np
import cv2
import math
import datetime

import matplotlib.pyplot as plt

# from image_changer_func import *
from pixel_warping_effect import *


def gauss(x, a=1, mu=0, sigma=10):
    return a * np.exp(-(x - mu)**2 / (2*sigma**2))

def dist(y=0, x=0):
    # return math.sqrt(y*y + x*x)
    return (y**2 + x**2)**0.5

def pixel_warping_effect2(src, center=(100,100), size=0, rad=0):
    dst = src.copy()
    height, width, colors = src.shape
    # print(height, width, colors)
    
    '''
    init declare
    '''
    power = size / 4
    move = [power*np.cos(np.radians(rad)), power*np.sin(np.radians(rad))]
    move = np.round(move, 3)
    # print(move)
    start = [center[0]-size, center[1]-size]
    finish = [center[0]+size, center[1]+size]
    
    '''
    make black square
    '''
    dst[start[0]:finish[0], start[1]:finish[1], :] = 0
    
    '''
    make warp square
    '''
    # stackで宣言する方法も考えてみる
    warp_height = np.tile(np.arange(height), width).reshape(width, height).T
    warp_width = np.tile(np.arange(width), height).reshape(height, width)
    warp_square = np.stack([warp_height, warp_width])
    warp_square = np.transpose(warp_square, (1,2,0))
    
    center_point_square = np.zeros([height, width, 2]) + center
    warp_dist_square = center_point_square - warp_square
    
    '''
    pixel warping
    '''
    warp_square = warp_square + (gauss(dist(warp_dist_square)) * move)
    print(warp_square)
    warp_square = warp_square.astype(int)
    print(warp_square, width, height)
    
    for y in range(start[0], finish[0]):
        for x in range(start[1], finish[1]):
            dst[y, x, :] = src[warp_square[y, x, 0], warp_square[y, x, 1], :]
    
    '''
    simple pixel complement
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
    '''
    
    
    '''
    plot
    '''
    scope = 0 # +:scale up, -:scale down
    for y in range(start[0], finish[0]):
        for x in range(start[1], finish[1]):
            plt.plot(warp_square[y,x,1], warp_square[y,x,0], marker='.')
    plt.show()
    
    cv2.circle(dst, center=(start[1], start[0]), radius=3, color=(255,0,0))
    cv2.circle(dst, center=(start[1], finish[0]), radius=3, color=(255,0,0))
    cv2.circle(dst, center=(finish[1], start[0]), radius=3, color=(255,0,0))
    cv2.circle(dst, center=(finish[1], finish[0]), radius=3, color=(255,0,0))
    
    return dst

# height 10 width 5
def test(height, width):
    '''
    mat_height = np.tile(np.arange(height), width).reshape(width, height).T
    mat_width = np.tile(np.arange(width), height).reshape(height, width)
    mat = np.stack([mat_height, mat_width])
    mat = np.transpose(mat, (1,2,0))
    print(mat_height)
    print(mat_width)
    print(mat)
    print(mat[0,0], mat[2,3], mat[7,4])
    '''
    pass
    

if __name__ == '__main__':
    # test(10, 5)
    '''
    src = cv2.imread('./input.jpg')
    height, width, color = src.shape
    src = cv2.resize(src, (int(width/2), int(height/2)))
    '''
    src = cv2.imread('./Lenna.bmp')
    dst = src.copy()
    # 変顔モード
    '''
    dst = pixel_warping_effect(dst, center=(200,400), size=80, rad=0)
    dst = pixel_warping_effect(dst, center=(200,320), size=80, rad=0)
    '''
    # 小顔モード
    '''
    dst = pixel_warping_effect(dst, center=(250,250), size=80, rad=75)
    dst = pixel_warping_effect(dst, center=(250,490), size=80, rad=285)
    dst = pixel_warping_effect(dst, center=(330,370), size=80, rad=0)
    '''
    # dst = oil_painter(dst)
    dst = pixel_warping_effect(dst, center=(100,100), size=80, rad=180)
    
    cv2.imshow('input', src)
    cv2.imshow('output', dst)
    cv2.waitKey()
    cv2.destroyAllWindows()