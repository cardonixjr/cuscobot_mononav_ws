#!/usr/bin/env python3
import cv2
import numpy as np

def equalizeHist(img):
    '''Histogram equalization'''
    return cv2.equalizeHist(img)

def clahe(img):
    '''Contrast Limited Adaptive Histogram Equalization'''
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    return clahe.apply(img)

def laplacian(img):
    '''Laplacian Edge augmentation'''
    lap = cv2.Laplacian(img, cv2.CV_64F)      
    sharp = img - 0.5 * lap
    return np.clip(sharp, 0, 255).astype(np.uint8)

def normalize(img):
    '''Normalizarino'''
    return cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    
def edges(img):
    '''sobel multi-scale augmentation'''  
    edges = cv2.Canny(img, 50, 150)
    return cv2.addWeighted(img, 0.7, edges, 0.3, 0)

def sharpening(img):
    kernel = np.array([[-1, -1, -1],
                               [-1,  9, -1],
                               [-1, -1, -1]])

    return cv2.filter2D(img, -1, kernel)
