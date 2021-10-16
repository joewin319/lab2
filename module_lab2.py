# -*- coding: utf-8 -*-
"""
Created on Sun Sep 20 10:40:06 2020

@author: haris
"""
from scipy.fft import dct, idct
from numpy import r_
import numpy as np
import os
import matplotlib.pyplot as plt
import cv2


#%% Student code definition

# Write the function to compute the PSNR and SNR
# computePSNR(ori, recon) and computeSNR(ori, recon)
def computePSNR(ori, recon):
    psnr=0.0
    #Your code
    ori=ori.astype('float64')
    recon=recon.astype('float64')
    mse=np.mean((ori-recon)**2)
    searchforpeak=ori**2
    peak=np.max(searchforpeak)
    psnr=10*np.log10(peak/mse)
    return psnr
      

def computeSNR(ori, recon):
    snr=0.0
    #Your code
    ori=ori.astype('float64')
    recon=recon.astype('float64')
    x=np.mean(ori**2)
    mse=np.mean((ori-recon)**2)
    snr=10*np.log10(x/mse)
    return snr

def RLencode(vec):
    tuppleList = None
    # Your code 
    tuppleList=[]
    count=0
    for i in vec[:-1]:
        if i==0:
            count=count+1
        else:
            tuppleList.append((count,i))
            count=0
    if vec[-1]==0:
        tuppleList.append((0,0))
    else:
        tuppleList.append((count,vec[-1]))
    
    return tuppleList

#%% DCT Transform Followed by Masking

def compressImage(filename, mask):
    im = readImageAsGray(filename) # convert to gray as well 
    im = resize(im,8,8) # resize to be multiple of 8 rows and col
    dctIm=None 
    size= None
    dctIm=im*mask
    size=dctIm.shape[0]*dctIm.shape[1]
    # Your code
    return (dctIm, size)

def compressImageThres(filename, thres):
    im = readImageAsGray(filename) # convert to gray as well
    im = resize(im,8,8) # resize to be multiple of 8 rows and col
    dctIm=None 
    size= None
    mask=np.full((8,8), thres)
    print(mask)
    dctIm=im*mask
    size=dctIm.shape[0]*dctIm.shape[1]
    # your code
    return (dctIm, size)


def unCompressImage(dctIm):
    imsize = dctIm.shape
    imRecon = np.zeros(imsize)
    imRecon=None 
    #your code
            
    return imRecon


#%%
def readImageAsGray(filename):
  img = cv2.imread(filename)
  img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  #print(f' {imgfile} has shape {img.shape} ')
  return img


def showImage(imgfile, str):
  plt.figure(figsize = (10,10))
  img = cv2.imread(imgfile)
  img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  plt.xticks([]),plt.yticks([])
  plt.imshow(img, cmap='gray')
  plt.title(str)
  print(f'Image shape for {imgfile} = {img.shape} ')


#Assume 10 bits for each rlc code
# Get data size from run length code extracted from each 8x8 block
def getDataSize(mat):
    mat = np.round(mat,2)
    series = zigzag(mat)
    rlc = RLencode(series)
    dataSize = len(rlc)
    return dataSize*10 # assume 10 bits fixed length code for each symbol, size in bits

def getBaseName(filename):
    base =  os.path.basename(filename)
    ext =os.path.splitext(base)[1]
    baseName= os.path.splitext(base)[0]
    return baseName

def show2DMatrix(mat):
    (nr,nc) = mat.shape
    for i in r_[:nr]:
        for j in r_[:nc] :
            print("{:05.02f}".format(mat[i,j]),  end=' ')
        
        print("\n")
        

# DCT Transform using scipy library function
def resize(mat,nr,nc):
    numRow = mat.shape[0]
    numCol = mat.shape[1]
    remR = numRow % nr
    remC =  numCol % nc
    if remR==0: remR = -numRow
    if remC==0: remC = -numCol
    mat2 = mat[:-remR,:-remC]
    return mat2

# implement 2D DCT
def dct2(a, mask):
    dctMat = dct( dct( a, axis=0, norm='ortho' ), axis=1, norm='ortho' )
    dctMat2 = dctMat * mask
    return dctMat2
 
def dct2_thres(a, thresh):
    dctMat = dct( dct( a, axis=0, norm='ortho' ), axis=1, norm='ortho' )
    dct_thresh = dctMat * (abs(dctMat) > (thresh*np.max(dctMat)))
    return dct_thresh

# implement 2D IDCT
def idct2(a):
    return idct( idct( a, axis=0 , norm='ortho'), axis=1 , norm='ortho')   

#%%
# from Matlab Code:
# Alexey S. Sokolov a.k.a. nICKEL, Moscow, Russia
# June 2007
# alex.nickel@gmail.com
# # Function returns a 1-by-(m*n) array

def zigzag(input):
    #initializing the variables
    #----------------------------------
    h = 0
    v = 0

    vmin = 0
    hmin = 0

    vmax = input.shape[0]
    hmax = input.shape[1]
    
    #print(vmax ,hmax )

    i = 0

    output = np.zeros(( vmax * hmax))
    #----------------------------------

    while ((v < vmax) and (h < hmax)):
    	
        if ((h + v) % 2) == 0:                 # going up
            
            if (v == vmin):
            	#print(1)
                output[i] = input[v, h]        # if we got to the first line

                if (h == hmax):
                    v = v + 1
                else:
                    h = h + 1                        

                i = i + 1

            elif ((h == hmax -1 ) and (v < vmax)):   # if we got to the last column
            	#print(2)
            	output[i] = input[v, h] 
            	v = v + 1
            	i = i + 1

            elif ((v > vmin) and (h < hmax -1 )):    # all other cases
            	#print(3)
            	output[i] = input[v, h] 
            	v = v - 1
            	h = h + 1
            	i = i + 1

        
        else:                                    # going down

        	if ((v == vmax -1) and (h <= hmax -1)):       # if we got to the last line
        		#print(4)
        		output[i] = input[v, h] 
        		h = h + 1
        		i = i + 1
        
        	elif (h == hmin):                  # if we got to the first column
        		#print(5)
        		output[i] = input[v, h] 

        		if (v == vmax -1):
        			h = h + 1
        		else:
        			v = v + 1

        		i = i + 1

        	elif ((v < vmax -1) and (h > hmin)):     # all other cases
        		#print(6)
        		output[i] = input[v, h] 
        		v = v + 1
        		h = h - 1
        		i = i + 1




        if ((v == vmax-1) and (h == hmax-1)):          # bottom right element
        	#print(7)        	
        	output[i] = input[v, h] 
        	break

    #print ('v:',v,', h:',h,', i:',i)
    return output

#%%

#%% Mask type
# Use 64 coefficients
mask64 = np.ones((8,8))

# Use 54 coefficients
mask54 =np.array( [[1, 1, 1, 1, 1, 1, 1, 1],
                  [1, 1, 1, 1, 1, 1, 1, 1],
                  [1, 1, 1, 1, 1, 1, 1, 1],
                  [1, 1, 1, 1, 1, 1, 1, 1],
                  [1, 1, 1, 1, 1, 1, 1, 0],
                  [1, 1, 1, 1, 1, 1, 0, 0],
                  [1, 1, 1, 1, 1, 0, 0, 0],
                  [1, 1, 1, 1, 0, 0, 0, 0]
                  
                 ]
               )


# Use 49 coefficients
mask49 =np.array( [[1, 1, 1, 1, 1, 1, 1, 1],
                  [1, 1, 1, 1, 1, 1, 1, 1],
                  [1, 1, 1, 1, 1, 1, 1, 1],
                  [1, 1, 1, 1, 1, 1, 1, 0],
                  [1, 1, 1, 1, 1, 1, 0, 0],
                  [1, 1, 1, 1, 1, 0, 0, 0],
                  [1, 1, 1, 1, 0, 0, 0, 0],
                  [1, 1, 1, 0, 0, 0, 0, 0]
                  
                 ]
               )

# Use 43 coefficients
mask43 =np.array( [[1, 1, 1, 1, 1, 1, 1, 1],
                  [1, 1, 1, 1, 1, 1, 1, 1],
                  [1, 1, 1, 1, 1, 1, 1, 0],
                  [1, 1, 1, 1, 1, 1, 0, 0],
                  [1, 1, 1, 1, 1, 0, 0, 0],
                  [1, 1, 1, 1, 0, 0, 0, 0],
                  [1, 1, 1, 0, 0, 0, 0, 0],
                  [1, 1, 0, 0, 0, 0, 0, 0]
                  
                 ]
               )


# Use only 10 coefficients
mask10 =np.array( [[1, 1, 1, 1, 0, 0, 0, 0],
                  [1, 1, 1, 0, 0, 0, 0, 0],
                  [1, 1, 0, 0, 0, 0, 0, 0],
                  [1, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0]
                  
                 ]
               )

# Use only 3 coefficients
mask3 =np.array([[1, 1, 0, 0, 0, 0, 0, 0],
                  [1, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0]
                  
                 ]
               )
mask6 = np.array([[1, 1, 1, 0, 0, 0, 0, 0],
                  [1, 1, 0, 0, 0, 0, 0, 0],
                  [1, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0]
                  
                 ]
               )

mask21=np.array([[1, 1, 1, 1, 1, 1, 0, 0],
                  [1, 1, 1, 1, 1, 0, 0, 0],
                  [1, 1, 1, 1, 0, 0, 0, 0],
                  [1, 1, 1, 0, 0, 0, 0, 0],
                  [1, 1, 0, 0, 0, 0, 0, 0],
                  [1, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0]
                  
                 ]
               )

