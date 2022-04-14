#!/usr/bin/env python
# coding: utf-8

# In[1]:


import glob
import joblib
import pandas as pd
import numpy as np
from itertools import islice

import sunpy.map
from sunpy.coordinates import frames
from sunpy.map.mapbase import GenericMap
from sunpy.map.sources import MDIMap
from sunpy.util.metadata import MetaDict

import matplotlib.pyplot as plt

import cv2
import os
from PIL import Image

from IPython.display import Video

import warnings
warnings.filterwarnings('ignore')


# In[5]:


class video_loader:
    def __init__(self):
            pass
        
    def build_images(self, b_path, f_path):
        backgrounds = glob.glob(b_path + "/*magnetogram*.fits") 
        foregrounds_pil = glob.glob(f_path + "/*_PIL*.png")
        foregrounds_ropi = glob.glob(f_path + "/*RoPI*.png")
        
        def mask_img(img):
            return np.ma.masked_where(img.astype(float) == 0, img.astype(float))
      
        def apply_params(background, pil, ropi, date):
            hmi_magmap = sunpy.map.Map(background)
            
            ropi_mask = mask_img(plt.imread(ropi))
            pil_mask = mask_img(plt.imread(pil))
            
            cmap = plt.cm.spring
            cmap = cmap.set_bad(color='white')

            fig = plt.figure(figsize=(10,8))
            hmi_magmap.plot()
            plt.xlabel('Carrington Longitude [deg]', fontsize = 16)
            plt.ylabel('Latitude [deg]', fontsize = 16)
            plt.imshow(ropi_mask, 'cool', interpolation='none', alpha=0.2)
            plt.imshow(pil_mask, cmap, interpolation='none', alpha=1)
            
            file_path = os.path.join("/home/nkhasayeva1/Results", date + '.png')
            plt.savefig(file_path)
        
        for bg in backgrounds:
            b_date = bg.split('_TAI')[0].split('.', 3)[-1].replace('_', '')
            print(b_date)
            for pil in foregrounds_pil:
                date = pil.split('_BLOS')[0].split('_', 2)[-1]
                pil_date = date.replace('-','').replace(':','')
                print(pil_date)
                if(b_date == pil_date): 
                    print('hi')
                    ropi = glob.glob(f_path + "/*" + date + "*RoPI*.png")
                    apply_params(bg, pil, ropi[0], date)
                    
    def display_video(self, path):
        img_array = []
        file_name = 'mag.mp4'
        for filename in glob.glob(path + '*.png'): 
            img = cv2.imread(filename)
            height, width, layers = img.shape
            size = (width, height)
            img_array.append(img)
        
            out = cv2.VideoWriter(file_name, cv2.VideoWriter_fourcc(*'VP90'), 1, size)
         
        for i in range(len(img_array)):
            out.write(img_array[i])
            out.release()
        
        Video(file_name, embed=True)

