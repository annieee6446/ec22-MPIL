import os
import cv2
import glob
import sunpy.map
import numpy as np
from itertools import islice
import matplotlib.pyplot as plt

from IPython.display import Video

import warnings
warnings.filterwarnings('ignore')

class video_loader:
    def __init__(self):
            pass
        
    def build_images(self, bg_path, fg_path, results_path):
        backgrounds = orderedglob.glob(bg_path + "/*magnetogram*.fits") 
        foregrounds_pil = glob.glob(fg_path + "/*_PIL*.png")
        foregrounds_ropi = glob.glob(fg_path + "/*RoPI*.png")
        
        def mask_img(img):
            return np.ma.masked_where(img.astype(float) == 0, img.astype(float))
      
        def apply_params(background, pil, ropi, date):
            hmi_magmap = sunpy.map.Map(background)
            
            ropi_mask = mask_img(plt.imread(ropi))
            pil_mask = mask_img(plt.imread(pil))
            
            cmap = plt.cm.spring
            cmap = cmap.set_bad(color='white')
            
            fig = plt.figure(figsize=(10,8))
            
            hmi_magmap.plot_settings['cmap'] = 'hmimag'
            hmi_magmap.plot_settings['norm'] = plt.Normalize(-1500, 1500)
            im_hmi = hmi_magmap.plot()
            cb = plt.colorbar(im_hmi, fraction=0.019, pad=0.1)
            
            plt.xlabel('Carrington Longitude [deg]',fontsize = 16)
            plt.ylabel('Latitude [deg]',fontsize = 16)
            plt.imshow(mask, cmap, interpolation='none', alpha=1)
            plt.imshow(mask1, 'cool', interpolation='none', alpha=0.6)
            
            cb.set_label("LOS Magnetic Field [Gauss]")
            
            file_path = os.path.join(results_path, date + '.png')
            plt.savefig(file_path)
        
        for bg in backgrounds:
            b_date = bg.split('_TAI')[0].split('.', 3)[-1].replace('_', '')
            for pil in foregrounds_pil:
                date = pil.split('_BLOS')[0].split('_', 2)[-1]
                pil_date = date.replace('-','').replace(':','')
                if(b_date == pil_date): 
                    ropi = glob.glob(fg_path + "/*" + date + "*RoPI*.png")
                    apply_params(bg, pil, ropi[0], date)
                    
    def display_video(self, path):
        img_array = []
        file_name = 'mag.mp4'
        for filename in sorted(glob.glob(path + '*.png')): 
            img = cv2.imread(filename)
            height, width, layers = img.shape
            size = (width, height)
            img_array.append(img)
        
            out = cv2.VideoWriter(file_name, cv2.VideoWriter_fourcc(*'VP90'), 1, size)
         
        for i in range(len(img_array)):
            out.write(img_array[i])
            out.release()
        
        Video(file_name, embed=True)