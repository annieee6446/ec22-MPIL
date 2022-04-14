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
        """
        A constructor that initializes the class variables.
        """
        pass
    
    def build_images(self, path, results_path):
        backgrounds = sorted(glob.glob(path + "/*magnetogram*.fits"))
        foregrounds_pil = sorted(glob.glob(path + "/*_PIL*.png"))
        
        def mask_img(img):
            return np.ma.masked_where(img.astype(float) == 0, img.astype(float))
      
        def apply_params(background, pil, ropi, chpil, date):
            hmi_magmap = sunpy.map.Map(background)
     
            ropi_mask = mask_img(plt.imread(ropi))
            pil_mask = mask_img(plt.imread(pil))
            chpil_mask =  mask_img(plt.imread(chpil))
            
            cmap = plt.cm.spring
            cmap = cmap.set_bad(color='white')
            
            fig = plt.figure(figsize=(10,8))
            
            hmi_magmap.plot_settings['cmap'] = 'hmimag'
            hmi_magmap.plot_settings['norm'] = plt.Normalize(-1500, 1500)
            im_hmi = hmi_magmap.plot()
            cb = plt.colorbar(im_hmi, fraction=0.019, pad=0.1)
            
            plt.xlabel('Carrington Longitude [deg]',fontsize = 16)
            plt.ylabel('Latitude [deg]',fontsize = 16)
            plt.imshow(chpil_mask, 'bone', interpolation='none', alpha=0.6)
            plt.imshow(ropi_mask, 'cool', interpolation='none', alpha=0.8)
            plt.imshow(pil_mask, cmap, interpolation='none', alpha=1)
            
            cb.set_label("LOS Magnetic Field [Gauss]")
            file_path = os.path.join(results_path, date + '.png')
            plt.savefig(file_path)
        
        for bg in backgrounds:
            b_date = bg.split('_TAI')[0].rsplit('.')[-1].replace('_', '')

            for pil in foregrounds_pil:
                date = pil.split('_BLOS')[0].split('_', 2)[-1]
                pil_date = date.replace('-','').replace(':','').replace('_','')

                if(b_date == pil_date): 
                    ropi = glob.glob(path + "/*" + date + "*RoPI*.png")
                    chpil = glob.glob(path + "/*" + date + "*CHPIL*.png")
                    apply_params(bg, pil, ropi[0], chpil[0], date)
                    
                        
    def display_video(self, path):
        img_array = []
        file_name = path + '_video/mag.mp4'
        size = (None, None)
        for filename in sorted(glob.glob(path + '/*.png')): 
            img = cv2.imread(filename)
            height, width, layers = img.shape
            size = (width, height)
            img_array.append(img)
        out = cv2.VideoWriter(file_name, cv2.VideoWriter_fourcc(*'VP90'), 2, size)
         
        for i in range(len(img_array)):
            out.write(img_array[i])
        out.release()
        
        return Video(file_name, embed=True)