<<<<<<< HEAD:utils/video_loader.py
import os
import cv2
=======
>>>>>>> 0cc3d0f1930c05bf2d5399fef6e3bc062d92a96d:utils/video_loading.py
import glob
import sunpy.map
import numpy as np
from itertools import islice
import matplotlib.pyplot as plt

from IPython.display import Video

import warnings
warnings.filterwarnings('ignore')

<<<<<<< HEAD:utils/video_loader.py
=======

>>>>>>> 0cc3d0f1930c05bf2d5399fef6e3bc062d92a96d:utils/video_loading.py
class video_loader:
    def __init__(self):
            pass
        
    def build_images(self, path, results_path):
        backgrounds = sorted(glob.glob(path + "/*magnetogram*.fits"))
        foregrounds_pil = sorted(glob.glob(path + "/*_PIL*.png"))
        
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
            plt.imshow(ropi_mask, 'cool', interpolation='none', alpha=0.6)
            plt.imshow(pil_mask, cmap, interpolation='none', alpha=1)
            
            cb.set_label("LOS Magnetic Field [Gauss]")
            file_path = os.path.join(results_path, date + '.png')
            plt.ioff()
            plt.savefig(file_path)
        
        for bg in backgrounds:
            b_date = bg.split('_TAI')[0].split('.', 3)[-1].replace('_', '')
            
            for pil in foregrounds_pil:
                date = pil.split('_BLOS')[0].split('_', 2)[-1]
                pil_date = date.replace('-','').replace(':','').replace('_','')
                
                if(b_date == pil_date): 
                    ropi = glob.glob(path + "/*" + date + "*RoPI*.png")
                    apply_params(bg, pil, ropi[0], date)
                    
    def display_video(self, path):
        img_array = []
        file_name = 'mag.mp4'
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