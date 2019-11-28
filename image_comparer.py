import cv2
from astropy.io import fits
import numpy as np
from image_processor import restrict_intensities, rescale_intensities
from matplotlib import pyplot as plt

'''
    This code stacks two wide field images in different colour bands to emphasise any differences.
    
'''

filename1="final_A85_HSC-i_cal_shifted.fits"
filename2="final_A85_HSC-g_cal.fits"

data1 = fits.open(filename1)[0].data[:20889,:24890]
data2 = fits.open(filename2)[0].data

data1[np.where(np.isnan(data1))]=0
data2[np.where(np.isnan(data2))]=0

data1=restrict_intensities(data1,0,800)
data2=restrict_intensities(data2,0,2000)

data1=rescale_intensities(data1,0,255)
data2=rescale_intensities(data2,0,255)

print(data1.shape)
print(data2.shape)

print("Fusing...")
fused = np.dstack((data1,data2, data1))

print("Plotting...")
plt.imshow(fused)
plt.gca().invert_yaxis()
plt.show()
