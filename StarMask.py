from PIL import Image, ImageDraw
import numpy as np
from astropy import wcs
from astropy.io import fits

'''
    A class that creates a mask for stars in the wide field image. Used in image_processor.py.
'''

class StarMask(object):
    def __init__(self, image_file, regions_file):
        self.image_file = image_file
        self.regions_file = regions_file
    
    def getCircle(self, regName):
        lines  = open(regName).readlines()
        coords = []
        radius = []
        for line in lines:
            if 'circle(' in line:
                line   = line.replace('circle(','').split(')')
                coord  = line[0].split(',')
                coord2 = coord[2].split('"')
                P      = [np.float(coord[0]), np.float(coord[1])]
                rad    = np.float(coord2[0])
                coords.append(P)
                radius.append(rad)
        return coords, radius

    def produce_mask(self, boolean_mask=0):
        ### Read the wcs of the image :
        im, hdr         = fits.getdata(self.image_file, header = True)
        wcs_im          = wcs.WCS(hdr)
        pixscale        = 0.17  # arcsec/pixel

        #### Read the regions
        coords, radius = self.getCircle(self.regions_file)
        height, width  = np.shape(im)
        mask_poly      = Image.new('L', (width, height), 0)
        coords_reg     = wcs_im.wcs_world2pix(coords, 0)
        mask           = np.zeros_like(im)


        for poly in np.arange(0, len(coords_reg)):
                x,y = np.array(coords_reg[poly])
                x   = np.float(x)
                y   = np.float(y)
                rad = np.round(np.float(radius[poly])/pixscale)
                ImageDraw.Draw(mask_poly).ellipse((x-rad, y-rad, x+rad,y+rad), fill = 1)

        mask_poly        = np.array(mask_poly, dtype = bool)
        if boolean_mask:
            return mask_poly
        else:
            mask[mask_poly]  = np.nan
            return mask
### So you end up with a mask where the good values  are 0 and the bad ones (the stars) are nan. But you can also  leave it in the boolean one.
