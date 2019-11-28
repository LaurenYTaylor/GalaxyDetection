'''
    Created October 2019.
    @author: Lauren Taylor, any questions please email lauren.y.taylor96@gmail.com.
    This is the main code used to find galaxies within the wide field image. It applies a binary threshold to the image, then uses the connected components algorithm to identify clusters of bright pixels. The image is rethresholded with a lower threshold to pick up finer detail around the previously found bright clusters. The permieter and area of the clusters is found, and used to calculate the compactness. Cluster with low compactness are recorded and moved into their own folder, as they may be galaxies with interesting structure. Finding kinds of galaxies is the aim of this code.
    
    '''
import cv2
import sys
import numpy as np
from matplotlib import pyplot as plt
from skimage import measure
from astropy.io import fits
from astropy.utils.data import get_pkg_data_filename
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from PIL import Image, ImageDraw
from astropy import wcs
import StarMask
from helper_functions import find_uncompact_clusters, move_uncompact_plots

MAX_ROW_PAD = 25
MAX_COL_PAD = 25

#Make all of the stars black
def mask_stars(filename, star_regions_file, new_data):
    sm = StarMask.StarMask(filename, star_regions_file)
    mask = sm.produce_mask(boolean_mask=1)
    new_data[mask] = 0
    return new_data

#Restrict the displayed intensities, this shows the galaxies better
def restrict_intensities(new_data, min, max):
    idxs = np.where(new_data<min)
    new_data[idxs]=min
    idxs = np.where(new_data>max)
    new_data[idxs]=max
    return new_data

#Rescale the intensities between 0 and 255, change type to uint8 (cv2 prefers it)
def rescale_intensities(data, min, max):
    mms = MinMaxScaler((min,max))
    img = mms.fit_transform(data)
    return img.astype(np.uint8)

#Observe the histogram if necessary to find the best threshold
def view_intensity_histogram(image):
    plt.subplot(2,1,1), plt.imshow(image,cmap = 'gray')
    plt.title('Original')
    plt.subplot(2,1,2), plt.hist(image.ravel(), 256)
    plt.title('Histogram')
    plt.show()
    plt.close()

#See the individual clusters highlighted within the larger field
def highlight_clusters(clusters):
    for cluster in clusters:
       cluster_highlight = np.zeros(threshold.shape)
       cluster_xs,cluster_ys = np.where(labels==cluster)
       cluster_highlight[cluster_xs,cluster_ys]=1
       plt.imshow(cluster_highlight, cmap='gray')
       plt.show()
       plt.close()

#Find clusters within the wide field
def find_clusters_within_cluster(blurred, threshold, cluster, labels):
    '''Analyses clusters of bright pixels within a 2D wide field image (with binary thresholding applied). A window is created centered on the cluster with some padding around it, and a new, lower binary threshold applied to bring up any faint, 'wispy' structure around galaxies. The lower threshold was not originally applied because of its tendency to highlight noise and image artefacts.
        
        Parameters
        __________
        
        blurred : numpy array
            The entire 2D wide field image with some Gaussian/median blurring applied.
        threshold : numpy array
            The 2D wide field image with (fairly high) binary thresholding applied.
        cluster : int
            The cluster number to be analysed.
        labels : numpy array
                The image array with the pixels numbered with their associated cluster number.
        __________
        
    '''
    rows, cols = np.where(labels==cluster)

    if (((np.max(rows)-np.min(rows)) and (np.max(cols)-np.min(cols))) <= 10):
        print(f"Cluster {cluster} less than 10x10 pixels")
        return None, None, None, None
    if ((np.max(rows)-np.min(rows)) >= (np.max(cols)-np.min(cols))*4):
        print(f"Cluster {cluster} too vertically oblong (probably a star)")
        return None, None, None, None

    print(f"Processing Cluster {cluster}...")
    min_row = np.max([np.min(rows)-MAX_ROW_PAD, 0])
    max_row = np.min([np.max(rows)+MAX_ROW_PAD, threshold.shape[0]])
    min_col = np.max([np.min(cols)-MAX_COL_PAD, 0])
    max_col = np.min([np.max(cols)+MAX_COL_PAD, threshold.shape[1]])

    original = blurred[min_row:max_row,min_col:max_col] #Only needed to plot a visual comparison with the thresholded image later on
    first_thresh = threshold[min_row:max_row,min_col:max_col] #Window centered around cluster with original threshold
    retval, new_thresh_all = cv2.threshold(blurred, 40, 255, cv2.THRESH_BINARY)
    new_thresh = new_thresh_all[min_row:max_row,min_col:max_col] #Window centered around cluster with new threshold

    retval, new_labels, new_stats, new_centroids = cv2.connectedComponentsWithStats(new_thresh)

    return original, first_thresh, new_thresh, new_labels

#Find the central cluster within the re-thresholded cluster window
def final_cluster(new_thresh, new_cluster, new_labels):
    ''' Locates the original cluster within the window with the new threshold applied. Returns a window with only that cluster displayed.
       
       Parameters
       __________
       
        new_thresh : numpy array
            A window centered around the new cluster with the new threshold applied.
        new_cluster : int
            The cluster number to be analysed.
        new_labels : numpy array
            The newly-thresholded image array with the pixels numbered with their associated cluster number.
        __________
        
    '''
    if new_cluster==0:
        print(f"Subcluster {new_cluster} is background")
        return None
    new_rows, new_cols = np.where(new_labels==new_cluster)
    new_thresh_centrex = int(np.round(new_thresh.shape[0]/2))
    new_thresh_centrey = int(np.round(new_thresh.shape[1]/2))
    if (new_thresh_centrex,new_thresh_centrey) not in list(zip(new_rows,new_cols)):
        print(f"Subcluster {new_cluster} isn't central")
        return None
    mask = np.zeros(new_thresh.shape, dtype=np.uint8)
    mask[new_rows, new_cols]=1
    return mask

#Find shape statistics for the final cluster
def compute_statistics(cluster):
    ''' Compute some intersting shape statistics for the final cluster.
    
    Parameters
    __________
    
    cluster : numpy array
        The window around the cluster with the new threshold applied.
    
    __________
    '''
    
    contours, hierarchy = cv2.findContours(cluster, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contour_length = np.round(cv2.arcLength(contours[0], True),2)
    area=np.round(cv2.contourArea(contours[0]),2)
    compactness = np.round((4*np.pi*area)/contour_length**2,3)
    return contour_length, area, compactness

#Plot the final cluster
def plot_final_cluster(original, first_threshold, final_threshold, cluster_num, centroids, save_dest='', show=1):
    fig, ax = plt.subplots(1, 3, figsize=(8,2))
    fig.subplots_adjust(top=0.8)
    plt.rcParams["axes.titlesize"] = 8
    ax[0].imshow(original, cmap='gray')
    ax[0].set_title(f"Original image: Cluster {cluster_num}")
    ax[1].imshow(first_threshold, cmap='gray')
    ax[1].set_title("Cluster found \n with threshold t=100")
    ax[2].imshow(final_threshold, cmap='gray')
    ax[2].set_title("Central subcluster with t=40 \n and other subclusters removed")
    if save_dest:
        save_final_cluster(final_threshold, save_dest, cluster_num, centroids)
    if show:
        plt.show()
    plt.close()

#Save the final cluster image to 'folder' and information to text file 'filename'
def save_final_cluster(cluster_img, save_dest, cluster_num, centroids):
    plt.savefig(f"{save_dest}/{cluster_num}.png")
    contour_length, area, compactness = compute_statistics(cluster_img)
    row, col = centroids[cluster_num]
    with open(f"{save_dest}/cluster_descriptions.txt", "a") as f:
        fstring=f"Cluster {cluster_num} - Original Centre Row: {int(row)}, Original Centre Column: {int(col)}, Shape Area: {area}, Contour length: {contour_length}, Compactness: {compactness} \n"
        f.write(fstring)


def main():
    filename1="final_A85_HSC-i_cal_shifted.fits"
    filename2="final_A85_HSC-g_cal.fits"
    star_regions_file="A85_regions.reg"
    BLUR_WINDOW=5
    folder=""
    max_int=0
    for fn in [filename1, filename2]:
        if fn==filename1:
            folder="ical-thresh40"
            max_int=2000
        else:
            folder="gcal-thresh40"
            max_int=800

        image_file = get_pkg_data_filename(fn)
        data = fits.open(image_file)[0].data

        #Make any NAN pixels black
        data[np.where(np.isnan(data))]=0

        star_masked=mask_stars(fn, star_regions_file, data)
        restricted_data=restrict_intensities(star_masked,0,max_int)
        img=rescale_intensities(restricted_data,0,255)
        cv2.imwrite(f"{folder}/unblurred.png", img)
        blurred = cv2.GaussianBlur(img, (BLUR_WINDOW, BLUR_WINDOW), 4)
        cv2.imwrite(f"{folder}/blurred.png", blurred)
        #Apply a global binary threshold to highlight the bright cluster centres
        retval1, threshold = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY)
        cv2.imwrite(f"{folder}/first_thresh.png", threshold)
        retval1, threshold2 = cv2.threshold(blurred, 40, 255, cv2.THRESH_BINARY)
        cv2.imwrite(f"{folder}/final_thresh.png", threshold2)
        continue
        #Get number of blobs
        retval, labels, stats, centroids = cv2.connectedComponentsWithStats(threshold)

        unique_clusters=np.unique(labels)
        len_unique_clusters=len(unique_clusters)

        for cluster in unique_clusters:
            print(f"{folder[:4]} Cluster {cluster}/{len_unique_clusters}")
            if cluster==0:
                continue

            orig, first_thresh, new_thresh, new_labels = find_clusters_within_cluster(blurred, threshold, cluster, labels)
            if new_thresh is not None:
                unique_new_clusters=np.unique(new_labels)
                for new_cluster in unique_new_clusters:
                    final = final_cluster(new_thresh, new_cluster, new_labels)
                    if final is not None:
                        print(f"Plotting Subcluster {new_cluster} from Cluster {cluster}")
                        plot_final_cluster(orig, first_thresh, final, cluster, centroids, save_dest=folder, show=0)

        #Find uncompact clusters
        cluster_nums = find_uncompact_clusters(folder, threshold=0.6, record_cluster_nums=1)
        #move_uncompact_plots(folder, "uncompact_clusters.txt", "uncompact")

if __name__ == "__main__":
    main()
