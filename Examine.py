#%%
import cv2
from numpy.core.defchararray import title
import skimage
import numpy as np
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.color import label2rgb
from skimage import io
import matplotlib.pyplot as plt
from skimage.morphology import closing, square
import matplotlib.patches as mpatches
    
# %%
###Functions
def GetAreas(img, verbose=False,
    min_size = 500,  
    max_size = 900,
    number_of_bacteria = -1
    ):
    blur = cv2.boxFilter(img,-1,(5,5), normalize = True)
    if verbose:
        plt.imshow(blur)
        plt.show()
    ret,thresh1 = cv2.threshold(blur,90,255,cv2.THRESH_BINARY)
    thresh1=255-thresh1
    if verbose:
        plt.imshow(thresh1)
    kernel = np.ones((5,5),np.uint8)
    closing = cv2.morphologyEx(thresh1, cv2.MORPH_CLOSE, kernel)
    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)
    if verbose:
        plt.imshow(opening)
        plt.show()
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(opening, connectivity=8)
    sizes = stats[1:, -1]; nb_components = nb_components - 1

    #your answer image
    img2 = np.zeros((output.shape))
    #for every component in the image, you keep it only if it's above min_size
    for i in range(0, nb_components):
        if max_size>=sizes[i] >= min_size:
            img2[output == i + 1] = 255
    if verbose:
        plt.imshow(img2)
        plt.show()
    cleared = clear_border(img2)
    if verbose:
        plt.imshow(cleared)
        plt.show()
    label_image = label(cleared)
    if verbose:
        plt.imshow(label_image)
        plt.show()
    image_label_overlay = label2rgb(label_image, image=cleared, bg_label=0)
    areas = []
    if number_of_bacteria==-1:
        number_of_bacteria=len(regionprops(label_image))
    for region in regionprops(label_image):
        areas.append(region.area)
    if verbose:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.imshow(image_label_overlay)
    
        for region in regionprops(label_image):
            # take regions with large enough areas
            # draw rectangle around segmented coins
            if region.feret_diameter_max>30:
                minr, minc, maxr, maxc = region.bbox
                rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                        fill=False, edgecolor='red', linewidth=2)
                ax.add_patch(rect)

        ax.set_axis_off()
        plt.tight_layout()
        plt.show()
    #print(np.mean(areas), np.std(areas))
    return areas,number_of_bacteria

#%%
import os
files = [f for f in os.listdir("./") if f.endswith("tif")]
results = {}
for fl in files:
    print(fl)
    #fl = "./20210429_KP_0140_50_Stack_3dstack.tif"
    img=io.imread(f"./{fl}")
    img_norm=(img-np.min(img))/(np.max(img)-np.min(img))
    img_norm_0_255 = (img_norm*255).astype(np.uint8)
    from tqdm import tqdm
    Areas = []
    counter =0
    expov = [2**i for i in range(16)]
    #print(expov)
    number_of_backteria=-1
    for img in tqdm(img_norm_0_255):
        area,number_of_backteria=GetAreas(img, False,150, 900**expov[counter],number_of_backteria)
        Areas.append([area,number_of_backteria])
        counter+=1
    results[fl]=Areas
#%%
    for i,ar in enumerate(Areas):
        print(i, np.mean(ar[0]), np.std(ar[0]))
    for i,ar in enumerate(Areas):
        print(i, np.sum(ar[0])/ar[1])
    #%%
    plt.title(fl)
    plt.plot( range(15,91,5),[np.mean(ar[0]) for ar in Areas])
    plt.show()
    plt.title(fl)
    plt.plot( range(15,91,5),[np.sum(ar[0])/ar[1]for ar in Areas])
    plt.show()
# %%
plt.figure(figsize=(10,10))
for fl in results:
    Areas=results[fl]
    plt.plot( range(15,91,5),[np.mean(ar[0])/np.mean(Areas[0][0]) for ar in Areas])
plt.title("Fold change of Area growth over time in minutes")
plt.legend(list(results.keys()))
plt.show()

# %%
plt.figure(figsize=(10,10))
for fl in results:
    Areas=results[fl]
    itms=[(np.sum(ar[0])/ar[1])/(np.sum(Areas[0][0])/Areas[0][1])for ar in Areas]
    plt.plot( range(15,91,5),itms)
plt.title("Fold change of Area growth per initial bacteria number over time in minutes")
plt.legend(list(results.keys()))
plt.show()
# %%
