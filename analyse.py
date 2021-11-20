#%%
import cv2
import skimage
import numpy as np
#%%
from skimage import io
# %%
img=io.imread("./20210429_KP_0140_50_Stack_3dstack.tif")
# %%
img.shape
# %%
img_norm=(img-np.min(img))/(np.max(img)-np.min(img))
# %%
import matplotlib.pyplot as plt
plt.imshow(img_norm[0], cmap="gray")
# %%
img_norm_0_255 = (img_norm*255).astype(np.uint8)
#img_norm_0_255 = cv2.medianBlur(img_norm_0_255,)
blur = cv2.boxFilter(img_norm_0_255[0],-1,(5,5), normalize = True)
plt.imshow(blur)
#%%
ret,thresh1 = cv2.threshold(blur,90,255,cv2.THRESH_BINARY)
thresh1=255-thresh1
plt.imshow(thresh1)
# %%
kernel = np.ones((5,5),np.uint8)
closing = cv2.morphologyEx(thresh1, cv2.MORPH_CLOSE, kernel)
opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)
plt.imshow(opening)
# %%
nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(opening, connectivity=8)
sizes = stats[1:, -1]; nb_components = nb_components - 1

min_size = 500  # 0.5 µm
max_size = 900 # 1 µm
#your answer image
img2 = np.zeros((output.shape))
#for every component in the image, you keep it only if it's above min_size
for i in range(0, nb_components):
    if max_size>=sizes[i] >= min_size:
        img2[output == i + 1] = 255
# %%
plt.imshow(img2)
# %%

from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.color import label2rgb
#%%
cleared = clear_border(img2)
plt.imshow(cleared)

# %%
label_image = label(cleared)
plt.imshow(label_image)
# %%
image_label_overlay = label2rgb(label_image, image=cleared, bg_label=0)
# %%
fig, ax = plt.subplots(figsize=(10, 6))
ax.imshow(image_label_overlay)
from skimage.morphology import closing, square
import matplotlib.patches as mpatches
areas = []
for region in regionprops(label_image):
    # take regions with large enough areas
    # draw rectangle around segmented coins
    if region.feret_diameter_max>30:
        minr, minc, maxr, maxc = region.bbox
        rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                  fill=False, edgecolor='red', linewidth=2)
        ax.add_patch(rect)
        areas.append(region.area)

ax.set_axis_off()
plt.tight_layout()
plt.show()
# %%
np.mean(areas), np.std(areas)
# %%
