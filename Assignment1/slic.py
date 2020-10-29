import cv2
# import matplotlib.pyplot as plt
import numpy as np
# import matplotlib.patches as mpatches
# from sklearn.cluster import KMeans

from functions import *
# for i in im_list:
#     plot_image(cv2.imread(i),i.split("/")[-1])

im = cv2.imread(im_list[0])
seg = cv2.imread(im_list[0].replace("_s","_s_GT"))

plot_image(im,"Image")
plot_image(seg,"Segmentation")

Q1 : K-means on RBG
for k in [5,10,50]:
    clusters = cluster_pixels(im,k)
    _ = rgb_segment(clusters,n = k, title =  "naive clustering: Pixelwise class plot: Clusters: " + str(k),legend = False)
    superpixel_plot(im,clusters,title =  "naive clustering: Superpixel plot: Clusters: "+ str(k))

# Q2 : K-means on RBGXY
# for k in [5,10,25,50,150]:
#     clusters = cluster_rgbxy(im,k)
#     _ = rgb_segment(clusters,n = k, title =  "naive clustering: Pixelwise class plot: Clusters: " + str(k),legend = False)
#     superpixel_plot(im,clusters,title =  "naive clustering: Superpixel plot: Clusters: "+ str(k))

#TODO diplay your SLIC results.

print('done')

