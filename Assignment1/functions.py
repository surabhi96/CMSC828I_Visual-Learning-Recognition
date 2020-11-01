import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.cluster import KMeans
import numpy as np
import math
from sklearn.metrics.pairwise import euclidean_distances
import sys

# im_list = ['MSRC_ObjCategImageDatabase_v1/1_22_s.bmp',
#            'MSRC_ObjCategImageDatabase_v1/1_27_s.bmp',
#            'MSRC_ObjCategImageDatabase_v1/3_3_s.bmp',
#            'MSRC_ObjCategImageDatabase_v1/3_6_s.bmp',
#            'MSRC_ObjCategImageDatabase_v1/6_5_s.bmp',
#            'MSRC_ObjCategImageDatabase_v1/7_19_s.bmp']
im_list = ['1_22_s.bmp',
           '1_27_s.bmp',
           '3_3_s.bmp',
           '3_6_s.bmp',
           '6_5_s.bmp',
           '7_19_s.bmp']

def plot_image(im,title,xticks=[],yticks= [],isCv2 = True):
    """
    im :Image to plot
    title : Title of image 
    xticks : List of tick values. Defaults to nothing
    yticks :List of tick values. Defaults to nothing 
    cv2 :Is the image cv2 image? cv2 images are BGR instead of RGB. Default True
    """
    plt.figure()
    if isCv2:
        im = im[:,:,::-1]
    plt.imshow(im)
    plt.title(title)
    plt.xticks(xticks)
    plt.yticks(yticks)
    plt.show()

def superpixel_plot(im,seg,title = "Superpixels"):
    """
    Given an image (nXmX3) and pixelwise class mat (nXm), 
    1. Consider each class as a superpixel
    2. Calculate mean superpixel value for each class
    3. Replace the RGB value of each pixel in a class with the mean value  
    
    Inputs:
    im: Input image
    seg: Segmentation map
    title: Title of the plot 
    
    Output: None
    Creates a plot    
    """
    clust = np.unique(seg)
    mapper_dict = {i: im[seg == i].mean(axis = 0)/255. for i in clust}

    seg_img =  np.zeros((seg.shape[0],seg.shape[1],3))
    for i in clust:
        seg_img[seg == i] = mapper_dict[i]
    
    plot_image(seg_img,title)
    
    return    

def rgb_segment(seg,n = None,plot = True,title=None,legend = True,color = None):
    """
    Given a segmentation map, get the plot of the classes
    """
    clust = np.unique(seg)
    if n is None:
        n = len(clust)
    if color is None:
        cm = plt.cm.get_cmap('hsv',n+1)
        # mapper_dict = {i:np.array(cm(i/n)) for i in clust}
        mapper_dict = {i:np.random.rand(3,) for i in clust} 
    #elif color == 'mean':
        #TODO..get the mean color of cluster center and assign that to mapper_dict

    seg_img =  np.zeros((seg.shape[0],seg.shape[1],3))
    for i in clust:
        seg_img[seg == i] = mapper_dict[i][:3]

    if plot: 
        plot_image(seg_img,title = title)
    if legend:
        # get the colors of the values, according to the 
        # colormap used by imshow
        patches = [ mpatches.Patch(color=mapper_dict[i], label=" : {l}".format(l=i) ) for i in range(n) ]
        # put those patched as legend-handles into the legend
        plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0. )
        plt.grid(True)
        plt.show()

    return seg_img

# Q1 : K-means on RBG
def cluster_pixels(im,k):    
    #TODO Pixelwise clustering   
    
    # convert BGR to RGB
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    h, w, d = np.shape(im)
    im_vector = im.reshape((-1,3))
    # cluster_centers ~ initialization

    # pick random x and y and choose their rgb value
    np.random.seed(100)
    cluster_centers = im[np.random.choice(h, k), np.random.choice(w, k)]
    # cluster_centers = (np.random.choice(255, k*3)).reshape((-1,3))
    
    # dist = np.linalg.norm( np.tile(im_vector[0] , (k,1)) - cluster_centers, axis=1)
    # dist = np.linalg.norm( np.repeat(im_vector, repeats=k, axis=0) - np.tile(cluster_centers , (len(im_vector),1)), axis=1 )
    # dist = dist.reshape(len(dist),-1)

    im_vec_dist = np.repeat(im_vector, repeats=k, axis=0)
    distance_moved = float('inf')
    # convergence threshold
    thresh = 180
    if (k == 50):
        thresh = 850
    final_cluster_points = 0

    while(distance_moved > thresh):

        # initialize empty cluster bins
        cluster_points = []
        for i in range(k):
            cluster_points.append([])

        dist = np.linalg.norm( im_vec_dist - np.tile(cluster_centers , (len(im_vector),1)), axis=1 )

        # allocate cluster center to each pixel
        it = 0
        for i in range(0, len(dist), k):
            current_dist = dist[i : i+k]
            min_value = min(current_dist)
            min_position = np.argmin(current_dist)    
            # segregate the points in k bins
            # append point index 
            cluster_points[min_position].append(it)
            it+=1
            
        # recompute cluster centers
        distance_moved = 0
        for i in range (k):
            cp = cluster_points[i]
            # get points
            a = im_vector[cp]
            if (len(a)):
                new_centers = a.mean(axis=0)
                distance_moved += np.linalg.norm(cluster_centers[i] - new_centers)
                # update new cluster centers
                cluster_centers[i] = new_centers

        # for cp in cluster_points: 
        #     print(np.shape(cp))

        final_cluster_points = cluster_points
        print(distance_moved)

    index_vector = np.empty([len(im_vector),1])
    # cp is the point instances 
    for i in range(k):
        cp = final_cluster_points[i] 
        # print(np.shape(cp))
        index_vector[cp] = i
    segmap = index_vector.reshape((h,w))
    # print(segmap)
    # print(np.shape(segmap))
    # at every location, put the corresponding cluster number
    # segmap = np.empty()
    #segmap is nXm. Each value in the 2D array is the cluster assigned to that pixel
    return segmap

#TODO: clustering r,b,g,x,y values 
#try k = 20,80,200,400,800
def cluster_rgbxy(im,k):
    """
    Given image im and asked for k clusters, return nXm size 2D array
    segmap[0,0] is the class of pixel im[0,0,:]
    """
    # convert BGR to RGB
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    h, w, d = np.shape(im)
    im_vector = im.reshape((-1, 3))
    arr = [[[j, i] for i in range(w)] for j in range(h)]
    arr = np.asarray(arr)
    im_vector_xy = arr.reshape((-1, 2))
    # cluster_centers ~ initialization

    # pick random x and y and choose their rgb value
    np.random.seed(100)
    cluster_centers_xy = [np.random.choice(h, k), np.random.choice(w, k)]
    cluster_centers_xy = np.asarray(cluster_centers_xy)
    cluster_centers_xy.reshape((k,-1))
    cluster_centers_xy = cluster_centers_xy.transpose()
    # cluster_centers_rgb = [[im[cluster[0]][cluster[1]] for cluster in cluster_centers_xy]]
    cluster_centers_rgb = []
    for cluster in cluster_centers_xy:
        cluster_centers_rgb.append(im[cluster[0]][cluster[1]])
    cluster_centers_rgb = np.asarray(cluster_centers_rgb)
    # cluster_centers_rgb = im[cluster_centers_xy[:, 0], cluster_centers_xy[:, 1]]
    # cluster_centers = (np.random.choice(255, k*3)).reshape((-1,3))

    # dist = np.linalg.norm( np.tile(im_vector[0] , (k,1)) - cluster_centers, axis=1)
    # dist = np.linalg.norm( np.repeat(im_vector, repeats=k, axis=0) - np.tile(cluster_centers , (len(im_vector),1)), axis=1 )
    # dist = dist.reshape(len(dist),-1)

    im_vec_dist = np.repeat(im_vector, repeats=k, axis=0)
    im_vec_dist_xy = np.repeat(im_vector_xy, repeats=k, axis=0)
    distance_moved = float('inf')
    # convergence threshold
    thresh = 180
    if (k >= 10):
        thresh = 400
    if (k >= 25):
        thresh = 1200
    if (k >= 50):
        thresh = 1850
    if (k >= 100):
        thresh = 3500
    final_cluster_points = 0

    while (distance_moved > thresh):

        # initialize empty cluster bins
        cluster_points = []
        for i in range(k):
            cluster_points.append([])

        dist_rgb = np.linalg.norm(im_vec_dist - np.tile(cluster_centers_rgb, (len(im_vector), 1)), axis=1)
        dist_xy = np.linalg.norm(im_vec_dist_xy - np.tile(cluster_centers_xy, (len(im_vector_xy), 1)), axis=1)

        dist = 0.5 * dist_rgb + 0.5 * dist_xy

        # allocate cluster center to each pixel
        it = 0
        for i in range(0, len(dist), k):
            current_dist = dist[i: i + k]
            min_value = min(current_dist)
            min_position = np.argmin(current_dist)
            # segregate the points in k bins
            # append point index
            cluster_points[min_position].append(it)
            it += 1

        # recompute cluster centers
        distance_moved = 0
        for i in range(k):
            cp = cluster_points[i]
            # get points
            a = im_vector[cp]
            b = im_vector_xy[cp]
            if (len(a)):
                new_centers = a.mean(axis=0)
                distance_moved += np.linalg.norm(cluster_centers_rgb[i] - new_centers)
                # update new cluster centers
                cluster_centers_rgb[i] = new_centers
            if (len(b)):
                new_centers = b.mean(axis=0)
                distance_moved += np.linalg.norm(cluster_centers_xy[i] - new_centers)
                # update new cluster centers
                cluster_centers_xy[i] = new_centers

        # for cp in cluster_points:
        #     print(np.shape(cp))

        final_cluster_points = cluster_points

    print(distance_moved)

    index_vector = np.empty([len(im_vector), 1])
    # cp is the point instances
    for i in range(k):
        cp = final_cluster_points[i]
        # print(np.shape(cp))
        index_vector[cp] = i
    segmap = index_vector.reshape((h, w))
    # print(segmap)
    # print(np.shape(segmap))
    # at every location, put the corresponding cluster number
    # segmap = np.empty()
    # segmap is nXm. Each value in the 2D array is the cluster assigned to that pixel
    return segmap

# Modified k-means with weighted distances. 
#TODO: clustering r,b,g,x,y values with lambdas and display outputs
# def cluster_rgbxy(im,k, lambda_1, lambda_2):
#     """
#     Given image im and asked for k clusters, return nXm size 2D array
#     segmap[0,0] is the class of pixel im[0,0,:]
#     """
#     assert 1==2,"NOT IMPLEMENTED"
#     return segmap

def calculate_gradient(im, x, y, h, w):
    if (x + 1) >= w or (y+1) >= h or (x-1) < 0 or (y-1) < 0:
        return sys.maxsize
    return euclidean_distances(im[x+1][y] - im[x-1][y])^2 + euclidean_distances(im[x][y+1] - im[x][y-1])^2

def calculate_distance(im, x_i, y_i, x_k, y_k, m, S):
    d_lab = np.sqrt((im[x_k][y_k][0] - im[x_i][y_i][0])^2 + (im[x_k][y_k][1] - im[x_i][y_i][1])^2 + (im[x_k][y_k][2] - im[x_i][y_i][2])^2)
    d_xy = np.sqrt((x_k - x_i)^2 + (y_k - y_i)^2)
    return d_lab + m/S*d_xy

def define_search_region(cluster, im, h, w, S):
    x = cluster[3]
    y = cluster[4]
    if (x - S) < 0:
        left = 0
    else:
        left = x - S
    if (x + S) >= w:
        right = w-1
    else:
        right = x+S
    if (y - S) < 0:
        top = 0
    else:
        top = y - S
    if (y + S) >= h:
        bottom = h-1
    else:
        right = y+S

    sub_im = im[left:right+1][top:bottom+1]
    im_vector = sub_im.reshape((-1, 3))
    arr = [[[j, i] for i in range(left, right+1)] for j in range(top, bottom+1)]
    arr = np.asarray(arr)
    im_vector_xy = arr.reshape((-1, 2))
    # im_vec_dist = np.repeat(im_vector, repeats=k, axis=0)
    # im_vec_dist_xy = np.repeat(im_vector_xy, repeats=k, axis=0)
    return im_vector, im_vector_xy, left, right, top, bottom
#TODO
############Algorithm############
#Compute grid steps: S
#you can explore different values of m
#initialize cluster centers [l,a,b,x,y] using  
#Perturb for minimum G
#while not converged
##for every pixel:
####  compare distance D_s with each cluster center within 2S X 2S. 
####  Assign to nearest cluster
##calculate new cluster center 
def SLIC(im, k):
    """
    Input arguments: 
    im: image input
    k: number of cluster segments


    Compute
    S: As described in the paper
    m: As described in the paper (use the same value as in the paper)
    follow the algorithm..
    
    returns:
    segmap: 2D matrix where each value corresponds to the image pixel's cluster number
    """
    # convert to lab space
    im = cv2.cvtColor(im, cv2.COLOR_BGR2LAB)
    h, w, d = np.shape(im)
    # compute S
    S = np.sqrt((h * w) / k)
    m = 10  # from paper

    # initialize cluster centers
    cluster_centers = [[0 for x in range(5)] for y in range(k)]
    h_temp = int(S/2)
    w_temp = int(S/2)
    count = 0
    while h_temp < h:
        while w_temp < w:
            cluster_centers[count] = [im[h_temp][w_temp][0], im[h_temp][w_temp][1], im[h_temp][w_temp][2], h_temp, w_temp]
            w_temp = int(w_temp + S)
            count = count + 1
        w_temp = int(S/2)
        h_temp = int(h_temp + S)

    # perturb to lowest gradient position
    for ind, cluster in enumerate(cluster_centers):
        center_gradient = calculate_gradient(im, cluster[3], cluster[4], h, w)
        for i in range(-1, 2):
            for j in range(-1, 2):
                gradient = calculate_gradient(im, cluster[3] + i, cluster[4] + j, h, w)
                if gradient < center_gradient:
                    cluster_centers[ind][3] = cluster[3] + i
                    cluster_centers[ind][4] = cluster[4] + i
                    cluster_centers[ind][0] = im[cluster[3]][cluster[4]][0]
                    cluster_centers[ind][1] = im[cluster[3]][cluster[4]][1]
                    cluster_centers[ind][2] = im[cluster[3]][cluster[4]][2]
                    center_gradient = gradient

    thresh = 850
    dist_moved = float('inf')

    pixel_distances = [[sys.maxsize for i in range(w)] for j in range(h)]
    segmap = [[0 for i in range(w)] for j in range(h)]

    while dist_moved > thresh:

        cluster_points = []
        for i in range(k):
            cluster_points.append([])

        for k_ind, cluster in enumerate(cluster_centers):
            im_vector, im_vector_xy, left, right, top, bottom = define_search_region(cluster, im, h, w, S)

            d_lab = np.linalg.norm(im_vector - np.tile(cluster[:3], (len(im_vector), 1)), axis=1)
            d_xy = np.linalg.norm(im_vector_xy - np.tile(cluster[3:], (len(im_vector_xy), 1)), axis=1)

            d = d_lab + m/S*d_xy
            count = 0
            for i in range(top, bottom+1):
                for j in range(left, right+1):
                    if d[count] < pixel_distances[i][j]:
                        pixel_distances[i][j] = d[count]
                        segmap[i][j] = k_ind
                    count = count + 1

            for i in range(h):
                for j in range(w):
                    cluster_points[int(segmap[i][j])].append([i,j])

        dist_moved = 0
        for i in range(k):
            cp = cluster_points[i]
            cp = np.asarray(cp)
            # get points
            # points = [[im[row][col] for row in cp[:,0]] for col in cp[:,1]]
            if (len(cp)):
                new_centers = cp.mean(axis=0)
                dist_moved += np.linalg.norm(np.asarray(cluster_centers[i]) - new_centers)
                # update new cluster centers
                cluster_centers[i] = new_centers

        # for cp in cluster_points:
        #     print(np.shape(cp))

        final_cluster_points = cluster_points





    segmap = []
    return segmap
