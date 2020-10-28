
import numpy as np
import cv2
from matplotlib import pyplot as plt
#import imutils
from PIL import Image
import skimage.measure





def extractComponents(img_mat,num_labels,labels,z_dict):

    '''

    extract the identfied connected component from img_mat
    return all_pts list for different clusters -> only cenetr points [[x array, y array, z array],...,...]

    '''
    # print("extract component")
    all_pts = []
   
    for i in range(1,num_labels+1):
        
        pts = np.where(labels==i)
        if (len(pts[0])==0):
            continue
        x_array = []
        y_array = []
        z_array = []
        # extract points for the ith cluster
        for k in range(len(pts[0])):
            
            if((pts[0][k],pts[1][k]) not in z_dict):
                #tmp_z_lists.append(None)
                # do not have a centern point in (pts[0][k],pts[1][k]), ignore this point
                continue
            # only record center points
            _z_list = z_dict[(pts[0][k],pts[1][k])]
            for _z in _z_list:
                if(_z !=None):
                    x_array.append(pts[0][k])
                    y_array.append(pts[1][k])
                    z_array.append(_z)
            
        all_pts.append([np.asarray(x_array),np.asarray(y_array),np.asarray(z_array)])
    return all_pts


def extractComponentsByCoors(img_mat,num_labels,labels,cluster_pts):

    for i in range(1,num_labels+1):

        pts = np.where(labels==i)
        if (len(pts[0])==0):
            continue


    pass

def watershed(img):
    img2 = cv2.merge((img,img,img)) 
    ret0, thresh0 = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY)
    
    ret2, markers = cv2.connectedComponents(thresh0)
    

    markers3 = cv2.watershed(img2,markers)
    
    
    return markers3.max()+1,markers3



def Morphological_Transformations(img):

    # erosion
    kernel = np.ones((3,3),np.uint8)
    opening = cv2.erode(img,kernel,iterations = 2)

    # dilation
    img = cv2.dilate(opening,kernel,iterations=3)

    
    return img

    

def cluster_by_connectedComponents(img_mat,my_connectivity,Morphological_Transformations=False):
    
    # original version
    
    img = cv2.threshold(img_mat, 0, 255, cv2.THRESH_BINARY)[1]  # ensure binary
    #print('img.shape',img.shape,my_connectivity)
      
    if(Morphological_Transformations):
        img = Morphological_Transformations(img) 
    
    num_labels, labels_im = cv2.connectedComponents(img,connectivity=my_connectivity)
    #print("labels_im.max()",labels_im.max())
    return num_labels,labels_im

    pass









if __name__ == '__main__':

    #img_mat,myData = loadData('../data/250-499.csv')
     

    '''
    N=9
    for i in range(1, num_labels+1):
        pts = np.where(labels_im==i)
        #print(pts)
        #print(len(pts[0]))
    
        # Remove the connected components with only one pixel 
        #if len(pts[0]) <= N:
        #    print(len(pts[0]),i)
        #    labels_im[pts] = 0
        

    
    label_hue = np.uint8(179 * labels_im / np.max(labels_im))
    blank_ch = 255 * np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])

    # cvt to BGR for display
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)

    # set bg label to black
    labeled_img[label_hue == 0] = 0

    #cv2.imshow('labeled.png', labeled_img)
    cv2.imwrite("labeled.png", labeled_img)
    #cv2.waitKey()
    '''
