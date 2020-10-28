import pandas as pd
import math
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.image import imread
import matplotlib.pyplot as plt
import cv2


class DataLoader(object):
    '''
    Generate the image by different algorithms

    '''
    def __init__(self,method, lateral_shifted,nm_per_pixel,validRegion):
        # settings
        self.method = method
        self.nm_per_pixel = 10
        
        if(method=='Average shifted histogram'):
            self.lateral_shifted = lateral_shifted 
        self.validRegion = validRegion
        pass
    
    def genHeatMap(self,img):

        '''
        mat: 2D array
        
        return matplotlib figure, axes

        '''
        figure, ax = plt.subplots()
        ax.imshow(img, interpolation='nearest',cmap="hot",origin="upper")
        ax.axis('off')
        return figure,ax

    def visualize(self,coors,y_gap=False,x_gap=False,bbox=None):
        '''
        for convinient, use plt heatmap to generate the image 
        
        '''
        if(self.method == 'Average shifted histogram'):
            img, _ = self.averageShiftedHistogram(coors)
        
        if(bbox!=None):
            img = img[bbox['x1']:bbox['x2']+1,bbox['y1']:bbox['y2']+1] 
        
         
        
        if(y_gap):
            img = img.T
            
            # gapped one line in y direction 
            new_size = (img.shape[0]*2+2,img.shape[1]+2)
            new_img = np.zeros(new_size)
            k = 1
            for i in range(img.shape[0]):
                for j in range(img.shape[1]):
                    
                    new_img[k][j] = img[i][j] 
                    pass 
                k += 2
            img = new_img   
            img = np.flipud(img)     
            pass
        elif(x_gap):
            
            img = img.T

            # gapped one line in x direction
            new_size = (img.shape[0]+2,img.shape[1]*2+2)
            new_img = np.zeros(new_size)
            k = 1
            #print('img.shape',img.shape,'new_img.shape',new_img.shape)
            for j in range(img.shape[1]):
                for i in range(img.shape[0]):

                    new_img[i][k] = img[i][j]
                    
                k += 2
            img = new_img

            
        
        else:
            img = img.T 
        
          


        
        figure,ax = self.genHeatMap(img)


        
        return figure,ax
        
    def loadImage(self,srcData):
        '''
        ''' 
        coors = self.readCoordinates(srcData)
        #print("coors",coors)
        if(self.method=='Average shifted histogram'):
        
            return self.averageShiftedHistogram(coors)
        pass
    



   

    def genNewImg(self,pts,original_img,copy_flag,bbox=None):

        '''
        generate a new image by update a specific region of an old image
        
        pts: [x array, y array, z array]
        original_img: 2D numpy array
        bbox: dictionary, updating region
        copy_flag: whether copy the original_img. Set it as False means it only generate the bbox region content
 
        '''
        if(copy_flag):
            new_img =  np.array(original_img, copy=True)

            # clear the region of interest
            for i in range(original_bbox['x1'],original_bbox['x2']+1):

                for j in range(original_bbox['y1'],original_bbox['y2']+1):

                    new_img[i][j] = 0  
        else:
            new_img = np.zeros(original_img.shape)
        
        # update the new_img
        if(self.method=='Average shifted histogram'):

            return self.averageShiftedHistogram(pts,new_img) 

        pass    


    

 
    def readCoordinates(self,srcData):
        '''
        param srcData: DataFrame, must have 'X_magnificated','Y_magnificated' and 'z'
        return coordinates [x array, y array, z array], this coordinates are all of center points
        validRegion = {'x1':int,'y1':int,'x2':int,'y2':int,'z1':int,'z2':int}

        '''
        x_coors = []
        y_coors = []
        z_coors = []
        validRegion = self.validRegion
        for index, row in srcData.iterrows():
            cur_x = int(row["X_magnificated"])
            cur_y = int(row["Y_magnificated"])
            cur_z = int(row["z"])
            
          
            if( validRegion['x1']!=None and cur_x < validRegion['x1']):
                continue
            if( validRegion['x2']!=None and cur_x > validRegion['x2']):
                continue
            if( validRegion['y1'] !=None and cur_y > validRegion['y2']):
                continue
            
            if( validRegion['y2'] !=None and cur_y < validRegion['y1']):
                continue  

            if( validRegion['z1']!=None and cur_z < validRegion['z1']):
                continue
            if( validRegion['z1']!=None and cur_z < validRegion['z2']):
                continue 
            
               
           
            
            x_coors.append(cur_x)
            y_coors.append(cur_y)
            z_coors.append(cur_z)

        coors = [np.asarray(x_coors),np.asarray(y_coors),np.asarray(z_coors)]
        return coors
        pass    


    def averageShiftedHistogram(self,coors,img=None):

        '''

        This method was first proposed by ThunderStorm
        Generate the 2D visualization image for the csv data
        params: srcData, DataFrame

        '''
    
        
        if('numpy' not in str(type(img)) and img==None):
            # create a new image
            # get the size of the image for the visualization
            minX = coors[0].min()
            maxX = coors[0].max()
            minY = coors[1].min()
            maxY = coors[1].max()
            #print(coors)
            imgSize = [math.ceil(maxX)+self.lateral_shifted+10,math.ceil(maxY)+self.lateral_shifted+10]
            img = np.zeros(imgSize)

        max_point_gray_value = pow(2,self.lateral_shifted)

        # z_value is to store the z axis value with specifican (x,y)
        # z_value = {(x,y):[z1,z2]}
        z_value  = {}
        
        for index in range(coors[0].shape[0]):

            center_x = coors[0][index]
            center_y  = coors[1][index]
            #print('center_x',center_x,'center_y',center_y)
            for i in range(-1*int(self.lateral_shifted/2),int(self.lateral_shifted/2)+1):
                for j in range(-1*int(self.lateral_shifted/2),int(self.lateral_shifted/2)+1):
                    shift_value = abs(i) + abs(j)
                    
                    tmp_gray_value = max_point_gray_value/pow(2,shift_value)
                    
                    img[center_x+i][center_y+j] += tmp_gray_value
                   
            if(len(coors)>2):
                # record the z value of the center x and center y
                x_y_pair = (center_x,center_y)
                if(x_y_pair not in z_value ):
                    z_value[x_y_pair]  = []
                z_value[x_y_pair].append(coors[2][index])

        return img.astype(np.uint8), z_value


    def gassianRendering(self,srcData):
        '''
        Generate the image using gassian 
        TODO 
        '''
        pass
     

    def jittering(self,srcData):
        '''
        TODO
        '''
        pass
