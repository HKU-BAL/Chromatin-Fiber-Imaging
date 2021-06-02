import pandas as pd
import math
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.image import imread
import matplotlib.pyplot as plt
import cv2



import logging
logging.getLogger().setLevel(logging.INFO)
class DataLoader(object):
    '''
    Generate the image by different algorithms

    '''
    def __init__(self,method, lateral_shifted,axial_shifted,nm_per_pixel,validRegion):
        # settings
        self.method = method
        self.nm_per_pixel = 10
        
        if(method=='Average shifted histogram'):
            self.lateral_shifted = lateral_shifted
            self.axial_shifted = axial_shifted
 
        self.validRegion = validRegion
        pass
    
    def genHeatMap(self,img,colors="hot"):

        '''
        mat: 2D array
        
        return matplotlib figure, axes

        '''
        img = np.clip(img,0,255)
        img = img.astype(np.uint8)
        figure, ax = plt.subplots()
        ax.imshow(img, interpolation='nearest',cmap=colors,origin="upper")
        ax.axis('off')
        return figure,ax



    def img_resize(self,image):
        height, width = image.shape[0], image.shape[1] 
        width_new = 480
        height_new = 620
    
        if width / height >= width_new / height_new:
            img_new = cv2.resize(image, (width_new, int(height * width_new / width)),interpolation = cv2.INTER_AREA)
        else:
            img_new = cv2.resize(image, (int(width * height_new / height), height_new),interpolation = cv2.INTER_AREA)
        return img_new
    def apply_brightness_contrast(self,input_img, brightness = 0, contrast = 0):
        if brightness != 0:
            if brightness > 0:
                shadow = brightness
                highlight = 255
            else:
                shadow = 0
                highlight = 255 + brightness
            alpha_b = (highlight - shadow)/255
            gamma_b = shadow
        
            buf = cv2.addWeighted(input_img, alpha_b, input_img, 0, gamma_b)
        else:
            buf = input_img.copy()
    
        if contrast != 0:
            f = 131*(contrast + 127)/(127*(131-contrast))
            alpha_c = f
            gamma_c = 127*(1-f)
        
            buf = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)

        return buf
    def genRGB(self,img,colorDirection):

        '''
        '''
        # initialize the HSV mat,size=(x,y,3)
        original_row = img.shape[0]
        original_col = img.shape[1]
        max_gray_value = np.max(img)
        
        '''
        hsv = np.zeros((original_row,original_col,3))
        if(colorDirection=='row'):
            color_step = max(int(360/original_row),1) 

            pass
        elif(colorDirection=='col'):
            color_step = max(int(360/original_col),1)

        max_gray_value = np.max(img)
         
        
        for i in range(original_row):
             for j in range(original_col):
                 if(img[i][j]==0):
                     continue
                 if(colorDirection=='row'):
                     #hsv[i][j][0] = (i*color_step)%360
                     hsv[i][j][1] = 30+img[i][j]*25 
                     hsv[i][j][1] = min(img[i][j]/max_gray_value+0.5,1)

                 elif(colorDirection=='col'):
                     #hsv[i][j][0] = (j*color_step)%360
                     hsv[i][j][0] = 30+img[i][j]*25
                     hsv[i][j][1] = min(img[i][j]/max_gray_value+0.5,1)
                 
                 #hsv[i][j][2] = min(int((img[i][j]/max_gray_value+0.3)*255),255)
                 hsv[i][j][2] = min(int(img[i][j]*25+30),255)
                 pass
        
        hsv = self.img_resize(hsv)
        hsv = np.float32(hsv)
        bgrimg = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
        '''
        # adjust the brightness and contrast  
        '''
        for i in range(img.shape[0]):
            tmp_line = ''
            for j in range(img.shape[1]):
                tmp_line += str(img[i][j])+'\t'
            tmp_line = tmp_line.strip()
            print(tmp_line) 
        '''      
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                if(img[i][j]==0):
                    continue
                before_gray = img[i][j]
                #img[i][j] = min(int(img[i][j]/max_gray_value)*255)+30,255)
                tmp_val = math.ceil((img[i][j]/max_gray_value)*254)
                img[i][j] =min(254,10+int(tmp_val*2))
                
                after_gray = img[i][j]
                #logging.info("adjust:"+str(i)+' '+str(j)+' '+str(before_gray)+' '+str(after_gray)) 
                #print("img[i][j]",img[i][j]) 
        im_color =  cv2.applyColorMap(img, cv2.COLORMAP_HOT)
        
        
        return self.img_resize(im_color) 
        return bgrimg 
        pass

    def visualize(self,coors,y_gap=False,x_gap=False,bbox=None,projection='xy',adjust_BC=True):
        '''
        for convinient, use plt heatmap to generate the image 
        
        return matplotlib figure, ax, original gray matrix
    
        '''
        if(self.method == 'Average shifted histogram'):
            
            img, _ = self.averageShiftedHistogram(coors,projection=projection)
        

        # img: numpy 
        if(bbox!=None):
            
            img = img[bbox['x1']:bbox['x2']+1,bbox['y1']:bbox['y2']+1] 
        
         
        original_img = img         
        if(y_gap):
            #img = img.T
            #img = np.flipud(img)
            if(adjust_BC):
                img = np.clip(img,0,255)
                img = img.astype(np.uint8)
                img = img.T
                img = np.flipud(img)    
                cv2_img = self.genRGB(img,colorDirection='row')
            else:
                cv2_img = img
            return cv2_img,None,original_img
            pass
        elif(x_gap):
                         
            #img = img.T
            if(adjust_BC):
                img = np.clip(img,0,255)
                img = img.astype(np.uint8)
                img = img.T 
                cv2_img = self.genRGB(img,colorDirection='col')
            else:
                cv2_img = img 
            return cv2_img,None,original_img
        else:
            img = img.T         
            figure,ax = self.genHeatMap(img) 
            return figure,ax, original_img
        
    def loadImage(self,srcData,projection='xy'):
        '''
        ''' 
        coors = self.readCoordinates(srcData)
        #print("coors",coors)
        if(self.method=='Average shifted histogram'):
        
            return self.averageShiftedHistogram(coors,projection=projection)
        pass
    



   

    def genNewImg(self,pts,original_img,copy_flag,bbox=None,projection='xy'):

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

            return self.averageShiftedHistogram(pts,new_img,projection=projection) 

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


    def averageShiftedHistogram(self,coors,img=None,projection='xy'):

        '''

        This method was first proposed by ThunderStorm
        Generate the 2D visualization image for the csv data
        params: srcData, DataFrame
                projection: 'xy' lateral_shift in "X" and lateral_shift in 'Y'
                            'xz' lateral_shift in 'X' and axial_shift in 'Y'
                            'yz' axial_shift in 'X' and lateral_shift in 'Y'
              
      
        '''
        x_shift = 1
        y_shift = 1
        #print("averageShiftedHistogram",img)
        if(projection=='xy'):
            x_shift = self.lateral_shifted
            y_shift = self.lateral_shifted
        elif(projection=='xz'):
            x_shift = self.lateral_shifted
            y_shift = self.axial_shifted
        elif(projection=='yz'):
            x_shift = self.axial_shifted
            y_shift = self.lateral_shifted 
        #print("projection",projection,'coors',coors) 
        if('numpy' not in str(type(img)) and img==None):
            # create a new image
            # get the size of the image for the visualization
            minX = coors[0].min()
            maxX = coors[0].max()
            minY = coors[1].min()
            maxY = coors[1].max()
            
            imgSize = [math.ceil(maxX)+x_shift*2+10,math.ceil(maxY)+y_shift*2+10]
            
            img = np.zeros(imgSize)

        
        
        max_point_gray_value = x_shift * y_shift
         
        x_bin_size = int(max_point_gray_value/x_shift)
        y_bin_size = int(max_point_gray_value/y_shift)
         

        # z_value is to store the z axis value with specifican (x,y)
        # z_value = {(x,y):[z1,z2]}
        z_value  = {}
        
 
        for index in range(coors[0].shape[0]):

            center_x = coors[0][index]
            center_y  = coors[1][index]

            for j in range(-1*int(y_shift-1),int(y_shift-1)+1):
                for i in range(-1*int(x_shift-1),int(x_shift-1)+1):
                                    
                    tmp_x_value  = max_point_gray_value - abs(i) * x_bin_size
                    tmp_y_bin = int(tmp_x_value/(y_shift))
                              
                    tmp_gray_value = tmp_x_value - tmp_y_bin * abs(j)
                    before = img[center_x+i][center_y+j] 
                     
                    img[center_x+i][center_y+j] += tmp_gray_value
                    #img[center_x+i][center_y+j] = min(255,img[center_x+i][center_y+j])
                     
            if(len(coors)>2):
                # record the z value of the center x and center y
                x_y_pair = (center_x,center_y)
                if(x_y_pair not in z_value ):
                    z_value[x_y_pair]  = []
                z_value[x_y_pair].append(coors[2][index])
        
        

        return img, z_value
        

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
