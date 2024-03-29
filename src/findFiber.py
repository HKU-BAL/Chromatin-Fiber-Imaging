import pandas as pd
from seg import cluster_by_connectedComponents,extractComponents
import numpy as np
from lmfit import Model as lmModel
import matplotlib
matplotlib.use("agg")
#from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
import cv2
import math
import os
from lmfit.models import GaussianModel
import logging
logging.getLogger().setLevel(logging.INFO)


class DataProcessor(object):
    '''
    Data preprocessor is to split the frames and 
    remove some data by user's request, like filter out too large location error
 
    '''
    def __init__(self):
        
        pass
    
    def prepare(self,DATA_PATH,splitFrameLen=None,locPrecThreshold=None,magnificationRatio=1):

       '''
       prepapre the data: split the frame, magnificate the coordinates and filter out rows as required
       
       '''
       srcData = pd.read_csv(DATA_PATH)
       # check the input data
       requiredCol = set(["x",'y','z',"Frame Number"])
       optionalCol = set(["PhotoCount","X Fitting Error","X Fitting Error"])
       if(not requiredCol.issubset(srcData)):
           return data_list,0
       if(not optionalCol.issubset(srcData)):
           dataCheckFlag = -1
       else:
           dataCheckFlag = 1

       data_list = []
       srcData["Frame Number"] = srcData["Frame Number"].astype(int)
       #print("prepare locPrecThreshold",locPrecThreshold)
       if(locPrecThreshold!=None):
          f1 = srcData["X Fitting Error"]<= locPrecThreshold
          f2 = srcData["Y Fitting Error"] <= locPrecThreshold
          srcData = srcData[f1&f2]
       #print("srcData.shape",srcData.shape)       
       srcData['X_magnificated'] = (srcData['x'] * magnificationRatio).astype(int)
       srcData['Y_magnificated'] = (srcData['y'] * magnificationRatio).astype(int)        
       srcData['Z_normalized'] = srcData['z'].astype(int)
       if(splitFrameLen!=None): 
           # split the frame
           minFrameNum = srcData["Frame Number"].min()
           maxFrameNum = srcData["Frame Number"].max()
           for i in range(minFrameNum,maxFrameNum+1,splitFrameLen):
               f1 = srcData['Frame Number'] >= i
               f2 = srcData['Frame Number'] <= (i+splitFrameLen-1)
               tmpData = srcData[f1&f2]
               data_list.append(tmpData) 
       else:
               data_list.append(srcData)
        
       return data_list,dataCheckFlag
       
    


class DataManager(object):

    #save_index= 0
    #data_list = []
    def __init__(self,SAVE_ROOT,second_per_frame,contour_pts=False):
        self.save_index = 0
        self.data_list = []
        self.save_root = SAVE_ROOT
        if(not os.path.exists(self.save_root)):
            os.mkdir(self.save_root) 
        self.second_per_frame = second_per_frame
        
        self.batch_num = 0
        self.contour_pts = contour_pts
        
        
    def saveData(self,stat_info,csvData,imgs,splitFrameBatch):
        '''
        every batch in one directory and every 1000 file per batch
        stat_info: dictionary
        csvData: DataFrame
        imgs: big_xy, xy, yz, xz projection images
        splitFrameBatch: the index of the current frame interval

        '''

        cur_root = os.path.join(self.save_root,'split_frame_batch'+str(splitFrameBatch)) 
        if(not os.path.exists(cur_root)):
            os.mkdir(cur_root)

        
        # to avoid too many files store in one directory         
        if(self.save_index%1000==0):
            self.batch_num += 1
        print("self.save_index: ",self.save_index)
        cur_batch_dir =  os.path.join(cur_root,'result_batch_'+str(self.batch_num))
        stat_path = os.path.join(cur_batch_dir,'stat')
        xy_path = os.path.join(cur_batch_dir,'xy')
        xz_path = os.path.join(cur_batch_dir,'xz')
        yz_path = os.path.join(cur_batch_dir,'yz')
        big_xy_path = os.path.join(cur_batch_dir,'big_xy')
        original_data_path = os.path.join(cur_batch_dir,'source_csv')
        # add a z slice column in source_csv

        # select points with minimun/maximum x/y in each of the z slice
         
        #selected_points_path = os.path.join(cur_batch_dir,"contour_pts")




        if(not os.path.exists(cur_batch_dir)):
            os.mkdir(cur_batch_dir)
            os.mkdir(stat_path)
            os.mkdir(xy_path)
            os.mkdir(xz_path)
            os.mkdir(yz_path)
            os.mkdir(big_xy_path)
            os.mkdir(original_data_path)
        
        # save original csv data
        #print("self.save_index",self.save_index)
        csvData.to_csv(os.path.join(original_data_path,str(self.save_index)+'.csv'),index=False)                
         
        # save stat data
        stat_info = pd.DataFrame(stat_info)
        # set the order of the columns
        #print(stat_info.columns.tolist())

        # format the order of the stat file
        cols = ['x_mean_FWHM','x_mean_FWHM_err','y_mean_FWHM','y_mean_FWHM_err','x_median_FWHM','x_median_FWHM_err','y_median_FWHM','y_median_FWHM_err','x_whole_FWHM','x_whole_err','y_whole_FWHM','y_whole_err','xz_mean_FWHM','xz_mean_FWHM_err','yz_mean_FWHM','yz_mean_FWHM_err','xz_whole_FWHM','xz_whole_err','yz_whole_FWHM','yz_whole_err','X average localization precision','Y average localization precision','Pseudo Z size','Z range','Life time','avg_photon_num','Max frame gap','Frame length','sum of gray value','upper left','lower right','angle_xz','angle_yz']
        
        #cols = ['x_mean_FWHM','x_mean_FWHM_err','y_mean_FWHM','y_mean_FWHM_err','x_whole_FWHM','x_whole_err','y_whole_FWHM','y_whole_err','xz_mean_FWHM','xz_mean_FWHM_err','yz_mean_FWHM','yz_mean_FWHM_err','xz_whole_FWHM','xz_whole_err','yz_whole_FWHM','yz_whole_err','X average localization precision','Y average localization precision','Z size','Z range','Life time','avg_photon_num','Max frame gap','Frame length','sum of gray value','upper left','lower right','angle_xz','angle_yz']
        stat_info = stat_info[cols] 
        stat_info.to_csv(os.path.join(stat_path,str(self.save_index)+'_stat.csv'),index=False)

        # generate visualization image
        
        # save xy project image
         
        imgs['xy'].savefig(os.path.join(xy_path,str(self.save_index)+'_xy.png'),transparent=True) 
        
        # save big_xy projectiong image
        
        imgs['big_xy'].savefig(os.path.join(big_xy_path,str(self.save_index)+'_big_xy.png'),transparent=True) 
        

        # save xz projection
        # save matplotlib object 
        #imgs['xz'].savefig(os.path.join(xz_path,str(self.save_index)+'_xz.png'),transparent=True) 
        cv2.imwrite(os.path.join(xz_path,str(self.save_index)+'_xz.png'),imgs['xz'])

        # save yz projection
        # save matplotlib object
        #imgs['yz'].savefig(os.path.join(yz_path,str(self.save_index)+'_yz.png'),transparent=True) 
        cv2.imwrite(os.path.join(yz_path,str(self.save_index)+'_yz.png'),imgs['yz'])
        self.save_index += 1 
        plt.close('all')    
    

    def extractCsvFromZSlicePts(self,z_slice_pts,csvData):

        
        '''
        '''
        index = []
        z_slice_record = []
        #print("extractCsvFromZSlicePts z_slice_pts:",z_slice_pts)
        for slice_index in range(len(z_slice_pts)):
            if(z_slice_pts[slice_index]==None):
                continue
            for i in range(len(z_slice_pts[slice_index][0])):
                _x = z_slice_pts[slice_index][0][i]
                _y = z_slice_pts[slice_index][1][i]
                _z = z_slice_pts[slice_index][2][i]
                f1 = csvData['X_magnificated'] == _x
                f2 = csvData['Y_magnificated'] == _y
                f3 = csvData['Z_normalized'] ==  _z
             
                tmpData = csvData[f1&f2&f3]
                 
                index.append(tmpData.index[0])
                z_slice_record.append(slice_index)
         
        newData = csvData.loc[index]
        newData['z slice index'] = z_slice_record
        return newData

    
    def extractData(self,pts,csvData,sliceData=False):

        '''
        extract the data from the csv data
        pts: points, [[x array, y array, z array], ...]
        csvData: original splitted input csv data
        
        return cluster statistics and cluster original csv data

        '''
        #index = []
        
        

        newData = self.extractCsvFromZSlicePts(pts,csvData)

        min_frame = min(newData["Frame Number"])
        max_frame = max(newData["Frame Number"])
        # life time of this found fiber
        life_time = ( max_frame - min_frame + 1 )*self.second_per_frame
        avg_photon_num = float(sum(newData["PhotoCount"])/newData.shape[0])

        x_location_precision = newData["X Fitting Error"].values.mean()
        y_location_precision = newData["Y Fitting Error"].values.mean()
        loc_prec = [x_location_precision,y_location_precision]
        frame_length = newData["Frame Number"].max() - newData["Frame Number"].min() + 1
        max_frame_gap = self.calMaxFrameGap(newData["Frame Number"].values)
        stat_info = {"X average localization precision":[x_location_precision],\
                     "Y average localization precision":[y_location_precision],\
                     "Life time":[life_time],\
                     "avg_photon_num":[avg_photon_num],\
                     "Max frame gap":[max_frame_gap],\
                      "Frame length":[frame_length]}

        return stat_info,newData
         
    def calMaxFrameGap(self,frames):

        '''
        calculate the max frame gap

        '''


        frames = sorted(frames)
        max_gap = 0
        start = 0
        end = 0
        for i in range(1,len(frames)):
            tmpGap = abs(frames[i] - frames[i-1])
            if(tmpGap > max_gap):
                max_gap = tmpGap
                start = frames[i]
                end = frames[i-1]

        return max_gap 
    
        
    pass

class Model(object):

    def __init__(self):
        pass
    

    @staticmethod 
    def gauss(x, y0,xc,w,A):

        '''
        gauss function used in Origin
        '''
        PI = 3.1415926 
        return y0 + (A/(w*np.sqrt(PI/2)))*np.exp(-2*((x-xc)/w)*((x-xc)/w))
    @staticmethod
    def gaussAMP(x, y0,xc,w,A):


        return y0 + A*np.exp(-1*((x-xc)*(x-xc))/(2*w*w))
        pass
    def calFWHM(self,box,img,box_flag,nm_per_pixel):
        '''
        Given a line, calculate the FWHM
        box: dictionary, the line, line width is a variable
        box_flag: 'x' or 'y'
        nm_per_pixel: float
 
        '''
        PI = 3.1415926    
        distance = []
        gray = []


        # data: dictionary, distance:gray value
        data = {}
        num_of_pixels = 0
        #print("box",box,"nm_per_pixel",nm_per_pixel)
        # extract data point from the image
        for i in range(box['x1'],box['x2']+1):
            for j in range(box['y1'],box['y2']+1):

                if(box_flag == 'y'):

                    # the distance
                    _tmp_dist = (i - box['x1'] ) * nm_per_pixel
                elif(box_flag == 'x'):
                    _tmp_dist = (j - box['y1']) * nm_per_pixel

                if(_tmp_dist not in data):
                    data[_tmp_dist] = 0
                #print("i",i,'j',j,"img.shape[0]",img.shape[0],"img.shape[1]",img.shape[1],"img[i][j]",img[i][j])
                #print("i<img.shape[0]",i<img.shape[0],"")
                if(i<img.shape[0] and j < img.shape[1] and i >=0 and j >=0): 
                    data[_tmp_dist] += img[i][j]
                    #print("data[_tmp_dist]",data[_tmp_dist])
                else:
                    continue
        
        # x value
        
        #print(data)
        data_x = np.asarray(list(data.keys()))
        v = list(data.values())
       
         
        # normalize y values
        if(box_flag == 'y'):
            data_y = np.asarray(v) / float(box['y2']-box['y1']+1)
        elif(box_flag == 'x'):
            data_y = np.asarray(v) / float(box['x2']-box['x1']+1)
        
        if(len(np.take(data_y,np.argwhere(data_y>0),0))<4):
            
            
            return -1,np.Inf
 
        
        gmodel = lmModel(self.gaussAMP)
         
        # initialize the parameters
        init_xc = np.take(data_x, np.argmax(data_y),0)
        init_sigma = np.std(data_x)
        init_w = init_sigma
        init_y0 = min(data_y)
        init_A = max(data_y) - min(data_y)
        params = gmodel.make_params()
       
        params['y0'].set(init_y0)
        params['xc'].set(init_xc)
        params['A'].set(init_A)
        params['w'].set(init_w,min=0)
        
        result = gmodel.fit(data_y, params,x=data_x)
       
        FWHM =  2.35482004503 *  result.params['w']
        
        my_k_param = 'w'
        if (result.params[my_k_param].stderr == None):
            #err = 'err'
            err = np.Inf
        else:
           
            err = abs(result.params[my_k_param].stderr*2.354)
         
        return FWHM,err
        


    def calMaxGrayVal(self,img,bbox):

        '''
     

        return the max and the sum of the gray value of the current image

        '''        
        
        max_gray_val = img.max()
        sum_gray_val = np.sum(img)

        return max_gray_val, sum_gray_val 
    
     

class FindFiber(object):


    '''
    Find fibers with the specific conditions

    '''
    def __init__(self,minX,maxX,minY,maxY,minZ,zSlice,dataManager,dataLoader,minGray,error_threshold,nm_per_pixel,max_removel_gray,z_gap_tolerance,z_FWHM_error_threshold,max_z_removal_gray=None):
        # basic settings
       
        self.minX = minX
        self.maxX = maxX
        self.minY = minY
        self.maxY = maxY
        self.minZ = minZ
        self.zSlice = zSlice
        self.myDataManager = dataManager
        self.myDataLoader = dataLoader
        self.connectedFlag = 8
        self.minGray = minGray 
        self.myModel = Model()
        self.ERROR_THRESHOLD = error_threshold 
        self.nm_per_pixel = nm_per_pixel
        self.max_removel_gray = max_removel_gray
        self.z_gap_tolerance = z_gap_tolerance
        self.max_z_removal_gray = max_z_removal_gray
        self.z_FWHM_error_threshold = z_FWHM_error_threshold
    def identifyFiber(self):

        '''
        identify the fiber from the dataManager
        no return, save the identified data by using data manager to the save path
 
        '''

        logging.info("total %d frame intervals"%(len(self.myDataManager.data_list)))
        print(self.myDataManager.data_list) 
        for index in range(len(self.myDataManager.data_list)):
            

            logging.info("process the %d frame interval" %(index)) 
             
            csvData = self.myDataManager.data_list[index]
            
            curImg,curZDict = self.myDataLoader.loadImage(csvData)
            
            self.processOneBatch(csvData,curImg,curZDict,index)

        
    
   

    def findAllBox2(self,img_mat,bbox,axis):
        
        '''
        img_may: numpy
        bbox: dictionary
        axis: 'x' or 'y'
        
        with lines only covers left and right or up and bottom

        '''
        shift_val = 5
        # the left 
        x1 = max(0,bbox['x1'])
        # the right
        x2 = min(bbox['x2'],img_mat.shape[0])
        
        # the upper 
        y1 = max(0,bbox['y1'])
        # the bottom
        y2 = min(bbox['y2'],img_mat.shape[1])
        #print("x1",x1,'y1',y1,'x2',x2,'y2',y2)
        all_boxes = []
        
        if(axis=='x'):
            max_line_width = bbox['x2'] - bbox['x1'] + 1
           
            for tmp_line_width in range(max_line_width):
                
                # move the line 
                
                for left in range(bbox['x1'],bbox['x2']+1):
                    right = left + tmp_line_width
                    if(right > bbox['x2']):
                        break
                    # check if the line covers the left and the right
                    cover_flag_left = False
                    cover_flag_right = False

                    for k in range(left,right+1):
                                           
                        if(img_mat[k][y1]>0):
                            cover_flag_left = True
                        if(img_mat[k][y2]>0):
                            cover_flag_right = True
                        continue
                    #print('find box axis',axis,left,right,y1,y2,cover_flag_left,cover_flag_right) 
                    if(cover_flag_left and cover_flag_right):
                         
                        tmp_box = {'x1':left,'x2':right,'y1':y1-shift_val,'y2':y2+shift_val}               
                        #tmp_box = {'x1':left,'x2':right,'y1':max(0,y1-shift_val),'y2':min(img_mat.shape[1],y2+shift_val)}
                        
                        all_boxes.append(tmp_box)
            all_boxes.append({'x1':x1,'x2':x2,'y1':y1-shift_val,'y2':y2+shift_val})       
            #all_boxes.append({'x1':x1,'x2':x2,'y1':max(0,y1-shift_val),'y2':min(y2+shift_val,img_mat.shape[1])})

        elif(axis=='y'):
            max_line_width = bbox['y2'] - bbox['y1'] + 1
            
            for tmp_line_width in range(max_line_width):
               
                for left in range(bbox['y1'],bbox['y2']+1):
                    right = left + tmp_line_width
                    if(right > bbox['y2']):
                        break
                    
                    cover_flag_up = False
                    cover_flag_down = False
 
                    for k in range(left, right+1):
                        if(img_mat[x1][k]>0):
                           cover_flag_up = True
                        if(img_mat[x2][k]>0):
                            cover_flag_down = True
                        continue
                    if(cover_flag_up and cover_flag_down):
                        tmp_box = {'x1':x1-shift_val,'x2':x2+shift_val,'y1':left,'y2':right}
                        #tmp_box = {'x1':max(0,x1-shift_val),'x2':min(img_mat.shape[0],x2+shift_val),'y1':left,'y2':right}
                        all_boxes.append(tmp_box)
            all_boxes.append({'x1':x1-shift_val,'x2':x2+shift_val,'y1':y1,'y2':y2})
            #all_boxes.append({'x1':max(0,x1-shift_val),'x2':min(img_mat.shape[0],x2+shift_val),'y1':y1,'y2':y2})
        return all_boxes
        pass
    
    def _getCenter(self,point_idx):
        '''
        '''
        if(point_idx.shape[0]==1):
            return point_idx[0][0]
        else:
            # more than one point, return the center
            min_pts = np.amin(point_idx)
            max_pts = np.amax(point_idx)
            center_pts = int((max_pts - min_pts)/2) + min_pts
            return center_pts
        pass

    def _getAngleByEdge(self,a,b,c):
        '''
        return the angle of a,c
              
           |\            /|
         b | \ c     c  / |
           |  \        /  | b
           |___\      /___|
             a          a

        '''
        #print('a',a,'b',b,'c',c)
        cos_beta = (a*a + b*b - c*c)  / (2*a*b)  
        try:
            beta = round(math.acos(cos_beta),6)
            #print('beta',beta)
            degree = round(beta*(180/math.pi),6) 
        except:
            degree = -1
        return degree   
        pass
    def calAngle(self,imgMatData):
        '''
        calculate the angle for Z-axis

         xz

             |\
          z  | \ k
             |  \
             |---\
               x
        '''
        degree_res = {'xz':None,'yz':None}
        for k_name in imgMatData:
            #print(k_name)
            #print(imgMatData[k_name])
            n_row,n_col = imgMatData[k_name].shape
            non_zero_coors = np.nonzero(imgMatData[k_name])
            #print('non_zero_coors',non_zero_coors)
            if(k_name=='xz'):
                # row for the x, col for the z
                # first, find center (the x coordinates of max gray value) of the bottom z slice
                
                bottom_index  = min(non_zero_coors[1])
                up_index = max(non_zero_coors[1])
                # bottom points with max gray value
                bottom_max_value_idx = np.argwhere(imgMatData[k_name][:,bottom_index]==np.amax(imgMatData[k_name][:,bottom_index]))
                # top points with max gray value
                top_max_value_idx = np.argwhere(imgMatData[k_name][:,up_index]==np.amax(imgMatData[k_name][:,up_index]))
                bottom_center = self._getCenter(bottom_max_value_idx)
                top_center = self._getCenter(top_max_value_idx)
                #print('bottom_max_value_idx: ',bottom_max_value_idx,'top_max_value_idx: ',top_max_value_idx)
                #print("bottom_center: ",bottom_center,"top_center: ",top_center)
                
                # calculate the angle
                if(top_center==bottom_center):
                     degree_res['xz'] = 90
                     continue 
                z_length = (up_index - bottom_index) * self.zSlice
                # point1 (bottom_center*self.nm_per_pixel,bottom_index*self.zSlice),
                # point2 (top_center*self.nm_per_pixel,up_index*self.zSlice)
                x1_minus_x2 = (bottom_center*self.nm_per_pixel) - (top_center*self.nm_per_pixel)
                z1_minus_z2 = (bottom_index*self.zSlice) - (up_index*self.zSlice)
                k_length = np.sqrt((x1_minus_x2*x1_minus_x2)+(z1_minus_z2*z1_minus_z2)) 
                x_length = abs(top_center-bottom_center)*self.nm_per_pixel   
                xz_degree = self._getAngleByEdge(x_length,z_length,k_length)
                #print('xz\n','bottom_center',bottom_center,'top_center',top_center,'\nx_length',x_length,'z_length',z_length,'k_length',k_length)
                #print('xz_degree:',xz_degree) 
                degree_res['xz'] = xz_degree
                pass
            elif(k_name=='yz'):
                # row for z, and col for y
                bottom_z_idx = non_zero_coors[0][0] 
                top_z_idx = non_zero_coors[0][-1]
                bottom_max_value_y_idx = np.argwhere(imgMatData[k_name][bottom_z_idx,:]==np.amax(imgMatData[k_name][bottom_z_idx,:])) 
                bottom_center_y = self._getCenter(bottom_max_value_y_idx) 
                top_max_value_y_idx = np.argwhere(imgMatData[k_name][top_z_idx,:]==np.amax(imgMatData[k_name][top_z_idx,:])) 
                top_center_y = self._getCenter(top_max_value_y_idx)
                
                # calculate the angle 
                if(bottom_center_y==top_center_y):
                    degree_res['yz'] = 90
                    continue
                z_length  = (up_index - bottom_index) * self.zSlice
                y_length = abs(top_center_y-bottom_center_y)*self.nm_per_pixel
                y1_minus_y2 = (bottom_center_y*self.nm_per_pixel) - (top_center_y*self.nm_per_pixel)
                z1_minus_z2 = (bottom_z_idx*self.zSlice) - (top_z_idx*self.zSlice)
                k_length = np.sqrt(y1_minus_y2*y1_minus_y2+z1_minus_z2*z1_minus_z2)
                #print('yz\n','bottom_center_y',bottom_center_y,'top_center_y',top_center_y,'\ny_length',y_length,'z_length',z_length,'k_length',k_length)
                yz_degree = self._getAngleByEdge(y_length,z_length,k_length)
                degree_res['yz'] = yz_degree
                pass
        #print(degree_res)
        return degree_res 
        pass  
    def calLength(self,bbox,img,axis,nm,cur_error_threshold):

        '''
        calculate the X or Y length of the cluster by gaussian FWHM

        '''
         
        allBoxes = self.findAllBox2(img,bbox,axis) 
        all_FWHM = []
        all_FWHM_err = []
        max_length = -1
        
      
        for i in range(len(allBoxes)-1):
           cur_box = allBoxes[i] 
           cur_FWHM, cur_err = self.myModel.calFWHM(cur_box,img,axis,nm)
            
           
              
           if(type(cur_err) == str and (cur_err == 'err' or cur_err=='nan')):
               
               continue
           if(np.isnan(cur_err)):
               
               continue
   
           if(cur_err>cur_error_threshold):
               
               continue
           all_FWHM.append(cur_FWHM)
           all_FWHM_err.append(cur_err)
           if(cur_FWHM>max_length):
               max_length = cur_FWHM
        
        all_FWHM = np.asarray(all_FWHM)
        all_FWHM_err = np.asarray(all_FWHM_err)
        
        if(len(all_FWHM)>0): 
            mean_FWHM = all_FWHM.mean()
            mean_FWHM_err = all_FWHM_err.mean()
            median_FWHM = np.median(all_FWHM)
            median_FWHM_err = np.median(all_FWHM_err)
        else:
           # there not exists any valid  box
            mean_FWHM = -1
            mean_FWHM_err = ''
            median_FWHM = -1
            median_FWHM_err = ''
        
        whole_FWHM, whole_err  = self.myModel.calFWHM(allBoxes[-1],img,axis,nm)
        
        if(max_length==-1):
            max_length = whole_FWHM    
        clipFlag = False
        if(axis=='x' and max_length>self.minX):
            clipFlag = True
        elif(axis=='y' and max_length>self.minY):
            clipFlag = True

        if(whole_err>cur_error_threshold):
            whole_FWHM = -1
            whole_err = '' 
        cur_info = {'mean_FWHM':mean_FWHM,'mean_FWHM_err':mean_FWHM_err,\
                    'median_FWHM':median_FWHM,'median_FWHM_err':median_FWHM_err,\
                    'whole_FWHM':whole_FWHM,'whole_err':whole_err  }
        return cur_info,clipFlag
    def _setInfoToStat(self,stat_info,add_info,prefix=None):
        
        for key in add_info:
            if(prefix):
                new_key = prefix + '_' + key
            else:
                new_key = key
            if(add_info[key]==-1):
                add_info[key]=''
            stat_info[new_key] = [add_info[key]]
        return stat_info
        pass
    def fetchCoordinates(self,xyCoors,xyzCoors,coor_type='xy',real_z_list=[]):
        '''
         

        '''
        
        xyCoors= np.vstack((xyCoors[0], xyCoors[1])).T
        
        
        xyzCoors = np.vstack((xyzCoors[0],xyzCoors[1],xyzCoors[2])).T
        new_x = []
        new_y = []
        new_z = []
          
        for i in range(xyCoors.shape[0]):
            for k in range(xyzCoors.shape[0]):
                if(coor_type=='xy' and xyCoors[i][0]==xyzCoors[k][0] and xyCoors[i][1]==xyzCoors[k][1]):
                    new_x.append(xyCoors[i][0])
                    new_y.append(xyCoors[i][1]) 
                    new_z.append(xyzCoors[k][2])
                elif(coor_type=='yz' and xyCoors[i][0]==xyzCoors[k][1] and xyCoors[i][1]==xyzCoors[k][2]):
                    new_x.append(xyzCoors[k][0])
                    new_y.append(xyzCoors[k][1])
                    #new_z.append(xyzCoors[k][2])
                    new_z.append(real_z_list[k])
                elif(coor_type=='xz' and xyCoors[i][0]==xyzCoors[k][0] and xyCoors[i][1]==xyzCoors[k][2]):
                    new_x.append(xyzCoors[k][0])
                    new_y.append(xyzCoors[k][1])
                    #new_z.append(xyzCoors[k][2])
                    new_z.append(real_z_list[k])
         
        return [np.asarray(new_x),np.asarray(new_y),np.asarray(new_z)]   
        pass 

    def _rmZPts(self,z_slice_pts,direction='1'):
        
        '''

        iterate the z slices and remove the head points or tail points

        '''
        if(direction==1):
            start = 0
            end = len(z_slice_pts) - 1
        elif(direction==-1):
            start = len(z_slice_pts) - 1
            end = 0
        clipped_z_slice_pts = None
         
        for i in range(start,end,direction):
        
            tmp_slice_pts = z_slice_pts[i]
            
            if(tmp_slice_pts == None):
                # ignore none center points
                continue
             
            if(len(tmp_slice_pts[0])>1):
                # stop when encountering dense slice
                break
            next_slice_index = i
            # move forward to get the next slice with center points
            for j in range(i+direction,end+1,direction): 
                 
                if(z_slice_pts[j]!=None):
                     
                    next_slice_index = j
                    break

            cur_gap = abs(next_slice_index - i) * self.zSlice
            # remove the slices with large z gap
            if(cur_gap >= self.z_gap_tolerance):
                            
                if(direction==1):
                     clipped_z_slice_pts = z_slice_pts[next_slice_index:]
                     
                elif(direction==-1):
                     clipped_z_slice_pts = z_slice_pts[:next_slice_index+1]
                 
                break
            cur_x = tmp_slice_pts[0][0]
            cur_y = tmp_slice_pts[1][0]
            next_slice_pts = z_slice_pts[next_slice_index]
            

            ''' 
            if(len(next_slice_pts[0])>1):
                print("dense next slice",next_slice_pts)
                # z gap < tolerance and next slice is dense slice, do not need to remove any points
                break
            '''
            if((cur_x not in z_slice_pts[next_slice_index][0]) and (cur_y not in z_slice_pts[next_slice_index][1])):
                # not in a line ,we can accept this as noise even when z gap is relatively small
                # keep searching
               
                continue

            else:
                
                #print("overlapped x or y")
                # do not keep searching
                break


        
        if(clipped_z_slice_pts!=None):
            
            return clipped_z_slice_pts
        return z_slice_pts
        

    def rmOutlierZ(self,z_slice_pts):

        '''
        based on localized coordinates, remove outlier points with z

        remove condition: 1. only one point in the z slice; 
                          2. the z gap satisfy removing threshold 
                          3. only remove the head and tail, once encountered   
                          4. can not distory the connectivity of XY projection image
        
        starts from the head: if(more than two points in the slice) stop
                              if(gap < threshod and x or y not the same between the neiboring points) continue to the next slice
                              else:stop

        starts from the tail: smiliar as the above

        check xy connectivity
        
        return cluster_pts


        '''
        #print("rm outlier Z module ...")
        #print("current z_slice_pts",z_slice_pts) 
   
        new_z_slice_pts = self._rmZPts(z_slice_pts,direction=1)
        
        new_z_slice_pts = self._rmZPts(new_z_slice_pts,direction=-1)
        # add shifted points to z_slice_pts,z_ranges
        # get cluter points
        #if(len(new_z_slice_pts)==len(z_slice_pts)):
        #     return None
        clipped_flag = not(len(new_z_slice_pts)==len(z_slice_pts))
        #print("len(new_z_slice_pts)",len(new_z_slice_pts),"len(z_slice_pts)",len(z_slice_pts))
        
        tmp_x = []
        tmp_y = []
        tmp_z = []
        #print("new_z_slice_pts: ",new_z_slice_pts)
         
        if(clipped_flag):
            for tmp_z_slice in new_z_slice_pts:
                
                if(tmp_z_slice==None):
                    continue
                for i in range(len(tmp_z_slice[0])):

                    tmp_x.append(tmp_z_slice[0][i])
                    tmp_y.append(tmp_z_slice[1][i])
                    tmp_z.append(tmp_z_slice[2][i])

            cur_cluster = [np.asarray(tmp_x),np.asarray(tmp_y),np.asarray(tmp_z)]
        else:
            cur_cluster = None
            
        return new_z_slice_pts,[cur_cluster],clipped_flag
    

    def checkAround(self,img,x,y):
        '''
        
        '''
        n_row = img.shape[0]
        n_col = img.shape[1]
        center_gray = img[x][y]
        for i in range(-1,2):
            for j in range(-1,2):
                if(i==j and i==0):
                    continue
                if(img[i][j]>=center_gray):
                    return False
        return True

       
    def clipZCluster(self,z_slice_pts,projection='xz'):


        '''
        remove points with gray value <= threshold && it is the local maximum

        '''
        # format Z slice points
        new_x = []
        new_y = []
        new_z = []
        real_z = []
        #print("clipZCluster: ",z_slice_pts)
        for i in range(len(z_slice_pts)):
            cur_slice_pts = z_slice_pts[i]
            if(cur_slice_pts==None):
                continue
            new_x.extend(cur_slice_pts[0])
            new_y.extend(cur_slice_pts[1])
            # (x,y) belongs the ith z slice
            tmp = [i]*len(cur_slice_pts[0])
            new_z.extend(tmp)
            real_z.extend(cur_slice_pts[2])
            pass
        
        z_format_pts = [np.asarray(new_x),np.asarray(new_y),np.asarray(new_z)]
        
        if(self.max_z_removal_gray==None):    
            rm_z_threshold = 2*self.myDataLoader.lateral_shifted * self.myDataLoader.axial_shifted
        else:
            rm_z_threshold = self.max_z_removal_gray
        if(projection=='xz'):
            cur_bbox = {'x1':0,'x2':z_format_pts[0].max()+self.myDataLoader.lateral_shifted+2,'y1':0,'y2':z_format_pts[2].max()+self.myDataLoader.axial_shifted-1}
       
            xz_figure, _, cur_img = self.myDataLoader.visualize([z_format_pts[0],z_format_pts[2]],y_gap=True,bbox=cur_bbox,projection='xz',adjust_BC=False)
        
            '''
            for i in range(z_format_pts[0].min()-2,z_format_pts[0].max()+3):
                 tmp_line = ''
                 for j in range(xz_img.shape[1]):
                    tmp_line  = tmp_line + str(xz_img[i][j]) + '\t'
                 print(tmp_line.strip('\t'))
            '''
        elif(projection=='yz'):
            cur_bbox = {'x1':0,'x2':z_format_pts[1].max()+self.myDataLoader.lateral_shifted+2,'y1':0,'y2':z_format_pts[2].max()+self.myDataLoader.axial_shifted-1}
            yz_figure, _, cur_img = self.myDataLoader.visualize([z_format_pts[2],z_format_pts[1]],x_gap=True,bbox=cur_bbox,projection='yz',adjust_BC=False)

        all_clipped_pts = []
        for k in range(1,rm_z_threshold+1):
            #print("projection",projection,"clip z cluster gray", k)
            cliped_pts = self.clipCluster(cur_img,cur_bbox,z_format_pts,k,projection,real_z)
            #print("len(cliped_pts)",len(cliped_pts))
            #print(cliped_pts)
            all_clipped_pts.extend(cliped_pts)
         
        return all_clipped_pts

    def clipCluster(self,curClusterImg,curBBox,clusterPts,gray_threshold,projection='xy',real_z_list=[]):
        '''
        This is a brute force way, we might try other algorthims in the future 
        1. set points that <= gray_threshould as 0 in the image mat
        2. recalculate the connected componets
        3. collect the information of points and return them 
        '''
        # step 1, iterate cluster points and record which point has low gray threshold
        new_img  = np.array(curClusterImg, copy=True) 
        found_valid_flag = False
         
        rm_coors = np.argwhere((curClusterImg<=gray_threshold)&(curClusterImg>0))
       
        #print("rm_coors",rm_coors)
        if(len(rm_coors)==0):
            return []
        #print("clusterPts",clusterPts) 
        for k in range(rm_coors.shape[0]):
            # remove the point
            x = rm_coors[k][0]
            y = rm_coors[k][1]
            #print('x',x,'y',y,'curClusterImg',curClusterImg[x][y])
            new_img[x][y] = 0
 
            if(not found_valid_flag):
                # check if the current position at least has one center point
                for i in range(len(clusterPts[0])):
                    #print(clusterPts[0][i],clusterPts[1][i],clusterPts[2][i])
                    if(projection=='xy' and clusterPts[0][i]==x and clusterPts[1][i]==y):
                        found_valid_flag = True
                        break
                    elif(projection=='yz' and clusterPts[1][i]==x and clusterPts[2][i]==y):
                        found_valid_flag = True
                        break
                    elif(projection=='xz' and clusterPts[0][i]==x and clusterPts[2][i]==y):

                        found_valid_flag = True
                        break
        
        num_labels, labels = cluster_by_connectedComponents(new_img,self.connectedFlag)
        #print(num_labels,labels) 
        # if do not result in more than 1 connected component and not valid points were removed
        #print("num_labels<=min_cliped_cluster",num_labels<=2,"not found_valid_flag",not found_valid_flag)
        if(num_labels<=2 and (not found_valid_flag)):
            
            return [] 
        
        all_pts = []
        for i in range(1,num_labels+1):

            pts = np.where(labels==i)
            if (len(pts[0])==0):
                continue
            
            new_cluster_pts = self.fetchCoordinates(pts,clusterPts,coor_type=projection,real_z_list=real_z_list)
            all_pts.append(new_cluster_pts)
        return all_pts                    
      
        





    def genProjections(self,pts,z_slice_pts,bbox,wholeImg,clusterImg):
        '''
        generate xy,yz,xz images

        '''
        figure_data = {}
        imgMat_data = {}
        # generate xy_figure
        clusterImg = clusterImg[bbox['x1']:bbox['x2']+1,bbox['y1']:bbox['y2']+1]
        clusterImg_transpose = clusterImg.T
        #print("clusterImg_transpose.shape",clusterImg_transpose.shape) 
        xy_figure,_ = self.myDataLoader.genHeatMap(clusterImg_transpose)
        figure_data['xy'] = xy_figure

        # generate a bigger xy projection from the original whole image
        xy_x = bbox['x2'] - bbox['x1'] + 1
        xy_y = bbox['y2'] - bbox['y1'] + 1

        margin_x = int((100 - xy_x)/2)
        margin_y = int((100 - xy_y)/2)
        
        bigger_x1 = max(bbox['x1']-margin_x,0)
        bigger_x2 = min(bbox['x2']+margin_x,wholeImg.shape[0])
        bigger_y1 = max(bbox['y1']-margin_y,0)
        bigger_y2 = min(bbox['y1']+margin_y,wholeImg.shape[1])
       
        big_xy_img = wholeImg[bigger_x1:bigger_x2+1,bigger_y1:bigger_y2+1]
        big_xy_img_transpose = big_xy_img.T
        
        big_xy_figure,_ = self.myDataLoader.genHeatMap(big_xy_img_transpose)
        figure_data['big_xy'] = big_xy_figure
        
        # format Z slice points
        new_x = []
        new_y = []
        new_z = []
        for i in range(len(z_slice_pts)):
            cur_slice_pts = z_slice_pts[i]
            if(cur_slice_pts==None):
                continue
            new_x.extend(cur_slice_pts[0])
            new_y.extend(cur_slice_pts[1])
            # (x,y) belongs the ith z slice
            tmp = [i]*len(cur_slice_pts[0])
            new_z.extend(tmp)
            pass
       
        z_format_pts = [np.asarray(new_x),np.asarray(new_y),np.asarray(new_z)]
        # generate xz projection
        
        xz_bbox = {'x1':max(z_format_pts[0].min()-self.myDataLoader.lateral_shifted-2,0),'x2':min(z_format_pts[0].max()+self.myDataLoader.lateral_shifted+2,wholeImg.shape[0]),'y1':max(z_format_pts[2].min()-self.myDataLoader.axial_shifted+1,0),'y2':min(z_format_pts[2].max()+self.myDataLoader.axial_shifted-1,wholeImg.shape[1])}        
        #xz_bbox = {'x1':max(z_format_pts[0].min()-4,0),'x2':min(z_format_pts[0].max()+4,wholeImg.shape[0]),'y1':z_format_pts[2].min(),'y2':z_format_pts[2].max()}       
        #xz_bbox = {'x1':max(z_format_pts[0].min()-self.myDataLoader.lateral_shifted-2,0),'x2':min(z_format_pts[0].max()+self.myDataLoader.lateral_shifted+2,wholeImg.shape[0]),'y1':max(z_format_pts[2].min()-self.myDataLoader.axial_shifted-5+1,0),'y2':z_format_pts[2].max()+self.myDataLoader.axial_shifted*2-1+5}        
        xz_figure, _, xz_img = self.myDataLoader.visualize([z_format_pts[0],z_format_pts[2]],y_gap=True,bbox=xz_bbox,projection='xz')
        #xz_figure, _, xz_img = self.myDataLoader.visualize([z_format_pts[0],z_format_pts[2]],y_gap=True,projection='xz')
        figure_data['xz'] = xz_figure   
        imgMat_data['xz'] = xz_img
 
        # generate yz projection
        yz_bbox = {'y1':max(z_format_pts[1].min()-4,0),'y2':min(z_format_pts[1].max()+4,wholeImg.shape[1]),'x1':max(z_format_pts[2].min()-self.myDataLoader.axial_shifted+1,0),'x2':min(z_format_pts[2].max()+self.myDataLoader.axial_shifted-1,wholeImg.shape[1])}

        #yz_bbox = {'y1':max(z_format_pts[1].min()-4,0),'y2':min(z_format_pts[1].max()+4,wholeImg.shape[1]),'x1':max(z_format_pts[2].min()-self.myDataLoader.axial_shifted-5+1,0),'x2':z_format_pts[2].max()+self.myDataLoader.axial_shifted*2-1+5}
        yz_figure, _, yz_img = self.myDataLoader.visualize([z_format_pts[2],z_format_pts[1]],x_gap=True,bbox=yz_bbox,projection='yz')
        #yz_figure, _, yz_img = self.myDataLoader.visualize([z_format_pts[2],z_format_pts[1]],x_gap=True,projection='yz')
        figure_data['yz'] = yz_figure
        imgMat_data['yz'] = yz_img
 
        return figure_data,imgMat_data 

        
        pass 
    def processOneBatch(self,csvData,curImg,curZDict,splitFrameIndex=0):
        '''
        Identify the fiber for one batch data
        curZDict: {(x,y):[z list]}
        '''
        logging.info("Clustering by 8 connectivity..")    
        num_labels, labels = cluster_by_connectedComponents(curImg,self.connectedFlag,False)
        
        logging.info('extractComponents...')
        all_pts = extractComponents(curImg,num_labels,labels,curZDict)
         
        logging.info('Number of clusters: '+str(num_labels)) 
        find_coordinates = {}
        count = 0
        
        while(len(all_pts)>0):
            
              
            # cur_cluster only record the center points    
            cur_cluster = all_pts[0]
            
            
            #print("cur_cluster",cur_cluster)
            count += 1
            if(len(all_pts)>1):
                all_pts = all_pts[1:]
            else:
                all_pts = []
            if(len(cur_cluster[0])==0):
                logging.info("empty current cluster")
                continue
            
             
            # gen a new image by the current points,only the current cluster
            cur_img,cur_zDict = self.myDataLoader.genNewImg(cur_cluster,curImg,False)
            bbox = self._bbox(cur_img)
            #print(cur_img[bbox['x1']:bbox['x2']+1,bbox['y1']:bbox['y2']+1])

            # check connectivity 
            check_num_labels, _ = cluster_by_connectedComponents(cur_img,self.connectedFlag,False) 
            if(check_num_labels>2):
                logging.info("current cluster is not connected")
                continue 
            # check the max gray value of the cluster 
            cur_max_gray,cur_sum_gray = self.myModel.calMaxGrayVal(cur_img,bbox)
            
            '''
            # fingerprint of this molecular cluster            
            #tmp_coors =  str(bbox['x1']) + '.' + str(bbox['y1']) + '.' + str(bbox['x2']) + '.' + str(bbox['y2']) + '.' + str(len(cur_cluster[0]))
            tmp_coors = str(bbox['x1']) + '.' + str(bbox['y1']) + '.' + str(bbox['x2']) + '.' + str(bbox['y2'])
            
            if(tmp_coors in find_coordinates and cur_sum_gray<find_coordinates[tmp_coors]):
                
                logging.info('current cluster has been identified before')
                continue
            '''
             
            if(self.minGray and cur_max_gray < self.minGray):
                continue
            
            # check if the cluster has continue Z slice
            logging.info("start check Z continuous")       
            flag,z_slice_pts,z_slice_ranges = self.checkZContinues(cur_cluster)
            logging.info("check continue Z complete")
            
            pseudo_z_length = len(z_slice_ranges) * self.zSlice
            tmp_coors = str(bbox['x1']) + '.' + str(bbox['y1']) + '.' + str(bbox['x2']) + '.' + str(bbox['y2']) + '.' +str(pseudo_z_length)
            if(tmp_coors in find_coordinates):
                
                logging.info('current cluster has been identified before')
                continue

            find_coordinates[tmp_coors] = cur_sum_gray




            # if the total Z length of this cluster is smaller than the Z threshold, 
            # there is no need to analyze this cluster  
            if(pseudo_z_length<self.minZ):
                #print("pseudo_z_length",pseudo_z_length)
                logging.info("z length is not satisfying")
                continue
                      
            # if the current cluster does not has continuous Z slice
            # we extract the maximum continue Z 
            if(flag==1):
                
                          
                       
                x_info,clipFlagX = self.calLength(bbox,cur_img,'x',self.nm_per_pixel,self.ERROR_THRESHOLD) 
               
                y_info,clipFlagY = self.calLength(bbox,cur_img,'y',self.nm_per_pixel,self.ERROR_THRESHOLD)
                logging.info("X Y dimension calculation complete") 
                
                if((x_info['whole_FWHM']>=self.minX or x_info['mean_FWHM']>=self.minX or x_info['median_FWHM'] >= self.minX) and (x_info['whole_FWHM']<=self.maxX or x_info['mean_FWHM']<=self.maxX or x_info['median_FWHM'] <= self.maxX) and (y_info['whole_FWHM']>=self.minY or y_info['mean_FWHM']>=self.minY or y_info['median_FWHM'] >= self.minY) and (y_info['whole_FWHM']<=self.maxY or y_info['mean_FWHM']<=self.maxY or y_info['median_FWHM'] <= self.maxY)):
                    
                    
                    logging.info("Save the cluster")
                    # extract the found fiber statistics
                    #stat_info,clusterData = self.myDataManager.extractData(cur_cluster,csvData)
                    stat_info,clusterData = self.myDataManager.extractData(z_slice_pts,csvData)
                    logging.info("extract data complete")
                    stat_info = self._setInfoToStat(stat_info,x_info,'y')
                    stat_info = self._setInfoToStat(stat_info,y_info,'x')
                    stat_info['Pseudo Z size'] =[ pseudo_z_length]
                    stat_info['Z range'] = ['[' + str(z_slice_ranges[0][0] ) + ',' + str(z_slice_ranges[-1][1]) + ']']
                    #bbox
                    stat_info['upper left'] = ['['+str(bbox['x1'])+','+str(bbox['y1'])+']']
                    stat_info['lower right'] = ['['+str(bbox['x2'])+','+str(bbox['y2'])+']']
                    stat_info['sum of gray value'] = [cur_sum_gray] 
                     
                    # generate visualizations
                    figureData,imgMatData = self.genProjections(cur_cluster,z_slice_pts,bbox,curImg,cur_img)
                    logging.info("figure rendering complete")

                    # FWHM for xz 
                    xz_bbox = self._bbox(imgMatData['xz']) 
                    #print("imgMatData['xz']",imgMatData['xz'],imgMatData['xz'].shape)  
                    xz_info, _ = self.calLength(xz_bbox,imgMatData['xz'],'x',self.zSlice,self.z_FWHM_error_threshold) 
                    stat_info = self._setInfoToStat(stat_info,xz_info,'xz')
                     
                    # FWHM for yz
                    yz_bbox = self._bbox(imgMatData['yz'])
                    #print("imgMatData['yz']",imgMatData['yz'],imgMatData['yz'].shape) 
                    yz_info,_ =self.calLength(yz_bbox,imgMatData['yz'],'y',self.zSlice,self.z_FWHM_error_threshold)
                    stat_info = self._setInfoToStat(stat_info,yz_info,'yz') 
                     
                    angle_res = self.calAngle(imgMatData) 
                    stat_info = self._setInfoToStat(stat_info,angle_res,'angle')
                    self.myDataManager.saveData(stat_info,clusterData,figureData,splitFrameIndex)
                    
                    logging.info("saveData complete")
                    #find_coordinates[tmp_coors] = True
                     
                       
                if((clipFlagX or clipFlagY)and (self.max_removel_gray!=None)):
                    logging.info("Apply the noise removal module..") 
                    for clip_threshold in range(min(self.max_removel_gray,int(cur_max_gray/2)),0,-1):
                         
                        clipped_pts = self.clipCluster(cur_img,bbox,cur_cluster,clip_threshold)
                        
                        if(len(clipped_pts)!=0):
                            pass
                            clipped_pts.extend(all_pts)
                            all_pts = clipped_pts
                         
                    pass

                
                zCliped_pts,zCliped_cluster_pts,zClipped_flag = self.rmOutlierZ(z_slice_pts)
                #print("zCliped_pts",zCliped_pts)            
                if(zClipped_flag):
                    #print("z clipped!")
                    #print("zCliped_cluster_pts",zCliped_cluster_pts)
                    #print("zCliped_pts",zCliped_pts)
                    zCliped_cluster_pts.extend(all_pts)
                    all_pts = zCliped_cluster_pts
                    clipping_z_pts = zCliped_pts
                else:
                    
                    clipping_z_pts = z_slice_pts

                xz_clipped_pts = self.clipZCluster(clipping_z_pts,projection='xz')
                
                if(len(xz_clipped_pts)!=0):
                     
                    xz_clipped_pts.extend(all_pts)
                    all_pts = xz_clipped_pts
               
                      
                yz_clipped_pts = self.clipZCluster(clipping_z_pts,projection='yz')
                if(len(yz_clipped_pts)!=0):
                     
                    yz_clipped_pts.extend(all_pts)
                    all_pts= yz_clipped_pts
                
               
            else:
                logging.info("Current cluster is not continue in Z-axis") 
                sorted_continue_z_slice_pts = self.findContinueZ(z_slice_pts,z_slice_ranges)
                
                connectedComponents = self.splitContinueZAsConnected(sorted_continue_z_slice_pts,cur_img)             
                
                # add the connectedComponents to the front of the queue
                #print("connectedComponents",connectedComponents)        
                connectedComponents.extend(all_pts)
                all_pts = connectedComponents
                pass
        
        pass

    
    def _bbox(self,img):
    
        bbox = {}
        
        nonZeroCoor = np.nonzero(img)
        #print('nonZeroCoor',nonZeroCoor)
        bbox['x1'] = nonZeroCoor[0].min()
        bbox['x2'] = nonZeroCoor[0].max()
        bbox['y1'] = nonZeroCoor[1].min()
        bbox['y2'] = nonZeroCoor[1].max() 

        return bbox
    def _formatZSlicePts(self,z_slice_pts):

        '''
        input: a continue z_slice_pts : [[slice1:[x array, y array, z array]], slice2:[x array, y array z array]]
      
        output: pts: [all x array, all y array, all z array]
       
        '''
        all_x_array = []
        all_y_array = []
        all_z_array = []
        for cur_slice in z_slice_pts:
            all_x_array.extend(cur_slice[0])
            all_y_array.extend(cur_slice[1])
            all_z_array.extend(cur_slice[2])
        pts = [np.asarray(all_x_array),np.asarray(all_y_array),np.asarray(all_z_array)]
        return pts
    
    def _consZDic(pts):
        '''
        '''
        zDict = {}
        for i in range(len(pts[0])):
            tmp = (pts[0][i],pts[1][i])
            if(tmp not in zDict):
                zDict[tmp] = []
            zDict[tmp].append(pts[2][i])
        return zDict 
        pass
    def splitContinueZAsConnected(self,sorted_continue_z_slice_pts,cur_img):
        
        '''

        '''
      
        connnected_components_list = []
        for cur_z_slice_pts in sorted_continue_z_slice_pts:
            cur_pts= self._formatZSlicePts(cur_z_slice_pts)
                        
            # judge if the continue z cluster is connected or not 
            cur_cluster_img,curZDict = self.myDataLoader.genNewImg(cur_pts,cur_img,False)
             
            cur_bbox= self._bbox(cur_cluster_img)
                   
            num_labels, labels = cluster_by_connectedComponents(cur_cluster_img,self.connectedFlag)
            if(num_labels<=2):
                # the current cluster is connected
                connnected_components_list.append(cur_pts)
            else:
                # the current cluster is not connected
                # split them, but it is possible to generate not continue Z cluster
                 
                connected_all_pts = extractComponents(cur_cluster_img,num_labels,labels,curZDict)
                connnected_components_list.extend(connected_all_pts)
        
        
        return connnected_components_list  
                  
        pass 

    def checkZContinues(self,cur_cluster):
        '''
        check if the cluster has continue Z 
        input: [x array, y array, z array], only center points
        
        return z_continue_flag: Boolean, sorted points by Z slice
        '''
        # whether to ignor too less point here?
            
        x_array = cur_cluster[0]
        y_array = cur_cluster[1]
        z_array = cur_cluster[2]
 
        sort_z_args = np.argsort(z_array) 
         
        # get min and max z from the exact z value

        # min z of the slice: min z minus shifted value
        # clip head and tail version 
        #min_z =  math.floor(z_array[sort_z_args[0]]/self.zSlice)*self.zSlice - self.zSlice*int(self.myDataLoader.axial_shifted-1)  
        #max_z = min_z + math.ceil((z_array[sort_z_args[-1]]-min_z)/self.zSlice)*self.zSlice +  self.zSlice*int(self.myDataLoader.axial_shifted-1)
         
        # do not clip head and tail version
        min_z = math.floor(z_array[sort_z_args[0]]/self.zSlice)*self.zSlice        
        max_z = min_z + math.ceil((z_array[sort_z_args[-1]]-min_z)/self.zSlice)*self.zSlice
        
        if((z_array[sort_z_args[-1]]-min_z)%self.zSlice==0):
            max_z += self.zSlice 
         
        z_slice_num = int(math.ceil((max_z - min_z)/self.zSlice)) + (2*self.myDataLoader.axial_shifted-2)
        
    
        # check if every z slice has a valid point 
        existed_flag = np.zeros(z_slice_num)
        
        # clip version 
        #start = min_z
        # not clip version
        start = min_z - self.zSlice*(self.myDataLoader.axial_shifted-1)
        end = start + self.zSlice
        
        z_slice_pts = [None]*z_slice_num
        # x_y_coors is to store which pts belongs which slice
         
        # the index of the slice
        index = 0
        existed_z_values = [None]*z_slice_num
        # existed_z_values: [[slice_start1, slice_end2],...] None means no point falls into that Z slice

        for i in sort_z_args:
            # find the point belongs to which slice
            #print("z_array[i]",z_array[i])

            # find the current z slice
            while(not(z_array[i]>=start and z_array[i]<end)):
                start += self.zSlice
                end += self.zSlice
                index += 1

            # apply the axial shifts to Z axis, 
            # thus, set the neibor slice as existed, record z slice range
            for k in range(int(self.myDataLoader.axial_shifted-1)*-1,int(self.myDataLoader.axial_shifted-1)+1,1):
                #print("k",k,'index',index) 
                existed_z_values[index+k] = [start+self.zSlice*k,end+self.zSlice*k]
            if(z_slice_pts[index]==None):
                # record exactly points for the current slice, pts: [x array, y array, z array]
                z_slice_pts[index] = [[],[],[]]
            
            #print(x_array[i],y_array[i],z_array[i])        
            z_slice_pts[index][0].append(x_array[i])
            z_slice_pts[index][1].append(y_array[i])
            z_slice_pts[index][2].append(z_array[i])
        
        
        if(None in existed_z_values):
            return 0, z_slice_pts, existed_z_values
        else:
            return 1, z_slice_pts, existed_z_values
        

    def findContinueZ(self,z_slice_pts,z_ranges):

        '''
        z_slice_pts: [[x array, y array, z array],...]
        z_range:

        '''


        
        interval = []
        gray = []
        tmp_gray = 0
        x=0
        y=0

        my_flag = 0


        z_values_list = []
        interval_size = []  

        for i in range(len(z_ranges)):

            if(z_ranges[i]!=None):
                my_flag += 1
                # the first interval
                if(my_flag ==1):
                    # record the left point index
                    x = i
            # the gap
            else:
                # record the interval
                if(my_flag>0):
                    y = i-1
                    #print('x',x,'y',y,'z_ranges[i]',z_ranges[i])
                    # record the interval index
                    
                    interval_size.append(y-x+1)
                    tmp_z_slice_pts = []
                    for k in z_slice_pts[x:y+1]:
                        if(k==None):
                            continue
                        tmp_z_slice_pts.append(k)
                    #print('tmp_z_slice_pts',tmp_z_slice_pts)
                    z_values_list.append(tmp_z_slice_pts)
                    my_flag = 0

        # add the last interval
        if(my_flag!=0):
            
            interval_size.append(len(z_ranges)-x)
            tmp_z_slice_pts = []
            for k in z_slice_pts[x:len(z_ranges)]:
                if(k==None):
                    continue
                tmp_z_slice_pts.append(k)
            z_values_list.append(tmp_z_slice_pts)
        #print("z_values_list",z_values_list)


        index = 0

        sorted_index = list(np.argsort(interval_size))[::-1]
        sorted_interval = []
        #sorted_interval = np.take(interval,sorted_index,0)

        #new_z_values_list = np.take(z_values_list,sorted_index,0)
        
                 
        # sorted the intervals
        new_z_values_list = []
        for k in range(len(sorted_index)):
            index = sorted_index[k]
        
            #print('z_values_list[index]',z_values_list[index])
            new_z_values_list.append(z_values_list[index])
        
        return new_z_values_list





