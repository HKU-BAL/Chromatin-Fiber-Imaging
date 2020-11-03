import pandas as pd
from seg import cluster_by_connectedComponents,extractComponents
import numpy as np
from lmfit import Model as lmModel
from matplotlib import pyplot as plt
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
    def __init__(self,SAVE_ROOT,second_per_frame):
        self.save_index = 0
        self.data_list = []
        self.save_root = SAVE_ROOT
        if(not os.path.exists(self.save_root)):
            os.mkdir(self.save_root) 
        self.second_per_frame = second_per_frame
        
        self.batch_num = 0
        
        
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

        cur_batch_dir =  os.path.join(cur_root,'result_batch_'+str(self.batch_num))
        stat_path = os.path.join(cur_batch_dir,'stat')
        xy_path = os.path.join(cur_batch_dir,'xy')
        xz_path = os.path.join(cur_batch_dir,'xz')
        yz_path = os.path.join(cur_batch_dir,'yz')
        big_xy_path = os.path.join(cur_batch_dir,'big_xy')
        original_data_path = os.path.join(cur_batch_dir,'source_csv')

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
        cols = ['x_mean_FWHM','x_mean_FWHM_err','y_mean_FWHM','y_mean_FWHM_err','x_median_FWHM','x_median_FWHM_err','y_median_FWHM','y_median_FWHM_err','x_whole_FWHM','x_whole_err','y_whole_FWHM','y_whole_err','X average location precision','Y average location precision','Z size','Z range','Life time','avg_photon_num','Max frame gap','Frame length','sum of gray value','upper left','lower right']
        stat_info = stat_info[cols] 
        stat_info.to_csv(os.path.join(stat_path,str(self.save_index)+'_stat.csv'),index=False)

        # generate visualization image
        
        # save xy project image
         
        imgs['xy'].savefig(os.path.join(xy_path,str(self.save_index)+'_xy.png'),transparent=True) 
        
        # save big_xy projectiong image
        
        imgs['big_xy'].savefig(os.path.join(big_xy_path,str(self.save_index)+'_big_xy.png'),transparent=True) 
        

        # save xz projection
        
        imgs['xz'].savefig(os.path.join(xz_path,str(self.save_index)+'_xz.png'),transparent=True) 
        
        # save yz projection
        
        imgs['yz'].savefig(os.path.join(yz_path,str(self.save_index)+'_yz.png'),transparent=True) 

        self.save_index += 1 
        plt.close('all')    
        
    def extractData(self,pts,csvData):

        '''
        extract the data from the csv data
        pts: points, [[x array, y array, z array], ...]
        csvData: original splitted input csv data
        
        return cluster statistics and cluster original csv data

        '''
        index = []
        
         
        for i in range(len(pts[0])):
            
             _x = pts[0][i]
             _y = pts[1][i]
             _z = pts[2][i]
             f1 = csvData['X_magnificated'] == _x
             f2 = csvData['Y_magnificated'] == _y
             f3 = csvData['Z_normalized'] ==  _z
             #print('_x',_x,'_y',_y,'_z',_z)
             
            
             tmpData = csvData[f1&f2&f3]
             #print(tmpData.shape)
             index.append(tmpData.index[0])

        
        newData = csvData.iloc[index]

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
        stat_info = {"X average location precision":[x_location_precision],\
                     "Y average location precision":[y_location_precision],\
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
        # extract data point from the image
        for i in range(box['x1'],min((box['x2']+1),img.shape[0])):
            for j in range(box['y1'],min(box['y2']+1,img.shape[1])):

                if(box_flag == 'y'):

                    # the distance
                    _tmp_dist = (i - box['x1'] ) * nm_per_pixel
                elif(box_flag == 'x'):
                    _tmp_dist = (j - box['y1']) * nm_per_pixel

                if(_tmp_dist not in data):
                    data[_tmp_dist] = 0
              
                data[_tmp_dist] += img[i][j]
                 
        
        # x value
        

        data_x = np.asarray(list(data.keys()))
        v = list(data.values())
       
         
        # normalize y values
        if(box_flag == 'y'):
            data_y = np.asarray(v) / float(box['y2']-box['y1']+1)
        elif(box_flag == 'x'):
            data_y = np.asarray(v) / float(box['x2']-box['x1']+1)
        
       
        if(len(np.take(data_y,np.argwhere(data_y>0),0))<4):
            
            return -1,'err'
 
        
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
            err = 'err'
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
    def __init__(self,minX,maxX,minY,maxY,minZ,zSlice,dataManager,dataLoader,minGray,error_threshold,nm_per_pixel):
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
    def identifyFiber(self):

        '''
        identify the fiber from the dataManager
        no return, save the identified data by using data manager to the save path
 
        '''
        
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
                        all_boxes.append(tmp_box)
            all_boxes.append({'x1':x1,'x2':x2,'y1':y1-shift_val,'y2':y2+shift_val})

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
                        all_boxes.append(tmp_box)
            all_boxes.append({'x1':x1-shift_val,'x2':x2+shift_val,'y1':y1,'y2':y2})
        return all_boxes
        pass  
    def calLength(self,bbox,img,axis):

        '''
        calculate the X or Y length of the cluster by gaussian FWHM

        '''
         
        allBoxes = self.findAllBox2(img,bbox,axis) 
        all_FWHM = []
        all_FWHM_err = []
        max_length = -1
        
      
        for i in range(len(allBoxes)-1):
           cur_box = allBoxes[i] 
           cur_FWHM, cur_err = self.myModel.calFWHM(cur_box,img,axis,self.nm_per_pixel)
           
           
              
           if(type(cur_err) == str and (cur_err == 'err' or cur_err=='nan')):
               
               continue
           if(np.isnan(cur_err)):
               
               continue
   
           if(cur_err>self.ERROR_THRESHOLD):
               
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
            mean_FWHM_err = -1
            median_FWHM = -1
            median_FWHM_err = -1
        
        whole_FWHM, whole_err  = self.myModel.calFWHM(allBoxes[-1],img,axis,self.nm_per_pixel)
        
        
        clipFlag = False
        if(axis=='x' and max_length>self.minX):
            clipFlag = True
        elif(axis=='y' and max_length>self.minY):
            clipFlag = True

         
        cur_info = {'mean_FWHM':mean_FWHM,'mean_FWHM_err':mean_FWHM_err,\
                    'median_FWHM':median_FWHM,'median_FWHM_err':median_FWHM_err,\
                    'whole_FWHM':whole_FWHM,'whole_err':whole_err  }
        return cur_info,clipFlag
    def _setInfoToStat(self,stat_info,add_info,prefix=''):
        
        for key in add_info:
            new_key = prefix + '_' + key
            stat_info[new_key] = [add_info[key]]
        return stat_info
        pass
    def fetchCoordinates(self,xyCoors,xyzCoors):
        '''
         

        '''
        #print("fetchCoordinates vstack before",xyCoors)
        xyCoors= np.vstack((xyCoors[0], xyCoors[1])).T
        #print('#'*10)
        #print("fetchCoordinates vstack after",xyCoors)
        xyzCoors = np.vstack((xyzCoors[0],xyzCoors[1],xyzCoors[2])).T
        new_x = []
        new_y = []
        new_z = []
        #print("xyCoors.shape",xyCoors.shape,'xyzCoors.shape',xyzCoors.shape)
        for i in range(xyCoors.shape[0]):
            for k in range(xyzCoors.shape[0]):
                if(xyCoors[i][0]==xyzCoors[k][0] and xyCoors[i][1]==xyzCoors[k][1]):
                    new_x.append(xyCoors[i][0])
                    new_y.append(xyCoors[i][1]) 
                    new_z.append(xyzCoors[k][2])
        
        return [np.asarray(new_x),np.asarray(new_y),np.asarray(new_z)]   
        pass 

 
    def clipCluster(self,curClusterImg,curBBox,clusterPts,gray_threshold):
        '''
        This is a brute force way, we might need to try other algorthims in the future 
        1. set points that <= gray_threshould as 0 in the image mat
        2. recalculate the connected componets
        3. collect the information of points and return them 
        '''
        # step 1, iterate cluster points and record which point has low gray threshold
        new_img  = np.array(curClusterImg, copy=True) 
        found_valid_flag = False
        #print('new_img.shape',new_img.shape) 
        rm_coors = np.argwhere((curClusterImg<=gray_threshold)&(curClusterImg>0))
       

        if(len(rm_coors)==0):
            return []
        #print("rm_coors",rm_coors)    
        for k in range(rm_coors.shape[0]):
            # remove the point
            x = rm_coors[k][0]
            y = rm_coors[k][1]
            #print('x',x,'y',y,'curClusterImg',curClusterImg[x][y])
            new_img[x][y] = 0

            if(not found_valid_flag):
                # check if the current position at least has one center point
                for i in range(len(clusterPts[0])):
                    if(clusterPts[0][i]==x and clusterPts[1][i]==y):
                        found_valid_flag = True
                        break
        
        num_labels, labels = cluster_by_connectedComponents(new_img,self.connectedFlag)
 
        #print("num_labels",num_labels,"found_valid_flag",found_valid_flag)
        # if do not result in more than 1 connected component and not valid points were removed
        if(num_labels<=2 and (not found_valid_flag)):
            return [] 
        
        all_pts = []
        for i in range(1,num_labels+1):

            pts = np.where(labels==i)
            if (len(pts[0])==0):
                continue
            
            new_cluster_pts = self.fetchCoordinates(pts,clusterPts)
            all_pts.append(new_cluster_pts)
        return all_pts                    
      
        pass





    def genProjections(self,pts,z_slice_pts,bbox,wholeImg,clusterImg):
        '''
        generate xy,yz,xz images

        '''
        figure_data = {}
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
        #print("big_xy_img_transpose.shape",big_xy_img_transpose.shape)
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
        
        xz_bbox = {'x1':max(z_format_pts[0].min()-4,0),'x2':min(z_format_pts[0].max()+4,wholeImg.shape[0]),'y1':z_format_pts[2].min(),'y2':z_format_pts[2].max()}       
        xz_figure, _ = self.myDataLoader.visualize([z_format_pts[0],z_format_pts[2]],y_gap=True,bbox=xz_bbox)
        figure_data['xz'] = xz_figure   
 
        # generate yz projection
        yz_bbox = {'y1':max(z_format_pts[1].min()-4,0),'y2':min(z_format_pts[1].max()+4,wholeImg.shape[1]),'x1':z_format_pts[2].min(),'x2':z_format_pts[2].max()}
        yz_figure, _ = self.myDataLoader.visualize([z_format_pts[2],z_format_pts[1]],x_gap=True,bbox=yz_bbox)
        figure_data['yz'] = yz_figure 
        return figure_data 

        
        pass 
    def processOneBatch(self,csvData,curImg,curZDict,splitFrameIndex=0):
        '''
        Identify the fiber for one batch data
        curZDict: {(x,y):[z list]}
        '''
        
        num_labels, labels = cluster_by_connectedComponents(curImg,self.connectedFlag,False)
        
        logging.info('extractComponents...')
        all_pts = extractComponents(curImg,num_labels,labels,curZDict)
         
        logging.info('Number of clusters',num_labels) 
        find_coordinates = {}
        count = 0
        
        while(len(all_pts)>0):
            
              
            # cur_cluster only record the center points    
            cur_cluster = all_pts[0]
            

            count += 1
            if(len(all_pts)>1):
                all_pts = all_pts[1:]
            else:
                all_pts = []
             
            # gen a new image by the current points,only the current cluster
            cur_img,cur_zDict = self.myDataLoader.genNewImg(cur_cluster,curImg,False)
            
            # bbox 
            bbox = self._bbox(cur_img)
            


            ''' 
            tmp_coors =  str(bbox['x1']) + '.' + str(bbox['y1']) + '.' + str(bbox['x2']) + '.' + str(bbox['y2'])
            if(tmp_coors in find_coordinates):
                continue 
            
            logging.info("current region %s" %(tmp_coors)) 
            '''
            # check the max gray value of the cluster 
            cur_max_gray,cur_sum_gray = self.myModel.calMaxGrayVal(cur_img,bbox)

            # fingerprint of this molecular cluster            
            tmp_coors =  str(bbox['x1']) + '.' + str(bbox['y1']) + '.' + str(bbox['x2']) + '.' + str(bbox['y2']) + '.' + str(cur_sum_gray)
            if(tmp_coors in find_coordinates):
                continue
            find_coordinates[tmp_coors] = True 
            logging.info("current region %s" %(tmp_coors))

 
            if(self.minGray and cur_max_gray < self.minGray):
                continue
  
            # check if the cluster has continue Z slice
           
            flag,z_slice_pts,z_slice_ranges = self.checkZContinues(cur_cluster)

            # remove the first and end slices
            pseudo_z_length = (len(z_slice_ranges) - self.myDataLoader.lateral_shifted)* self.zSlice

            # if the total Z length of this cluster is smaller than the Z threshold, 
            # there is no need to analyze this cluster  
            if(pseudo_z_length<self.minZ):
                continue
                        
            # if the current cluster does not has continuous Z slice
            # we extract the maximum continue Z 
            if(flag==1):
                
                       
                x_info,clipFlagX = self.calLength(bbox,cur_img,'x') 
               
                y_info,clipFlagY = self.calLength(bbox,cur_img,'y')
                
               
                if((x_info['whole_FWHM']>=self.minX or x_info['mean_FWHM']>=self.minX or x_info['median_FWHM'] >= self.minX) and (x_info['whole_FWHM']<=self.maxX or x_info['mean_FWHM']<=self.maxX or x_info['median_FWHM'] <= self.maxX) and (y_info['whole_FWHM']>=self.minY or y_info['mean_FWHM']>=self.minY or y_info['median_FWHM'] >= self.minY) and (y_info['whole_FWHM']<=self.maxY or y_info['mean_FWHM']<=self.maxY or y_info['median_FWHM'] <= self.maxY)):
                    
                    
                    logging.info("Save the cluster")
                    # extract the found fiber statistics
                    stat_info,clusterData = self.myDataManager.extractData(cur_cluster,csvData)
                    stat_info = self._setInfoToStat(stat_info,x_info,'y')
                    stat_info = self._setInfoToStat(stat_info,y_info,'x')
                    stat_info['Z size'] =[ pseudo_z_length]
                    stat_info['Z range'] = ['(' + str(z_slice_ranges[0][0] ) + ',' + str(z_slice_ranges[-1][1]) + ')']
                    #bbox
                    stat_info['upper left'] = ['('+str(bbox['x1'])+','+str(bbox['y1'])+')']
                    stat_info['lower right'] = ['('+str(bbox['x2'])+','+str(bbox['y2'])+')']
                    stat_info['sum of gray value'] = [sum(sum(cur_img))] 
                    
                    
                    # generate visualizations
                    figureData = self.genProjections(cur_cluster,z_slice_pts,bbox,curImg,cur_img) 
                    self.myDataManager.saveData(stat_info,clusterData,figureData,splitFrameIndex)
                    find_coordinates[tmp_coors] = True
                     
                       
                if(clipFlagX or clipFlagY):
                    logging.info("Apply the noise removal module..") 
                    for clip_threshold in range(min(8,int(cur_max_gray/2)),0,-1):
                         
                        clipped_pts = self.clipCluster(cur_img,bbox,cur_cluster,clip_threshold)
                        
                        if(len(clipped_pts)!=0):
                             
                            clipped_pts.extend(all_pts)
                            all_pts = clipped_pts
                         
                    pass
            else:
                logging.info("Current cluster is not continue in Z-axis") 
                sorted_continue_z_slice_pts = self.findContinueZ(z_slice_pts,z_slice_ranges)
                
                connectedComponents = self.splitContinueZAsConnected(sorted_continue_z_slice_pts,cur_img)             
                
                # add the connectedComponents to the front of the queue
                
                connectedComponents.extend(all_pts)
                all_pts = connectedComponents
                pass
        
        pass

    
    def _bbox(self,img):
    
        bbox = {}
        
        nonZeroCoor = np.nonzero(img)
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
         
        min_z = math.floor(z_array[sort_z_args[0]]/self.zSlice)*self.zSlice - self.zSlice*int(self.myDataLoader.lateral_shifted/2) 
        
        max_z = min_z + math.ceil((z_array[sort_z_args[-1]]-min_z)/self.zSlice)*self.zSlice +  self.zSlice*int(self.myDataLoader.lateral_shifted/2)
        if((z_array[sort_z_args[-1]]-min_z)%self.zSlice==0):
            max_z += self.zSlice*int(self.myDataLoader.lateral_shifted/2) 
         
        z_slice_num = math.ceil((max_z - min_z)/self.zSlice) 
        
        
        # check if every z slice has a valid point 
        existed_flag = np.zeros(z_slice_num)
        start = min_z
        end = min_z + 20
        z_slice_pts = [None]*z_slice_num
        # x_y_coors is to store which pts belongs which slice
         
        # the index of the slice
        index = 0
        existed_z_values = [None]*z_slice_num
        # existed_z_values: [[slice_start1, slice_end2],...] None means no point falls into that Z slice

        for i in sort_z_args:
            # find the point belongs to which slice
            #print("z_array[i]",z_array[i])
            while(not(z_array[i]>=start and z_array[i]<end)):
                start += 20
                end += 20
                index += 1

            # lateral shift also applies to Z axis, 
            # thus, set the neibor slice as existed, record z slice range
            for k in range(int(self.myDataLoader.lateral_shifted/2)*-1,int(self.myDataLoader.lateral_shifted/2)+1,1):
                #print('index',index,'k',k)
                existed_z_values[index+k] = [start+self.zSlice*k,end+self.zSlice*k]
            if(z_slice_pts[index]==None):
                # record exactly points for the current slice, pts: [x array, y array, z array]
                z_slice_pts[index] = [[],[],[]]
            
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





