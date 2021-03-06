

import argparse

from findFiber import DataManager,DataProcessor,FindFiber

from loadData import DataLoader


import logging
logging.getLogger().setLevel(logging.INFO)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Argument for running the program')
    parser.add_argument('--DATA_PATH', metavar='N', type=str,
                   help='INPUT CSV PATH')

    parser.add_argument('--SECOND_PER_FRAME',type=float,help='secod per frame of the input data, it is probably set by the miscroscopy')
    parser.add_argument('--NM_PER_PIXEL',type=float,help='nanometer per pixel. Please set the value of after performing magnification (if any)')


    parser.add_argument('--MAGNIFICATION', metavar='N', type=float,
                   default=1,help='magnification ratio')

    parser.add_argument('--SAVE_PATH', metavar='N', type=str,help='SAVE PATH, if it not existed, the program will creat one')

    parser.add_argument('--X_MIN',type=float,default=20,help='minimum X size of the target object')
    parser.add_argument('--X_MAX',type=float,default=50, help='maximum X size of the target object')

    parser.add_argument('--Y_MIN',type=float,default=20,help='minimum Y size of the target object')
    parser.add_argument('--Y_MAX',type=float,default=50, help='maximum Y size of the target object')

    parser.add_argument('--Z_MIN',type=float,default=100, help="minimum Z size of the target object")
    

    # valid region settings, optional
    parser.add_argument('--X1',type=int,default=None,help="(optional) the left coordinate of the valid region")
    parser.add_argument('--Y1',type=int,default=None, help="(optional) the low coordinate of the valid region")
    parser.add_argument('--X2',type=int,default=None, help="(optional) the right coordinate of the valid region")
    parser.add_argument('--Y2',type=int,default=None, help="(optional) the up coordinate of the valid region")
    parser.add_argument('--Z1',type=int,default=None,help="(optional) the valid minimum Z coordinate")
    parser.add_argument('--Z2',type=int,default=None,help="(optional) the valid maximum Z coordinate")

    parser.add_argument('--Z_SLICE',type=float,default=10, help='set the length of slices on z-axis')
    parser.add_argument('--ERROR_THRESHOLD',type=float,default=6,help='fitting error threshold for gussian')
    parser.add_argument('--Z_FWHM_ERROR_THRESHOLD',type=float,default=40,help='Guassian fitting error threshold')
   
    parser.add_argument('--MAX_GRAY_RM_THRESHOLD',type=int,default=8,help='(required) The pixels with the max gray value of the molecular clusters should bigger than this parameter. Otherwise, the cluster will not be analyzed.')
    parser.add_argument('--FRAME_RANGE',type=int,default=None,help='frame range for each of the frame intervals')
    parser.add_argument('--LOC_PREC_THRESHOLD',type=float,default=100,help='filtering options for the localization precision')
    parser.add_argument('--VIS_METHOD',type=str,default='Average shifted histogram',help='visualization method')
    parser.add_argument('--LATERAL_SHIFT',type=int,default=2,help='lateral shifts')
    parser.add_argument('--AXIAL_SHIFT',type=int,default=4,help='axial shift')

    parser.add_argument('--MAX_NOISE_GRAY',type=int,default=8,help='max removal gray value, which is used in noise removal module')
    parser.add_argument('--Z_GAP_TOLERANCE',type=float,default=50,help='z gap tolerance threshold')
    parser.add_argument('--MAX_Z_NOISE_GRAY',type=int,default=None,help='max removal gray value for xz and yz projections')
 
    args = parser.parse_args()



    myValidRegion = {'x1':args.X1,'x2':args.X2,'y1':args.Y1,'y2':args.Y2,'z1':args.Z1,'z2':args.Z2}
    myDataManager = DataManager(args.SAVE_PATH,args.SECOND_PER_FRAME)
    myDataProcessor = DataProcessor()
    myDataLoader = DataLoader(args.VIS_METHOD,lateral_shifted=args.LATERAL_SHIFT,axial_shifted=args.AXIAL_SHIFT,nm_per_pixel=args.NM_PER_PIXEL,validRegion=myValidRegion)
    # prepare csv data
    myDataManager.data_list,dataCheckFlag = myDataProcessor.prepare(DATA_PATH=args.DATA_PATH,splitFrameLen=args.FRAME_RANGE,locPrecThreshold=args.LOC_PREC_THRESHOLD,magnificationRatio=args.MAGNIFICATION)
    
    if(dataCheckFlag!=1):
        logging.info("incorrent input!")
        exit()
 
    myFiberIdentifier = FindFiber(minX=args.X_MIN,maxX=args.X_MAX,minY=args.Y_MIN,maxY=args.Y_MAX,minZ=args.Z_MIN,zSlice=args.Z_SLICE,dataManager=myDataManager,dataLoader=myDataLoader,minGray=args.MAX_GRAY_RM_THRESHOLD,error_threshold=args.ERROR_THRESHOLD,nm_per_pixel=args.NM_PER_PIXEL,max_removel_gray=args.MAX_NOISE_GRAY,z_gap_tolerance=args.Z_GAP_TOLERANCE,max_z_removal_gray=args.MAX_Z_NOISE_GRAY,z_FWHM_error_threshold=args.Z_FWHM_ERROR_THRESHOLD)
    myFiberIdentifier.identifyFiber()













