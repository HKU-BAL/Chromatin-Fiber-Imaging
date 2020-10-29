# Chromatin-Fiber-Imaging
## Introduction
The main aim of this tool is to automatically detect the molecular clusters with user-defined sizes from single-molecule localization microscopy (SMLM) data and provide XY, YZ, and XZ projections for the observation and further filtering for the experts. The corresponding statistical information is also provided for reference, such as FWHM of X and Y dimension, length of Z, average localization precision, average photon count, position, max frame gap, etc.  A web server version is also provided: [Click here](http://www.bio8.cs.hku.hk/CFI)

## Dependency
* Numpy
* Pandas
* CV2
* LMfit
* matplotlib

It is recommended to establish the running environment via conda enviromental by the enviroment.yml

![image](https://drive.google.com/uc?export=view&id=1a1wYN44hSXHKOUAY77lPCk4tq-RM3JZK)

## Input data 


The input is SMLM data, which should be in CSV format. Each line represents one localized single-molecule. The columns should contains:
| Column name | Description |
| --- | --- |
| x | The pixel coordinate of the x-axis of the single molecule’s centroid.|
|y | The pixel coordinate of the y-axis of the single molecule’s centroid. |
| z | The pixel coordinate of the z-axis of the single molecule’s centroid.|
| Photon count | Photon count of the single-molecule. |
| X fitting error | The X fitting error of the localized single-molecules. |
| Y fitting error | The Y fitting error of the localized single-molecules. |


## Parameters

For details of the algorithm, please refer to the paper: [link]().

### Visualization options

Here, we re-implemented the "average shifted histogram" for image rendering, which is proposed by [ThunderStorm](https://github.com/zitmen/thunderstorm)
| Paramter | Description |
| --- | --- |
|VIS_METHOD|Image rendering method for reconstructing XY, YZ, XZ projection images. The default method is the average shifted histogram, which is proposed by ThunderSTORM.|
| MAGNIFICATION | Magnification ratio for the positions of pixels. For example, the original coordinate of one single-molecule is \(x,y,z\), and the magnified coordinates id \(x\*r,y\*r,z\*r \) |
| NM_PER_PIXEL | The length of a pixel after doing magnification. |
| LATERAL_SHIFT |The lateral shift of the X and Y-axis. |
| FRAME_RANGE | Split the input CSV data into multiple frame intervals. | 

### size estimation

The length of the Z-axis of a molecular cluster was estimated based on the number of Z slices. And the X, Y dimensions of the molecular clusters were estimated based on the full width at half maximum (FWHM).

| Paramter | Description |
| --- | --- |
| X_MIN | Minimum X for the target fiber |
| X_MAX | Maximum X for the target fiber | 
| Y_MIN | Minimum Y for the target fiber |
| Y_MAX | Maximum Y for the target fiber |
| Z_MIN | Minimum Z length for the target fiber |
| Z_MAX | Maximum Z length for the target fiber |
| Z_SLICE | The length of slices on Z-axis | 

###  



### Example

* Find fibers with X and Y range from 20nm to 45nm. And their minimum length of Z slices should larger than 100nm

````python

python SMLMAnalyzer.py --DATA_PATH SMLM.csv --SECOND_PER_FRAME 0.0017 --NM_PER_PIXEL 10 --MAGNIFICATION 10.6 --SAVE_PATH save_dir/ --X_MIN 20 --X_MAX 45 --Y_MIN 20 --Y_MAX 45 --Z_MIN 100 --FRAME_RANGE 250

```



