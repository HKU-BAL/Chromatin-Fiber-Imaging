# Chromatin-Fiber-Imaging
## Introduction
The main aim of this tool is to automatically detect the molecular clusters with user-defined sizes from single-molecule localization microscopy (SMLM) data and provide XY, YZ, and XZ projections for the observation and further filtering for the experts. The corresponding statistical information is also provided for reference, such as FWHM of X and Y dimension, length of Z, average localization precision, average photon count, position, max frame gap, etc. We also recommend to use the web-server version [Click here](http://www.bio8.cs.hku.hk/CFI)


## Dependency
* Numpy
* Pandas
* opencv-python
* LMFIT
* matplotlib

It is recommended to establish the running environment via conda enviromental by the enviroment.yml

![image](https://drive.google.com/uc?export=view&id=1a1wYN44hSXHKOUAY77lPCk4tq-RM3JZK)


## Usage
### Input data 

The input is SMLM data, which should be in CSV format. Each line represents one localized single-molecule. The columns should contains:
| Column name | Description |
| --- | --- |
| x | The pixel coordinate of the x-axis of the single molecule’s centroid.|
|y | The pixel coordinate of the y-axis of the single molecule’s centroid. |
| z | The pixel coordinate of the z-axis of the single molecule’s centroid.|
| X Fitting Error | The X fitting error of the localized single-molecules. |
| Y Fitting Error | The Y fitting error of the localized single-molecules. |
| PhotoCount | Photon count of the single-molecule. |
| Frame Number | The frame index of the localized single-molecule | 


### Parameters

For details of the algorithm, please refer to: [link]().
You may also view the options by:
````python
python SMLMAnalyzer.py --help

````

#### Visualization options

Here, we re-implemented the "average shifted histogram" for image rendering, which is proposed by [ThunderStorm](https://github.com/zitmen/thunderstorm)
| Paramter | Description |
| --- | --- |
|VIS_METHOD| (required) Image rendering method for reconstructing XY, YZ, XZ projection images. The default method is the average shifted histogram, which is proposed by ThunderSTORM.|
| MAGNIFICATION | (optional) Magnification ratio for the positions of pixels. For example, the original coordinate of one single-molecule is \(x,y,z\), and the magnified coordinates id \(x\*r,y\*r,z\*r \) |
| NM_PER_PIXEL | (required) The length of a pixel after doing magnification. |
| LATERAL_SHIFT | (required) The lateral shift of the X, Y and Z-axis. |
| FRAME_RANGE | (optional) Split the input CSV data into multiple frame intervals to avoid large frame gap within a molecular cluster | 

#### Size estimation

The length of the Z-axis of a molecular cluster was estimated based on the number of Z slices. And the X, Y dimensions of the molecular clusters were estimated based on the full width at half maximum (FWHM).

| Paramter | Description |
| --- | --- |
| X_MIN | (required) Minimum X for the target fiber |
| X_MAX | (required) Maximum X for the target fiber | 
| Y_MIN | (required) Minimum Y for the target fiber |
| Y_MAX | (required) Maximum Y for the target fiber |
| Z_MIN | (required) Minimum Z length for the target fiber |
| Z_SLICE | (required) The length of slices on Z-axis | 

### Filtering
This is an optional setting. It enables users to screen unnecessary data. For now, we provide 3 filtering options: 
| Paramter | Description |
| --- | --- |
|MAX_GRAY_RM_THRESHOLD| (required) The pixels with the max gray value of the molecular clusters should bigger than this parameter. Otherwise, the cluster will not be analyzed.|
| LOC_PREC_THRESHOLD | (optional) The localized single-molecules with X or Y fitting errors bigger than this parameter will be discarded. |
| ERROR_THRESHOLD |(optional) Fitting error tolerence for FWHM calcuation. |
|Valid region (X1,X2,Y1,Y2,Z1,Z2) | (optional) Minimum and maximum coordinates of the valid region on X,Y,Z axis |

### Example

* Find fibers with X and Y range from 20nm to 45nm. And their minimum length of Z slices should larger than 100nm

````python

python SMLMAnalyzer.py --DATA_PATH SMLM.csv --SECOND_PER_FRAME 0.0017 --NM_PER_PIXEL 10 --MAGNIFICATION 10.6 --SAVE_PATH save_dir/ --X_MIN 20 --X_MAX 45 --Y_MIN 20 --Y_MAX 45 --Z_MIN 100 --FRAME_RANGE 250

````


## Output

* source_csv/*.csv: localized single-mocule data of the identified fiber
* stat/*_stat.csv: statistics of the identified fibers, including  x,y,z sizes, average photon count, average localization precision, etc.
* big_xy/*_big_xy.png: XY projection of a 100px\*100px region around the identified fiber.
* xy/*_xy.png: XY projection of the identified fiber.
* xz/*_xz.png: XZ projection of the identified fiber.
* yz/*_yz.png: YZ projection of the identified fiber.


