# Chromatin-Fiber-Imaging
## Introduction
The main aim of this tool is to automatically detect the molecular clusters with user-defined sizes from single-molecule localization microscopy (SMLM) data and provide XY, YZ, and XZ projections for the observation and further filtering for the experts. The corresponding statistical information is also provided for reference, such as FWHM of X and Y dimension, length of Z, average localization precision, average photon count, position, max frame gap, etc.  A web server version is also provided: [Click here](http://www.bio8.cs.hku.hk/CFI)

![image](https://drive.google.com/uc?export=view&id=1a1wYN44hSXHKOUAY77lPCk4tq-RM3JZK)

## Input data 


The input is SMLM data, which should be in CSV format. Each line represents one localized single-molecule. The columns should contains:
| column name | description |
| --- | --- |
| x | The pixel coordinate of the x-axis of the single molecule’s centroid.|
|y | The pixel coordinate of the y-axis of the single molecule’s centroid. |
| z | The pixel coordinate of the z-axis of the single molecule’s centroid.|
| Photon count | Photon count of the single-molecule. |
| X fitting error | The X fitting error of the localized molecules. |
| Y fitting error | The Y fitting error of the localized molecules. |

