# DNN model for segmentation of brain tumor from MRI scans by
# Roy Hirsch and Ori Chayoot, 2018, BGU.

The models contains two neural networks that are connected in serial connection.
The first net is a Unet network that detected the whole tumor region.
The second net is Vgg16 based network that classifies the whole tumor region into it sub-classes.
The second network gets it's input from the first network output.

The folder hierarchy:
 - Data - contains the raw BRATS database (placeholder)
 - UnetModel - contains the first network
 - Vgg16Model - contains the second network
 - Utilities - general utilities and data handling functions

Each network contains a netLuncher script which is the main script for running the network (for train of prediction).
Further comments may be found in the different files.

All rights reserved to Roy Hirsch and Ori Chayoot.
Roy Hirsch: roymhirsch@gmail.com
Ori Chayoot: orich@post.bgu.ac.il


