# BrainTumorSegmentation
Final Bsc project, semanting segmentaion of brain tumor from MRI scans using deep learning.

Brain tumor is a state in which cells form in an abnormal way and without supervision.
In order to evaluate the disease’s progression and give suitable treatment a 3D MRI scan is used. In current clinical routine, the resulting images are evaluated based on qualitative criteria manually.
In this project we will developed an optimal algorithm using neural networks for segmenting Glioma tumors from MR images

Using the BRATS2012 dataset (http://braintumorsegmentation.org).
![alt text](https://raw.githubusercontent.com/RoyHirsch/BrainTumorSegmentaion/master/Documents/dataset.png)

The classic image processing model for semantic segmentaion of the whole tumor area:
![alt text](https://raw.githubusercontent.com/RoyHirsch/BrainTumorSegmentaion/master/Documents/classicProject.png)

The deep learning networks architecture we used:
Cascading two neural network, a detection network and a classification network:
![alt text](https://raw.githubusercontent.com/RoyHirsch/BrainTumorSegmentaion/master/Documents/deepLearningProject.png)

The neural network model's results:
![alt text](https://raw.githubusercontent.com/RoyHirsch/BrainTumorSegmentaion/master/Documents/results.png)
