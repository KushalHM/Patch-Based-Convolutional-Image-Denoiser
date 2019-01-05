# Patch-Based-Convolutional-Image-Denoiser
Patch Convolutional Image Denoising - PACAID - algorithm that processes an image, subject to spatial varying noise, through a 22 layers Convolutional Neural Network - CNN - to deliver a cleaner image
Execution Flow
Open PaCaidMain.py and update the parameters - file_path (path to the test images), patchSize, fileSaveExtension
Execute the file - python PaCaidMain.py
Output will be stored in a new folder located at the same base location as the input images. e.g if input folder is ‘/data/test’, then output will be ‘/data/test_cleaned_ps_65’ for a patch size of 65

Filelist
core/ - Contains the core network architecture
extras/ - Code to generate train/test data set by adding spatially varying noise.
weights/ - Stores the trained weight files
data/ - COntains the training and testing image files
PaCaidMain.py - The main code file. Run this to test out the algorithm.
patchifier.py - Contains code to convert image to patches and back.

Required modules
keras
opencv-python
numpy
gc
psutil
