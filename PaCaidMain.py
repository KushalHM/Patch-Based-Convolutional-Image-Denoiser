import core.test_blind_sup as test_sup
from core.models import make_model
import keras.backend.tensorflow_backend
from keras.backend import clear_session
import tensorflow as tf

import cv2, os, time, gc, psutil
import numpy as np

from patchifier import *


file_path = './data/demo'
patchSize = 65
fileSaveExtension = '.jpg'

file_path_cleaned = file_path+'_cleaned_ps_'+str(patchSize)
if not os.path.exists(file_path_cleaned):
	os.makedirs(file_path_cleaned)


programStarTime = time.time()
for path, subdirs, files in os.walk(file_path):
	for fName in files:
		startTime = time.time()
		print ("\n\n\n\nProcessing: ",fName)

		inFilePath = os.path.join(path, fName)		
		filename = inFilePath.split('\\')[-1].split('.')[0]

		outPath = os.path.join(file_path_cleaned, filename+'_cleaned_ps_'+str(patchSize)+fileSaveExtension)

		noisy_image = cv2.imread(inFilePath)
		patches = image2Patch(noisy_image, patchSize, savePatches=False)
		totalPatches = 1 if (len(patches) == 0) else len(patches)
		

		print ("Image shape: ",noisy_image.shape)
		print ("Number of patches: ", totalPatches)
		print ("Patch Size: ", patchSize)

		denoisedPatches = []
		idx = 0
		patchShape0 = patches[0].shape[0]
		patchShape1 = patches[0].shape[1]

		model = make_model(patchShape0, patchShape1)
		model.load_weights('./weights/' + 'blind' +'.hdf5')
		
		for patch in patches:
			if (patch.shape[0] != patchShape0 or patch.shape[1] != patchShape1):
				patchShape0 = patch.shape[0]
				patchShape1 = patch.shape[1]
				model = make_model(patchShape0, patchShape1)
				model.load_weights('./weights/' + 'blind' +'.hdf5')

			gc.collect()
			redPatch = patch[:,:,0]
			greenPatch = patch[:,:,1]
			bluePatch = patch[:,:,2]

			denoisedPatchR = test_sup.denoising(redPatch, model)
			denoisedPatchG = test_sup.denoising(greenPatch, model)
			denoisedPatchB = test_sup.denoising(bluePatch, model)

			denoisedPatch = np.zeros(patch.shape, 'uint8')

			denoisedPatch[:,:, 0] = denoisedPatchR*255
			denoisedPatch[:,:, 1] = denoisedPatchG*255
			denoisedPatch[:,:, 2] = denoisedPatchB*255

			denoisedPatches.append(denoisedPatch)
			
			idx += 1
			if (idx % int(totalPatches/10) == 0):
				print('Processed: {}/{} Time elapsed: {}.'.format(idx,totalPatches,str(time.time()-startTime)))
				print('Memory Details: {}.'.format(psutil.virtual_memory()))

		clear_session()
		if keras.backend.tensorflow_backend._SESSION:
			tf.reset_default_graph()
			keras.backend.tensorflow_backend._SESSION.close()
			keras.backend.tensorflow_backend._SESSION = None
			
		denoisedImg = patches2Image(denoisedPatches, noisy_image)

		cv2.imwrite(outPath, denoisedImg)
		print ("Image saved to: ", outPath)
		print ("Time taken (seconds): ", str(time.time()-startTime))

print ("Thank you for your patience. :)")
print ("All images done. Time Taken: ", str(time.time()-programStarTime))