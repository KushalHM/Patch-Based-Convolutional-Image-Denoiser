import cv2, os
import numpy as np
import time

def image2Patch(img, patchSize, savePatches=True, patchSaveDir='imagePatches',patchBaseName='patch',patchSaveExt='.jpg'):
	width = img.shape[1]
	height = img.shape[0]
	hStride = (width % patchSize)
	vStride = (height % patchSize)
	startTime = time.time()
	
	hStart = 0
	vStart = 0

	hEnd = hStart + patchSize
	vEnd = vStart + patchSize

	idx = 0;
	patches = []
	while (vEnd < height):
		hStart = 0
		hEnd = hStart + patchSize
		vEnd = vStart + patchSize
		while (hEnd < width):
			hEnd = hStart + patchSize
			patch = img[vStart:vEnd,hStart:hEnd,:]
			if (savePatches):
				if not os.path.exists(patchSaveDir):
					os.makedirs(patchSaveDir)
				cv2.imwrite(os.path.join(patchSaveDir, patchBaseName+str(idx)+patchSaveExt),patch)
			hStart += hStride
			
			idx += 1
			if (idx % 5000 == 0):
				print ("Processed: ", idx)
			patches.append(patch)
		vStart +=vStride
		

	print ("All Patches Saved. Count: ",idx)
	print ("Time Taken: ",time.time()-startTime)
	return patches



def patches2Image(patches, baseImg):
	newImage = np.zeros(baseImg.shape)
	patchSizeW = 0
	patchSizeH = 0
	# print (patches.shape)
	# print (patchSize)
	width = baseImg.shape[1]
	height = baseImg.shape[0]

	patchSizeW = patches[0].shape[1]
	patchSizeH = patches[0].shape[0]

	hStride = (width % patchSizeH)
	vStride = (height % patchSizeW)
	
	hStart = 0
	vStart = 0
	hEnd = hStart + patchSizeH
	vEnd = vStart + patchSizeW

	
	idx = 0;
	while (vEnd < height):
		hStart = 0
		hEnd = 0
		vEnd = vStart + patchSizeH
		while (hEnd < width):
			patch = patches[idx]
			patchSizeW = patch.shape[1]
			patchSizeH = patch.shape[0]
			hEnd = hStart + patchSizeW
			newImage[vStart:vEnd, hStart:hEnd,:] = patch
			hStart = hStart + hStride
			idx += 1
		vStart += vStride
	return newImage

